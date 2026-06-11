"""Background worker for annotation-task evaluator runs.

Routes the LLM-as-judge work through the calibrate CLI's STT `--eval-only` mode
so the judge logic isn't duplicated between backend and CLI.

Reuses the existing eval-output helpers (`_read_results_csv`, `_read_metrics_json`,
`upload_top_level_files_to_s3`, `build_evaluator_cli_payload`) — nothing here
re-implements parsing or S3 layout from scratch.
"""

from __future__ import annotations

import csv
import datetime
import json
import logging
import os
import subprocess
import tempfile
import threading
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def _utcnow_str() -> str:
    """SQLite CURRENT_TIMESTAMP-style UTC string. Used in details when we need
    to record times in JSON (the row's status column is updated separately)."""
    return datetime.datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")

from db import (
    create_evaluator_runs,
    get_annotation_items_for_task,
    get_annotation_task,
    get_eval_job_items,
    get_evaluator,
    get_evaluator_version,
    get_job,
    snapshot_eval_job_items,
    update_job,
)
from llm_judge import (
    build_evaluator_cli_payload,
    build_evaluator_cli_payload_unrendered,
)
from utils import (
    TaskStatus,
    capture_exception_to_sentry,
    coerce_evaluator_score,
    get_s3_client,
    get_s3_output_config,
    is_job_timed_out,
    kill_process_group,
    register_job_starter,
    try_start_queued_job,
    upload_file_to_s3,
    upload_top_level_files_to_s3,
)


# No-progress watchdog window. Each poll tick samples the output_dir's file
# count + total byte size; the heartbeat fires only when that snapshot has
# changed since the previous tick (i.e. calibrate has actually written
# something to disk). 15 minutes of no on-disk progress → the subprocess is
# considered stuck and SIGTERM/SIGKILL'd.
#
# Why 15 min and not 5: calibrate `--eval-only` may batch disk writes (e.g.
# `results.csv` / `metrics.json` written only after a chunk of items is
# done), and the judge LLM can be slow under load. 15 min gives a real
# stall a clear signal without false-killing legitimate runs that just
# write less frequently than once per tick. A genuinely hung subprocess
# (waiting forever on a stalled HTTP call) still gets killed, just later.
ANNOTATION_EVAL_TIMEOUT_SECONDS = 15 * 60


def _output_dir_snapshot(output_dir: Path) -> Tuple[int, int]:
    """Cheap progress signal: (file_count, total_size_bytes) for everything
    under output_dir. Comparing tuples between ticks tells us whether
    calibrate has written anything since last check."""
    if not output_dir.exists():
        return (0, 0)
    count = 0
    total_size = 0
    for root, _dirs, files in os.walk(output_dir):
        for fname in files:
            count += 1
            try:
                total_size += (Path(root) / fname).stat().st_size
            except OSError:
                # File may have been removed mid-walk; ignore — next tick
                # picks up the right number.
                pass
    return (count, total_size)


# Eval queue types this module participates in (must match `EVAL_JOB_TYPES`
# elsewhere). Importing the constant from utils would create a circular import,
# so we redeclare it here intentionally — see also routers/stt.py and
# job_recovery.py. Keep the three lists in sync.
EVAL_JOB_TYPES = ["stt-eval", "tts-eval", "annotation-eval"]
ANNOTATION_EVAL_JOB_TYPE = "annotation-eval"

# Task types whose annotation rows we know how to evaluate via the CLI's
# --eval-only modes. `tts` is omitted because annotation tasks don't store
# audio S3 keys today; `voice` simulation isn't supported by the CLI in
# eval-only mode. `llm-general` (non-conversational input -> output) uses the
# dedicated `calibrate general` command — see `_build_llm_general_dataset`.
SUPPORTED_EVAL_TASK_TYPES = ("stt", "llm", "llm-general", "conversation")

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Output parsing — generic eval-only output dir (same layout as STT --eval-only)
# ---------------------------------------------------------------------------


def _read_results_csv(output_dir: Path) -> Optional[List[dict]]:
    """One row per dataset entry: id, gt, pred, wer, <evaluator>, <evaluator>_reasoning, ..."""
    p = output_dir / "results.csv"
    if not p.exists():
        return None
    try:
        with open(p, "r", encoding="utf-8") as f:
            return [dict(row) for row in csv.DictReader(f)]
    except Exception as e:
        logger.warning(f"Failed to parse {p}: {e}")
        return None


def _read_metrics_json(output_dir: Path) -> Optional[dict]:
    p = output_dir / "metrics.json"
    if not p.exists():
        return None
    try:
        with open(p, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        logger.warning(f"Failed to parse {p}: {e}")
        return None


# ---------------------------------------------------------------------------
# Resolving evaluators (linked + version-pinned)
# ---------------------------------------------------------------------------


class EvaluatorResolutionError(ValueError):
    pass


def _resolve_evaluator_dicts(
    requested: List[Dict[str, Optional[str]]],
    linked_evaluator_uuids: set[str],
) -> List[Dict[str, Any]]:
    """Combine each (evaluator_id, evaluator_version_id?) pair into a single dict
    that `build_evaluator_cli_payload` understands. Validates that every
    requested evaluator is linked to the task and the version belongs to it."""
    resolved: List[Dict[str, Any]] = []
    for entry in requested:
        evaluator_id = entry.get("evaluator_id")
        if not evaluator_id:
            raise EvaluatorResolutionError("evaluator_id is required")
        if evaluator_id not in linked_evaluator_uuids:
            raise EvaluatorResolutionError(
                f"Evaluator {evaluator_id} is not linked to this task"
            )
        evaluator = get_evaluator(evaluator_id)
        if not evaluator:
            raise EvaluatorResolutionError(f"Evaluator {evaluator_id} not found")
        version_uuid = entry.get("evaluator_version_id") or evaluator.get(
            "live_version_id"
        )
        if not version_uuid:
            raise EvaluatorResolutionError(
                f"Evaluator {evaluator_id} has no live version; pass evaluator_version_id"
            )
        version = get_evaluator_version(version_uuid)
        if not version or version["evaluator_id"] != evaluator_id:
            raise EvaluatorResolutionError(
                f"Version {version_uuid} does not belong to evaluator {evaluator_id}"
            )
        # `output_type`, `kind`, `data_type` are required identity columns on
        # the evaluator row (set at create time, backfilled by migration). A
        # missing value here means the row is malformed — surface it instead
        # of silently miscoercing rating evaluators as "binary", side-by-side
        # as "single", or audio as "text".
        missing = [
            field
            for field in ("output_type", "kind", "data_type")
            if not evaluator.get(field)
        ]
        if missing:
            raise EvaluatorResolutionError(
                f"Evaluator {evaluator_id} is missing required field(s) "
                f"{missing}; check the evaluators table"
            )
        # Shape that build_evaluator_cli_payload + downstream want.
        resolved.append(
            {
                "uuid": evaluator_id,
                "name": evaluator["name"],
                "judge_model": version["judge_model"],
                "system_prompt": version["system_prompt"],
                "output_type": evaluator["output_type"],
                "output_config": version.get("output_config"),
                "variables": version.get("variables"),
                "variable_values": {},
                "kind": evaluator["kind"],
                "data_type": evaluator["data_type"],
                # extra fields for our own bookkeeping (not seen by CLI):
                "_evaluator_version_id": version_uuid,
            }
        )
    _dedupe_evaluator_names(resolved)
    return resolved


def _dedupe_evaluator_names(evaluators: List[Dict[str, Any]]) -> None:
    """Mutate `evaluators` so every `name` is unique within the list.

    Calibrate keys its result columns / JSON entries by evaluator name. If two
    linked evaluators share the same name, calibrate's outputs collapse them
    and our results parser silently attributes everything to one UUID. We
    sidestep this the same way `build_test_evaluators_payload` does for the
    regular LLM-tests flow: when a name collides, suffix `-{uuid8}`.

    Mutates in place — downstream consumers (dataset builder, calibrate
    payload builder, results parser, stored job details) all see the same
    deduped name. In the common case where names are already unique this is
    a no-op.
    """
    used: set = set()
    for ev in evaluators:
        base = ev.get("name") or ev["uuid"]
        name = base
        if name in used:
            name = f"{base}-{ev['uuid'][:8]}"
            # Extreme edge case: even the suffixed form collides (two evaluators
            # with both same name AND same first-8 uuid chars). Append more.
            extra = 9
            while name in used and extra <= len(ev["uuid"]):
                name = f"{base}-{ev['uuid'][:extra]}"
                extra += 1
            logger.warning(
                f"[annotation-eval] evaluator name collision on {base!r}; "
                f"renaming evaluator {ev['uuid']} → {name!r} for this run"
            )
        used.add(name)
        ev["name"] = name


# ---------------------------------------------------------------------------
# Dataset construction (STT --eval-only shape)
# ---------------------------------------------------------------------------


class DatasetBuildError(ValueError):
    pass


# ---------------------------------------------------------------------------
# Per-task-type dataset builders. The annotation_item.uuid is always the row
# identifier so output rows map back unambiguously.
# ---------------------------------------------------------------------------


def _payload_dict(item: Dict[str, Any]) -> Dict[str, Any]:
    payload = item.get("payload")
    if not isinstance(payload, dict):
        raise DatasetBuildError(
            f"Item {item['uuid']}: payload must be a JSON object"
        )
    return payload


def _build_stt_dataset(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """STT --eval-only: [{id, gt, pred}]."""
    out: List[Dict[str, Any]] = []
    for it in items:
        payload = _payload_dict(it)
        pred = payload.get("predicted_transcript")
        gt = payload.get("reference_transcript")
        if pred is None or gt is None:
            raise DatasetBuildError(
                f"Item {it['uuid']}: STT items need "
                "`predicted_transcript` and `reference_transcript` in payload"
            )
        out.append({"id": it["uuid"], "gt": str(gt), "pred": str(pred)})
    return out


def _build_llm_dataset(
    items: List[Dict[str, Any]], evaluators_resolved: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """LLM --eval-only: [{test_case: {id, history, evaluation}, output: {response, tool_calls}}].

    Annotation convention for payload (response-mode only — tool-call eval is
    not exposed via this endpoint today):
      { "chat_history": [...], "agent_response": "...", "tool_calls"?: [...],
        "evaluator_variables"?: { "<evaluator_uuid>": { "<var>": <value>, ... } } }

    `evaluator_variables` lets each item supply per-evaluator `{{variable}}`
    values for evaluators whose prompts contain placeholders. Values are
    threaded into each test case's `evaluation.criteria[].arguments` so
    calibrate substitutes them at judge time. Missing entries → no
    arguments → calibrate falls back to the placeholder (or its declared
    default if any).
    """
    _require_evaluators(evaluators_resolved, "LLM --eval-only")
    out: List[Dict[str, Any]] = []
    for it in items:
        payload = _payload_dict(it)
        history = payload.get("chat_history")
        response = payload.get("agent_response")
        if not isinstance(history, list) or response is None:
            raise DatasetBuildError(
                f"Item {it['uuid']}: LLM items need `chat_history` (list) and "
                "`agent_response` in payload"
            )
        tool_calls = payload.get("tool_calls", [])
        if not isinstance(tool_calls, list):
            raise DatasetBuildError(
                f"Item {it['uuid']}: `tool_calls` must be a list if provided"
            )
        criteria_refs = _criteria_refs_for_item(it, payload, evaluators_resolved)
        out.append(
            {
                "test_case": {
                    "id": it["uuid"],
                    "history": history,
                    "evaluation": {
                        "type": "response",
                        "criteria": criteria_refs,
                    },
                },
                "output": {"response": str(response), "tool_calls": tool_calls},
            }
        )
    return out


def _require_evaluators(evaluators_resolved: List[Dict[str, Any]], what: str) -> None:
    """Guard for builders whose CLI mode judges against evaluators — raise if
    none were resolved. `what` names the flow for the error message."""
    if not evaluators_resolved:
        raise DatasetBuildError(
            f"{what} requires at least one evaluator (criteria)"
        )


def _validated_evaluator_variables(
    it: Dict[str, Any], payload: Dict[str, Any]
) -> Dict[str, Any]:
    """Extract + validate an item's optional `evaluator_variables` map
    (`{evaluator_uuid: {var: value}}`). Shared by the `llm` and `llm-general`
    dataset builders, which both resolve it per-evaluator via
    `_criteria_refs_for_item` (the `llm` path threads them as
    `criteria[].arguments`; `llm-general` reshapes them into a per-row
    `arguments` object keyed by evaluator name)."""
    per_evaluator_vars = payload.get("evaluator_variables") or {}
    if not isinstance(per_evaluator_vars, dict):
        raise DatasetBuildError(
            f"Item {it['uuid']}: `evaluator_variables` must be a dict "
            "keyed by evaluator UUID"
        )
    return per_evaluator_vars


def _criteria_refs_for_item(
    it: Dict[str, Any],
    payload: Dict[str, Any],
    evaluators_resolved: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Build the per-test `evaluation.criteria` refs from an item's optional
    `evaluator_variables` map. Each ref is `{name, arguments?}`; missing entries
    → no arguments → calibrate falls back to the prompt placeholder/default.
    Used by the `llm` dataset builder."""
    per_evaluator_vars = _validated_evaluator_variables(it, payload)
    criteria_refs: List[Dict[str, Any]] = []
    for ev in evaluators_resolved:
        ref: Dict[str, Any] = {"name": ev["name"]}
        args = per_evaluator_vars.get(ev["uuid"]) or {}
        if args:
            ref["arguments"] = args
        criteria_refs.append(ref)
    return criteria_refs


def _build_llm_general_dataset(
    items: List[Dict[str, Any]], evaluators_resolved: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """`calibrate general` dataset: a flat list of non-conversational
    `{id, input, output, arguments?}` rows. No agent and no conversation —
    `calibrate general` grades each pre-existing input/output pair against the
    config's evaluators (see `calibrate general --dataset … -c …`).

    Annotation convention for payload:
      { "input": "...", "output": "...",
        "evaluator_variables"?: { "<evaluator_uuid>": { "<var>": <value>, ... } } }

    `evaluator_variables` uses the SAME payload contract as the `llm` task type,
    and is resolved per-evaluator the SAME way (`_criteria_refs_for_item`).
    `calibrate general` takes those as a per-row `arguments` object keyed by
    evaluator **name** — it renders each evaluator's prompt against
    `arguments[ev_name]`, exactly mirroring `llm`'s per-criteria `arguments`
    (no shared bag, so two evaluators using the same `{{var}}` never collide).
    Omitted/empty → no `arguments` key.
    """
    _require_evaluators(evaluators_resolved, "general eval")
    out: List[Dict[str, Any]] = []
    for it in items:
        payload = _payload_dict(it)
        input_text = payload.get("input")
        output_text = payload.get("output")
        if input_text is None or output_text is None:
            raise DatasetBuildError(
                f"Item {it['uuid']}: llm-general items need `input` and "
                "`output` in payload"
            )
        # Identical per-evaluator resolution to the `llm` builder; just reshape
        # the criteria refs into general's name-keyed `arguments` map.
        criteria_refs = _criteria_refs_for_item(it, payload, evaluators_resolved)
        arguments = {
            ref["name"]: ref["arguments"]
            for ref in criteria_refs
            if "arguments" in ref
        }
        row: Dict[str, Any] = {
            "id": it["uuid"],
            "input": str(input_text),
            "output": str(output_text),
        }
        if arguments:
            row["arguments"] = arguments
        out.append(row)
    return out


def _build_simulation_dataset(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Text-simulation --eval-only: [{name, conversation_history}].

    `name` is metadata only (per the calibrate spec — duplicate / missing
    names are safe; calibrate uses its own internal `row_<i>` ids for output
    directories). We pass the annotation_item.uuid as the name so it shows
    up in the per-simulation output for human inspection, but the parser
    maps results back via dataset_map.json + dataset order, NOT via name.
    """
    out: List[Dict[str, Any]] = []
    for it in items:
        payload = _payload_dict(it)
        transcript = payload.get("transcript")
        if not isinstance(transcript, list) or not transcript:
            raise DatasetBuildError(
                f"Item {it['uuid']}: simulation items need a non-empty "
                "`transcript` (list of {role, content}) in payload"
            )
        out.append({"name": it["uuid"], "conversation_history": transcript})
    return out


def build_dataset_for_task_type(
    task_type: str,
    items: List[Dict[str, Any]],
    evaluators_resolved: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    if task_type == "stt":
        return _build_stt_dataset(items)
    if task_type == "llm":
        return _build_llm_dataset(items, evaluators_resolved)
    if task_type == "llm-general":
        return _build_llm_general_dataset(items, evaluators_resolved)
    if task_type == "conversation":
        return _build_simulation_dataset(items)
    raise DatasetBuildError(
        f"Evaluator runs are not supported for task type {task_type!r} "
        "(supported: stt, llm, llm-general, conversation)"
    )


def calibrate_command_for_task_type(
    task_type: str, dataset_path: Path, output_dir: Path, config_path: Path
) -> List[str]:
    if task_type == "stt":
        return [
            "calibrate", "stt", "--eval-only",
            "--dataset", str(dataset_path),
            "-o", str(output_dir),
            "--config", str(config_path),
        ]
    if task_type == "llm":
        return [
            "calibrate", "llm",
            "-c", str(config_path),
            "--eval-only",
            "--dataset", str(dataset_path),
            "-o", str(output_dir),
        ]
    if task_type == "llm-general":
        # Dedicated non-conversational `input -> output` judge. No agent, no
        # conversation, no `--eval-only` flag — `calibrate general` only ever
        # grades pre-existing pairs.
        return [
            "calibrate", "general",
            "--dataset", str(dataset_path),
            "-c", str(config_path),
            "-o", str(output_dir),
        ]
    if task_type == "conversation":
        return [
            "calibrate", "simulations",
            "-t", "text",
            "-c", str(config_path),
            "--eval-only",
            "--dataset", str(dataset_path),
            "-o", str(output_dir),
        ]
    raise DatasetBuildError(f"Unsupported task type: {task_type!r}")


# ---------------------------------------------------------------------------
# Results -> evaluator_runs mapping
# ---------------------------------------------------------------------------


def _read_config_evaluators_map(output_dir: Path) -> Dict[str, str]:
    """Read calibrate's `config.json` and invert `evaluators_map` so we can
    go from evaluator-name-in-results → evaluator UUID we sent. This is the
    canonical authority for the (name → uuid) mapping; the runner never
    matches results by name alone."""
    p = output_dir / "config.json"
    if not p.exists():
        return {}
    try:
        with open(p, "r", encoding="utf-8") as f:
            cfg = json.load(f)
    except Exception as e:
        logger.warning(f"Failed to parse {p}: {e}")
        return {}
    raw = cfg.get("evaluators_map") or {}
    # `evaluators_map` is `{evaluator_id: name}` per the codebase convention;
    # invert to {name: evaluator_id}.
    out: Dict[str, str] = {}
    for k, v in raw.items():
        if isinstance(k, str) and isinstance(v, str):
            out[v] = k
    return out


# `_coerce_score` previously lived here as a private helper. Moved to
# `utils.coerce_evaluator_score` so the API-facing post-processor can
# share the same implementation — keeps binary "1.0"/"0.0" handling
# consistent across persistence and read paths. Local alias retained
# so the call sites below stay readable.
_coerce_score = coerce_evaluator_score


def _row_evaluator_value(
    row: Dict[str, Any], column_name: str, output_type: str
) -> Optional[Dict[str, Any]]:
    """Pick one evaluator's score + reasoning out of a results.csv row by its
    calibrate-side column name. The caller is responsible for resolving the
    column name via the config.json `evaluators_map` so we don't accidentally
    read a built-in row column (id, gt, pred, wer, …) when an evaluator
    happens to share its name."""
    if column_name not in row:
        return None
    raw_score = row[column_name]
    reasoning = row.get(f"{column_name}_reasoning")
    if raw_score in (None, ""):
        return None
    out: Dict[str, Any] = {"value": _coerce_score(raw_score, output_type)}
    if reasoning:
        out["reasoning"] = reasoning
    return out


def _parse_results_stt(
    output_dir: Path,
    evaluators_resolved: List[Dict[str, Any]],
    job_uuid: str,
) -> List[Dict[str, Any]]:
    """STT outputs: <output_dir>/results.csv with columns id, gt, pred, wer,
    <column_name>, <column_name>_reasoning per evaluator.
    Maps row.id → annotation_item.uuid (we set id = item.uuid on the way out).

    Iteration is driven by the `evaluators_map` calibrate writes into
    `config.json` (`{column_name -> evaluator_uuid}`). That map is the only
    string we *know* identifies an evaluator output column rather than a
    built-in CSV column — without it, an evaluator named `wer` (or any other
    reserved column name) would silently lift the built-in WER value into
    `evaluator_runs.value` and persist it to the DB. Falls back to the
    snapshot's display name only when the map is missing (legacy pre-map
    runs); that fallback is collision-prone but preserved for back-compat.
    """
    rows = _read_results_csv(output_dir) or []
    name_to_uuid_via_config = _read_config_evaluators_map(output_dir)
    snapshot_by_uuid = {ev["uuid"]: ev for ev in evaluators_resolved}

    if name_to_uuid_via_config:
        # Authoritative: iterate calibrate's own column→uuid record.
        lift_pairs: List[Tuple[str, Optional[Dict[str, Any]]]] = [
            (col, snapshot_by_uuid.get(uid))
            for col, uid in name_to_uuid_via_config.items()
        ]
    else:
        # Back-compat fallback: legacy data without an evaluators_map.
        # Vulnerable to reserved-column / duplicate-name collisions.
        logger.warning(
            f"[annotation-eval] no evaluators_map in {output_dir}/config.json; "
            "falling back to display-name lookup (collision-prone)"
        )
        lift_pairs = [(ev["name"], ev) for ev in evaluators_resolved]

    runs: List[Dict[str, Any]] = []
    for column_name, ev in lift_pairs:
        if not ev:
            # Map referenced an evaluator UUID we don't have a snapshot for
            # — odd, but skip rather than crash; nothing reproducible we can
            # do here without the snapshot's output_type / version_id.
            logger.warning(
                f"[annotation-eval] evaluators_map column {column_name!r} "
                "has no matching snapshot; skipping"
            )
            continue
        evaluator_id = ev["uuid"]
        version_uuid = ev["_evaluator_version_id"]
        output_type = ev["output_type"]
        for row in rows:
            item_id = row.get("id")
            if not item_id:
                continue
            value = _row_evaluator_value(row, column_name, output_type)
            runs.append(
                {
                    "job_id": job_uuid,
                    "item_id": item_id,
                    "evaluator_id": evaluator_id,
                    "evaluator_version_id": version_uuid,
                    "value": value,
                    "status": "completed" if value is not None else "failed",
                }
            )
    return runs


def _parse_results_general(
    output_dir: Path,
    evaluators_resolved: List[Dict[str, Any]],
    job_uuid: str,
) -> List[Dict[str, Any]]:
    """`calibrate general` writes the same `results.csv` shape as STT: an `id`
    column plus a `<name>`/`<name>_reasoning` pair per evaluator, with the
    column→uuid mapping recorded in `config.json`'s `evaluators_map`. The only
    difference is the built-in columns (`input, output` instead of `gt, pred,
    wer`), which the column-map-driven parser ignores — so the STT CSV parser
    handles general results verbatim."""
    return _parse_results_stt(output_dir, evaluators_resolved, job_uuid)


def _parse_results_llm(
    output_dir: Path,
    evaluators_resolved: List[Dict[str, Any]],
    job_uuid: str,
) -> List[Dict[str, Any]]:
    """LLM outputs: <output_dir>/results.json — list of per-test-case results.
    Each entry is `{output, metrics, test_case, test_case_id?}`; per-row
    judgements live in `metrics.judge_results = {<name>: {reasoning, match|score}}`
    (`match` for binary, `score` for rating). Tool-call evaluation is not
    surfaced into evaluator_runs (this endpoint only runs response-mode
    criteria evaluators)."""
    p = output_dir / "results.json"
    if not p.exists():
        logger.warning(f"[annotation-eval] missing {p}")
        return []
    try:
        with open(p, "r", encoding="utf-8") as f:
            results = json.load(f)
    except Exception as e:
        logger.warning(f"[annotation-eval] failed to parse {p}: {e}")
        return []
    name_to_uuid_via_config = _read_config_evaluators_map(output_dir)
    by_name: Dict[str, Dict[str, Any]] = {ev["name"]: ev for ev in evaluators_resolved}

    runs: List[Dict[str, Any]] = []
    for entry in results if isinstance(results, list) else []:
        if not isinstance(entry, dict):
            continue
        item_id = (
            entry.get("test_case_id")
            or (entry.get("test_case") or {}).get("id")
        )
        if not item_id:
            continue
        metrics = entry.get("metrics") or {}
        judge_results = (
            metrics.get("judge_results") if isinstance(metrics, dict) else None
        )
        if not isinstance(judge_results, dict):
            continue
        for ev_name, judgement in judge_results.items():
            ev = by_name.get(ev_name)
            if not ev:
                continue
            evaluator_id = name_to_uuid_via_config.get(ev_name, ev["uuid"])
            if evaluator_id != ev["uuid"]:
                logger.warning(
                    f"[annotation-eval] evaluators_map round-trip mismatch for "
                    f"{ev_name!r}: sent {ev['uuid']}, got {evaluator_id}"
                )
            output_type = ev["output_type"]
            value: Optional[Dict[str, Any]] = None
            if isinstance(judgement, dict):
                # Calibrate emits `match` for binary, `score` for rating.
                raw = judgement.get("match")
                if raw is None:
                    raw = judgement.get("score")
                if raw is not None:
                    value = {"value": _coerce_score(raw, output_type)}
                    reasoning = judgement.get("reasoning")
                    if reasoning:
                        value["reasoning"] = reasoning
            runs.append(
                {
                    "job_id": job_uuid,
                    "item_id": item_id,
                    "evaluator_id": evaluator_id,
                    "evaluator_version_id": ev["_evaluator_version_id"],
                    "value": value,
                    "status": "completed" if value is not None else "failed",
                }
            )
    return runs


def _read_simulation_dataset_map(output_dir: Path) -> Dict[str, int]:
    """Read calibrate's `dataset_map.json` for the simulation flow.

    Returns `{row_id: 0-based-index}`. Calibrate writes one entry per input
    dataset row. We use the index to map calibrate's internal `row_<i>`
    directory back to the annotation_item we sent at that position.
    """
    p = output_dir / "dataset_map.json"
    if not p.exists():
        return {}
    try:
        with open(p, "r", encoding="utf-8") as f:
            raw = json.load(f)
    except Exception as e:
        logger.warning(f"[annotation-eval] failed to parse {p}: {e}")
        return {}
    out: Dict[str, int] = {}
    for row_id, entry in (raw or {}).items():
        if not isinstance(entry, dict):
            continue
        idx = entry.get("index")
        if isinstance(idx, int):
            out[row_id] = idx
    return out


def _parse_results_simulation(
    output_dir: Path,
    evaluators_resolved: List[Dict[str, Any]],
    job_uuid: str,
    items: Optional[List[Dict[str, Any]]] = None,
) -> List[Dict[str, Any]]:
    """Simulation outputs: per-simulation
    `<output_dir>/<row_id>/evaluation_results.csv` with one row per evaluator
    (`name, type, value, reasoning, evaluator_id?, scale_min?, scale_max?`).

    Calibrate uses internal `row_<1-based-index>` ids for the subdirectories
    (per the eval-only spec — the `name` we send is metadata only). We map
    `row_id → 0-based-index → items[index].uuid` via `dataset_map.json` plus
    the items list we built the dataset from. Falls back to `dataset_map`'s
    `name` field (where we stashed the item uuid) if `items` isn't passed.
    """
    name_to_uuid_via_config = _read_config_evaluators_map(output_dir)
    by_name: Dict[str, Dict[str, Any]] = {ev["name"]: ev for ev in evaluators_resolved}
    row_id_to_index = _read_simulation_dataset_map(output_dir)

    def _resolve_item_id(row_id: str) -> Optional[str]:
        idx = row_id_to_index.get(row_id)
        if idx is None:
            return None
        if items is not None and 0 <= idx < len(items):
            return items[idx]["uuid"]
        # Fallback: read the `name` field calibrate echoed into dataset_map
        # (we set it to the item uuid in `_build_simulation_dataset`).
        try:
            with open(output_dir / "dataset_map.json", "r", encoding="utf-8") as f:
                raw = json.load(f)
            entry = (raw or {}).get(row_id) or {}
            name = entry.get("name")
            return name if isinstance(name, str) else None
        except Exception:
            return None

    runs: List[Dict[str, Any]] = []
    if not output_dir.exists():
        return runs
    for sim_dir in sorted(p for p in output_dir.iterdir() if p.is_dir()):
        item_id = _resolve_item_id(sim_dir.name)
        if not item_id:
            logger.warning(
                f"[annotation-eval] no dataset_map entry for {sim_dir.name}; "
                "skipping its evaluation_results.csv"
            )
            continue
        csv_path = sim_dir / "evaluation_results.csv"
        if not csv_path.exists():
            continue
        try:
            with open(csv_path, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    ev_name = row.get("name") or ""
                    ev = by_name.get(ev_name)
                    if not ev:
                        continue
                    # Prefer the per-row `evaluator_id` from the CSV (calibrate
                    # echoes the id we sent); fall back to config.json's map;
                    # last-resort the local resolved record.
                    evaluator_id = (
                        row.get("evaluator_id")
                        or name_to_uuid_via_config.get(ev_name)
                        or ev["uuid"]
                    )
                    if evaluator_id != ev["uuid"]:
                        logger.warning(
                            f"[annotation-eval] evaluator id mismatch for "
                            f"{ev_name!r}: sent {ev['uuid']}, got {evaluator_id}"
                        )
                    output_type = ev["output_type"]
                    raw = row.get("value")
                    if raw in (None, ""):
                        value: Optional[Dict[str, Any]] = None
                    else:
                        value = {"value": _coerce_score(raw, output_type)}
                        reasoning = row.get("reasoning")
                        if reasoning:
                            value["reasoning"] = reasoning
                    runs.append(
                        {
                            "job_id": job_uuid,
                            "item_id": item_id,
                            "evaluator_id": evaluator_id,
                            "evaluator_version_id": ev["_evaluator_version_id"],
                            "value": value,
                            "status": "completed" if value is not None else "failed",
                        }
                    )
        except Exception as e:
            logger.warning(f"[annotation-eval] failed to parse {csv_path}: {e}")
    return runs


def parse_results_for_task_type(
    task_type: str,
    output_dir: Path,
    evaluators_resolved: List[Dict[str, Any]],
    job_uuid: str,
    items: Optional[List[Dict[str, Any]]] = None,
) -> List[Dict[str, Any]]:
    """Dispatch per task type. `items` (in dataset order) is required for
    simulation — calibrate uses internal `row_<i>` ids and we map back to
    annotation_item.uuid via `dataset_map.json` + this list."""
    if task_type == "stt":
        return _parse_results_stt(output_dir, evaluators_resolved, job_uuid)
    if task_type == "llm":
        return _parse_results_llm(output_dir, evaluators_resolved, job_uuid)
    if task_type == "llm-general":
        return _parse_results_general(output_dir, evaluators_resolved, job_uuid)
    if task_type == "conversation":
        return _parse_results_simulation(
            output_dir, evaluators_resolved, job_uuid, items=items
        )
    return []


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


class AnnotationEvalTimeoutError(RuntimeError):
    """Raised when the polling loop's `updated_at` watchdog fires. The outer
    handler treats this like any other failure (mark job FAILED, drain queue,
    upload partials)."""


def _run_calibrate_eval_only(
    cmd: List[str],
    cwd: Path,
    log_dir: Path,
    on_started: Optional[Any] = None,
    heartbeat_seconds: int = 2,
    job_uuid: Optional[str] = None,
    timeout_seconds: int = ANNOTATION_EVAL_TIMEOUT_SECONDS,
) -> Tuple[int, str, str]:
    """Spawn the calibrate subprocess; redirect stdout/stderr to disk to avoid
    pipe-buffer deadlocks; poll until done; return (returncode, stdout, stderr).

    If `job_uuid` is provided, each tick:
      1. Snapshots `log_dir` (the calibrate output dir) — file count and
         total byte size. If the snapshot changed since the last tick,
         calibrate has written new bytes → heartbeat `update_job(job_uuid)`
         to bump `updated_at`.
      2. Re-reads the job's `updated_at` and raises
         `AnnotationEvalTimeoutError` if it hasn't advanced in
         `timeout_seconds`. With (1) in place, this fires only when the
         disk has been silent for the full window — i.e. calibrate is
         genuinely stuck. Before raising, the process group is SIGTERM/
         SIGKILL'd via the standard helper.
    """
    stdout_path = log_dir / "stdout.log"
    stderr_path = log_dir / "stderr.log"
    with open(stdout_path, "w") as out_f, open(stderr_path, "w") as err_f:
        proc = subprocess.Popen(
            cmd,
            stdout=out_f,
            stderr=err_f,
            text=True,
            start_new_session=True,
            cwd=str(cwd),
        )
        # Notify caller of pid/pgid so it can be persisted (used for recovery
        # to kill an orphaned process if the backend restarts mid-run).
        if on_started is not None:
            try:
                on_started(proc.pid)
            except Exception as e:
                logger.warning(f"on_started callback raised: {e}")
        last_snapshot: Tuple[int, int] = _output_dir_snapshot(log_dir)
        while proc.poll() is None:
            time.sleep(heartbeat_seconds)
            if job_uuid is None:
                continue
            # Heartbeat ONLY when calibrate has written new bytes since the
            # last tick. A subprocess that's hung (no disk activity) lets
            # `updated_at` age until the timeout window below kills it.
            if proc.poll() is None:
                current_snapshot = _output_dir_snapshot(log_dir)
                if current_snapshot != last_snapshot:
                    last_snapshot = current_snapshot
                    try:
                        update_job(job_uuid)
                    except Exception as e:
                        # DB blip — log but don't fail the run; the timeout
                        # check below catches sustained DB issues.
                        logger.warning(
                            f"[annotation-eval] heartbeat update_job failed "
                            f"for {job_uuid}: {e}"
                        )
            job = get_job(job_uuid)
            updated_at = job.get("updated_at") if job else None
            if updated_at and is_job_timed_out(
                updated_at, timeout_seconds=timeout_seconds
            ):
                logger.warning(
                    f"[annotation-eval] job {job_uuid} timed out "
                    f"(no progress for {timeout_seconds}s); killing pgid {proc.pid}"
                )
                kill_process_group(proc.pid, job_uuid)
                # Wait briefly for the kill to propagate before reading logs.
                try:
                    proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    pass
                raise AnnotationEvalTimeoutError(
                    f"calibrate --eval-only timed out after {timeout_seconds}s "
                    "with no progress"
                )
    try:
        with open(stdout_path, "r") as f:
            stdout = f.read()
    except Exception:
        stdout = ""
    try:
        with open(stderr_path, "r") as f:
            stderr = f.read()
    except Exception:
        stderr = ""
    return proc.returncode, stdout, stderr


def _extract_calibrate_error(stdout: str, stderr: str) -> str:
    """Pick the most useful error string out of calibrate's outputs.

    calibrate eval-only emits `{"status": "error", "error": "..."}` to stdout
    on validation failures (per the CLI contract). Fall back to the last
    non-empty line of stderr, then stdout, then a generic message.
    """
    for blob in (stdout, stderr):
        for line in reversed(blob.splitlines()):
            line = line.strip()
            if not line:
                continue
            try:
                parsed = json.loads(line)
            except (TypeError, ValueError):
                continue
            if (
                isinstance(parsed, dict)
                and parsed.get("status") == "error"
                and parsed.get("error")
            ):
                return str(parsed["error"])
            break  # only check the last line for JSON
    if stderr.strip():
        return stderr.strip().splitlines()[-1]
    if stdout.strip():
        return stdout.strip().splitlines()[-1]
    return "calibrate exited non-zero with no diagnostic output"


def _persist_pgid(job_uuid: str, pid: int) -> None:
    """Write pid/pgid into the job's details so recovery can kill the
    orphaned subprocess if the backend dies mid-run. Uses update_job's merge
    semantics so existing keys (evaluators, item_ids, ...) survive.

    `os.getpgid(pid)` resolves the actual process-group id rather than
    assuming pid==pgid (which is true today only because the subprocess is
    spawned with `start_new_session=True`)."""
    try:
        pgid = os.getpgid(pid)
    except (ProcessLookupError, OSError):
        # Process already exited / pgid lookup failed — fall back to pid so
        # recovery still has *something* to try.
        pgid = pid
    update_job(job_uuid, details={"pid": pid, "pgid": pgid})


def _try_upload_partial_outputs(
    output_dir: Optional[Path], task_uuid: str, job_uuid: str
) -> Optional[str]:
    """Best-effort: upload whatever the calibrate CLI managed to write before
    failing (logs, partial config). Mirrors the STT failure path. Returns the
    s3 prefix on success, None on failure (and never raises)."""
    if output_dir is None or not output_dir.exists():
        return None
    try:
        s3_bucket = get_s3_output_config()
        s3_prefix = (
            f"annotation-tasks/{task_uuid}/evaluator-runs/{job_uuid}/outputs"
        )
        s3 = get_s3_client()
        for root, _dirs, files in os.walk(output_dir):
            for fname in files:
                local_path = Path(root) / fname
                rel = local_path.relative_to(output_dir)
                upload_file_to_s3(s3, local_path, s3_bucket, f"{s3_prefix}/{rel}")
        return s3_prefix
    except Exception as e:
        logger.warning(f"[annotation-eval] partial upload failed: {e}")
        return None


def _run_job(
    job_uuid: str,
    task_uuid: str,
    user_id: str,
    evaluators_resolved: List[Dict[str, Any]],
    item_ids: Optional[List[str]] = None,
):
    """Synchronous worker — runs in a background thread.

    `item_ids=None` means "run on every item in the task." A list filters to
    that subset (UUIDs already validated upstream); soft-deleted UUIDs are
    silently skipped here so a recovered run after a delete still progresses.
    """
    logger.info(f"[annotation-eval] starting job {job_uuid} for task {task_uuid}")
    output_dir_for_partial: Optional[Path] = None
    try:
        task = get_annotation_task(task_uuid)
        if not task:
            raise RuntimeError(f"Task {task_uuid} disappeared")

        # Read the snapshot the submission endpoint wrote into
        # `annotation_eval_job_items`. This is the exact item set + payloads
        # the user submitted, frozen against subsequent edits / soft-deletes
        # on `annotation_items`. Order is preserved by the snapshot's
        # insertion order so calibrate sees the same sequence on retry.
        snapshot_items = get_eval_job_items(job_uuid)
        if snapshot_items:
            items = snapshot_items
        else:
            # Backwards-compat: jobs created before snapshotting was added
            # have no rows in `annotation_eval_job_items`. Fall back to the
            # live list (old behavior) and write the snapshot now so a later
            # recovery / re-read goes through the snapshot path. This
            # branch is also the only path for any in-flight job that
            # spans the deploy of this change.
            all_items = get_annotation_items_for_task(task_uuid)
            if not all_items:
                raise RuntimeError("Task has no items")
            if item_ids is None:
                items = all_items
            else:
                by_id = {it["uuid"]: it for it in all_items}
                items = [by_id[i] for i in item_ids if i in by_id]
                if not items:
                    raise RuntimeError(
                        "None of the requested item_ids are still live on this task"
                    )
            try:
                snapshot_eval_job_items(job_uuid, items)
            except Exception as e:
                # Snapshotting is a perf optimisation for next-read; a
                # failure here shouldn't block the run.
                logger.warning(
                    f"[annotation-eval] backfill snapshot failed for "
                    f"{job_uuid}: {e}"
                )

        task_type = task.get("type", "stt")

        with tempfile.TemporaryDirectory() as temp_dir:
            tmp = Path(temp_dir)
            input_dir = tmp / "input"
            input_dir.mkdir()
            output_dir = tmp / "output"
            output_dir.mkdir()
            output_dir_for_partial = output_dir

            # 1. Dataset.
            dataset = build_dataset_for_task_type(
                task_type, items, evaluators_resolved
            )
            dataset_path = input_dir / "dataset.json"
            with open(dataset_path, "w", encoding="utf-8") as f:
                json.dump(dataset, f, ensure_ascii=False)

            # 2. Config (evaluators only). For LLM + llm-general tasks, leave
            # {{variable}} placeholders unrendered so calibrate substitutes them
            # per row from each item's `evaluator_variables` (llm → per-test
            # `criteria[].arguments`; general → a per-row `arguments` object
            # keyed by evaluator name).
            # STT/simulation flows have no per-row arguments mechanism, so we
            # pre-render the evaluator's own `variable_values` (typically empty
            # in the annotation flow).
            if task_type in ("llm", "llm-general"):
                evaluator_payload = build_evaluator_cli_payload_unrendered(
                    evaluators_resolved
                )
            else:
                evaluator_payload = build_evaluator_cli_payload(
                    evaluators_resolved
                )
            config_path = input_dir / "config.json"
            with open(config_path, "w", encoding="utf-8") as f:
                json.dump({"evaluators": evaluator_payload}, f, ensure_ascii=False)

            # 3. Spawn the right calibrate subcommand for this task type.
            cmd = calibrate_command_for_task_type(
                task_type, dataset_path, output_dir, config_path
            )
            logger.info(f"[annotation-eval] spawning: {' '.join(cmd)}")

            rc, proc_stdout, proc_stderr = _run_calibrate_eval_only(
                cmd,
                cwd=tmp,
                log_dir=output_dir,
                on_started=lambda pid: _persist_pgid(job_uuid, pid),
                job_uuid=job_uuid,
            )
            if rc != 0:
                logger.error(
                    f"[annotation-eval] calibrate exited rc={rc}\nstderr:\n{proc_stderr}"
                )
                raise subprocess.CalledProcessError(
                    rc, cmd, output=proc_stdout, stderr=proc_stderr
                )

            # 4. Parse outputs (per task type) into evaluator_runs rows. The
            # mapping name → evaluator_id is verified against the CLI's
            # `config.json` evaluators_map so the stored evaluator_id is
            # authoritative — name/description on the evaluator can change
            # later and the runs still resolve correctly.
            runs_to_insert = parse_results_for_task_type(
                task_type, output_dir, evaluators_resolved, job_uuid, items=items
            )
            metrics = _read_metrics_json(output_dir)
            if runs_to_insert:
                create_evaluator_runs(runs_to_insert)

            # 6. Upload all artifacts to S3 (mirrors normal STT eval layout).
            s3_bucket = get_s3_output_config()
            s3_prefix = (
                f"annotation-tasks/{task_uuid}/evaluator-runs/{job_uuid}/outputs"
            )
            s3 = get_s3_client()
            for root, _dirs, files in os.walk(output_dir):
                for fname in files:
                    local_path = Path(root) / fname
                    rel = local_path.relative_to(output_dir)
                    upload_file_to_s3(
                        s3, local_path, s3_bucket, f"{s3_prefix}/{rel}"
                    )

            # 7. Mark job completed (status + completed_at on the generic
            # jobs table; metrics + s3_prefix go into details so polling
            # endpoints can surface them).
            update_job(
                job_uuid,
                status=TaskStatus.DONE.value,
                details={
                    "s3_prefix": s3_prefix,
                    "metrics": metrics,
                    "item_count": len(items),
                    "completed_at": _utcnow_str(),
                },
            )
            logger.info(f"[annotation-eval] job {job_uuid} completed")
    except subprocess.CalledProcessError as e:
        # The calibrate CLI exited non-zero. Surface the actual stderr (or the
        # structured `{"status":"error","error":"..."}` blob if present) so the
        # API caller sees what went wrong, mirroring stt/tts/etc.
        traceback.print_exc()
        capture_exception_to_sentry(e)
        cli_err = _extract_calibrate_error(
            getattr(e, "stdout", "") or "", getattr(e, "stderr", "") or ""
        )
        s3_prefix = _try_upload_partial_outputs(
            output_dir_for_partial, task_uuid, job_uuid
        )
        details_patch = {"completed_at": _utcnow_str()}
        if s3_prefix:
            details_patch["s3_prefix"] = s3_prefix
        update_job(
            job_uuid,
            status=TaskStatus.FAILED.value,
            results={"error": f"calibrate --eval-only failed: {cli_err}"},
            details=details_patch,
        )
    except Exception as e:
        traceback.print_exc()
        logger.exception(f"[annotation-eval] job {job_uuid} failed: {e}")
        capture_exception_to_sentry(e)
        s3_prefix = _try_upload_partial_outputs(
            output_dir_for_partial, task_uuid, job_uuid
        )
        details_patch = {"completed_at": _utcnow_str()}
        if s3_prefix:
            details_patch["s3_prefix"] = s3_prefix
        update_job(
            job_uuid,
            status=TaskStatus.FAILED.value,
            results={
                "error": f"Unexpected error during annotation evaluator run: {str(e)}"
            },
            details=details_patch,
        )
    finally:
        # Whichever way the job ended, free the slot for the next queued job.
        try:
            try_start_queued_job(EVAL_JOB_TYPES)
        except Exception as e:
            logger.warning(f"[annotation-eval] failed to drain queue: {e}")


def start_annotation_eval_job(
    job_uuid: str,
    task_uuid: str,
    user_id: str,
    evaluators_resolved: List[Dict[str, Any]],
    item_ids: Optional[List[str]] = None,
) -> None:
    """Spawn the runner thread. Returns immediately."""
    t = threading.Thread(
        target=_run_job,
        args=(job_uuid, task_uuid, user_id, evaluators_resolved, item_ids),
        daemon=True,
    )
    t.start()


def _start_annotation_eval_job_from_queue(job_row: Dict[str, Any]) -> None:
    """Queue starter — invoked by `try_start_queued_job` when a queued
    annotation-eval job is ready to run. Re-resolves evaluators from the
    persisted `details` (so a queued job picks up the same versions it was
    submitted with) and spawns the runner."""
    details = job_row.get("details") or {}
    task_uuid = details.get("task_id")
    user_id = job_row.get("user_id")
    if not task_uuid or not user_id:
        raise RuntimeError(
            f"annotation-eval job {job_row.get('uuid')} missing task_id/user_id"
        )
    evaluator_refs = details.get("evaluators") or []
    if not evaluator_refs:
        raise RuntimeError(
            f"annotation-eval job {job_row.get('uuid')} has no evaluators in details"
        )
    from db import get_evaluators_for_annotation_task

    linked = {e["uuid"] for e in get_evaluators_for_annotation_task(task_uuid)}
    requested = [
        {
            "evaluator_id": ref.get("evaluator_id"),
            "evaluator_version_id": ref.get("evaluator_version_id"),
        }
        for ref in evaluator_refs
    ]
    resolved = _resolve_evaluator_dicts(requested, linked)
    start_annotation_eval_job(
        job_uuid=job_row["uuid"],
        task_uuid=task_uuid,
        user_id=user_id,
        evaluators_resolved=resolved,
        item_ids=details.get("item_ids"),
    )


# Register at import time so the existing `try_start_queued_job` machinery
# can resume queued annotation-eval jobs without any explicit wiring.
register_job_starter(ANNOTATION_EVAL_JOB_TYPE, _start_annotation_eval_job_from_queue)


def resume_annotation_eval_job(job_row: Dict[str, Any]) -> None:
    """Recovery entry point — called from job_recovery for `in_progress` jobs.

    Clears any `evaluator_runs` rows from the partially-completed previous run
    to avoid duplicates, then dispatches via the standard queue starter so
    capacity rules are respected on resume.
    """
    from db import clear_evaluator_runs_for_job

    job_uuid = job_row["uuid"]
    cleared = clear_evaluator_runs_for_job(job_uuid)
    if cleared:
        logger.info(
            f"[annotation-eval] cleared {cleared} stale evaluator_run rows for {job_uuid}"
        )
    _start_annotation_eval_job_from_queue(job_row)
