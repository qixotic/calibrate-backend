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
    get_evaluator,
    get_evaluator_version,
    get_job,
    update_job,
)
from llm_judge import build_evaluator_cli_payload
from utils import (
    TaskStatus,
    capture_exception_to_sentry,
    get_s3_client,
    get_s3_output_config,
    register_job_starter,
    try_start_queued_job,
    upload_file_to_s3,
    upload_top_level_files_to_s3,
)


# Eval queue types this module participates in (must match `EVAL_JOB_TYPES`
# elsewhere). Importing the constant from utils would create a circular import,
# so we redeclare it here intentionally — see also routers/stt.py and
# job_recovery.py. Keep the three lists in sync.
EVAL_JOB_TYPES = ["stt-eval", "tts-eval", "annotation-eval"]
ANNOTATION_EVAL_JOB_TYPE = "annotation-eval"

# Task types whose annotation rows we know how to evaluate via the CLI's
# --eval-only modes. `tts` is omitted because annotation tasks don't store
# audio S3 keys today; `voice` simulation isn't supported by the CLI in
# eval-only mode.
SUPPORTED_EVAL_TASK_TYPES = ("stt", "llm", "simulation")

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
        # Shape that build_evaluator_cli_payload + downstream want.
        resolved.append(
            {
                "uuid": evaluator_id,
                "name": evaluator["name"],
                "judge_model": version["judge_model"],
                "system_prompt": version["system_prompt"],
                "output_type": evaluator.get("output_type", "binary"),
                "output_config": version.get("output_config"),
                "variables": version.get("variables"),
                "variable_values": {},
                "kind": evaluator.get("kind", "single"),
                "data_type": evaluator.get("data_type", "text"),
                # extra fields for our own bookkeeping (not seen by CLI):
                "_evaluator_version_id": version_uuid,
            }
        )
    return resolved


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
    items: List[Dict[str, Any]], evaluator_names: List[str]
) -> List[Dict[str, Any]]:
    """LLM --eval-only: [{test_case: {id, history, evaluation}, output: {response, tool_calls}}].

    Annotation convention for payload (response-mode only — tool-call eval is
    not exposed via this endpoint today):
      { "chat_history": [...], "agent_response": "...", "tool_calls"?: [...] }
    The set of evaluators requested for the run becomes the
    `evaluation.criteria` references on every test case.
    """
    if not evaluator_names:
        raise DatasetBuildError(
            "LLM --eval-only requires at least one evaluator (criteria)"
        )
    criteria_refs = [{"name": n} for n in evaluator_names]
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
        out.append(
            {
                "test_case": {
                    "id": it["uuid"],
                    "history": history,
                    "evaluation": {
                        "type": "response",
                        "criteria": list(criteria_refs),
                    },
                },
                "output": {"response": str(response), "tool_calls": tool_calls},
            }
        )
    return out


def _build_simulation_dataset(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Text-simulation --eval-only: [{name, conversation_history}].
    `name` = annotation_item.uuid so per-simulation output dirs map back."""
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
    evaluator_names: List[str],
) -> List[Dict[str, Any]]:
    if task_type == "stt":
        return _build_stt_dataset(items)
    if task_type == "llm":
        return _build_llm_dataset(items, evaluator_names)
    if task_type == "simulation":
        return _build_simulation_dataset(items)
    raise DatasetBuildError(
        f"Evaluator runs are not supported for task type {task_type!r} "
        "(supported: stt, llm, simulation)"
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
    if task_type == "simulation":
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


def _coerce_score(raw: Any, output_type: str) -> Any:
    """Coerce a raw value out of CSV/JSON into the right Python type per
    output_type. Falls back to passthrough on unparseable input."""
    if output_type == "binary":
        if isinstance(raw, bool):
            return raw
        s = str(raw).strip().lower()
        if s in ("true", "1", "yes", "pass"):
            return True
        if s in ("false", "0", "no", "fail"):
            return False
        return raw
    if output_type == "rating":
        try:
            return int(float(raw))
        except (TypeError, ValueError):
            try:
                return float(raw)
            except (TypeError, ValueError):
                return raw
    return raw


def _row_evaluator_value(
    row: Dict[str, Any], evaluator_name: str, output_type: str
) -> Optional[Dict[str, Any]]:
    """Pick one evaluator's score + reasoning out of a results.csv row."""
    if evaluator_name not in row:
        return None
    raw_score = row[evaluator_name]
    reasoning = row.get(f"{evaluator_name}_reasoning")
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
    <evaluator_name>, <evaluator_name>_reasoning per evaluator.
    Maps row.id → annotation_item.uuid (we set id = item.uuid on the way out)."""
    rows = _read_results_csv(output_dir) or []
    name_to_uuid_via_config = _read_config_evaluators_map(output_dir)
    runs: List[Dict[str, Any]] = []
    for ev in evaluators_resolved:
        name = ev["name"]
        # Trust the config.json's evaluators_map first; fall back to the local
        # resolved record's uuid if the map isn't present (defensive).
        evaluator_id = name_to_uuid_via_config.get(name, ev["uuid"])
        if evaluator_id != ev["uuid"]:
            logger.warning(
                f"[annotation-eval] evaluators_map round-trip mismatch for "
                f"{name!r}: sent {ev['uuid']}, got {evaluator_id}"
            )
        version_uuid = ev["_evaluator_version_id"]
        output_type = ev["output_type"]
        for row in rows:
            item_id = row.get("id")
            if not item_id:
                continue
            value = _row_evaluator_value(row, name, output_type)
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


def _parse_results_llm(
    output_dir: Path,
    evaluators_resolved: List[Dict[str, Any]],
    job_uuid: str,
) -> List[Dict[str, Any]]:
    """LLM outputs: <output_dir>/results.json — list of per-test-case results
    with `metrics`. Each test case carries the input `test_case` (for `id`)
    and a `metrics.criteria` map keyed by evaluator name. Tool-call evaluation
    is not surfaced into evaluator_runs (this endpoint only runs response-mode
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
        # Calibrate's documented shape is `metrics.criteria = {name: <judgement>}`.
        # Be defensive: also accept a flat `metrics = {name: <judgement>}`.
        per_criterion = metrics.get("criteria") if isinstance(metrics, dict) else None
        if not isinstance(per_criterion, dict):
            per_criterion = metrics if isinstance(metrics, dict) else {}
        for ev_name, judgement in per_criterion.items():
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
                # Pull the score from any of the conventional keys.
                raw = (
                    judgement.get("value")
                    or judgement.get("score")
                    or judgement.get("rating")
                    or judgement.get("pass")
                    or judgement.get("passed")
                )
                if raw is not None:
                    value = {"value": _coerce_score(raw, output_type)}
                    reasoning = judgement.get("reasoning")
                    if reasoning:
                        value["reasoning"] = reasoning
            elif judgement is not None:
                value = {"value": _coerce_score(judgement, output_type)}
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


def _parse_results_simulation(
    output_dir: Path,
    evaluators_resolved: List[Dict[str, Any]],
    job_uuid: str,
) -> List[Dict[str, Any]]:
    """Simulation outputs: per-simulation `<output_dir>/<name>/evaluation_results.csv`
    with one row per evaluator: name, type, value, reasoning, evaluator_id?, ...
    `<name>` was set to annotation_item.uuid when we built the dataset."""
    name_to_uuid_via_config = _read_config_evaluators_map(output_dir)
    by_name: Dict[str, Dict[str, Any]] = {ev["name"]: ev for ev in evaluators_resolved}

    runs: List[Dict[str, Any]] = []
    if not output_dir.exists():
        return runs
    for sim_dir in sorted(p for p in output_dir.iterdir() if p.is_dir()):
        item_id = sim_dir.name  # uuid we set as `name`
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
) -> List[Dict[str, Any]]:
    if task_type == "stt":
        return _parse_results_stt(output_dir, evaluators_resolved, job_uuid)
    if task_type == "llm":
        return _parse_results_llm(output_dir, evaluators_resolved, job_uuid)
    if task_type == "simulation":
        return _parse_results_simulation(output_dir, evaluators_resolved, job_uuid)
    return []


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


def _run_calibrate_eval_only(
    cmd: List[str],
    cwd: Path,
    log_dir: Path,
    on_started: Optional[Any] = None,
    heartbeat_seconds: int = 2,
) -> Tuple[int, str, str]:
    """Spawn the calibrate subprocess; redirect stdout/stderr to disk to avoid
    pipe-buffer deadlocks; poll until done; return (returncode, stdout, stderr)."""
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
        while proc.poll() is None:
            time.sleep(heartbeat_seconds)
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
    semantics so existing keys (evaluators, item_ids, ...) survive."""
    update_job(job_uuid, details={"pid": pid, "pgid": pid})


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

        task_type = task.get("type", "stt")

        with tempfile.TemporaryDirectory() as temp_dir:
            tmp = Path(temp_dir)
            input_dir = tmp / "input"
            input_dir.mkdir()
            output_dir = tmp / "output"
            output_dir.mkdir()
            output_dir_for_partial = output_dir

            # 1. Dataset.
            evaluator_names = [ev["name"] for ev in evaluators_resolved]
            dataset = build_dataset_for_task_type(task_type, items, evaluator_names)
            dataset_path = input_dir / "dataset.json"
            with open(dataset_path, "w", encoding="utf-8") as f:
                json.dump(dataset, f, ensure_ascii=False)

            # 2. Config (evaluators only — same shape across all eval-only modes).
            evaluator_payload = build_evaluator_cli_payload(evaluators_resolved)
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
                task_type, output_dir, evaluators_resolved, job_uuid
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
