"""Judge a batch of already-produced traces without re-running the agent.

A trace is a turn the agent already served in production, so every mode here
grades the stored output as-is. That is exactly what the annotation eval runner
already does for annotation items, so this module reuses its dataset builders,
CLI command builder, subprocess runner and result parsers wholesale — only the
adapter from a trace to an annotation-item-shaped dict and the persistence side
are new.

The three inferred trace types map onto three eval-only CLI modes:

    response      -> calibrate llm --eval-only
    conversation  -> calibrate simulations -t text --eval-only
    tool_call     -> calibrate general

`conversation` results come back keyed by calibrate's own `row_<i>` directory
names, mapped to traces by dataset ORDER, so the trace list, the item list and
the dataset rows must stay in the same order end to end.
"""

from __future__ import annotations

import json
import logging
import tempfile
import threading
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from annotation_eval_runner import (
    _dedupe_evaluator_names,
    _extract_calibrate_error,
    _run_calibrate_eval_only,
    build_dataset_for_task_type,
    calibrate_command_for_task_type,
    parse_results_for_task_type,
)
from llm_judge import (
    _scale_bounds,
    build_evaluator_cli_payload,
    refresh_evaluators_to_live,
)
from traces import eval_store
from traces.inference import TYPE_CONVERSATION, TYPE_RESPONSE, TYPE_TOOL_CALL
from utils import TaskStatus, capture_exception_to_sentry, coerce_evaluator_score

logger = logging.getLogger(__name__)

# Which annotation-runner task type judges each inferred trace type. The values
# collide with `inference.EVALUATOR_TYPE_FOR` today, but they answer a different
# question (which CLI mode to spawn, not which evaluator judges), so they are
# spelled out separately.
CLI_TASK_TYPE_FOR = {
    TYPE_RESPONSE: "llm",
    TYPE_CONVERSATION: "conversation",
    TYPE_TOOL_CALL: "llm-general",
}


# ---------------------------------------------------------------------------
# Evaluators
# ---------------------------------------------------------------------------


def resolve_evaluators(evaluators: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Pin each linked evaluator to its live version at run time.

    Mirrors the STT/TTS re-hydration path: the run reads whatever prompt and
    rubric are live when it starts, and the resolved version id is frozen into
    the run's snapshot so finished results still render after the evaluator is
    edited.
    """
    resolved: List[Dict[str, Any]] = []
    for ev in refresh_evaluators_to_live(evaluators):
        scale_min, scale_max = _scale_bounds(ev.get("output_config"))
        resolved.append(
            {
                "uuid": ev["uuid"],
                "name": ev.get("name") or ev["uuid"],
                "judge_model": ev.get("judge_model"),
                "system_prompt": ev.get("system_prompt") or "",
                "output_type": ev.get("output_type") or "binary",
                "output_config": ev.get("output_config"),
                "variables": ev.get("variables"),
                "variable_values": ev.get("variable_values") or {},
                "kind": ev.get("kind") or "single",
                "data_type": ev.get("data_type") or "text",
                # Underscored keys are bookkeeping for the shared parsers and
                # the result rows; the CLI payload builder ignores them.
                "_evaluator_version_id": ev.get("evaluator_version_id"),
                "_scale_min": scale_min,
                "_scale_max": scale_max,
            }
        )
    _dedupe_evaluator_names(resolved)
    return resolved


def evaluator_snapshot(evaluators_resolved: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """The identity a result needs to render, frozen onto the run row."""
    return [
        {
            "uuid": ev["uuid"],
            "name": ev["name"],
            "evaluator_version_id": ev["_evaluator_version_id"],
            "output_type": ev["output_type"],
            "scale_min": ev["_scale_min"],
            "scale_max": ev["_scale_max"],
        }
        for ev in evaluators_resolved
    ]


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


def _text_of(content: Any) -> str:
    if isinstance(content, str):
        return content
    return json.dumps(content, ensure_ascii=False)


def _last_user_message(turns: List[Any]) -> str:
    for turn in reversed(turns or []):
        if isinstance(turn, dict) and turn.get("role") == "user":
            return _text_of(turn.get("content"))
    return ""


def trace_to_item(trace: Dict[str, Any], inferred_type: str) -> Dict[str, Any]:
    """Reshape one trace into the annotation-item shape the shared dataset
    builders consume.

    A trace's `input` IS the history and its `output` IS the output, so this is
    renaming rather than transformation. The one exception is `tool_call`, whose
    CLI mode grades a flat input/output pair of strings and therefore needs the
    calls serialized for the judge to read.
    """
    turns = trace.get("input") or []
    output = trace.get("output") or {}
    response = output.get("response") or ""
    tool_calls = output.get("tool_calls") or []

    if inferred_type == TYPE_RESPONSE:
        payload: Dict[str, Any] = {
            "chat_history": turns,
            "agent_response": response,
            "tool_calls": tool_calls,
        }
    elif inferred_type == TYPE_CONVERSATION:
        payload = {
            "transcript": list(turns) + [{"role": "assistant", "content": response}]
        }
    elif inferred_type == TYPE_TOOL_CALL:
        payload = {
            "input": _last_user_message(turns),
            "output": json.dumps(tool_calls, ensure_ascii=False, indent=2),
        }
    else:
        raise ValueError(f"Traces cannot be judged as {inferred_type!r}")

    return {"uuid": trace["uuid"], "payload": payload}


def build_trace_dataset(
    inferred_type: str,
    traces: List[Dict[str, Any]],
    evaluators_resolved: List[Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Return `(items, dataset_rows)`, both in the given trace order."""
    items = [trace_to_item(t, inferred_type) for t in traces]
    dataset = build_dataset_for_task_type(
        CLI_TASK_TYPE_FOR[inferred_type], items, evaluators_resolved
    )
    return items, dataset


# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------


def _verdict(output_type: str, raw: Any) -> Tuple[Optional[bool], Optional[float]]:
    """Split one judged value into the binary and rating columns."""
    coerced = coerce_evaluator_score(raw, output_type)
    if output_type == "binary":
        return (coerced if isinstance(coerced, bool) else None), None
    if isinstance(coerced, bool) or not isinstance(coerced, (int, float)):
        return None, None
    return None, float(coerced)


def to_trace_results(
    parsed: List[Dict[str, Any]], evaluators_resolved: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """Turn the shared parsers' evaluator-run rows into `record_results` rows."""
    by_uuid = {ev["uuid"]: ev for ev in evaluators_resolved}
    results: List[Dict[str, Any]] = []
    for row in parsed:
        value = row.get("value")
        if value is None:
            continue
        ev = by_uuid.get(row.get("evaluator_id"))
        if ev is None:
            logger.warning(
                "[trace-eval] result for unknown evaluator %s; skipping",
                row.get("evaluator_id"),
            )
            continue
        passed, score = _verdict(ev["output_type"], value.get("value"))
        results.append(
            {
                "trace_uuid": row["item_id"],
                "evaluator_uuid": ev["uuid"],
                "evaluator_version_id": row.get("evaluator_version_id")
                or ev["_evaluator_version_id"],
                "evaluator_name": ev["name"],
                "output_type": ev["output_type"],
                "passed": passed,
                "score": score,
                "scale_min": ev["_scale_min"],
                "scale_max": ev["_scale_max"],
                "reasoning": value.get("reasoning"),
            }
        )
    return results


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


def judge_batch(
    run_uuid: str,
    inferred_type: str,
    traces: List[Dict[str, Any]],
    evaluators_resolved: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Spawn one eval-only calibrate run and return its verdicts."""
    task_type = CLI_TASK_TYPE_FOR[inferred_type]
    with tempfile.TemporaryDirectory() as temp_dir:
        tmp = Path(temp_dir)
        input_dir = tmp / "input"
        input_dir.mkdir()
        output_dir = tmp / "output"
        output_dir.mkdir()

        items, dataset = build_trace_dataset(inferred_type, traces, evaluators_resolved)
        dataset_path = input_dir / "dataset.json"
        with open(dataset_path, "w", encoding="utf-8") as f:
            json.dump(dataset, f, ensure_ascii=False)

        # Rendered, not unrendered: a trace carries no per-row variable values,
        # so each evaluator's own values (or its declared defaults) are the only
        # substitution available.
        config_path = input_dir / "config.json"
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(
                {"evaluators": build_evaluator_cli_payload(evaluators_resolved)},
                f,
                ensure_ascii=False,
            )

        cmd = calibrate_command_for_task_type(
            task_type, dataset_path, output_dir, config_path
        )
        logger.info("[trace-eval] run %s spawning: %s", run_uuid, " ".join(cmd))
        rc, stdout, stderr = _run_calibrate_eval_only(cmd, cwd=tmp, log_dir=output_dir)
        if rc != 0:
            raise RuntimeError(
                "calibrate --eval-only failed: "
                + _extract_calibrate_error(stdout, stderr)
            )

        parsed = parse_results_for_task_type(
            task_type, output_dir, evaluators_resolved, run_uuid, items=items
        )
        return to_trace_results(parsed, evaluators_resolved)


def _fail_run(run_uuid: str, error: str) -> None:
    try:
        eval_store.update_eval_run(
            run_uuid,
            status=TaskStatus.FAILED.value,
            error=error,
            finished_at="now",
        )
    finally:
        # Released last so the traces go back to the queue even if the status
        # write is what blew up; leaving them claimed strands them forever.
        eval_store.release_claims(run_uuid)


def run_batch(
    org_uuid: str,
    run_uuid: str,
    inferred_type: str,
    traces: List[Dict[str, Any]],
    evaluators_resolved: List[Dict[str, Any]],
) -> None:
    """Worker body — judge the claimed traces and record the outcome."""
    try:
        results = judge_batch(run_uuid, inferred_type, traces, evaluators_resolved)
        eval_store.record_results(org_uuid, run_uuid, results)
        judged = list(dict.fromkeys(r["trace_uuid"] for r in results))
        eval_store.mark_traces_evaluated(run_uuid, judged)
        eval_store.update_eval_run(
            run_uuid, status=TaskStatus.DONE.value, finished_at="now"
        )
        logger.info(
            "[trace-eval] run %s judged %d of %d traces",
            run_uuid,
            len(judged),
            len(traces),
        )
    except Exception as e:
        traceback.print_exc()
        logger.exception("[trace-eval] run %s failed: %s", run_uuid, e)
        capture_exception_to_sentry(e)
        _fail_run(run_uuid, str(e))


def launch_trace_eval(
    org_uuid: str,
    agent: Dict[str, Any],
    inferred_type: str,
    traces: List[Dict[str, Any]],
    evaluators: List[Dict[str, Any]],
    trigger: str,
) -> Tuple[str, str]:
    """Create the run row, claim the traces, start the worker thread.

    Returns `(run_uuid, status)`. Only traces actually won by the claim are
    judged; the rest are left for whoever holds them.
    """
    evaluators_resolved = resolve_evaluators(evaluators)
    run = eval_store.create_eval_run(
        org_uuid,
        agent["uuid"],
        trigger=trigger,
        inferred_type=inferred_type,
        status=TaskStatus.QUEUED.value,
        evaluator_snapshot=evaluator_snapshot(evaluators_resolved),
    )
    run_uuid = run["uuid"]

    won = set(eval_store.claim_traces(org_uuid, run_uuid, [t["uuid"] for t in traces]))
    claimed = [t for t in traces if t["uuid"] in won]
    if not claimed:
        eval_store.update_eval_run(
            run_uuid,
            status=TaskStatus.DONE.value,
            trace_count=0,
            started_at="now",
            finished_at="now",
        )
        return run_uuid, TaskStatus.DONE.value

    eval_store.update_eval_run(
        run_uuid,
        status=TaskStatus.IN_PROGRESS.value,
        trace_count=len(claimed),
        started_at="now",
    )
    threading.Thread(
        target=run_batch,
        args=(org_uuid, run_uuid, inferred_type, claimed, evaluators_resolved),
        daemon=True,
    ).start()
    return run_uuid, TaskStatus.IN_PROGRESS.value
