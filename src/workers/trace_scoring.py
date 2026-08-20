"""Background worker pool that scores queued traces.

Started from the FastAPI lifespan (see main.py) -- not `BackgroundTasks`,
which is unbounded and tied to a single request's lifecycle. This pool has
its own fixed concurrency bound, deliberately independent of
`MAX_CONCURRENT_JOBS_PER_ORG` (which defaults to 1 and would let one
workspace's backfill starve every other org's live scoring).

Each worker claims a batch, judges each row via `calibrate-agent llm
--eval-only` (the same eval-only mechanism `annotation_eval_runner` uses),
and settles it. The subprocess cost is tracked as a follow-up to remove, not
addressed here.
"""

import asyncio
import json
import logging
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from annotation_eval_runner import calibrate_command_for_task_type
from db import (
    claim_trace_eval_queue_rows,
    get_evaluator,
    get_evaluator_version,
    get_trace,
    settle_trace_eval_failure,
    settle_trace_eval_success,
)
import trace_scoring_nudge
from llm_judge import build_evaluator_cli_payload_unrendered
from utils import coerce_evaluator_score

logger = logging.getLogger(__name__)

TRACE_SCORING_POOL_SIZE = int(os.getenv("TRACE_SCORING_POOL_SIZE", "4"))
TRACE_SCORING_BATCH_SIZE = 5
# Backstop for leases expiring while every worker is idle -- the nudge covers
# the common case (a fresh row lands), this covers the rest.
TRACE_SCORING_POLL_SECONDS = 5
TRACE_SCORING_SUBPROCESS_TIMEOUT_SECONDS = 120


def _resolve_evaluator_dict(evaluator_uuid: str) -> Dict[str, Any]:
    """Build the dict `build_evaluator_cli_payload_unrendered` expects, from
    the evaluator's CURRENT live version -- trace scoring always judges
    against live, there is no per-trace pin to honor (only the queue row's
    own `evaluator_version_id` is pinned, for score-history comparability)."""
    evaluator = get_evaluator(evaluator_uuid)
    if not evaluator or not evaluator.get("live_version_id"):
        raise ValueError(f"Evaluator {evaluator_uuid} has no live version")
    version = get_evaluator_version(evaluator["live_version_id"])
    if not version:
        raise ValueError(f"Evaluator {evaluator_uuid} live version not found")
    return {
        "uuid": evaluator_uuid,
        "name": evaluator["name"],
        "judge_model": version["judge_model"],
        "system_prompt": version["system_prompt"],
        "output_type": evaluator["output_type"],
        "output_config": version.get("output_config"),
        "variables": version.get("variables"),
        "variable_values": {},
        "kind": evaluator["kind"],
        "data_type": evaluator["data_type"],
    }


def _run_eval_only(trace: Dict[str, Any], evaluator: Dict[str, Any]) -> Tuple[Any, Optional[str]]:
    """Judge one trace's recorded response against one evaluator via
    `calibrate-agent llm --eval-only`, and return (score, reasoning).

    Raises on any failure (non-zero exit, missing/malformed results) so the
    caller can route it through settle_trace_eval_failure uniformly.
    """
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        dataset_path = tmp_path / "dataset.json"
        config_path = tmp_path / "config.json"
        output_dir = tmp_path / "output"
        output_dir.mkdir()

        output = trace.get("output") or {}
        dataset = [
            {
                "test_case": {
                    "id": trace["uuid"],
                    "history": trace["input"],
                    "evaluation": {
                        "type": "response",
                        "criteria": [{"name": evaluator["name"]}],
                    },
                },
                "output": {
                    "response": output.get("response"),
                    "tool_calls": output.get("tool_calls") or [],
                },
            }
        ]
        dataset_path.write_text(json.dumps(dataset))
        config_path.write_text(
            json.dumps({"evaluators": build_evaluator_cli_payload_unrendered([evaluator])})
        )

        cmd = calibrate_command_for_task_type("llm", dataset_path, output_dir, config_path)
        proc = subprocess.run(
            cmd,
            cwd=str(tmp_path),
            capture_output=True,
            text=True,
            timeout=TRACE_SCORING_SUBPROCESS_TIMEOUT_SECONDS,
        )
        if proc.returncode != 0:
            raise RuntimeError(
                f"calibrate-agent llm --eval-only exited {proc.returncode}: "
                f"{proc.stderr[-500:]}"
            )

        results_path = output_dir / "results.json"
        if not results_path.exists():
            raise RuntimeError("calibrate-agent llm --eval-only produced no results.json")
        results = json.loads(results_path.read_text())
        entry = next(
            (r for r in results if isinstance(r, dict) and r.get("test_case_id") == trace["uuid"]),
            None,
        )
        if entry is None:
            raise RuntimeError(f"No result entry for trace {trace['uuid']}")
        judge_results = (entry.get("metrics") or {}).get("judge_results") or {}
        judgement = judge_results.get(evaluator["name"])
        if not isinstance(judgement, dict):
            raise RuntimeError(f"No judge_results for evaluator {evaluator['name']!r}")
        raw = judgement.get("match")
        if raw is None:
            raw = judgement.get("score")
        if raw is None:
            raise RuntimeError(f"Judge result for {evaluator['name']!r} has no match/score")
        score = coerce_evaluator_score(raw, evaluator["output_type"])
        return float(score), judgement.get("reasoning")


def _score_one(row: Dict[str, Any]) -> None:
    """Claim -> judge -> settle for a single trace_eval_queue row. Blocking
    (DB + subprocess); callers run it via asyncio.to_thread."""
    queue_id = row["id"]
    trace_uuid = row["trace_uuid"]
    evaluator_uuid = row["evaluator_uuid"]
    org_uuid = row["org_uuid"]
    evaluator_version_id = row["evaluator_version_id"]
    attempts = row["attempts"]

    try:
        trace = get_trace(org_uuid, trace_uuid)
        if trace is None:
            # Soft-deleted mid-flight -- nothing left to score. Dead-letter
            # immediately rather than retrying a trace that will never exist.
            settle_trace_eval_failure(
                queue_id,
                trace_uuid=trace_uuid,
                evaluator_uuid=evaluator_uuid,
                evaluator_version_id=evaluator_version_id,
                org_uuid=org_uuid,
                attempts=attempts,
                error="trace no longer exists",
                max_attempts=0,
            )
            return

        evaluator = _resolve_evaluator_dict(evaluator_uuid)
        score, reasoning = _run_eval_only(trace, evaluator)
        settle_trace_eval_success(
            queue_id,
            trace_uuid=trace_uuid,
            evaluator_uuid=evaluator_uuid,
            evaluator_version_id=evaluator_version_id,
            org_uuid=org_uuid,
            score=score,
            reasoning=reasoning,
        )
    except Exception as e:
        logger.warning(f"[trace-scoring] queue row {queue_id} failed: {e}")
        settle_trace_eval_failure(
            queue_id,
            trace_uuid=trace_uuid,
            evaluator_uuid=evaluator_uuid,
            evaluator_version_id=evaluator_version_id,
            org_uuid=org_uuid,
            attempts=attempts,
            error=str(e),
        )


async def _worker_loop(
    worker_id: int, stop_event: asyncio.Event, nudge: asyncio.Event
) -> None:
    while not stop_event.is_set():
        rows = await asyncio.to_thread(
            claim_trace_eval_queue_rows, batch_size=TRACE_SCORING_BATCH_SIZE
        )
        if not rows:
            nudge.clear()
            try:
                await asyncio.wait_for(nudge.wait(), timeout=TRACE_SCORING_POLL_SECONDS)
            except asyncio.TimeoutError:
                pass
            continue
        # A claimed batch runs to completion even if stop_event flips
        # mid-batch -- each row's lease is already extended, so finishing is
        # cheaper and safer than abandoning it to expire on its own.
        for row in rows:
            await asyncio.to_thread(_score_one, row)


class TraceScoringPool:
    """Owns the worker tasks; started and stopped from the app lifespan."""

    def __init__(self, size: int = TRACE_SCORING_POOL_SIZE):
        self._size = size
        self._stop_event = asyncio.Event()
        self._tasks: List[asyncio.Task] = []

    def start(self) -> None:
        # Fresh Event per lifespan: asyncio.Event binds to whichever loop
        # first calls .wait() on it and raises if a later .wait() arrives
        # from a different loop -- reset() avoids reusing one bound to a
        # loop from a previous (e.g. test) lifespan that has since closed.
        nudge = trace_scoring_nudge.reset()
        self._tasks = [
            asyncio.create_task(_worker_loop(i, self._stop_event, nudge))
            for i in range(self._size)
        ]
        logger.info(f"[trace-scoring] started {self._size} worker(s)")

    async def shutdown(self) -> None:
        self._stop_event.set()
        trace_scoring_nudge.set()  # wake idle workers so they see stop_event
        if self._tasks:
            await asyncio.gather(*self._tasks, return_exceptions=True)
        logger.info("[trace-scoring] all workers stopped")
