"""Periodic driver that drains the pending-trace backlog into eval runs.

Traces arrive at production volume, so judging one per ingest would spawn a
judge subprocess per turn. This poller batches instead: each tick takes a
bounded slice of the pending queue for each agent that opted in, groups it by
the type inferred for each trace, and hands one batch at a time to
`eval_runner.launch_trace_eval`.
"""

import asyncio
import logging
from typing import Any, Dict, List, Tuple

import db
from traces import eval_store, inference
from traces.eval_runner import launch_trace_eval
from utils import TaskStatus, env_int

logger = logging.getLogger(__name__)

DEFAULT_POLL_SECONDS = 60
DEFAULT_BATCH_SIZE = 100
DEFAULT_MAX_CONCURRENT_RUNS = 2

ORPHANED_RUN_ERROR = "Run was interrupted by a backend restart"

# Marks the bookkeeping run that retires traces nothing can judge.
UNJUDGEABLE_RUN_TYPE = "skipped"


def poll_seconds() -> int:
    return max(1, env_int("TRACE_EVAL_POLL_SECONDS", DEFAULT_POLL_SECONDS))


def batch_size() -> int:
    return max(1, env_int("TRACE_EVAL_BATCH_SIZE", DEFAULT_BATCH_SIZE))


def max_concurrent_runs() -> int:
    # Zero is a valid setting: it pauses automatic judging without a redeploy.
    return max(0, env_int("TRACE_EVAL_MAX_CONCURRENT_RUNS", DEFAULT_MAX_CONCURRENT_RUNS))


def _auto_eval_enabled(agent: Dict[str, Any]) -> bool:
    return bool((agent.get("config") or {}).get("auto_eval_enabled"))


def _retire_unjudgeable(
    org_uuid: str, agent_id: str, skipped: List[Dict[str, str]]
) -> int:
    """Take traces nothing can judge out of the pending window.

    Selection is oldest-first, so leaving them pending is not merely untidy: a
    batch of traces no linked evaluator can judge would occupy the window on
    every tick and starve everything newer, silently stopping automatic judging
    for the agent. Retiring them under a bookkeeping run keeps the reason
    auditable instead of stamping the traces with no explanation.

    Consequence worth knowing: a trace retired for want of, say, a conversation
    evaluator is not revisited if one is linked later.
    """
    trace_uuids = [s["trace_uuid"] for s in skipped]
    run = eval_store.create_eval_run(
        org_uuid,
        agent_id,
        trigger=eval_store.TRIGGER_AUTO,
        inferred_type=UNJUDGEABLE_RUN_TYPE,
        status=TaskStatus.DONE.value,
        evaluator_snapshot={"skipped": skipped},
    )
    won = eval_store.claim_traces(org_uuid, run["uuid"], trace_uuids)
    eval_store.mark_traces_evaluated(run["uuid"], won)
    eval_store.update_eval_run(
        run["uuid"], skipped_count=len(won), finished_at="now"
    )
    logger.info(
        "Trace eval: retired %s unjudgeable trace(s) for agent %s under run %s",
        len(won),
        agent_id,
        run["uuid"],
    )
    return len(won)


def _launch_batches_for_agent(
    org_uuid: str, agent_id: str, headroom: int
) -> List[Tuple[str, str]]:
    """Launch up to `headroom` runs for one agent, one per inferred type.

    Returns the `(run_uuid, status)` of each run started, empty when the agent
    is not opted in, has nothing linked to judge with, or has no pending work.
    """
    agent = db.get_agent(agent_id)
    if not agent:
        return []
    # Without this gate, ingesting traces would spend judge tokens forever in
    # proportion to production traffic, invisible until the bill arrives.
    if not _auto_eval_enabled(agent):
        return []

    evaluators = db.get_evaluators_for_agent(agent_id)
    if not evaluators:
        return []

    traces = eval_store.list_pending_traces(org_uuid, agent_id, limit=batch_size())
    if not traces:
        return []

    batches, skipped = inference.plan_batches(traces, evaluators)
    if skipped:
        _retire_unjudgeable(org_uuid, agent_id, skipped)

    started: List[Tuple[str, str]] = []
    for inferred_type, batch in batches.items():
        if len(started) >= headroom:
            break
        run_uuid, status = launch_trace_eval(
            org_uuid=org_uuid,
            agent=agent,
            inferred_type=inferred_type,
            traces=batch,
            evaluators=inference.evaluators_for_type(inferred_type, evaluators),
            trigger=eval_store.TRIGGER_AUTO,
        )
        started.append((run_uuid, status))
        logger.info(
            "Trace eval run %s started: agent=%s type=%s traces=%s status=%s",
            run_uuid,
            agent_id,
            inferred_type,
            len(batch),
            status,
        )
    return started


def run_one_tick() -> int:
    """Launch eval runs for the current backlog. Returns how many were started."""
    ceiling = max_concurrent_runs()
    live = len(eval_store.unfinished_runs())
    launched = 0

    for org_uuid, agent_id in eval_store.agents_with_pending_traces():
        headroom = ceiling - live - launched
        if headroom <= 0:
            break
        try:
            launched += len(_launch_batches_for_agent(org_uuid, agent_id, headroom))
        except Exception:
            # One unhealthy workspace must not stop evaluation for the rest.
            logger.exception(
                "Trace eval tick failed for org=%s agent=%s", org_uuid, agent_id
            )

    return launched


def recover_orphaned_runs() -> int:
    """Fail runs left live by a restart, returning their traces to the queue.

    Their worker threads died with the process, so nothing will ever finish
    them. Claims are released rather than the traces failed: judging is
    idempotent from the trace's side, and any trace already judged keeps its
    `evaluated_at`, so recovery never re-runs work that landed.
    """
    recovered = 0
    for run in eval_store.unfinished_runs():
        run_uuid = run["uuid"]
        try:
            released = eval_store.release_claims(run_uuid)
            eval_store.update_eval_run(
                run_uuid,
                status=TaskStatus.FAILED.value,
                error=ORPHANED_RUN_ERROR,
                finished_at="now",
            )
        except Exception:
            logger.exception("Failed to recover orphaned trace eval run %s", run_uuid)
            continue
        recovered += 1
        logger.info(
            "Recovered orphaned trace eval run %s: %s trace(s) returned to the queue",
            run_uuid,
            released,
        )
    return recovered


async def poll_loop() -> None:
    """Run a tick every `TRACE_EVAL_POLL_SECONDS`, forever."""
    while True:
        try:
            launched = await asyncio.to_thread(run_one_tick)
            if launched:
                logger.info("Trace eval tick launched %s run(s)", launched)
        except Exception:
            logger.exception("Trace eval tick failed")
        await asyncio.sleep(poll_seconds())
