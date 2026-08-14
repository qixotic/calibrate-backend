"""Plain-function CRUD for trace evaluation runs and their results.

Split from `store.py` so the trace lifecycle and the judging lifecycle stay
separately readable; both sit behind the same session helper and return dicts,
so routers never see ORM objects.

Claiming is the load-bearing part. A trace is picked up by flipping its
`last_eval_run_uuid` from NULL under a WHERE that also requires NULL, so two
schedulers racing for the same backlog cannot both win a row: the second
UPDATE matches nothing. `evaluated_at` is stamped only once verdicts land, so a
claimed-but-unstamped row is exactly the crashed-run case that
`release_abandoned_claims` reopens.
"""

from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from sqlalchemy import func, select, update

from traces.engine import traces_session
from traces.models import Trace, TraceEvalResult, TraceEvalRun, utcnow

TRIGGER_AUTO = "auto"
TRIGGER_MANUAL = "manual"


def _iso(dt: Optional[datetime]) -> Optional[str]:
    return None if dt is None else dt.isoformat(timespec="seconds") + "Z"


def _run_to_dict(r: TraceEvalRun) -> Dict[str, Any]:
    return {
        "uuid": r.uuid,
        "org_uuid": r.org_uuid,
        "agent_id": r.agent_id,
        "trigger": r.trigger,
        "inferred_type": r.inferred_type,
        "status": r.status,
        "trace_count": r.trace_count,
        "skipped_count": r.skipped_count,
        "evaluator_snapshot": r.evaluator_snapshot,
        "error": r.error,
        "created_at": _iso(r.created_at),
        "updated_at": _iso(r.updated_at),
        "started_at": _iso(r.started_at),
        "finished_at": _iso(r.finished_at),
    }


def _result_to_dict(r: TraceEvalResult) -> Dict[str, Any]:
    return {
        "uuid": r.uuid,
        "run_uuid": r.run_uuid,
        "trace_uuid": r.trace_uuid,
        "evaluator_uuid": r.evaluator_uuid,
        "evaluator_version_id": r.evaluator_version_id,
        "evaluator_name": r.evaluator_name,
        "output_type": r.output_type,
        "passed": r.passed,
        "score": r.score,
        "scale_min": r.scale_min,
        "scale_max": r.scale_max,
        "reasoning": r.reasoning,
        "created_at": _iso(r.created_at),
    }


# ---------------------------------------------------------------------------
# Selecting and claiming work
# ---------------------------------------------------------------------------


def agents_with_pending_traces(limit: int = 100) -> List[Tuple[str, str]]:
    """Return `(org_uuid, agent_id)` pairs that have traces awaiting a verdict."""
    with traces_session() as s:
        rows = s.execute(
            select(Trace.org_uuid, Trace.agent_id)
            .where(
                Trace.deleted_at.is_(None),
                Trace.evaluated_at.is_(None),
                Trace.last_eval_run_uuid.is_(None),
            )
            .group_by(Trace.org_uuid, Trace.agent_id)
            .limit(limit)
        ).all()
        return [(r[0], r[1]) for r in rows]


def list_pending_traces(
    org_uuid: str, agent_id: str, *, limit: int
) -> List[Dict[str, Any]]:
    """Oldest-first traces for one agent that no run has claimed.

    Oldest-first so a high-volume agent cannot starve its own backlog: newest-
    first would let a steady stream of arrivals keep the oldest traces
    permanently unjudged.
    """
    with traces_session() as s:
        rows = s.scalars(
            select(Trace)
            .where(
                Trace.org_uuid == org_uuid,
                Trace.agent_id == agent_id,
                Trace.deleted_at.is_(None),
                Trace.evaluated_at.is_(None),
                Trace.last_eval_run_uuid.is_(None),
            )
            .order_by(Trace.created_at.asc(), Trace.id.asc())
            .limit(limit)
        ).all()
        return [
            {
                "uuid": t.uuid,
                "agent_id": t.agent_id,
                "input": t.input,
                "output": t.output,
                "message_id": t.message_id,
            }
            for t in rows
        ]


def claim_traces(org_uuid: str, run_uuid: str, trace_uuids: List[str]) -> List[str]:
    """Attach `run_uuid` to each still-unclaimed trace, returning those won.

    The NULL check inside the UPDATE is the whole concurrency guard: a second
    scheduler issuing the same statement flips zero rows and gets an empty list
    back, so the same trace is never judged twice.
    """
    if not trace_uuids:
        return []
    now = utcnow()
    with traces_session() as s:
        s.execute(
            update(Trace)
            .where(
                Trace.org_uuid == org_uuid,
                Trace.uuid.in_(trace_uuids),
                Trace.deleted_at.is_(None),
                Trace.evaluated_at.is_(None),
                Trace.last_eval_run_uuid.is_(None),
            )
            .values(last_eval_run_uuid=run_uuid, updated_at=now)
            .execution_options(synchronize_session=False)
        )
        won = s.scalars(
            select(Trace.uuid).where(
                Trace.org_uuid == org_uuid,
                Trace.uuid.in_(trace_uuids),
                Trace.last_eval_run_uuid == run_uuid,
            )
        ).all()
        return list(won)


def mark_traces_evaluated(run_uuid: str, trace_uuids: List[str]) -> int:
    """Stamp `evaluated_at`, taking the traces out of the pending set for good."""
    if not trace_uuids:
        return 0
    now = utcnow()
    with traces_session() as s:
        result = s.execute(
            update(Trace)
            .where(Trace.uuid.in_(trace_uuids), Trace.last_eval_run_uuid == run_uuid)
            .values(evaluated_at=now, updated_at=now)
            .execution_options(synchronize_session=False)
        )
        return result.rowcount or 0


def release_claims(run_uuid: str) -> int:
    """Hand a run's unjudged traces back to the pending set.

    Used when a run fails or is abandoned. Traces that already have verdicts
    keep their `evaluated_at` and are left alone, so releasing is safe to call
    on a partially finished run without re-judging what succeeded.
    """
    with traces_session() as s:
        result = s.execute(
            update(Trace)
            .where(
                Trace.last_eval_run_uuid == run_uuid,
                Trace.evaluated_at.is_(None),
            )
            .values(last_eval_run_uuid=None, updated_at=utcnow())
            .execution_options(synchronize_session=False)
        )
        return result.rowcount or 0


# ---------------------------------------------------------------------------
# Runs
# ---------------------------------------------------------------------------


def create_eval_run(
    org_uuid: str,
    agent_id: str,
    *,
    trigger: str,
    inferred_type: str,
    status: str,
    evaluator_snapshot: Optional[Any] = None,
    trace_count: int = 0,
    skipped_count: int = 0,
) -> Dict[str, Any]:
    with traces_session() as s:
        row = TraceEvalRun(
            org_uuid=org_uuid,
            agent_id=agent_id,
            trigger=trigger,
            inferred_type=inferred_type,
            status=status,
            evaluator_snapshot=evaluator_snapshot,
            trace_count=trace_count,
            skipped_count=skipped_count,
        )
        s.add(row)
        s.flush()
        return _run_to_dict(row)


def update_eval_run(run_uuid: str, **fields: Any) -> Optional[Dict[str, Any]]:
    """Patch a run. `started_at` / `finished_at` accept the sentinel `"now"`."""
    for key in ("started_at", "finished_at"):
        if fields.get(key) == "now":
            fields[key] = utcnow()
    with traces_session() as s:
        row = s.scalars(
            select(TraceEvalRun).where(TraceEvalRun.uuid == run_uuid)
        ).first()
        if not row:
            return None
        for key, value in fields.items():
            setattr(row, key, value)
        row.updated_at = utcnow()
        s.flush()
        return _run_to_dict(row)


def get_eval_run(org_uuid: str, run_uuid: str) -> Optional[Dict[str, Any]]:
    with traces_session() as s:
        row = s.scalars(
            select(TraceEvalRun).where(
                TraceEvalRun.org_uuid == org_uuid, TraceEvalRun.uuid == run_uuid
            )
        ).first()
        return _run_to_dict(row) if row else None


def list_eval_runs(
    org_uuid: str, agent_id: str, *, limit: int, offset: int
) -> Tuple[List[Dict[str, Any]], int]:
    conds = [TraceEvalRun.org_uuid == org_uuid, TraceEvalRun.agent_id == agent_id]
    with traces_session() as s:
        total = (
            s.scalar(select(func.count()).select_from(TraceEvalRun).where(*conds)) or 0
        )
        rows = s.scalars(
            select(TraceEvalRun)
            .where(*conds)
            .order_by(TraceEvalRun.created_at.desc(), TraceEvalRun.id.desc())
            .limit(limit)
            .offset(offset)
        ).all()
        return [_run_to_dict(r) for r in rows], total


def unfinished_runs() -> List[Dict[str, Any]]:
    """Runs still marked live, which after a restart means orphaned."""
    with traces_session() as s:
        rows = s.scalars(
            select(TraceEvalRun).where(
                TraceEvalRun.status.in_(("queued", "in_progress"))
            )
        ).all()
        return [_run_to_dict(r) for r in rows]


# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------


def record_results(org_uuid: str, run_uuid: str, results: List[Dict[str, Any]]) -> int:
    """Insert verdicts for one run. Returns the number written."""
    if not results:
        return 0
    with traces_session() as s:
        for r in results:
            s.add(
                TraceEvalResult(
                    org_uuid=org_uuid,
                    run_uuid=run_uuid,
                    trace_uuid=r["trace_uuid"],
                    evaluator_uuid=r["evaluator_uuid"],
                    evaluator_version_id=r.get("evaluator_version_id"),
                    evaluator_name=r["evaluator_name"],
                    output_type=r["output_type"],
                    passed=r.get("passed"),
                    score=r.get("score"),
                    scale_min=r.get("scale_min"),
                    scale_max=r.get("scale_max"),
                    reasoning=r.get("reasoning"),
                )
            )
        s.flush()
        return len(results)


def results_for_trace(org_uuid: str, trace_uuid: str) -> List[Dict[str, Any]]:
    """Every verdict recorded for one trace, newest run first."""
    with traces_session() as s:
        rows = s.scalars(
            select(TraceEvalResult)
            .where(
                TraceEvalResult.org_uuid == org_uuid,
                TraceEvalResult.trace_uuid == trace_uuid,
            )
            .order_by(TraceEvalResult.created_at.desc(), TraceEvalResult.id.desc())
        ).all()
        return [_result_to_dict(r) for r in rows]


def eval_summaries_for_traces(
    org_uuid: str, trace_uuids: List[str]
) -> Dict[str, Dict[str, Any]]:
    """Per-trace `{passed, total}` for the list badge, in one query.

    Batched deliberately: the list renders up to a full page of traces, and a
    per-row lookup would be an N+1 against the store on every page view.
    """
    if not trace_uuids:
        return {}
    scope = [
        TraceEvalResult.org_uuid == org_uuid,
        TraceEvalResult.trace_uuid.in_(trace_uuids),
    ]
    with traces_session() as s:
        totals = dict(
            s.execute(
                select(TraceEvalResult.trace_uuid, func.count())
                .where(*scope)
                .group_by(TraceEvalResult.trace_uuid)
            ).all()
        )
        passed = dict(
            s.execute(
                select(TraceEvalResult.trace_uuid, func.count())
                .where(*scope, TraceEvalResult.passed.is_(True))
                .group_by(TraceEvalResult.trace_uuid)
            ).all()
        )
    return {
        uuid: {"total": total, "passed": passed.get(uuid, 0)}
        for uuid, total in totals.items()
    }
