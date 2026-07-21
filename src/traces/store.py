"""Plain-function CRUD over the traces store.

Routers call these functions and get dicts back (mirroring db.py's
ergonomics); ORM sessions and Trace instances never leave this package.
Unlike the shared pagination helpers, search/filter/count here run in SQL —
traces are machine-written and can outgrow post-fetch filtering fast.
"""

from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from sqlalchemy import Text, cast, func, or_, select, update
from sqlalchemy.exc import IntegrityError

from traces.engine import traces_session
from traces.models import Trace, utcnow


def _iso(dt: Optional[datetime]) -> Optional[str]:
    # Explicit UTC marker so browsers don't parse the timestamp as local time
    # (same concern as _to_utc_iso in routers/api_keys.py).
    return None if dt is None else dt.isoformat(timespec="seconds") + "Z"


def _to_dict(t: Trace) -> Dict[str, Any]:
    return {
        "uuid": t.uuid,
        "org_uuid": t.org_uuid,
        "message_id": t.message_id,
        "conversation_id": t.conversation_id,
        "input": t.input,
        "output": t.output,
        "metadata": t.meta,
        "created_at": _iso(t.created_at),
        "updated_at": _iso(t.updated_at),
    }


def _live(org_uuid: str) -> List[Any]:
    return [Trace.org_uuid == org_uuid, Trace.deleted_at.is_(None)]


def _like_escape(needle: str) -> str:
    return needle.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")


def _search_condition(q: str):
    needle = f"%{_like_escape(q.strip().lower())}%"

    def _ilike(col):
        return func.lower(col).like(needle, escape="\\")

    # Matching the raw JSON text of input/output is a documented approximation
    # (it also matches keys and quoting artifacts); real JSON search waits for
    # a backend that has it.
    return or_(
        _ilike(Trace.message_id),
        _ilike(Trace.conversation_id),
        _ilike(cast(Trace.input, Text)),
        _ilike(cast(Trace.output, Text)),
    )


def _filters(
    org_uuid: str, q: Optional[str] = None, conversation_id: Optional[str] = None
) -> List[Any]:
    conds = _live(org_uuid)
    if conversation_id:
        conds.append(Trace.conversation_id == conversation_id)
    if q and q.strip():
        conds.append(_search_condition(q))
    return conds


def get_trace_by_message_id(org_uuid: str, message_id: str) -> Optional[Dict[str, Any]]:
    with traces_session() as s:
        row = s.scalars(
            select(Trace).where(*_live(org_uuid), Trace.message_id == message_id)
        ).first()
        return _to_dict(row) if row else None


def get_trace(org_uuid: str, trace_uuid: str) -> Optional[Dict[str, Any]]:
    with traces_session() as s:
        row = s.scalars(
            select(Trace).where(*_live(org_uuid), Trace.uuid == trace_uuid)
        ).first()
        return _to_dict(row) if row else None


def count_live_traces(org_uuid: str) -> int:
    with traces_session() as s:
        return (
            s.scalar(select(func.count()).select_from(Trace).where(*_live(org_uuid)))
            or 0
        )


def create_trace(
    org_uuid: str,
    message_id: str,
    conversation_id: str,
    input: Any,
    output: Any,
    metadata: Optional[Any] = None,
) -> Tuple[Dict[str, Any], bool]:
    """Insert a trace, returning `(row, created)`.

    Idempotent on (org_uuid, message_id): a duplicate returns the existing live
    row with created=False. SELECT-then-INSERT with an IntegrityError fallback
    is portable across SQLite and Postgres; the partial unique index closes the
    race window.
    """
    existing = get_trace_by_message_id(org_uuid, message_id)
    if existing:
        return existing, False
    try:
        with traces_session() as s:
            row = Trace(
                org_uuid=org_uuid,
                message_id=message_id,
                conversation_id=conversation_id,
                input=input,
                output=output,
                meta=metadata,
            )
            s.add(row)
            s.flush()
            return _to_dict(row), True
    except IntegrityError:
        existing = get_trace_by_message_id(org_uuid, message_id)
        if existing:
            return existing, False
        raise


def list_traces(
    org_uuid: str,
    *,
    limit: int,
    offset: int,
    q: Optional[str] = None,
    conversation_id: Optional[str] = None,
) -> Tuple[List[Dict[str, Any]], int]:
    """Return `(page, total)` newest-first; filters and count run in SQL."""
    conds = _filters(org_uuid, q, conversation_id)
    with traces_session() as s:
        total = s.scalar(select(func.count()).select_from(Trace).where(*conds)) or 0
        rows = s.scalars(
            select(Trace)
            .where(*conds)
            .order_by(Trace.created_at.desc(), Trace.id.desc())
            .limit(limit)
            .offset(offset)
        ).all()
        return [_to_dict(r) for r in rows], total


def select_traces(
    org_uuid: str,
    *,
    trace_ids: Optional[List[str]] = None,
    select_all: bool = False,
    q: Optional[str] = None,
    conversation_id: Optional[str] = None,
    limit: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """Return the full trace dicts a bulk action targets (same selection contract
    as `soft_delete_traces`), for callers that need the bodies rather than just a
    delete count (e.g. converting traces to tests).

    `select_all=True` returns every live trace matching `q`/`conversation_id`
    newest-first (pass `limit` to bound a large match); otherwise the given
    `trace_ids` in request order (missing/foreign ids skipped).
    """
    with traces_session() as s:
        if select_all:
            stmt = (
                select(Trace)
                .where(*_filters(org_uuid, q, conversation_id))
                .order_by(Trace.created_at.desc(), Trace.id.desc())
            )
            if limit is not None:
                stmt = stmt.limit(limit)
            return [_to_dict(r) for r in s.scalars(stmt).all()]
        if not trace_ids:
            return []
        rows = s.scalars(
            select(Trace).where(*_live(org_uuid), Trace.uuid.in_(trace_ids))
        ).all()
        by_uuid = {r.uuid: r for r in rows}
        return [_to_dict(by_uuid[tid]) for tid in trace_ids if tid in by_uuid]


def soft_delete_traces(
    org_uuid: str,
    *,
    trace_ids: Optional[List[str]] = None,
    select_all: bool = False,
    q: Optional[str] = None,
    conversation_id: Optional[str] = None,
) -> int:
    """Soft-delete traces, returning the number of rows flipped.

    Mirrors the annotation-task bulk contract: select_all=True targets every
    live trace matching q/conversation_id and ignores trace_ids; otherwise only
    the given trace_ids are deleted (empty list deletes nothing).
    """
    if select_all:
        conds = _filters(org_uuid, q, conversation_id)
    else:
        if not trace_ids:
            return 0
        conds = _live(org_uuid) + [Trace.uuid.in_(trace_ids)]
    now = utcnow()
    with traces_session() as s:
        result = s.execute(
            update(Trace)
            .where(*conds)
            .values(deleted_at=now, updated_at=now)
            .execution_options(synchronize_session=False)
        )
        return result.rowcount or 0
