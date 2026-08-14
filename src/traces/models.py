import uuid
from datetime import datetime
from typing import Any, Optional

from sqlalchemy import (
    Boolean,
    DateTime,
    Float,
    Index,
    Integer,
    MetaData,
    String,
    Text,
    text,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column
from sqlalchemy.types import JSON

# Stable constraint names keep Alembic autogenerate diffs deterministic.
NAMING_CONVENTION = {
    "ix": "ix_%(column_0_label)s",
    "uq": "uq_%(table_name)s_%(column_0_name)s",
    "ck": "ck_%(table_name)s_%(constraint_name)s",
    "fk": "fk_%(table_name)s_%(column_0_name)s_%(referred_table_name)s",
    "pk": "pk_%(table_name)s",
}

# Plain JSON on SQLite, JSONB on Postgres — part of the DSN-swap contract.
PORTABLE_JSON = JSON().with_variant(JSONB(), "postgresql")


def utcnow() -> datetime:
    # Naive UTC matches the house convention (SQLite CURRENT_TIMESTAMP in
    # pense.db is naive UTC); SQLite can't store a timezone anyway.
    return datetime.utcnow()


class TracesBase(DeclarativeBase):
    metadata = MetaData(naming_convention=NAMING_CONVENTION)


class Trace(TracesBase):
    __tablename__ = "traces"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    uuid: Mapped[str] = mapped_column(
        String(36), nullable=False, default=lambda: str(uuid.uuid4())
    )
    org_uuid: Mapped[str] = mapped_column(String(36), nullable=False)
    # No ForeignKey: `agents` lives in pense.db, a different database. Referential
    # integrity is enforced in the ingest handler instead. Becomes a real FK only
    # if agents ever move into this store.
    agent_id: Mapped[str] = mapped_column(String(36), nullable=False)
    message_id: Mapped[str] = mapped_column(String(255), nullable=False)
    conversation_id: Mapped[str] = mapped_column(String(255), nullable=False)
    input: Mapped[Any] = mapped_column(PORTABLE_JSON, nullable=False)
    output: Mapped[Any] = mapped_column(PORTABLE_JSON, nullable=False)
    # Attribute named `meta` because `metadata` is reserved on declarative
    # classes (collides with Base.metadata); the column keeps the wire name.
    meta: Mapped[Optional[Any]] = mapped_column("metadata", PORTABLE_JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, nullable=False, default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, default=utcnow, onupdate=utcnow
    )
    deleted_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    # Claim marker and completion stamp for the eval scheduler. `last_eval_run_uuid`
    # is set when a run claims the trace, `evaluated_at` only once results land,
    # so a crashed run is recognisable as claimed-but-unfinished and can be released.
    last_eval_run_uuid: Mapped[Optional[str]] = mapped_column(String(36), nullable=True)
    evaluated_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)

    __table_args__ = (
        Index("ix_traces_uuid", "uuid", unique=True),
        # The idempotency key: soft-deleting a trace frees its message_id for
        # re-ingestion, hence partial (live rows only) on both dialects.
        Index(
            "uq_traces_org_message_live",
            "org_uuid",
            "message_id",
            unique=True,
            sqlite_where=text("deleted_at IS NULL"),
            postgresql_where=text("deleted_at IS NULL"),
        ),
        Index("ix_traces_org_deleted", "org_uuid", "deleted_at"),
        Index("ix_traces_org_agent_deleted", "org_uuid", "agent_id", "deleted_at"),
        Index("ix_traces_org_conversation", "org_uuid", "conversation_id"),
        # The scheduler's hot path: oldest un-evaluated traces for one agent.
        # Partial so the index holds only the rows still awaiting a verdict,
        # which stays small even as the evaluated backlog grows without bound.
        Index(
            "ix_traces_pending_eval",
            "org_uuid",
            "agent_id",
            "created_at",
            sqlite_where=text("deleted_at IS NULL AND evaluated_at IS NULL"),
            postgresql_where=text("deleted_at IS NULL AND evaluated_at IS NULL"),
        ),
    )


class TraceEvalRun(TracesBase):
    """One batch of traces judged together by one set of evaluators.

    A batch is homogeneous by construction: the scheduler groups traces by the
    type it inferred for each, because a single calibrate invocation can only
    run one judging mode.
    """

    __tablename__ = "trace_eval_runs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    uuid: Mapped[str] = mapped_column(
        String(36), nullable=False, default=lambda: str(uuid.uuid4())
    )
    org_uuid: Mapped[str] = mapped_column(String(36), nullable=False)
    agent_id: Mapped[str] = mapped_column(String(36), nullable=False)
    trigger: Mapped[str] = mapped_column(String(16), nullable=False)
    inferred_type: Mapped[str] = mapped_column(String(16), nullable=False)
    status: Mapped[str] = mapped_column(String(16), nullable=False)
    trace_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    skipped_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    # Frozen at launch: which evaluators ran, at which version, with which
    # rubric. Read back to render a finished run even after the evaluator is
    # edited or deleted in pense.db, which this store cannot join against.
    evaluator_snapshot: Mapped[Optional[Any]] = mapped_column(
        PORTABLE_JSON, nullable=True
    )
    error: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, nullable=False, default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, default=utcnow, onupdate=utcnow
    )
    started_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    finished_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)

    __table_args__ = (
        Index("ix_trace_eval_runs_uuid", "uuid", unique=True),
        Index("ix_trace_eval_runs_org_agent", "org_uuid", "agent_id", "created_at"),
        Index("ix_trace_eval_runs_status", "status"),
    )


class TraceEvalResult(TracesBase):
    """One evaluator's verdict on one trace.

    Evaluator name, output type, and scale are copied here rather than joined:
    `evaluators` lives in pense.db, so the per-trace dialog would otherwise need
    a second database round trip per row, and a later rename or delete would
    strand finished results.
    """

    __tablename__ = "trace_eval_results"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    uuid: Mapped[str] = mapped_column(
        String(36), nullable=False, default=lambda: str(uuid.uuid4())
    )
    org_uuid: Mapped[str] = mapped_column(String(36), nullable=False)
    run_uuid: Mapped[str] = mapped_column(String(36), nullable=False)
    trace_uuid: Mapped[str] = mapped_column(String(36), nullable=False)
    evaluator_uuid: Mapped[str] = mapped_column(String(36), nullable=False)
    evaluator_version_id: Mapped[Optional[str]] = mapped_column(
        String(36), nullable=True
    )
    evaluator_name: Mapped[str] = mapped_column(String(255), nullable=False)
    output_type: Mapped[str] = mapped_column(String(16), nullable=False)
    # Exactly one of these carries the verdict: `passed` for binary evaluators,
    # `score` for rating ones. Both are null when the judge returned nothing
    # usable for that row.
    passed: Mapped[Optional[bool]] = mapped_column(Boolean, nullable=True)
    score: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    scale_min: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    scale_max: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    reasoning: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, nullable=False, default=utcnow)

    __table_args__ = (
        Index("ix_trace_eval_results_uuid", "uuid", unique=True),
        Index(
            "uq_trace_eval_results_row",
            "run_uuid",
            "trace_uuid",
            "evaluator_uuid",
            unique=True,
        ),
        # Drives the per-trace dialog.
        Index("ix_trace_eval_results_org_trace", "org_uuid", "trace_uuid"),
    )
