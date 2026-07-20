import uuid
from datetime import datetime
from typing import Any, Optional

from sqlalchemy import DateTime, Index, Integer, MetaData, String, text
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
        Index("ix_traces_org_conversation", "org_uuid", "conversation_id"),
    )
