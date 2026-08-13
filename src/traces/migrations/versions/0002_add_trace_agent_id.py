"""Bind each trace to an agent.

Revision ID: 0002
Revises: 0001
Create Date: 2026-08-14

"""
import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import JSONB

revision = "0002"
down_revision = "0001"
branch_labels = None
depends_on = None

_JSON = sa.JSON().with_variant(JSONB(), "postgresql")


def _create_traces(with_agent_id: bool) -> None:
    columns = [
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("uuid", sa.String(36), nullable=False),
        sa.Column("org_uuid", sa.String(36), nullable=False),
        sa.Column("message_id", sa.String(255), nullable=False),
        sa.Column("conversation_id", sa.String(255), nullable=False),
        sa.Column("input", _JSON, nullable=False),
        sa.Column("output", _JSON, nullable=False),
        sa.Column("metadata", _JSON, nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.Column("updated_at", sa.DateTime(), nullable=False),
        sa.Column("deleted_at", sa.DateTime(), nullable=True),
    ]
    if with_agent_id:
        columns.insert(3, sa.Column("agent_id", sa.String(36), nullable=False))
    op.create_table("traces", *columns)
    op.create_index("ix_traces_uuid", "traces", ["uuid"], unique=True)
    op.create_index(
        "uq_traces_org_message_live",
        "traces",
        ["org_uuid", "message_id"],
        unique=True,
        sqlite_where=sa.text("deleted_at IS NULL"),
        postgresql_where=sa.text("deleted_at IS NULL"),
    )
    op.create_index("ix_traces_org_deleted", "traces", ["org_uuid", "deleted_at"])
    if with_agent_id:
        op.create_index(
            "ix_traces_org_agent_deleted",
            "traces",
            ["org_uuid", "agent_id", "deleted_at"],
        )
    op.create_index(
        "ix_traces_org_conversation", "traces", ["org_uuid", "conversation_id"]
    )


def upgrade() -> None:
    # Drop-and-recreate rather than ALTER: SQLite cannot add a NOT NULL column
    # without a default, and its batch-mode rebuild does not reliably reflect
    # the partial unique index. Existing rows record no agent and cannot be
    # assigned one, so there is nothing to carry over.
    op.drop_table("traces")
    _create_traces(with_agent_id=True)


def downgrade() -> None:
    op.drop_table("traces")
    _create_traces(with_agent_id=False)
