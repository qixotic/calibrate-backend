"""Create the traces table.

Revision ID: 0001
Revises:
Create Date: 2026-07-20

"""
import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import JSONB

revision = "0001"
down_revision = None
branch_labels = None
depends_on = None

_JSON = sa.JSON().with_variant(JSONB(), "postgresql")


def upgrade() -> None:
    op.create_table(
        "traces",
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
    )
    op.create_index("ix_traces_uuid", "traces", ["uuid"], unique=True)
    # Partial unique index: the (org, message_id) idempotency key applies to
    # live rows only, so a soft-deleted trace frees its message_id.
    op.create_index(
        "uq_traces_org_message_live",
        "traces",
        ["org_uuid", "message_id"],
        unique=True,
        sqlite_where=sa.text("deleted_at IS NULL"),
        postgresql_where=sa.text("deleted_at IS NULL"),
    )
    op.create_index("ix_traces_org_deleted", "traces", ["org_uuid", "deleted_at"])
    op.create_index(
        "ix_traces_org_conversation", "traces", ["org_uuid", "conversation_id"]
    )


def downgrade() -> None:
    op.drop_table("traces")
