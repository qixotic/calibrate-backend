"""Store evaluator verdicts on traces.

Revision ID: 0003
Revises: 0002
Create Date: 2026-08-14

"""
import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import JSONB

revision = "0003"
down_revision = "0002"
branch_labels = None
depends_on = None

_JSON = sa.JSON().with_variant(JSONB(), "postgresql")


def upgrade() -> None:
    op.create_table(
        "trace_eval_runs",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("uuid", sa.String(36), nullable=False),
        sa.Column("org_uuid", sa.String(36), nullable=False),
        sa.Column("agent_id", sa.String(36), nullable=False),
        sa.Column("trigger", sa.String(16), nullable=False),
        sa.Column("inferred_type", sa.String(16), nullable=False),
        sa.Column("status", sa.String(16), nullable=False),
        sa.Column("trace_count", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("skipped_count", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("evaluator_snapshot", _JSON, nullable=True),
        sa.Column("error", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.Column("updated_at", sa.DateTime(), nullable=False),
        sa.Column("started_at", sa.DateTime(), nullable=True),
        sa.Column("finished_at", sa.DateTime(), nullable=True),
    )
    op.create_index("ix_trace_eval_runs_uuid", "trace_eval_runs", ["uuid"], unique=True)
    op.create_index(
        "ix_trace_eval_runs_org_agent",
        "trace_eval_runs",
        ["org_uuid", "agent_id", "created_at"],
    )
    op.create_index("ix_trace_eval_runs_status", "trace_eval_runs", ["status"])

    op.create_table(
        "trace_eval_results",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("uuid", sa.String(36), nullable=False),
        sa.Column("org_uuid", sa.String(36), nullable=False),
        sa.Column("run_uuid", sa.String(36), nullable=False),
        sa.Column("trace_uuid", sa.String(36), nullable=False),
        sa.Column("evaluator_uuid", sa.String(36), nullable=False),
        sa.Column("evaluator_version_id", sa.String(36), nullable=True),
        sa.Column("evaluator_name", sa.String(255), nullable=False),
        sa.Column("output_type", sa.String(16), nullable=False),
        sa.Column("passed", sa.Boolean(), nullable=True),
        sa.Column("score", sa.Float(), nullable=True),
        sa.Column("scale_min", sa.Float(), nullable=True),
        sa.Column("scale_max", sa.Float(), nullable=True),
        sa.Column("reasoning", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=False),
    )
    op.create_index(
        "ix_trace_eval_results_uuid", "trace_eval_results", ["uuid"], unique=True
    )
    op.create_index(
        "uq_trace_eval_results_row",
        "trace_eval_results",
        ["run_uuid", "trace_uuid", "evaluator_uuid"],
        unique=True,
    )
    op.create_index(
        "ix_trace_eval_results_org_trace",
        "trace_eval_results",
        ["org_uuid", "trace_uuid"],
    )

    op.add_column(
        "traces", sa.Column("last_eval_run_uuid", sa.String(36), nullable=True)
    )
    op.add_column("traces", sa.Column("evaluated_at", sa.DateTime(), nullable=True))
    op.create_index(
        "ix_traces_pending_eval",
        "traces",
        ["org_uuid", "agent_id", "created_at"],
        sqlite_where=sa.text("deleted_at IS NULL AND evaluated_at IS NULL"),
        postgresql_where=sa.text("deleted_at IS NULL AND evaluated_at IS NULL"),
    )


def downgrade() -> None:
    op.drop_index("ix_traces_pending_eval", table_name="traces")
    op.drop_column("traces", "evaluated_at")
    op.drop_column("traces", "last_eval_run_uuid")
    op.drop_table("trace_eval_results")
    op.drop_table("trace_eval_runs")
