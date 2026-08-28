"""Verify the DB indexes exist and are actually used by their query patterns.

Existence alone doesn't prove value — a plan can still fall back to a full
table SCAN. Each usage test runs EXPLAIN QUERY PLAN and asserts the intended
index shows up in the plan (`SEARCH <table> USING INDEX <name>`).
"""

import db

EXPECTED_INDEXES = [
    "idx_annotation_items_task",
    "idx_evaluator_runs_job",
    "idx_dataset_items_dataset",
    "idx_jobs_status_type_created",
    "idx_agent_test_jobs_agent_created",
    "idx_agent_test_jobs_status",
    "idx_agent_test_jobs_share",
    "idx_simulation_jobs_sim_created",
    "idx_simulation_jobs_status",
    "idx_simulation_jobs_share",
    "idx_annotation_jobs_task",
    "idx_annotation_jobs_annotator",
    "idx_traces_org_agent_active",
    "idx_traces_org_created",
    "ux_trace_eval_active",
    "ix_trace_eval_claim",
    "ix_trace_eval_agent_status",
    "ix_trace_eval_trace",
    "ix_trace_scores_trace",
    "ix_trace_scores_org_eval",
]


def _query_plan(query, params=()):
    with db.get_db_connection() as conn:
        rows = conn.execute("EXPLAIN QUERY PLAN " + query, params).fetchall()
    return " | ".join(str(r["detail"]) for r in rows)


def test_all_indexes_exist():
    with db.get_db_connection() as conn:
        names = {
            r["name"]
            for r in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='index'"
            ).fetchall()
        }
    missing = [n for n in EXPECTED_INDEXES if n not in names]
    assert not missing, f"missing indexes: {missing}"


def test_annotation_items_by_task_uses_index():
    plan = _query_plan(
        "SELECT * FROM annotation_items WHERE task_id = ? AND deleted_at IS NULL",
        (1,),
    )
    assert "idx_annotation_items_task" in plan, plan


def test_evaluator_runs_by_job_uses_index():
    plan = _query_plan(
        "SELECT * FROM evaluator_runs WHERE job_id = ? AND deleted_at IS NULL",
        (1,),
    )
    assert "idx_evaluator_runs_job" in plan, plan


def test_dataset_items_by_dataset_uses_index():
    plan = _query_plan(
        "SELECT * FROM dataset_items WHERE dataset_id = ? AND deleted_at IS NULL",
        (1,),
    )
    assert "idx_dataset_items_dataset" in plan, plan


def test_jobs_queue_scan_uses_an_index():
    # The planner may legitimately pick a different jobs index (e.g. an
    # org/created one) for this queue query; only require it isn't a bare SCAN.
    plan = _query_plan(
        "SELECT * FROM jobs WHERE status = 'queued' "
        "AND type IN ('stt-eval','tts-eval') ORDER BY created_at ASC"
    )
    assert "USING INDEX" in plan, plan


def test_agent_test_jobs_by_agent_uses_index():
    plan = _query_plan(
        "SELECT * FROM agent_test_jobs WHERE agent_id = ? ORDER BY created_at DESC",
        (1,),
    )
    assert "idx_agent_test_jobs_agent_created" in plan, plan


def test_agent_test_jobs_by_share_token_uses_index():
    plan = _query_plan(
        "SELECT * FROM agent_test_jobs WHERE share_token = ? AND is_public = 1",
        ("tok",),
    )
    assert "idx_agent_test_jobs_share" in plan, plan


def test_simulation_jobs_by_sim_uses_index():
    plan = _query_plan(
        "SELECT * FROM simulation_jobs WHERE simulation_id = ? ORDER BY created_at DESC",
        (1,),
    )
    assert "idx_simulation_jobs_sim_created" in plan, plan


def test_annotation_jobs_by_task_uses_index():
    plan = _query_plan(
        "SELECT * FROM annotation_jobs WHERE task_id = ? AND deleted_at IS NULL",
        (1,),
    )
    assert "idx_annotation_jobs_task" in plan, plan


def test_annotation_jobs_by_annotator_uses_index():
    plan = _query_plan(
        "SELECT * FROM annotation_jobs WHERE annotator_id = ? AND deleted_at IS NULL",
        (1,),
    )
    assert "idx_annotation_jobs_annotator" in plan, plan


def test_traces_by_agent_uses_index():
    # The count query behind the agent filter. Its page query is left out on
    # purpose: with an ORDER BY the planner prefers idx_traces_org_created,
    # which hands back the rows already sorted.
    plan = _query_plan(
        "SELECT COUNT(*) FROM traces WHERE org_uuid = ? AND deleted_at IS NULL "
        "AND agent_id = ?",
        ("org", "agent"),
    )
    assert "idx_traces_org_agent_active" in plan, plan


def test_traces_default_list_sorts_from_the_index():
    """The unfiltered list query must read rows already in newest-first order —
    a TEMP B-TREE here means every workspace row is sorted on each page."""
    plan = _query_plan(
        "SELECT * FROM traces WHERE org_uuid = ? AND deleted_at IS NULL "
        "ORDER BY created_at DESC, id DESC",
        ("org",),
    )
    assert "idx_traces_org_created" in plan, plan
    assert "TEMP B-TREE" not in plan, plan


def test_trace_eval_active_lookup_uses_index():
    plan = _query_plan(
        "SELECT * FROM trace_evaluations WHERE trace_uuid = ? "
        "AND status IN ('pending', 'processing')",
        ("trace",),
    )
    assert "ux_trace_eval_active" in plan, plan


def test_trace_eval_claim_uses_index():
    plan = _query_plan(
        "SELECT * FROM trace_evaluations WHERE status IN ('pending', 'processing') "
        "AND available_at <= ? ORDER BY available_at LIMIT 10",
        (0,),
    )
    assert "ix_trace_eval_claim" in plan, plan


def test_trace_eval_agent_status_uses_index():
    plan = _query_plan(
        "SELECT * FROM trace_evaluations WHERE agent_id = ? AND status = ? "
        "ORDER BY completed_at",
        ("agent", "failed"),
    )
    assert "ix_trace_eval_agent_status" in plan, plan


def test_trace_eval_history_uses_index():
    plan = _query_plan(
        "SELECT * FROM trace_evaluations WHERE trace_uuid = ? "
        "ORDER BY created_at DESC",
        ("trace",),
    )
    assert "ix_trace_eval_trace" in plan, plan
    assert "TEMP B-TREE" not in plan, plan


def test_trace_scores_by_trace_uses_index():
    plan = _query_plan(
        "SELECT * FROM trace_scores WHERE trace_uuid = ? ORDER BY completed_at",
        ("trace",),
    )
    assert "ix_trace_scores_trace" in plan, plan


def test_trace_scores_by_org_eval_uses_index():
    plan = _query_plan(
        "SELECT * FROM trace_scores WHERE org_uuid = ? AND evaluator_uuid = ? "
        "ORDER BY completed_at",
        ("org", "eval"),
    )
    assert "ix_trace_scores_org_eval" in plan, plan
