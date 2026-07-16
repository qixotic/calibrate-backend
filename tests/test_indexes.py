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
