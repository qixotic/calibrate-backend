"""Schema tests for trace_eval_queue, trace_scores, and trace_eval_errors.

PR 2 only ships the tables + soft-delete cascade -- no enqueue/claim/settle
logic exists yet, so these tests write directly to the tables via raw SQL.
"""

from __future__ import annotations

import uuid

import db


def _org() -> str:
    return str(uuid.uuid4())


def _ingest_trace(org: str, agent_id: str = "agent-1") -> dict:
    return db.create_trace(
        org_uuid=org,
        agent_id=agent_id,
        message_id=str(uuid.uuid4()),
        conversation_id="conv-1",
        input=[{"role": "user", "content": "hi"}],
        output={"response": "hello", "tool_calls": None},
        metadata=None,
    )


def _enqueue(org: str, trace_uuid: str, evaluator_uuid: str = "eval-1", **overrides):
    row = {
        "trace_uuid": trace_uuid,
        "evaluator_uuid": evaluator_uuid,
        "evaluator_version_id": 1,
        "org_uuid": org,
        "agent_id": "agent-1",
        "status": "pending",
        "available_at": 0,
        "attempts": 0,
    }
    row.update(overrides)
    with db.get_db_connection() as conn:
        conn.execute(
            "INSERT INTO trace_eval_queue "
            "(trace_uuid, evaluator_uuid, evaluator_version_id, org_uuid, "
            "agent_id, status, available_at, attempts) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (
                row["trace_uuid"],
                row["evaluator_uuid"],
                row["evaluator_version_id"],
                row["org_uuid"],
                row["agent_id"],
                row["status"],
                row["available_at"],
                row["attempts"],
            ),
        )
        conn.commit()


def _queue_rows_for_trace(trace_uuid: str):
    with db.get_db_connection() as conn:
        return conn.execute(
            "SELECT * FROM trace_eval_queue WHERE trace_uuid = ?", (trace_uuid,)
        ).fetchall()


def test_init_db_is_idempotent():
    # Running init_db() twice must not raise and must not change table
    # existence or column defaults.
    db.init_db()
    db.init_db()
    with db.get_db_connection() as conn:
        names = {
            r["name"]
            for r in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
        }
    assert {"trace_eval_queue", "trace_scores", "trace_eval_errors"} <= names


def test_new_columns_default_to_off():
    org = _org()
    trace = _ingest_trace(org)
    with db.get_db_connection() as conn:
        expected = conn.execute(
            "SELECT evaluators_expected FROM traces WHERE uuid = ?", (trace["uuid"],)
        ).fetchone()["evaluators_expected"]

        agent_uuid = str(uuid.uuid4())
        conn.execute(
            "INSERT INTO agents (uuid, org_uuid, name, config) VALUES (?, ?, ?, ?)",
            (agent_uuid, org, "test-agent", "{}"),
        )
        conn.commit()
        auto_score = conn.execute(
            "SELECT auto_score_traces FROM agents WHERE uuid = ?", (agent_uuid,)
        ).fetchone()["auto_score_traces"]

    assert expected == 0
    assert auto_score == 0


def test_trace_scores_upsert_is_idempotent_per_version():
    org = _org()
    trace = _ingest_trace(org)
    with db.get_db_connection() as conn:
        conn.execute(
            "INSERT INTO trace_scores "
            "(trace_uuid, evaluator_uuid, evaluator_version_id, org_uuid, "
            "score, reasoning, completed_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (trace["uuid"], "eval-1", 1, org, 1.0, "good", 100),
        )
        conn.commit()
        # Same (trace, evaluator, version) retried -- upsert, not a second row.
        conn.execute(
            "INSERT INTO trace_scores "
            "(trace_uuid, evaluator_uuid, evaluator_version_id, org_uuid, "
            "score, reasoning, completed_at) VALUES (?, ?, ?, ?, ?, ?, ?) "
            "ON CONFLICT (trace_uuid, evaluator_uuid, evaluator_version_id) "
            "DO UPDATE SET score = excluded.score, reasoning = excluded.reasoning, "
            "completed_at = excluded.completed_at",
            (trace["uuid"], "eval-1", 1, org, 0.5, "retried", 200),
        )
        conn.commit()
        rows = conn.execute(
            "SELECT * FROM trace_scores WHERE trace_uuid = ?", (trace["uuid"],)
        ).fetchall()
    assert len(rows) == 1
    assert rows[0]["score"] == 0.5

    # A later re-score under a NEW version coexists rather than overwriting.
    with db.get_db_connection() as conn:
        conn.execute(
            "INSERT INTO trace_scores "
            "(trace_uuid, evaluator_uuid, evaluator_version_id, org_uuid, "
            "score, reasoning, completed_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (trace["uuid"], "eval-1", 2, org, 1.0, "new version", 300),
        )
        conn.commit()
        rows = conn.execute(
            "SELECT * FROM trace_scores WHERE trace_uuid = ?", (trace["uuid"],)
        ).fetchall()
    assert len(rows) == 2


def test_soft_deleting_a_trace_deletes_its_pending_queue_rows():
    org = _org()
    trace = _ingest_trace(org)
    _enqueue(org, trace["uuid"], evaluator_uuid="eval-1", status="pending")
    _enqueue(org, trace["uuid"], evaluator_uuid="eval-2", status="processing")

    assert len(_queue_rows_for_trace(trace["uuid"])) == 2

    assert db.soft_delete_traces(org, trace_ids=[trace["uuid"]]) == 1

    assert _queue_rows_for_trace(trace["uuid"]) == []


def test_soft_deleting_a_trace_leaves_other_traces_queue_rows_alone():
    org = _org()
    kept = _ingest_trace(org)
    removed = _ingest_trace(org)
    _enqueue(org, kept["uuid"])
    _enqueue(org, removed["uuid"])

    db.soft_delete_traces(org, trace_ids=[removed["uuid"]])

    assert len(_queue_rows_for_trace(kept["uuid"])) == 1
    assert _queue_rows_for_trace(removed["uuid"]) == []


def test_soft_deleting_a_trace_scopes_queue_cascade_to_org():
    org_a = _org()
    org_b = _org()
    trace = _ingest_trace(org_a)
    _enqueue(org_a, trace["uuid"])

    # A different org's delete call for the same trace uuid (shouldn't
    # happen in practice, but the org filter must hold regardless) does
    # not touch it.
    deleted = db.soft_delete_traces(org_b, trace_ids=[trace["uuid"]])
    assert deleted == 0
    assert len(_queue_rows_for_trace(trace["uuid"])) == 1
