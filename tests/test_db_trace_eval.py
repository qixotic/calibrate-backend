"""Schema tests for trace_evaluations, trace_scores, and agents.auto_score_traces.

This slice ships tables + indexes only -- no enqueue/claim/settle logic yet,
so these tests write directly via raw SQL.
"""

from __future__ import annotations

import sqlite3
import uuid

import pytest

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


def _insert_run(org: str, trace_uuid: str, *, run_uuid: str | None = None, **overrides):
    row = {
        "uuid": run_uuid or str(uuid.uuid4()),
        "trace_uuid": trace_uuid,
        "org_uuid": org,
        "agent_id": "agent-1",
        "status": "pending",
        "criteria": None,
        "available_at": 0,
        "attempts": 0,
        "error": None,
        "created_at": 1,
        "updated_at": 1,
        "completed_at": None,
    }
    row.update(overrides)
    with db.get_db_connection() as conn:
        conn.execute(
            "INSERT INTO trace_evaluations "
            "(uuid, trace_uuid, org_uuid, agent_id, status, criteria, "
            "available_at, attempts, error, created_at, updated_at, completed_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                row["uuid"],
                row["trace_uuid"],
                row["org_uuid"],
                row["agent_id"],
                row["status"],
                row["criteria"],
                row["available_at"],
                row["attempts"],
                row["error"],
                row["created_at"],
                row["updated_at"],
                row["completed_at"],
            ),
        )
        conn.commit()
    return row["uuid"]


def _insert_score(org: str, run_uuid: str, trace_uuid: str, **overrides):
    row = {
        "run_uuid": run_uuid,
        "trace_uuid": trace_uuid,
        "evaluator_uuid": "eval-1",
        "evaluator_version_id": "version-1",
        "org_uuid": org,
        "match": 1,
        "score": None,
        "reasoning": "ok",
        "completed_at": 10,
    }
    row.update(overrides)
    with db.get_db_connection() as conn:
        conn.execute(
            "INSERT INTO trace_scores "
            "(run_uuid, trace_uuid, evaluator_uuid, evaluator_version_id, "
            "org_uuid, match, score, reasoning, completed_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                row["run_uuid"],
                row["trace_uuid"],
                row["evaluator_uuid"],
                row["evaluator_version_id"],
                row["org_uuid"],
                row["match"],
                row["score"],
                row["reasoning"],
                row["completed_at"],
            ),
        )
        conn.commit()


def test_init_db_is_idempotent():
    db.init_db()
    db.init_db()
    with db.get_db_connection() as conn:
        names = {
            r["name"]
            for r in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
        }
        indexes = {
            r["name"]
            for r in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='index'"
            ).fetchall()
        }
        cols = {
            r["name"]
            for r in conn.execute("PRAGMA table_info(agents)").fetchall()
        }
    assert {"trace_evaluations", "trace_scores"} <= names
    assert {
        "ux_trace_eval_active",
        "ix_trace_eval_claim",
        "ix_trace_eval_agent_status",
        "ix_trace_eval_trace",
        "ix_trace_scores_trace",
        "ix_trace_scores_org_eval",
    } <= indexes
    assert "auto_score_traces" in cols


def test_auto_score_traces_defaults_to_off():
    org = _org()
    agent_uuid = str(uuid.uuid4())
    with db.get_db_connection() as conn:
        conn.execute(
            "INSERT INTO agents (uuid, org_uuid, name, config) VALUES (?, ?, ?, ?)",
            (agent_uuid, org, "test-agent", "{}"),
        )
        conn.commit()
        auto_score = conn.execute(
            "SELECT auto_score_traces FROM agents WHERE uuid = ?", (agent_uuid,)
        ).fetchone()["auto_score_traces"]
    assert auto_score == 0


def test_active_run_uniqueness_rejects_a_second_open_run():
    org = _org()
    trace = _ingest_trace(org)
    _insert_run(org, trace["uuid"], status="pending")
    with pytest.raises(sqlite3.IntegrityError):
        _insert_run(org, trace["uuid"], status="processing")


def test_terminal_run_allows_a_new_open_run():
    org = _org()
    trace = _ingest_trace(org)
    first = _insert_run(org, trace["uuid"], status="completed", completed_at=5)
    second = _insert_run(org, trace["uuid"], status="pending", created_at=6)
    with db.get_db_connection() as conn:
        rows = conn.execute(
            "SELECT uuid, status FROM trace_evaluations WHERE trace_uuid = ? "
            "ORDER BY created_at",
            (trace["uuid"],),
        ).fetchall()
    assert [r["uuid"] for r in rows] == [first, second]
    assert [r["status"] for r in rows] == ["completed", "pending"]


def test_typed_result_check_accepts_binary_or_rating():
    org = _org()
    trace = _ingest_trace(org)
    run = _insert_run(org, trace["uuid"], status="completed", completed_at=5)
    _insert_score(org, run, trace["uuid"], match=0, score=None)
    _insert_score(
        org,
        run,
        trace["uuid"],
        evaluator_uuid="eval-rating",
        match=None,
        score=0.0,
    )
    with db.get_db_connection() as conn:
        rows = conn.execute(
            "SELECT evaluator_uuid, match, score FROM trace_scores "
            "WHERE run_uuid = ? ORDER BY evaluator_uuid",
            (run,),
        ).fetchall()
    assert len(rows) == 2
    by_eval = {r["evaluator_uuid"]: r for r in rows}
    assert by_eval["eval-1"]["match"] == 0
    assert by_eval["eval-1"]["score"] is None
    assert by_eval["eval-rating"]["match"] is None
    assert by_eval["eval-rating"]["score"] == 0.0


@pytest.mark.parametrize(
    "match,score",
    [
        (None, None),
        (1, 0.5),
        (2, None),
        (0, 0.0),
    ],
)
def test_typed_result_check_rejects_both_neither_and_invalid_match(match, score):
    org = _org()
    trace = _ingest_trace(org)
    run = _insert_run(org, trace["uuid"], status="completed", completed_at=5)
    with pytest.raises(sqlite3.IntegrityError):
        _insert_score(org, run, trace["uuid"], match=match, score=score)


def test_evaluator_version_id_is_required():
    org = _org()
    trace = _ingest_trace(org)
    run = _insert_run(org, trace["uuid"], status="completed", completed_at=5)
    with pytest.raises(sqlite3.IntegrityError):
        _insert_score(org, run, trace["uuid"], evaluator_version_id=None)


def test_same_version_scores_are_preserved_across_distinct_runs():
    org = _org()
    trace = _ingest_trace(org)
    first = _insert_run(org, trace["uuid"], status="completed", completed_at=5)
    _insert_score(
        org,
        first,
        trace["uuid"],
        match=1,
        score=None,
        evaluator_version_id="version-same",
        reasoning="first run",
        completed_at=5,
    )
    second = _insert_run(
        org, trace["uuid"], status="completed", created_at=6, completed_at=7
    )
    _insert_score(
        org,
        second,
        trace["uuid"],
        match=0,
        score=None,
        evaluator_version_id="version-same",
        reasoning="rescore",
        completed_at=7,
    )
    with db.get_db_connection() as conn:
        rows = conn.execute(
            "SELECT run_uuid, match, reasoning FROM trace_scores "
            "WHERE trace_uuid = ? ORDER BY completed_at",
            (trace["uuid"],),
        ).fetchall()
    assert len(rows) == 2
    assert rows[0]["run_uuid"] == first
    assert rows[0]["match"] == 1
    assert rows[0]["reasoning"] == "first run"
    assert rows[1]["run_uuid"] == second
    assert rows[1]["match"] == 0
    assert rows[1]["reasoning"] == "rescore"


def test_same_run_evaluator_is_unique():
    org = _org()
    trace = _ingest_trace(org)
    run = _insert_run(org, trace["uuid"], status="completed", completed_at=5)
    _insert_score(org, run, trace["uuid"], evaluator_version_id="v1")
    with pytest.raises(sqlite3.IntegrityError):
        _insert_score(org, run, trace["uuid"], evaluator_version_id="v2", match=0)
