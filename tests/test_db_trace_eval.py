"""Schema tests for trace_evaluations, trace_scores, and agents.auto_score_traces.

This slice ships tables + indexes only -- no enqueue/claim/settle logic yet,
so these tests write directly via raw SQL.
"""

from __future__ import annotations

import contextlib
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


def test_delete_pending_trace_evaluations_leaves_processing_and_terminal():
    org = _org()
    agent_id = str(uuid.uuid4())
    pending_trace = _ingest_trace(org, agent_id=agent_id)
    processing_trace = _ingest_trace(org, agent_id=agent_id)
    completed_trace = _ingest_trace(org, agent_id=agent_id)
    failed_trace = _ingest_trace(org, agent_id=agent_id)
    skipped_trace = _ingest_trace(org, agent_id=agent_id)
    other_agent_trace = _ingest_trace(org, agent_id="other-agent")

    pending = _insert_run(
        org, pending_trace["uuid"], agent_id=agent_id, status="pending"
    )
    processing = _insert_run(
        org, processing_trace["uuid"], agent_id=agent_id, status="processing"
    )
    completed = _insert_run(
        org,
        completed_trace["uuid"],
        agent_id=agent_id,
        status="completed",
        completed_at=5,
    )
    failed = _insert_run(
        org, failed_trace["uuid"], agent_id=agent_id, status="failed", completed_at=6
    )
    skipped = _insert_run(
        org, skipped_trace["uuid"], agent_id=agent_id, status="skipped", completed_at=7
    )
    other_pending = _insert_run(
        org, other_agent_trace["uuid"], agent_id="other-agent", status="pending"
    )

    deleted = db.delete_pending_trace_evaluations_for_agent(agent_id, org)
    assert deleted == 1
    with db.get_db_connection() as conn:
        remaining = {
            r["uuid"]: r["status"]
            for r in conn.execute(
                "SELECT uuid, status FROM trace_evaluations "
                "WHERE uuid IN (?, ?, ?, ?, ?, ?)",
                (pending, processing, completed, failed, skipped, other_pending),
            ).fetchall()
        }
    assert pending not in remaining
    assert remaining[processing] == "processing"
    assert remaining[completed] == "completed"
    assert remaining[failed] == "failed"
    assert remaining[skipped] == "skipped"
    assert remaining[other_pending] == "pending"


def test_delete_pending_without_org_uuid_still_scopes_to_agent():
    org = _org()
    agent_id = str(uuid.uuid4())
    trace = _ingest_trace(org, agent_id=agent_id)
    pending = _insert_run(org, trace["uuid"], agent_id=agent_id, status="pending")
    deleted = db.delete_pending_trace_evaluations_for_agent(agent_id)
    assert deleted == 1
    with db.get_db_connection() as conn:
        assert (
            conn.execute(
                "SELECT 1 FROM trace_evaluations WHERE uuid = ?", (pending,)
            ).fetchone()
            is None
        )


def test_update_agent_auto_score_traces_missing_row_is_false():
    assert db.update_agent(str(uuid.uuid4()), auto_score_traces=True) is False


def test_same_run_evaluator_is_unique():
    org = _org()
    trace = _ingest_trace(org)
    run = _insert_run(org, trace["uuid"], status="completed", completed_at=5)
    _insert_score(org, run, trace["uuid"], evaluator_version_id="v1")
    with pytest.raises(sqlite3.IntegrityError):
        _insert_score(org, run, trace["uuid"], evaluator_version_id="v2", match=0)


def _make_evaluator(org, *, output_type="binary", scale=None, name=None):
    ev = db.create_evaluator(
        name=name or f"ev-{uuid.uuid4().hex[:6]}",
        org_uuid=org,
        output_type=output_type,
        evaluator_type="llm",
    )
    output_config = None
    if output_type == "rating":
        output_config = {
            "scale": scale
            or [
                {"value": 1, "name": "Low"},
                {"value": 5, "name": "High"},
            ]
        }
    version = db.create_evaluator_version(
        ev, "openai/gpt-4.1", "Judge the reply.", output_config=output_config
    )
    return ev, version["uuid"]


def test_trace_evaluator_passed_matches_cli_rule():
    assert db.trace_evaluator_passed("binary", True, None, None) is True
    assert db.trace_evaluator_passed("binary", 1, None, None) is True
    assert db.trace_evaluator_passed("binary", False, None, None) is False
    assert db.trace_evaluator_passed("binary", 0, None, None) is False
    assert db.trace_evaluator_passed("rating", None, 5, 5) is True
    assert db.trace_evaluator_passed("rating", None, 5.0, 5) is True
    assert db.trace_evaluator_passed("rating", None, 4, 5) is False
    assert db.trace_evaluator_passed("rating", None, 0, 5) is False
    assert db.trace_evaluator_passed("rating", None, 5, None) is False


def test_latest_run_summary_picks_newest_created_at():
    org = _org()
    trace = _ingest_trace(org)
    ev, ver = _make_evaluator(org)
    older = _insert_run(
        org, trace["uuid"], status="completed", created_at=10, completed_at=11
    )
    newer = _insert_run(
        org, trace["uuid"], status="completed", created_at=20, completed_at=21
    )
    _insert_score(
        org, older, trace["uuid"], evaluator_uuid=ev, evaluator_version_id=ver, match=1
    )
    _insert_score(
        org, newer, trace["uuid"], evaluator_uuid=ev, evaluator_version_id=ver, match=0
    )

    summaries = db.get_latest_trace_run_summaries(org, [trace["uuid"]])
    assert summaries[trace["uuid"]] == {
        "status": "completed",
        "passed": False,
        "n_passed": 0,
        "n_total": 1,
    }


def test_latest_run_summary_tie_breaks_on_id():
    org = _org()
    trace = _ingest_trace(org)
    ev, ver = _make_evaluator(org)
    first = _insert_run(
        org, trace["uuid"], status="completed", created_at=50, completed_at=50
    )
    second = _insert_run(
        org, trace["uuid"], status="completed", created_at=50, completed_at=50
    )
    _insert_score(
        org, first, trace["uuid"], evaluator_uuid=ev, evaluator_version_id=ver, match=1
    )
    _insert_score(
        org,
        second,
        trace["uuid"],
        evaluator_uuid=ev,
        evaluator_version_id=ver,
        match=0,
    )
    with db.get_db_connection() as conn:
        ids = {
            r["uuid"]: r["id"]
            for r in conn.execute(
                "SELECT uuid, id FROM trace_evaluations WHERE uuid IN (?, ?)",
                (first, second),
            ).fetchall()
        }
    assert ids[second] > ids[first]

    summaries = db.get_latest_trace_run_summaries(org, [trace["uuid"]])
    assert summaries[trace["uuid"]]["passed"] is False
    assert summaries[trace["uuid"]]["n_passed"] == 0


def test_latest_run_summary_omits_traces_with_no_run():
    org = _org()
    trace = _ingest_trace(org)
    assert db.get_latest_trace_run_summaries(org, [trace["uuid"]]) == {}


@pytest.mark.parametrize(
    "status,error",
    [
        ("pending", None),
        ("processing", None),
        ("failed", "judge exploded"),
        ("skipped", "no_usable_evaluators"),
    ],
)
def test_latest_run_summary_non_completed_has_null_pass_counts(status, error):
    org = _org()
    trace = _ingest_trace(org)
    _insert_run(
        org,
        trace["uuid"],
        status=status,
        error=error,
        completed_at=None if status in ("pending", "processing") else 9,
    )
    summary = db.get_latest_trace_run_summaries(org, [trace["uuid"]])[trace["uuid"]]
    assert summary == {
        "status": status,
        "passed": None,
        "n_passed": None,
        "n_total": None,
    }


def test_latest_run_summary_mixed_types_are_a_conjunction():
    org = _org()
    trace = _ingest_trace(org)
    binary, binary_ver = _make_evaluator(org, output_type="binary")
    rating, rating_ver = _make_evaluator(org, output_type="rating")
    run = _insert_run(
        org, trace["uuid"], status="completed", created_at=1, completed_at=2
    )
    _insert_score(
        org,
        run,
        trace["uuid"],
        evaluator_uuid=binary,
        evaluator_version_id=binary_ver,
        match=1,
        score=None,
    )
    _insert_score(
        org,
        run,
        trace["uuid"],
        evaluator_uuid=rating,
        evaluator_version_id=rating_ver,
        match=None,
        score=4,
    )
    summary = db.get_latest_trace_run_summaries(org, [trace["uuid"]])[trace["uuid"]]
    assert summary == {
        "status": "completed",
        "passed": False,
        "n_passed": 1,
        "n_total": 2,
    }

    with db.get_db_connection() as conn:
        conn.execute(
            "UPDATE trace_scores SET score = 5 WHERE evaluator_uuid = ?", (rating,)
        )
        conn.commit()
    both_pass = db.get_latest_trace_run_summaries(org, [trace["uuid"]])[trace["uuid"]]
    assert both_pass == {
        "status": "completed",
        "passed": True,
        "n_passed": 2,
        "n_total": 2,
    }


def test_latest_run_summary_is_org_scoped():
    org_a, org_b = _org(), _org()
    trace_a = _ingest_trace(org_a)
    ev, ver = _make_evaluator(org_a)
    run = _insert_run(org_a, trace_a["uuid"], status="completed", completed_at=3)
    _insert_score(
        org_a, run, trace_a["uuid"], evaluator_uuid=ev, evaluator_version_id=ver, match=1
    )
    assert db.get_latest_trace_run_summaries(org_b, [trace_a["uuid"]]) == {}
    assert db.get_latest_trace_run_summaries(org_a, [trace_a["uuid"]])[
        trace_a["uuid"]
    ]["passed"] is True


def test_latest_run_summary_one_select_for_the_page(monkeypatch):
    org = _org()
    traces = [_ingest_trace(org) for _ in range(3)]
    ev, ver = _make_evaluator(org)
    for trace in traces:
        run = _insert_run(org, trace["uuid"], status="completed", completed_at=4)
        _insert_score(
            org,
            run,
            trace["uuid"],
            evaluator_uuid=ev,
            evaluator_version_id=ver,
            match=1,
        )

    executes = []
    real_connect = db.get_db_connection

    class _CountingConn:
        def __init__(self, conn):
            self._conn = conn

        def execute(self, sql, params=None):
            executes.append(sql)
            if params is None:
                return self._conn.execute(sql)
            return self._conn.execute(sql, params)

        def __getattr__(self, name):
            return getattr(self._conn, name)

    @contextlib.contextmanager
    def _counting():
        with real_connect() as conn:
            yield _CountingConn(conn)

    monkeypatch.setattr(db, "get_db_connection", _counting)
    result = db.get_latest_trace_run_summaries(org, [t["uuid"] for t in traces])
    assert len(result) == 3
    data_selects = [sql for sql in executes if "ROW_NUMBER()" in sql]
    assert len(data_selects) == 1


def test_scoring_history_is_newest_first_and_keeps_deleted_evaluator():
    org = _org()
    trace = _ingest_trace(org)
    ev, ver = _make_evaluator(org, name="Original")
    older = _insert_run(
        org, trace["uuid"], status="completed", created_at=1, completed_at=2
    )
    newer = _insert_run(
        org, trace["uuid"], status="completed", created_at=3, completed_at=4
    )
    _insert_score(
        org, older, trace["uuid"], evaluator_uuid=ev, evaluator_version_id=ver, match=1
    )
    _insert_score(
        org, newer, trace["uuid"], evaluator_uuid=ev, evaluator_version_id=ver, match=0
    )
    assert db.delete_evaluator(ev) is True

    runs = db.list_trace_scoring_runs(org, trace["uuid"])
    assert [r["run_uuid"] for r in runs] == [newer, older]
    assert runs[0]["results"][0]["name"] == "Original"
    assert runs[0]["results"][0]["passed"] is False
    assert runs[1]["results"][0]["passed"] is True
    assert db.list_trace_scoring_runs(_org(), trace["uuid"]) == []


def test_scoring_history_reads_pinned_soft_deleted_version_scale():
    org = _org()
    trace = _ingest_trace(org)
    ev, v1 = _make_evaluator(
        org,
        output_type="rating",
        scale=[{"value": 1, "name": "Low"}, {"value": 5, "name": "High"}],
    )
    v2 = db.create_evaluator_version(
        ev,
        "openai/gpt-4.1",
        "Judge the reply.",
        output_config={
            "scale": [{"value": 1, "name": "Low"}, {"value": 10, "name": "High"}]
        },
    )
    assert db.set_evaluator_live_version(ev, v2["uuid"]) is True
    assert db.soft_delete_evaluator_version(ev, v1) == "deleted"

    run = _insert_run(
        org, trace["uuid"], status="completed", created_at=1, completed_at=2
    )
    _insert_score(
        org,
        run,
        trace["uuid"],
        evaluator_uuid=ev,
        evaluator_version_id=v1,
        match=None,
        score=5,
    )
    history = db.list_trace_scoring_runs(org, trace["uuid"])
    result = history[0]["results"][0]
    assert result["scale_min"] == 1
    assert result["scale_max"] == 5
    assert result["passed"] is True
    summary = db.get_latest_trace_run_summaries(org, [trace["uuid"]])[trace["uuid"]]
    assert summary["passed"] is True
