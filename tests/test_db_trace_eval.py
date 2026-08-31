"""Schema tests for trace_eval_runs, trace_eval_scores, and agents.auto_score_traces.

This slice ships tables + indexes only -- no enqueue/claim/settle logic yet,
so these tests write directly via raw SQL.
"""

from __future__ import annotations

import contextlib
import sqlite3
import uuid

import pytest

import db
import trace_scoring as ts


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
        "scoring_plan": None,
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
            "INSERT INTO trace_eval_runs "
            "(uuid, trace_uuid, org_uuid, agent_id, status, scoring_plan, "
            "available_at, attempts, error, created_at, updated_at, completed_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                row["uuid"],
                row["trace_uuid"],
                row["org_uuid"],
                row["agent_id"],
                row["status"],
                row["scoring_plan"],
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
        "value": 1,
        "output_type": "binary",
        "reasoning": "ok",
        "completed_at": 10,
    }
    row.update(overrides)
    with db.get_db_connection() as conn:
        conn.execute(
            "INSERT INTO trace_eval_scores "
            "(run_uuid, trace_uuid, evaluator_uuid, evaluator_version_id, "
            "org_uuid, value, output_type, reasoning, completed_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                row["run_uuid"],
                row["trace_uuid"],
                row["evaluator_uuid"],
                row["evaluator_version_id"],
                row["org_uuid"],
                row["value"],
                row["output_type"],
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
    assert {"trace_eval_runs", "trace_eval_scores"} <= names
    assert {
        "ux_trace_eval_active",
        "ix_trace_eval_claim",
        "ix_trace_eval_agent_status",
        "ix_trace_eval_trace",
    } <= indexes
    assert "auto_score_traces" in cols


_OPEN_STATUS_IN_LIST = "status IN ('pending', 'processing')"


def test_trace_eval_run_ddl_is_frozen_not_interpolated_from_the_enum():
    """CREATE IF NOT EXISTS will not reshape; the birth DDL must stay literals
    that still agree with TraceEvalRunStatus / OPEN_TRACE_EVAL_RUN_STATUSES."""
    with db.get_db_connection() as conn:
        table_sql = conn.execute(
            "SELECT sql FROM sqlite_master WHERE type='table' AND name='trace_eval_runs'"
        ).fetchone()["sql"]
        index_sql = {
            r["name"]: r["sql"]
            for r in conn.execute(
                "SELECT name, sql FROM sqlite_master WHERE type='index' "
                "AND name IN ('ux_trace_eval_active', 'ix_trace_eval_claim')"
            )
        }
    assert "DEFAULT 'pending'" in table_sql
    assert _OPEN_STATUS_IN_LIST in index_sql["ux_trace_eval_active"]
    assert _OPEN_STATUS_IN_LIST in index_sql["ix_trace_eval_claim"]
    assert ts.TraceEvalRunStatus.PENDING.value == "pending"
    assert tuple(s.value for s in ts.OPEN_TRACE_EVAL_RUN_STATUSES) == (
        "pending",
        "processing",
    )


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
            "SELECT uuid, status FROM trace_eval_runs WHERE trace_uuid = ? "
            "ORDER BY created_at",
            (trace["uuid"],),
        ).fetchall()
    assert [r["uuid"] for r in rows] == [first, second]
    assert [r["status"] for r in rows] == ["completed", "pending"]


def test_typed_result_check_accepts_binary_or_rating():
    org = _org()
    trace = _ingest_trace(org)
    run = _insert_run(org, trace["uuid"], status="completed", completed_at=5)
    _insert_score(org, run, trace["uuid"], value=0, output_type="binary")
    _insert_score(
        org,
        run,
        trace["uuid"],
        evaluator_uuid="eval-rating",
        value=0.0,
        output_type="rating",
    )
    _insert_score(
        org,
        run,
        trace["uuid"],
        evaluator_uuid="eval-rating-high",
        value=4,
        output_type="rating",
    )
    with db.get_db_connection() as conn:
        rows = conn.execute(
            "SELECT evaluator_uuid, value, output_type FROM trace_eval_scores "
            "WHERE run_uuid = ? ORDER BY evaluator_uuid",
            (run,),
        ).fetchall()
    assert len(rows) == 3
    by_eval = {r["evaluator_uuid"]: r for r in rows}
    assert by_eval["eval-1"]["value"] == 0
    assert by_eval["eval-1"]["output_type"] == "binary"
    assert by_eval["eval-rating"]["value"] == 0.0
    assert by_eval["eval-rating"]["output_type"] == "rating"
    assert by_eval["eval-rating-high"]["value"] == 4
    assert by_eval["eval-rating-high"]["output_type"] == "rating"


@pytest.mark.parametrize(
    "kwargs",
    [
        {"value": None, "output_type": "binary"},
        {"value": 1, "output_type": None},
        {"value": 1, "output_type": "categorical"},
        {"value": 2, "output_type": "binary"},
        {"value": 0.5, "output_type": "binary"},
    ],
)
def test_typed_result_check_rejects_null_invalid_type_and_non_binary_value(kwargs):
    org = _org()
    trace = _ingest_trace(org)
    run = _insert_run(org, trace["uuid"], status="completed", completed_at=5)
    with pytest.raises(sqlite3.IntegrityError):
        _insert_score(org, run, trace["uuid"], **kwargs)


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
        value=1,
        output_type="binary",
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
        value=0,
        output_type="binary",
        evaluator_version_id="version-same",
        reasoning="rescore",
        completed_at=7,
    )
    with db.get_db_connection() as conn:
        rows = conn.execute(
            "SELECT run_uuid, value, reasoning FROM trace_eval_scores "
            "WHERE trace_uuid = ? ORDER BY completed_at",
            (trace["uuid"],),
        ).fetchall()
    assert len(rows) == 2
    assert rows[0]["run_uuid"] == first
    assert rows[0]["value"] == 1
    assert rows[0]["reasoning"] == "first run"
    assert rows[1]["run_uuid"] == second
    assert rows[1]["value"] == 0
    assert rows[1]["reasoning"] == "rescore"


def test_delete_pending_trace_eval_runs_leaves_processing_and_terminal():
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

    deleted = db.delete_pending_trace_eval_runs_for_agent(agent_id, org)
    assert deleted == 1
    with db.get_db_connection() as conn:
        remaining = {
            r["uuid"]: r["status"]
            for r in conn.execute(
                "SELECT uuid, status FROM trace_eval_runs "
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
    deleted = db.delete_pending_trace_eval_runs_for_agent(agent_id)
    assert deleted == 1
    with db.get_db_connection() as conn:
        assert (
            conn.execute(
                "SELECT 1 FROM trace_eval_runs WHERE uuid = ?", (pending,)
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
        _insert_score(org, run, trace["uuid"], evaluator_version_id="v2", value=0)


# The read helpers' user-facing contract (latest-run summary fields, full
# history, pass rule, hydration of deleted evaluators and pinned versions) is
# pinned in test_routers_traces.py. These three cover only what the endpoints
# cannot reach: the id tie-break, the helpers' own org filters, and the
# single-statement guarantee behind the "no N+1" rule.


def test_latest_run_summary_tie_breaks_on_id():
    org = _org()
    trace = _ingest_trace(org)
    first = _insert_run(
        org, trace["uuid"], status="completed", created_at=50, completed_at=50
    )
    second = _insert_run(
        org, trace["uuid"], status="completed", created_at=50, completed_at=50
    )
    _insert_score(org, first, trace["uuid"], value=1)
    _insert_score(org, second, trace["uuid"], value=0)
    with db.get_db_connection() as conn:
        ids = {
            r["uuid"]: r["id"]
            for r in conn.execute(
                "SELECT uuid, id FROM trace_eval_runs WHERE uuid IN (?, ?)",
                (first, second),
            ).fetchall()
        }
    assert ids[second] > ids[first]

    summary = db.get_latest_trace_run_summaries(org, [trace["uuid"]])[trace["uuid"]]
    assert summary["passed"] is False
    assert summary["n_passed"] == 0


def test_score_read_helpers_are_org_scoped():
    org_a, org_b = _org(), _org()
    trace = _ingest_trace(org_a)
    run = _insert_run(org_a, trace["uuid"], status="completed", completed_at=3)
    _insert_score(org_a, run, trace["uuid"], value=1)
    assert db.get_latest_trace_run_summaries(org_b, [trace["uuid"]]) == {}
    assert db.list_trace_scoring_runs(org_b, trace["uuid"]) == []
    assert (
        db.get_latest_trace_run_summaries(org_a, [trace["uuid"]])[trace["uuid"]][
            "passed"
        ]
        is True
    )
    assert db.list_trace_scoring_runs(org_a, trace["uuid"])[0]["results"][0][
        "passed"
    ] is True


def test_latest_run_summary_one_select_for_the_page(monkeypatch):
    org = _org()
    traces = [_ingest_trace(org) for _ in range(3)]
    for trace in traces:
        run = _insert_run(org, trace["uuid"], status="completed", completed_at=4)
        _insert_score(org, run, trace["uuid"], value=1)

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
    assert len(executes) == 1
