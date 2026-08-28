"""Trace CRUD tests for the traces helpers in src/db.py."""

from __future__ import annotations

import json
import sqlite3
import uuid

import pytest

import db


def _org() -> str:
    return str(uuid.uuid4())


def _ingest(
    org: str,
    message_id: str,
    conversation_id: str = "conv-1",
    agent_id: str = "agent-1",
    **overrides,
):
    payload = {
        "input": [
            {"role": "system", "content": "You are a vaccination assistant."},
            {"role": "user", "content": "When is the next vaccination?"},
        ],
        "output": {
            "response": "At 14 weeks, for OPV and DPT.",
            "tool_calls": [{"tool": "get_schedule", "arguments": {"child_age_weeks": 14}}],
        },
        "metadata": [{"key": "gen_ai.request.model", "value": "gpt-4"}],
    }
    payload.update(overrides)
    return db.create_trace(
        org_uuid=org,
        agent_id=agent_id,
        message_id=message_id,
        conversation_id=conversation_id,
        **payload,
    )


def test_create_and_get_roundtrip():
    org = _org()
    row = _ingest(org, "m-1")
    assert len(row["uuid"]) == 36
    assert row["message_id"] == "m-1"
    assert row["conversation_id"] == "conv-1"
    assert row["input"][0]["role"] == "system"
    assert row["output"]["tool_calls"][0]["tool"] == "get_schedule"
    assert row["metadata"][0]["key"] == "gen_ai.request.model"
    assert row["created_at"].endswith("Z") and "T" in row["created_at"]

    by_uuid = db.get_trace(org, row["uuid"])
    assert by_uuid is not None and by_uuid["uuid"] == row["uuid"]


def test_create_always_inserts():
    org = _org()
    first = _ingest(org, "m-dup")
    second = _ingest(
        org, "m-dup", output={"response": "different retry body", "tool_calls": None}
    )
    assert second["uuid"] != first["uuid"]
    assert second["output"]["response"] == "different retry body"
    assert db.count_live_traces(org) == 2


def test_soft_delete_then_reingest():
    org = _org()
    row = _ingest(org, "m-free")
    assert db.soft_delete_traces(org, trace_ids=[row["uuid"]]) == 1
    assert db.get_trace(org, row["uuid"]) is None
    assert db.count_live_traces(org) == 0

    again = _ingest(org, "m-free")
    assert again["uuid"] != row["uuid"]


def test_list_and_pagination():
    org = _org()
    _ingest(org, "m-a", conversation_id="conv-a")
    _ingest(
        org,
        "m-b",
        conversation_id="conv-b",
        input=[{"role": "user", "content": "Tell me about POLIO boosters"}],
        output={"response": "Polio boosters are due at 16 months.", "tool_calls": None},
    )
    _ingest(org, "m-c", conversation_id="conv-b")

    rows, total = db.list_traces(org, limit=50, offset=0)
    assert total == 3
    # Newest first: same-second timestamps fall back to id descending.
    assert [r["message_id"] for r in rows] == ["m-c", "m-b", "m-a"]

    page, total = db.list_traces(org, limit=1, offset=1)
    assert total == 3
    assert [r["message_id"] for r in page] == ["m-b"]


def test_bulk_delete_contract():
    org = _org()
    a = _ingest(org, "m-1", conversation_id="conv-x")
    _ingest(org, "m-2", conversation_id="conv-y")
    _ingest(org, "m-3", conversation_id="conv-y")

    # An empty id list deletes nothing.
    assert db.soft_delete_traces(org, trace_ids=[]) == 0
    assert db.count_live_traces(org) == 3
    # Unknown ids are ignored, and only the named rows go.
    assert db.soft_delete_traces(org, trace_ids=[a["uuid"], "not-a-real-uuid"]) == 1
    assert db.count_live_traces(org) == 2
    # Already-deleted rows don't count a second time.
    assert db.soft_delete_traces(org, trace_ids=[a["uuid"]]) == 0


def test_bulk_delete_splits_large_id_lists():
    """SQLite caps bound values per statement, so the delete chunks. Without
    that, a big enough list raises "too many SQL variables"."""
    org = _org()
    ids = [_ingest(org, f"m-{i}")["uuid"] for i in range(3)]
    # More IDs than SQLite allows in one statement, mostly unknown ones.
    padded = ids + [str(uuid.uuid4()) for _ in range(33_000)]

    assert db.soft_delete_traces(org, trace_ids=padded) == 3
    assert db.list_traces(org, limit=10, offset=0)[1] == 0


def test_org_isolation():
    org_a, org_b = _org(), _org()
    row_a = _ingest(org_a, "m-shared")
    row_b = _ingest(org_b, "m-shared")

    # Same message_id in two workspaces is two independent traces.
    assert row_a["uuid"] != row_b["uuid"]

    assert db.get_trace(org_a, row_b["uuid"]) is None
    rows, total = db.list_traces(org_a, limit=50, offset=0)
    assert total == 1 and rows[0]["uuid"] == row_a["uuid"]
    # Deletes never cross workspaces even with explicit foreign ids.
    assert db.soft_delete_traces(org_a, trace_ids=[row_b["uuid"]]) == 0
    assert db.count_live_traces(org_b) == 1


def test_agent_id_roundtrips():
    org = _org()
    row = _ingest(org, "m-agent", agent_id="agent-x")
    assert row["agent_id"] == "agent-x"

    by_uuid = db.get_trace(org, row["uuid"])
    assert by_uuid is not None and by_uuid["agent_id"] == "agent-x"
    rows, _ = db.list_traces(org, limit=50, offset=0)
    assert rows[0]["agent_id"] == "agent-x"


def test_list_filters_by_agent_id():
    org = _org()
    _ingest(org, "m-x1", agent_id="agent-x")
    _ingest(org, "m-x2", agent_id="agent-x")
    _ingest(org, "m-y1", agent_id="agent-y")

    rows, total = db.list_traces(org, limit=50, offset=0, agent_id="agent-x")
    assert total == 2
    assert {r["message_id"] for r in rows} == {"m-x1", "m-x2"}

    rows, total = db.list_traces(org, limit=50, offset=0, agent_id="agent-y")
    assert total == 1 and rows[0]["message_id"] == "m-y1"


def test_reused_message_id_keeps_both_turns():
    """Matching on message_id once discarded a turn; every call must store one."""
    org = _org()
    first = _ingest(org, "m-same", output={"response": "first answer"})
    second = _ingest(org, "m-same", output={"response": "second answer"})

    assert second["uuid"] != first["uuid"]
    rows, total = db.list_traces(org, limit=50, offset=0)
    assert total == 2
    assert {r["output"]["response"] for r in rows} == {"first answer", "second answer"}


def test_same_message_id_on_two_agents_is_two_rows():
    org = _org()
    first = _ingest(org, "m-dup", agent_id="agent-x")
    second = _ingest(org, "m-dup", agent_id="agent-y")

    assert second["uuid"] != first["uuid"]
    assert second["agent_id"] == "agent-y"
    assert db.count_live_traces(org) == 2


def test_get_by_uuids_keeps_caller_order_and_dedupes():
    org = _org()
    a = _ingest(org, "m-a")
    b = _ingest(org, "m-b")
    c = _ingest(org, "m-c")

    asked = [c["uuid"], a["uuid"], b["uuid"], a["uuid"]]
    rows = db.get_traces_by_uuids(org, asked)
    assert [r["uuid"] for r in rows] == [c["uuid"], a["uuid"], b["uuid"]]


def test_get_by_uuids_omits_unknown_deleted_and_foreign():
    org, other = _org(), _org()
    live = _ingest(org, "m-live")
    gone = _ingest(org, "m-gone")
    assert db.soft_delete_traces(org, trace_ids=[gone["uuid"]]) == 1
    foreign = _ingest(other, "m-foreign")

    rows = db.get_traces_by_uuids(
        org, [live["uuid"], gone["uuid"], foreign["uuid"], str(uuid.uuid4())]
    )
    assert [r["uuid"] for r in rows] == [live["uuid"]]


def test_get_by_uuids_empty_skips_the_database(monkeypatch):
    def _boom():
        raise AssertionError("empty uuid list must not open a connection")

    monkeypatch.setattr(db, "get_db_connection", _boom)
    assert db.get_traces_by_uuids(_org(), []) == []


def test_get_by_uuids_returns_parsed_rows():
    org = _org()
    row = _ingest(org, "m-shape")

    fetched = db.get_traces_by_uuids(org, [row["uuid"]])[0]
    assert fetched == db.get_trace(org, row["uuid"])
    assert fetched["input"][0]["role"] == "system"
    assert fetched["output"]["tool_calls"][0]["tool"] == "get_schedule"
    assert fetched["metadata"][0]["key"] == "gen_ai.request.model"
    assert fetched["created_at"].endswith("Z") and "T" in fetched["created_at"]


def _label_notnull() -> dict:
    with db.get_db_connection() as conn:
        return {
            row["name"]: row["notnull"]
            for row in conn.execute("PRAGMA table_info(traces)").fetchall()
            if row["name"] in ("message_id", "conversation_id")
        }


def test_create_allows_null_labels():
    org = _org()
    row = db.create_trace(
        org_uuid=org,
        agent_id="agent-1",
        input=[{"role": "user", "content": "hi"}],
        output={"response": "hello"},
    )
    assert row["message_id"] is None
    assert row["conversation_id"] is None
    assert db.get_trace(org, row["uuid"])["message_id"] is None


def _insert_agent(org: str, *, interaction_type="conversation", auto_score=False):
    agent_uuid = str(uuid.uuid4())
    with db.get_db_connection() as conn:
        conn.execute(
            "INSERT INTO agents "
            "(uuid, org_uuid, name, config, interaction_type, auto_score_traces) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (
                agent_uuid,
                org,
                f"agent-{agent_uuid[:8]}",
                "{}",
                interaction_type,
                1 if auto_score else 0,
            ),
        )
        conn.commit()
    return db.get_agent(agent_uuid)


def _eligible_evaluator(org: str, evaluator_type="llm"):
    ev = db.create_evaluator(
        name=f"eval-{uuid.uuid4().hex[:6]}",
        evaluator_type=evaluator_type,
        org_uuid=org,
        owner_user_id=str(uuid.uuid4()),
    )
    version = db.create_evaluator_version(ev, "openai/gpt-4.1", "Judge it.")
    db.set_evaluator_live_version(ev, version["uuid"])
    return ev, version["uuid"]


def _combined_ingest(org: str, agent: dict, **overrides):
    payload = {
        "message_id": None,
        "conversation_id": "conv-1",
        "input": [{"role": "user", "content": "hi"}],
        "output": {"response": "hello", "tool_calls": None},
        "metadata": None,
    }
    payload.update(overrides)
    return db.create_trace_with_eval_run(org_uuid=org, agent=agent, **payload)


def _runs_for(trace_uuid: str):
    with db.get_db_connection() as conn:
        return conn.execute(
            "SELECT * FROM trace_evaluations WHERE trace_uuid = ?",
            (trace_uuid,),
        ).fetchall()


def test_opted_out_agent_ingests_with_no_run():
    org = _org()
    agent = _insert_agent(org, auto_score=False)
    trace = _combined_ingest(org, agent)
    assert db.get_trace(org, trace["uuid"])["uuid"] == trace["uuid"]
    assert _runs_for(trace["uuid"]) == []


def test_opted_in_conversation_run_pins_response_snapshot():
    org = _org()
    agent = _insert_agent(org, interaction_type="conversation", auto_score=True)
    ev, version_id = _eligible_evaluator(org, "llm")
    db.add_evaluator_to_agent(agent["uuid"], ev)

    trace = _combined_ingest(org, agent)
    rows = _runs_for(trace["uuid"])
    assert len(rows) == 1
    run = rows[0]
    assert run["status"] == "pending"
    assert run["error"] is None
    assert run["completed_at"] is None
    assert run["org_uuid"] == org
    assert run["agent_id"] == agent["uuid"]
    assert run["criteria"] is not None
    snapshot = json.loads(run["criteria"])
    assert snapshot == {
        "type": "response",
        "evaluators": [
            {"evaluator_uuid": ev, "evaluator_version_id": version_id},
        ],
    }


def test_opted_in_general_run_pins_general_snapshot():
    org = _org()
    agent = _insert_agent(org, interaction_type="general", auto_score=True)
    ev, version_id = _eligible_evaluator(org, "llm-general")
    db.add_evaluator_to_agent(agent["uuid"], ev)

    trace = _combined_ingest(
        org,
        agent,
        input="Summarize the schedule.",
        output={"response": "Done.", "tool_calls": None},
    )
    rows = _runs_for(trace["uuid"])
    assert len(rows) == 1
    snapshot = json.loads(rows[0]["criteria"])
    assert snapshot["type"] == "general"
    assert snapshot["evaluators"] == [
        {"evaluator_uuid": ev, "evaluator_version_id": version_id},
    ]
    assert rows[0]["status"] == "pending"
    assert rows[0]["criteria"] is not None


def test_eligibility_drift_persists_skipped_run():
    org = _org()
    agent = _insert_agent(org, auto_score=True)
    ev, _ = _eligible_evaluator(org, "llm")
    db.add_evaluator_to_agent(agent["uuid"], ev)
    db.remove_evaluator_from_agent(agent["uuid"], ev)

    trace = _combined_ingest(org, agent)
    rows = _runs_for(trace["uuid"])
    assert len(rows) == 1
    assert rows[0]["status"] == "skipped"
    assert rows[0]["error"] == "no_usable_evaluators"
    assert rows[0]["criteria"] is None
    assert rows[0]["completed_at"] is not None


def test_unsupported_interaction_type_persists_skipped_run():
    org = _org()
    agent = _insert_agent(org, interaction_type="voice", auto_score=True)
    ev, _ = _eligible_evaluator(org, "llm")
    db.add_evaluator_to_agent(agent["uuid"], ev)

    trace = _combined_ingest(org, agent)
    rows = _runs_for(trace["uuid"])
    assert len(rows) == 1
    assert rows[0]["status"] == "skipped"
    assert rows[0]["error"] == "unsupported_interaction_type"
    assert rows[0]["criteria"] is None
    assert rows[0]["completed_at"] is not None


def test_empty_evaluators_plan_is_skipped_not_pending(monkeypatch):
    org = _org()
    agent = _insert_agent(org, auto_score=True)
    monkeypatch.setattr(
        "trace_scoring.resolve_scoring_plan",
        lambda _agent: {"type": "response", "evaluators": []},
    )

    trace = _combined_ingest(org, agent)
    rows = _runs_for(trace["uuid"])
    assert len(rows) == 1
    assert rows[0]["status"] == "skipped"
    assert rows[0]["error"] == "no_usable_evaluators"
    assert rows[0]["criteria"] is None


def test_trace_and_run_roll_back_together(monkeypatch):
    org = _org()
    agent = _insert_agent(org, auto_score=True)
    ev, _ = _eligible_evaluator(org, "llm")
    db.add_evaluator_to_agent(agent["uuid"], ev)

    def _boom(*_args, **_kwargs):
        raise sqlite3.IntegrityError("forced mid-transaction failure")

    monkeypatch.setattr(db, "_insert_trace_eval_run", _boom)

    with pytest.raises(sqlite3.IntegrityError):
        _combined_ingest(org, agent, message_id="m-atomic")

    with db.get_db_connection() as conn:
        trace_count = conn.execute(
            "SELECT COUNT(*) c FROM traces WHERE org_uuid = ?", (org,)
        ).fetchone()["c"]
        run_count = conn.execute(
            "SELECT COUNT(*) c FROM trace_evaluations WHERE org_uuid = ?", (org,)
        ).fetchone()["c"]
    assert trace_count == 0
    assert run_count == 0


def test_create_trace_still_inserts_without_a_run():
    org = _org()
    agent = _insert_agent(org, auto_score=True)
    ev, _ = _eligible_evaluator(org, "llm")
    db.add_evaluator_to_agent(agent["uuid"], ev)

    row = db.create_trace(
        org_uuid=org,
        agent_id=agent["uuid"],
        input=[{"role": "user", "content": "hi"}],
        output={"response": "hello"},
    )
    assert _runs_for(row["uuid"]) == []
