"""Tests for db.create_trace_with_eval_queue and usable_evaluators_for_agent.

db.create_trace stays covered separately in test_db_traces.py; this file only
exercises the new atomic insert-trace-and-enqueue path.
"""

from __future__ import annotations

import sqlite3
import uuid

import pytest

import db


def _org() -> str:
    return str(uuid.uuid4())


def _agent(auto_score_traces: bool) -> dict:
    return {"uuid": f"agent-{uuid.uuid4().hex[:8]}", "auto_score_traces": auto_score_traces}


def _usable_evaluator(org: str) -> str:
    """Create an evaluator with a live version and no {{placeholder}} variables."""
    evaluator_uuid = db.create_evaluator(
        name=f"eval-{uuid.uuid4().hex[:6]}",
        org_uuid=org,
        owner_user_id=str(uuid.uuid4()),
    )
    version = db.create_evaluator_version(
        evaluator_uuid, judge_model="openai/gpt-4o", system_prompt="Judge this."
    )
    db.set_evaluator_live_version(evaluator_uuid, version["uuid"])
    return evaluator_uuid


def _evaluator_with_variables(org: str) -> str:
    evaluator_uuid = db.create_evaluator(
        name=f"eval-vars-{uuid.uuid4().hex[:6]}",
        org_uuid=org,
        owner_user_id=str(uuid.uuid4()),
    )
    version = db.create_evaluator_version(
        evaluator_uuid,
        judge_model="openai/gpt-4o",
        system_prompt="Judge against {{criteria}}.",
        variables=[{"name": "criteria"}],
    )
    db.set_evaluator_live_version(evaluator_uuid, version["uuid"])
    return evaluator_uuid


def _evaluator_with_no_live_version(org: str) -> str:
    return db.create_evaluator(
        name=f"eval-nolive-{uuid.uuid4().hex[:6]}",
        org_uuid=org,
        owner_user_id=str(uuid.uuid4()),
    )


def _ingest(org, agent, **overrides):
    payload = {
        "message_id": None,
        "conversation_id": "conv-1",
        "input": [{"role": "user", "content": "hi"}],
        "output": {"response": "hello", "tool_calls": None},
        "metadata": None,
    }
    payload.update(overrides)
    return db.create_trace_with_eval_queue(org_uuid=org, agent=agent, **payload)


def _queue_rows(trace_uuid: str):
    with db.get_db_connection() as conn:
        return conn.execute(
            "SELECT * FROM trace_eval_queue WHERE trace_uuid = ?", (trace_uuid,)
        ).fetchall()


def test_opted_out_agent_ingests_with_no_queue_rows():
    org = _org()
    agent = _agent(auto_score_traces=False)
    trace = _ingest(org, agent)
    assert trace["evaluators_expected"] == 0
    assert _queue_rows(trace["uuid"]) == []


def test_opted_in_agent_with_no_linked_evaluators_ingests_cleanly():
    org = _org()
    agent = _agent(auto_score_traces=True)
    trace = _ingest(org, agent)
    assert trace["evaluators_expected"] == 0
    assert _queue_rows(trace["uuid"]) == []


def test_opted_in_agent_enqueues_usable_linked_evaluator():
    org = _org()
    agent = _agent(auto_score_traces=True)
    evaluator_uuid = _usable_evaluator(org)
    db.add_evaluator_to_agent(agent["uuid"], evaluator_uuid)

    trace = _ingest(org, agent)

    assert trace["evaluators_expected"] == 1
    rows = _queue_rows(trace["uuid"])
    assert len(rows) == 1
    assert rows[0]["evaluator_uuid"] == evaluator_uuid
    assert rows[0]["evaluator_version_id"] is not None
    assert rows[0]["status"] == "pending"
    assert rows[0]["org_uuid"] == org
    assert rows[0]["agent_id"] == agent["uuid"]


def test_opted_in_agent_skips_evaluator_with_variables():
    org = _org()
    agent = _agent(auto_score_traces=True)
    db.add_evaluator_to_agent(agent["uuid"], _evaluator_with_variables(org))

    trace = _ingest(org, agent)

    assert trace["evaluators_expected"] == 0
    assert _queue_rows(trace["uuid"]) == []


def test_opted_in_agent_skips_evaluator_with_no_live_version():
    org = _org()
    agent = _agent(auto_score_traces=True)
    db.add_evaluator_to_agent(agent["uuid"], _evaluator_with_no_live_version(org))

    trace = _ingest(org, agent)

    assert trace["evaluators_expected"] == 0
    assert _queue_rows(trace["uuid"]) == []


def test_opted_in_agent_enqueues_only_usable_among_mixed_links():
    org = _org()
    agent = _agent(auto_score_traces=True)
    usable = _usable_evaluator(org)
    db.add_evaluator_to_agent(agent["uuid"], usable)
    db.add_evaluator_to_agent(agent["uuid"], _evaluator_with_variables(org))
    db.add_evaluator_to_agent(agent["uuid"], _evaluator_with_no_live_version(org))

    trace = _ingest(org, agent)

    assert trace["evaluators_expected"] == 1
    rows = _queue_rows(trace["uuid"])
    assert len(rows) == 1
    assert rows[0]["evaluator_uuid"] == usable


def test_trace_and_queue_rows_commit_together(monkeypatch):
    """A mid-transaction failure (here, a real NOT NULL violation on the queue
    insert) must leave neither the trace nor any queue row behind."""
    org = _org()
    agent = _agent(auto_score_traces=True)

    monkeypatch.setattr(
        db,
        "usable_evaluators_for_agent",
        lambda agent_id: [{"uuid": str(uuid.uuid4()), "live_version_id": None}],
    )

    with pytest.raises(sqlite3.IntegrityError):
        _ingest(org, agent, message_id="m-atomic")

    with db.get_db_connection() as conn:
        trace_count = conn.execute(
            "SELECT COUNT(*) c FROM traces WHERE org_uuid = ?", (org,)
        ).fetchone()["c"]
        queue_count = conn.execute(
            "SELECT COUNT(*) c FROM trace_eval_queue WHERE org_uuid = ?", (org,)
        ).fetchone()["c"]
    assert trace_count == 0
    assert queue_count == 0
