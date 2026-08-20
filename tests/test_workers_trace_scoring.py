"""Integration tests for the trace-scoring worker pool (src/workers/trace_scoring.py).

Runs against the real FastAPI lifespan (so the pool actually starts/stops),
with FAKE_AI_PROVIDERS=1 for this module only so calibrate-agent invocations
route to the deterministic in-repo fake instead of a real provider call.

Isolation note: this module's TestClient starts a live worker pool for the
whole module. Any OTHER test that opts an agent into auto_score_traces AND
ingests a trace, inside a TestClient-backed test of its own, will have that
trace raced by whatever worker pool its own module's lifespan started --
harmless today because no other test file does both, but worth knowing if
one ever does.
"""

from __future__ import annotations

import os
import time
import uuid

import pytest
from fastapi.testclient import TestClient

import db


@pytest.fixture(scope="module", autouse=True)
def _fake_ai_providers():
    original = os.environ.get("FAKE_AI_PROVIDERS")
    os.environ["FAKE_AI_PROVIDERS"] = "1"
    yield
    if original is None:
        os.environ.pop("FAKE_AI_PROVIDERS", None)
    else:
        os.environ["FAKE_AI_PROVIDERS"] = original


@pytest.fixture(scope="module")
def app():
    import main as main_mod

    return main_mod.app


@pytest.fixture(scope="module")
def client(app):
    from unittest.mock import patch

    with patch("main.recover_pending_jobs"):
        with TestClient(app) as c:
            yield c


def _signup(client):
    suffix = uuid.uuid4().hex[:8]
    body = client.post(
        "/auth/signup",
        json={
            "first_name": "Wk",
            "last_name": "Er",
            "email": f"wk-{suffix}@example.com",
            "password": "passw0rd",
        },
    ).json()
    return {"Authorization": f"Bearer {body['access_token']}"}


def _create_agent(client, h, auto_score_traces=False):
    created = client.post(
        "/agents",
        json={"name": f"a-{uuid.uuid4().hex[:6]}", "type": "agent"},
        headers=h,
    ).json()
    if auto_score_traces:
        r = client.put(
            f"/agents/{created['uuid']}",
            json={"auto_score_traces": True},
            headers=h,
        )
        assert r.status_code == 200, r.text
    return created["uuid"]


def _create_evaluator(client, h):
    res = client.post(
        "/evaluators",
        json={
            "name": f"ev-{uuid.uuid4().hex[:6]}",
            "evaluator_type": "llm",
            "output_type": "binary",
            "version": {
                "judge_model": "openai/gpt-4.1",
                "system_prompt": "Judge the reply.",
            },
        },
        headers=h,
    )
    assert res.status_code == 200, res.text
    return res.json()["uuid"]


def _link_evaluator(client, h, agent_id, evaluator_id):
    r = client.post(
        f"/agents/{agent_id}/evaluators",
        json={"evaluator_ids": [evaluator_id]},
        headers=h,
    )
    assert r.status_code == 200, r.text


def _ingest_trace(client, h, agent_id):
    r = client.post(
        "/traces",
        json={
            "agent_id": agent_id,
            "message_id": f"m-{uuid.uuid4().hex[:8]}",
            "input": [{"role": "user", "content": "hi"}],
            "output": {"response": "hello there"},
        },
        headers=h,
    )
    assert r.status_code == 200, r.text
    return r.json()["uuid"]


def _wait_for_score(org_uuid, trace_uuid, evaluator_uuid, timeout=10):
    deadline = time.time() + timeout
    while time.time() < deadline:
        with db.get_db_connection() as conn:
            row = conn.execute(
                "SELECT * FROM trace_scores WHERE trace_uuid = ? AND evaluator_uuid = ?",
                (trace_uuid, evaluator_uuid),
            ).fetchone()
        if row:
            return row
        time.sleep(0.2)
    return None


def test_trace_on_opted_in_agent_gets_scored_end_to_end(client):
    h = _signup(client)
    agent_id = _create_agent(client, h, auto_score_traces=True)
    evaluator_id = _create_evaluator(client, h)
    _link_evaluator(client, h, agent_id, evaluator_id)

    trace_uuid = _ingest_trace(client, h, agent_id)

    with db.get_db_connection() as conn:
        org_uuid = conn.execute(
            "SELECT org_uuid FROM traces WHERE uuid = ?", (trace_uuid,)
        ).fetchone()["org_uuid"]

    score_row = _wait_for_score(org_uuid, trace_uuid, evaluator_id)
    assert score_row is not None, "trace was never scored by the worker pool"
    assert score_row["score"] == 1.0
    assert "Simulated judge reasoning" in score_row["reasoning"]

    with db.get_db_connection() as conn:
        queue_row = conn.execute(
            "SELECT * FROM trace_eval_queue WHERE trace_uuid = ?", (trace_uuid,)
        ).fetchone()
    assert queue_row is None


def test_opted_out_agent_traces_are_never_claimed(client):
    h = _signup(client)
    agent_id = _create_agent(client, h, auto_score_traces=False)

    trace_uuid = _ingest_trace(client, h, agent_id)

    # No queue row was ever created for an opted-out agent (PR 4's job), so
    # there is nothing for the worker pool to claim -- give it a moment
    # anyway and confirm no score/error ever materializes.
    time.sleep(1)
    with db.get_db_connection() as conn:
        scores = conn.execute(
            "SELECT * FROM trace_scores WHERE trace_uuid = ?", (trace_uuid,)
        ).fetchall()
        errors = conn.execute(
            "SELECT * FROM trace_eval_errors WHERE trace_uuid = ?", (trace_uuid,)
        ).fetchall()
    assert scores == []
    assert errors == []


def test_pool_shutdown_leaves_no_task_running():
    from workers.trace_scoring import TraceScoringPool

    import asyncio

    async def _exercise():
        pool = TraceScoringPool(size=2)
        pool.start()
        await asyncio.sleep(0.05)
        await pool.shutdown()
        return pool._tasks

    tasks = asyncio.run(_exercise())
    assert all(t.done() for t in tasks)
