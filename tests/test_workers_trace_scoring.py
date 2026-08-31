"""Tests for the trace-scoring lifespan worker pool."""

from __future__ import annotations

import asyncio
import os
import time
import uuid
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

import db
import trace_scoring_nudge
from workers import trace_scoring as pool_mod


@pytest.fixture(scope="module", autouse=True)
def _enable_pool_for_this_module():
    pool_mod.set_pool_enabled(True)
    yield
    pool_mod.set_pool_enabled(False)
    pool_mod._active_pool = None


@pytest.fixture(scope="module")
def app():
    import main as main_mod

    return main_mod.app


@pytest.fixture
def client(app, _enable_pool_for_this_module):
    original = os.environ.get("FAKE_AI_PROVIDERS")
    os.environ["FAKE_AI_PROVIDERS"] = "1"
    with db.get_db_connection() as conn:
        conn.execute(
            "UPDATE trace_eval_runs SET available_at = 2000000000 "
            "WHERE status IN ('pending', 'processing')"
        )
        conn.commit()
    with patch("main.recover_pending_jobs"):
        with TestClient(app) as c:
            yield c
    if original is None:
        os.environ.pop("FAKE_AI_PROVIDERS", None)
    else:
        os.environ["FAKE_AI_PROVIDERS"] = original
    pool_mod._active_pool = None


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
    )
    body.raise_for_status()
    data = body.json()
    return {"Authorization": f"Bearer {data['access_token']}"}


def _create_clean_evaluator(client, h, evaluator_type="llm"):
    resp = client.post(
        "/evaluators",
        json={
            "name": f"ev-{uuid.uuid4().hex[:6]}",
            "evaluator_type": evaluator_type,
            "output_type": "binary",
            "version": {
                "judge_model": "openai/gpt-4.1",
                "system_prompt": "Judge the reply.",
            },
        },
        headers=h,
    )
    assert resp.status_code == 200, resp.text
    return resp.json()["uuid"]


def _unlink_all_evaluators(client, h, agent_uuid):
    items = client.get(f"/agents/{agent_uuid}/evaluators", headers=h).json()["items"]
    for ev in items:
        r = client.delete(f"/agents/{agent_uuid}/evaluators/{ev['uuid']}", headers=h)
        assert r.status_code == 200, r.text


def _create_opted_in_agent(client, h):
    created = client.post(
        "/agents",
        json={"name": f"a-{uuid.uuid4().hex[:6]}", "type": "agent"},
        headers=h,
    ).json()
    agent_id = created["uuid"]
    ev_uuid = _create_clean_evaluator(client, h)
    _unlink_all_evaluators(client, h, agent_id)
    r = client.post(
        f"/agents/{agent_id}/evaluators",
        json={"evaluator_ids": [ev_uuid]},
        headers=h,
    )
    assert r.status_code == 200, r.text
    enabled = client.put(
        f"/agents/{agent_id}",
        json={"auto_score_traces": True},
        headers=h,
    )
    assert enabled.status_code == 200, enabled.text
    return agent_id, ev_uuid


def _ingest(client, h, agent_id, **extra):
    payload = {
        "agent_id": agent_id,
        "message_id": f"m-{uuid.uuid4().hex[:8]}",
        "input": [{"role": "user", "content": "hi"}],
        "output": {"response": "hello there"},
    }
    payload.update(extra)
    r = client.post("/traces", json=payload, headers=h)
    assert r.status_code == 200, r.text
    return r.json()["uuid"]


def _run_for_trace(trace_uuid):
    with db.get_db_connection() as conn:
        row = conn.execute(
            "SELECT * FROM trace_eval_runs WHERE trace_uuid = ?",
            (trace_uuid,),
        ).fetchone()
    return dict(row) if row else None


def _wait_for_status(trace_uuid, wanted, timeout=10):
    deadline = time.time() + timeout
    last = None
    while time.time() < deadline:
        last = _run_for_trace(trace_uuid)
        if last and last["status"] == wanted:
            return last
        time.sleep(0.1)
    return last


def test_nudge_get_creates_an_event_when_unset():
    trace_scoring_nudge._event = None
    ev = trace_scoring_nudge.get()
    assert ev is trace_scoring_nudge.get()
    trace_scoring_nudge.set()
    assert ev.is_set()


def test_nudge_reset_works_across_event_loops():
    async def first():
        ev = trace_scoring_nudge.reset()
        trace_scoring_nudge.set()
        assert ev.is_set()

    asyncio.run(first())

    async def second():
        ev = trace_scoring_nudge.reset()
        assert not ev.is_set()
        trace_scoring_nudge.set()
        await ev.wait()

    asyncio.run(second())


def test_runnable_ingest_sets_nudge(monkeypatch):
    calls = []
    monkeypatch.setattr(trace_scoring_nudge, "set", lambda: calls.append(1))
    org = str(uuid.uuid4())
    agent_uuid = str(uuid.uuid4())
    with db.get_db_connection() as conn:
        conn.execute(
            "INSERT INTO agents "
            "(uuid, org_uuid, name, config, interaction_type, auto_score_traces) "
            "VALUES (?, ?, ?, ?, ?, 1)",
            (agent_uuid, org, "a", "{}", "conversation"),
        )
        conn.commit()
    ev = db.create_evaluator(
        name=f"e-{uuid.uuid4().hex[:6]}",
        evaluator_type="llm",
        output_type="binary",
        org_uuid=org,
        owner_user_id=str(uuid.uuid4()),
    )
    version = db.create_evaluator_version(ev, "openai/gpt-4.1", "Judge.")
    db.set_evaluator_live_version(ev, version["uuid"])
    db.add_evaluator_to_agent(agent_uuid, ev)
    agent = db.get_agent(agent_uuid)
    db.create_trace_with_eval_run(
        org_uuid=org,
        agent=agent,
        input=[{"role": "user", "content": "hi"}],
        output={"response": "hello", "tool_calls": None},
    )
    assert calls == [1]


def test_skipped_and_opted_out_ingest_do_not_nudge(monkeypatch):
    calls = []
    monkeypatch.setattr(trace_scoring_nudge, "set", lambda: calls.append(1))
    org = str(uuid.uuid4())
    opted_out = str(uuid.uuid4())
    skipped = str(uuid.uuid4())
    with db.get_db_connection() as conn:
        conn.execute(
            "INSERT INTO agents "
            "(uuid, org_uuid, name, config, interaction_type, auto_score_traces) "
            "VALUES (?, ?, ?, ?, ?, 0)",
            (opted_out, org, "off", "{}", "conversation"),
        )
        conn.execute(
            "INSERT INTO agents "
            "(uuid, org_uuid, name, config, interaction_type, auto_score_traces) "
            "VALUES (?, ?, ?, ?, ?, 1)",
            (skipped, org, "on", "{}", "conversation"),
        )
        conn.commit()
    db.create_trace_with_eval_run(
        org_uuid=org,
        agent=db.get_agent(opted_out),
        input=[{"role": "user", "content": "hi"}],
        output={"response": "hello", "tool_calls": None},
    )
    db.create_trace_with_eval_run(
        org_uuid=org,
        agent=db.get_agent(skipped),
        input=[{"role": "user", "content": "hi"}],
        output={"response": "hello", "tool_calls": None},
    )
    assert calls == []


def test_disabled_start_creates_no_tasks():
    pool_mod.set_pool_enabled(False)
    try:
        pool = pool_mod.TraceScoringPool()
        pool.start()
        assert pool._tasks == []
        assert not pool.is_running
        asyncio.run(pool.shutdown())
    finally:
        pool_mod.set_pool_enabled(True)


def test_start_is_idempotent_and_shutdown_stops_workers():
    calls = []

    def fake_batch(**kwargs):
        calls.append("batch")
        return []

    async def _exercise():
        with patch.object(pool_mod, "_run_batch", fake_batch):
            pool = pool_mod.TraceScoringPool(size=1)
            pool.start()
            first = list(pool._tasks)
            assert len(first) == 1
            pool.start()
            assert pool._tasks == first
            await asyncio.sleep(0.05)
            await pool.shutdown()
            assert all(t.done() for t in first)
            assert pool._tasks == []
            n = len(calls)
            await asyncio.sleep(0.05)
            assert len(calls) == n

    asyncio.run(_exercise())


def test_shutdown_of_a_foreign_pool_leaves_the_active_one():
    async def _exercise():
        with patch.object(pool_mod, "_run_batch", lambda: []):
            pool_mod._active_pool = None
            active = pool_mod.start_trace_scoring_pool()
            other = pool_mod.TraceScoringPool()
            other._leases = 1
            await pool_mod.shutdown_trace_scoring_pool(other)
            assert active.is_running
            await pool_mod.shutdown_trace_scoring_pool(active)
            assert pool_mod._active_pool is None

    asyncio.run(_exercise())


def test_worker_loops_immediately_after_a_nonempty_claim():
    calls = []

    def batch():
        calls.append(1)
        return ["row"] if len(calls) == 1 else []

    async def _exercise():
        with patch.object(pool_mod, "_run_batch", batch):
            pool = pool_mod.TraceScoringPool(size=1)
            pool.start()
            deadline = time.time() + 2
            while time.time() < deadline and len(calls) < 2:
                await asyncio.sleep(0.02)
            await pool.shutdown()

    asyncio.run(_exercise())
    assert len(calls) >= 2


def test_worker_exception_backoff_aborts_on_shutdown():
    def boom():
        raise RuntimeError("boom")

    async def _exercise():
        with patch.object(pool_mod, "_run_batch", boom), patch.object(
            pool_mod, "capture_exception_to_sentry", lambda e: None
        ), patch.object(pool_mod, "_ERROR_BACKOFF_SECONDS", 30):
            pool = pool_mod.TraceScoringPool(size=1)
            pool.start()
            await asyncio.sleep(0.1)
            await pool.shutdown()
            assert not pool.is_running

    asyncio.run(_exercise())


def test_overlapping_start_does_not_duplicate_the_pool():
    async def _exercise():
        with patch.object(pool_mod, "_run_batch", lambda: []):
            pool_mod._active_pool = None
            a = pool_mod.start_trace_scoring_pool()
            b = pool_mod.start_trace_scoring_pool()
            assert a is b
            assert a._leases == 2
            running = list(a._tasks)
            assert len(running) == 1
            await pool_mod.shutdown_trace_scoring_pool(a)
            assert a.is_running
            await pool_mod.shutdown_trace_scoring_pool(b)
            assert all(t.done() for t in running)
            assert pool_mod._active_pool is None

    asyncio.run(_exercise())


def test_worker_survives_batch_exceptions_and_reports_sentry():
    hits = {"n": 0}
    captured = []

    def flaky():
        hits["n"] += 1
        if hits["n"] == 1:
            raise RuntimeError("boom")
        return []

    async def _exercise():
        with patch.object(pool_mod, "_run_batch", flaky), patch.object(
            pool_mod, "capture_exception_to_sentry", lambda e: captured.append(e)
        ), patch.object(pool_mod, "_ERROR_BACKOFF_SECONDS", 0.01):
            pool = pool_mod.TraceScoringPool(size=1)
            pool.start()
            deadline = time.time() + 2
            while time.time() < deadline and hits["n"] < 2:
                await asyncio.sleep(0.05)
            await pool.shutdown()

    asyncio.run(_exercise())
    assert hits["n"] >= 2
    assert captured
    assert isinstance(captured[0], RuntimeError)


def test_idle_worker_claims_after_poll_timeout_without_nudge():
    claimed = []

    def batch():
        claimed.append(time.time())
        return []

    async def _exercise():
        with patch.object(pool_mod, "_run_batch", batch), patch.object(
            pool_mod, "POLL_SECONDS", 0.05
        ):
            pool = pool_mod.TraceScoringPool(size=1)
            pool.start()
            deadline = time.time() + 2
            while time.time() < deadline and len(claimed) < 2:
                await asyncio.sleep(0.05)
            await pool.shutdown()

    asyncio.run(_exercise())
    assert len(claimed) >= 2


def test_nudge_wakes_idle_worker_before_poll():
    calls = []

    def batch():
        calls.append(time.time())
        return []

    async def _exercise():
        with patch.object(pool_mod, "_run_batch", batch), patch.object(
            pool_mod, "POLL_SECONDS", 30
        ):
            pool = pool_mod.TraceScoringPool(size=1)
            pool.start()
            deadline = time.time() + 2
            while time.time() < deadline and not calls:
                await asyncio.sleep(0.01)
            assert calls, "worker never claimed"
            await asyncio.sleep(0.05)
            n_before = len(calls)
            started = time.time()
            trace_scoring_nudge.set()
            while time.time() - started < 2 and len(calls) <= n_before:
                await asyncio.sleep(0.01)
            elapsed = time.time() - started
            await pool.shutdown()
            assert len(calls) > n_before
            return elapsed

    elapsed = asyncio.run(_exercise())
    assert elapsed < 5


def test_shutdown_lets_in_flight_batch_finish():
    started = asyncio.Event()
    finished = []

    def slow_batch():
        started.set()
        time.sleep(0.2)
        finished.append("done")
        return ["row"]

    async def _exercise():
        with patch.object(pool_mod, "_run_batch", slow_batch):
            pool = pool_mod.TraceScoringPool(size=1)
            pool.start()
            await asyncio.wait_for(started.wait(), timeout=2)
            await pool.shutdown()

    asyncio.run(_exercise())
    assert finished == ["done"]


def test_opted_in_trace_is_scored_end_to_end(client):
    h = _signup(client)
    agent_id, ev_uuid = _create_opted_in_agent(client, h)
    trace_uuid = _ingest(client, h, agent_id)
    run = _wait_for_status(trace_uuid, "completed")
    assert run is not None, "worker never created a run"
    assert run["status"] == "completed", run
    scores = db.get_trace_eval_scores(run["uuid"])
    assert len(scores) == 1
    assert scores[0]["evaluator_uuid"] == ev_uuid
    assert scores[0]["value"] == 1
    assert scores[0]["output_type"] == "binary"
    assert "Simulated judge reasoning" in (scores[0]["reasoning"] or "")
