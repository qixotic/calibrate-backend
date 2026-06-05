"""Integration tests for /api-keys and API-key-authenticated test runs."""

from __future__ import annotations

import uuid
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient


@pytest.fixture(scope="module")
def app():
    import main as main_mod

    return main_mod.app


@pytest.fixture(scope="module")
def client(app):
    with patch("main.recover_pending_jobs"):
        with TestClient(app) as c:
            yield c


def _signup(client):
    suffix = uuid.uuid4().hex[:8]
    body = client.post(
        "/auth/signup",
        json={
            "first_name": "AK",
            "last_name": "U",
            "email": f"ak-{suffix}@example.com",
            "password": "passw0rd",
        },
    ).json()
    return {"Authorization": f"Bearer {body['access_token']}"}


def _create_linked_agent(client, h):
    """Create an agent + a response test linked to it; return the agent dict."""
    agent = client.post(
        "/agents",
        json={"name": f"a-{uuid.uuid4().hex[:6]}", "type": "agent"},
        headers=h,
    ).json()
    evaluators = client.get("/evaluators", headers=h).json()
    llm_ev = next(e for e in evaluators if e.get("evaluator_type") == "llm")
    test = client.post(
        "/tests",
        json={
            "name": f"t-{uuid.uuid4().hex[:6]}",
            "type": "response",
            "config": {"history": [], "evaluation": {"type": "response"}},
            "evaluators": [{"evaluator_uuid": llm_ev["uuid"]}],
        },
        headers=h,
    ).json()
    client.post(
        "/agent-tests",
        json={"agent_uuid": agent["uuid"], "test_uuids": [test["uuid"]]},
        headers=h,
    )
    return agent


# ---------------------------------------------------------------------------
# CRUD
# ---------------------------------------------------------------------------


def test_api_key_crud(client):
    h = _signup(client)

    # Create — raw key returned exactly once.
    created = client.post("/api-keys", json={"name": "ci"}, headers=h)
    assert created.status_code == 201
    body = created.json()
    assert body["name"] == "ci"
    assert body["key"].startswith("sk_")
    # Only the last 4 chars survive; masked form is ready to render.
    assert body["last_four"] == body["key"][-4:]
    assert body["masked_key"] == f"sk_••••{body['key'][-4:]}"
    assert body["last_used_at"] is None
    # Timestamps are emitted as explicit UTC so the FE doesn't read them as local.
    assert body["created_at"].endswith("Z") and "T" in body["created_at"]
    key_uuid = body["uuid"]

    # List — no raw key; only last_four / masked_key are exposed.
    listed = client.get("/api-keys", headers=h)
    assert listed.status_code == 200
    rows = listed.json()
    row = next(r for r in rows if r["uuid"] == key_uuid)
    assert row["last_four"] == body["last_four"]
    assert row["masked_key"] == body["masked_key"]
    assert all("key" not in r for r in rows)

    # Revoke.
    assert client.delete(f"/api-keys/{key_uuid}", headers=h).status_code == 204
    assert all(r["uuid"] != key_uuid for r in client.get("/api-keys", headers=h).json())

    # Revoke again → 404.
    assert client.delete(f"/api-keys/{key_uuid}", headers=h).status_code == 404


def test_api_keys_scoped_to_org(client):
    """A's key must not appear in B's listing and vice versa."""
    ha = _signup(client)
    hb = _signup(client)
    a_key = client.post("/api-keys", json={"name": "a"}, headers=ha).json()["uuid"]

    b_rows = client.get("/api-keys", headers=hb).json()
    assert all(r["uuid"] != a_key for r in b_rows)
    # B cannot revoke A's key.
    assert client.delete(f"/api-keys/{a_key}", headers=hb).status_code == 404


def test_api_keys_require_auth(client):
    assert client.get("/api-keys").status_code in (401, 403)
    assert client.post("/api-keys", json={"name": "x"}).status_code in (401, 403)


# ---------------------------------------------------------------------------
# API-key-authenticated runs
# ---------------------------------------------------------------------------


def _raw_key(client, h, name="ci"):
    return client.post("/api-keys", json={"name": name}, headers=h).json()["key"]


def test_run_with_api_key_bearer_and_header(client):
    h = _signup(client)
    agent = _create_linked_agent(client, h)
    raw = _raw_key(client, h)

    with patch(
        "routers.agent_tests.can_start_agent_test_job", return_value=False
    ), patch("threading.Thread"):
        # Authorization: Bearer sk_…
        r1 = client.post(
            f"/agent-tests/agent/{agent['uuid']}/run",
            json={},
            headers={"Authorization": f"Bearer {raw}"},
        )
        assert r1.status_code == 200, r1.text
        task_id = r1.json()["task_id"]

        # X-API-Key header
        r2 = client.post(
            f"/agent-tests/agent/{agent['uuid']}/run",
            json={},
            headers={"X-API-Key": raw},
        )
        assert r2.status_code == 200, r2.text

    # Poll status with the key.
    got = client.get(
        f"/agent-tests/run/{task_id}", headers={"X-API-Key": raw}
    )
    assert got.status_code == 200

    # Using the key touches last_used_at — populated now, and stamped UTC.
    row = next(
        r
        for r in client.get("/api-keys", headers=h).json()
        if r["last_used_at"] is not None
    )
    assert row["last_used_at"].endswith("Z") and "T" in row["last_used_at"]

    # Clean up so the queued jobs don't leak into the session-shared DB and
    # perturb global queue-ordering assertions in other test modules.
    for r in (r1, r2):
        client.delete(f"/agent-tests/job/{r.json()['task_id']}", headers=h)


def test_invalid_api_key_rejected(client):
    h = _signup(client)
    agent = _create_linked_agent(client, h)
    r = client.post(
        f"/agent-tests/agent/{agent['uuid']}/run",
        json={},
        headers={"X-API-Key": "sk_not-a-real-key"},
    )
    assert r.status_code == 401


def test_run_requires_auth(client):
    # With neither a JWT nor a key, the run endpoint now rejects (no longer open).
    h = _signup(client)
    agent = _create_linked_agent(client, h)
    r = client.post(f"/agent-tests/agent/{agent['uuid']}/run", json={})
    assert r.status_code in (401, 403)


def test_api_key_cannot_cross_org(client):
    # Owner B creates an agent and a run; attacker A holds a key for a different org.
    hb = _signup(client)
    agent_b = _create_linked_agent(client, hb)

    with patch(
        "routers.agent_tests.can_start_agent_test_job", return_value=False
    ), patch("threading.Thread"):
        run_b = client.post(
            f"/agent-tests/agent/{agent_b['uuid']}/run", json={}, headers=hb
        )
        assert run_b.status_code == 200
        task_b = run_b.json()["task_id"]

    ha = _signup(client)
    a_key = _raw_key(client, ha)

    # A's key cannot trigger a run on B's agent…
    blocked = client.post(
        f"/agent-tests/agent/{agent_b['uuid']}/run",
        json={},
        headers={"X-API-Key": a_key},
    )
    assert blocked.status_code == 404

    # …nor read B's run.
    blocked_read = client.get(
        f"/agent-tests/run/{task_b}", headers={"X-API-Key": a_key}
    )
    assert blocked_read.status_code == 404

    # Clean up B's queued job (see note in test_run_with_api_key_*).
    client.delete(f"/agent-tests/job/{task_b}", headers=hb)
