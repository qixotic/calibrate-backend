"""Integration tests for /tests endpoints."""

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
            "first_name": "Test",
            "last_name": "User",
            "email": f"test-{suffix}@example.com",
            "password": "passw0rd",
        },
    ).json()
    return {"Authorization": f"Bearer {body['access_token']}"}


def _raw_key(client, h, name="ci"):
    return client.post("/api-keys", json={"name": name}, headers=h).json()["key"]


def _create_test(client, headers, name=None):
    r = client.post(
        "/tests",
        json={"name": name or f"t-{uuid.uuid4().hex[:6]}", "type": "response", "config": {}},
        headers=headers,
    )
    assert r.status_code == 200, r.text
    return r.json()["uuid"]


def test_create_test_with_api_key(client):
    """POST /tests must accept an API key — currently JWT-only so this should fail with 401."""
    jwt = _signup(client)
    key = _raw_key(client, jwt)
    r = client.post(
        "/tests",
        json={"name": f"t-{uuid.uuid4().hex[:6]}", "type": "response", "config": {}},
        headers={"X-API-Key": key},
    )
    assert r.status_code == 200


def test_list_tests_with_api_key(client):
    """GET /tests accepts an X-API-Key and lists the caller's org tests."""
    jwt = _signup(client)
    key = _raw_key(client, jwt)
    t_uuid = _create_test(client, {"X-API-Key": key})
    r = client.get("/tests", headers={"X-API-Key": key})
    assert r.status_code == 200, r.text
    assert t_uuid in {t["uuid"] for t in r.json()}


def test_get_test_with_api_key(client):
    """GET /tests/{uuid} accepts an X-API-Key."""
    jwt = _signup(client)
    key = _raw_key(client, jwt)
    t_uuid = _create_test(client, {"X-API-Key": key})
    r = client.get(f"/tests/{t_uuid}", headers={"X-API-Key": key})
    assert r.status_code == 200, r.text
    assert r.json()["uuid"] == t_uuid


def test_update_test_with_api_key(client):
    """PUT /tests/{uuid} accepts an X-API-Key."""
    jwt = _signup(client)
    key = _raw_key(client, jwt)
    t_uuid = _create_test(client, {"X-API-Key": key})
    new_name = f"t-upd-{uuid.uuid4().hex[:6]}"
    r = client.put(
        f"/tests/{t_uuid}", json={"name": new_name}, headers={"X-API-Key": key}
    )
    assert r.status_code == 200, r.text
    assert r.json()["name"] == new_name


def test_bulk_create_tests_with_api_key(client):
    """POST /tests/bulk accepts an X-API-Key."""
    jwt = _signup(client)
    key = _raw_key(client, jwt)
    evaluators = client.get("/evaluators", headers=jwt).json()
    llm_ev = next(e for e in evaluators if e.get("evaluator_type") == "llm")
    ev_ref = [{"evaluator_uuid": llm_ev["uuid"]}]
    r = client.post(
        "/tests/bulk",
        json={
            "type": "response",
            "tests": [
                {
                    "name": f"bulk-{uuid.uuid4().hex[:6]}",
                    "conversation_history": [{"role": "user", "content": "hi"}],
                    "evaluators": ev_ref,
                },
                {
                    "name": f"bulk-{uuid.uuid4().hex[:6]}",
                    "conversation_history": [{"role": "user", "content": "yo"}],
                    "evaluators": ev_ref,
                },
            ],
        },
        headers={"X-API-Key": key},
    )
    assert r.status_code == 200, r.text
    assert r.json()["count"] == 2


def test_create_test_invalid_api_key(client):
    """POST /tests with a bogus key must 401."""
    r = client.post(
        "/tests",
        json={"name": f"t-{uuid.uuid4().hex[:6]}", "type": "response", "config": {}},
        headers={"X-API-Key": "bad_key"},
    )
    assert r.status_code == 401


def test_get_test_wrong_org_api_key(client):
    """A key from another org must not read a test — 404 (existence-leak parity)."""
    jwt_a = _signup(client)
    t_uuid = _create_test(client, jwt_a)

    jwt_b = _signup(client)
    key_b = _raw_key(client, jwt_b)
    r = client.get(f"/tests/{t_uuid}", headers={"X-API-Key": key_b})
    assert r.status_code == 404


def test_create_test_bearer_sk_key(client):
    """POST /tests accepts the key via Authorization: Bearer sk_…."""
    jwt = _signup(client)
    key = _raw_key(client, jwt)
    r = client.post(
        "/tests",
        json={"name": f"t-{uuid.uuid4().hex[:6]}", "type": "response", "config": {}},
        headers={"Authorization": f"Bearer {key}"},
    )
    assert r.status_code == 200, r.text
