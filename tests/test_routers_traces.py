"""Integration tests for the /traces router."""

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
            "first_name": "Tr",
            "last_name": "U",
            "email": f"tr-{suffix}@example.com",
            "password": "passw0rd",
        },
    ).json()
    return {"Authorization": f"Bearer {body['access_token']}"}


def _api_key_headers(client, h):
    created = client.post("/api-keys", json={"name": "ingest"}, headers=h)
    assert created.status_code == 201, created.text
    return {"X-API-Key": created.json()["key"]}


def _payload(message_id: str, conversation_id: str = "conv-1", **overrides):
    payload = {
        "message_id": message_id,
        "conversation_id": conversation_id,
        "input": [
            {"role": "system", "content": "You are a vaccination assistant."},
            {"role": "user", "content": "When is my daughter's next vaccination?"},
        ],
        "output": {
            "response": "Aapki beti ka agla vaccination 14 weeks pe hai.",
            "tool_calls": [
                {"tool": "get_schedule", "arguments": {"child_age_weeks": 14}}
            ],
        },
        "metadata": [{"key": "gen_ai.request.model", "value": "gpt-4"}],
    }
    payload.update(overrides)
    return payload


def _mid() -> str:
    return f"m-{uuid.uuid4().hex[:10]}"


# ---------------------------------------------------------------------------
# Ingestion
# ---------------------------------------------------------------------------


def test_ingest_requires_auth(client):
    assert client.post("/traces", json=_payload(_mid())).status_code in (401, 403)
    assert (
        client.post(
            "/traces", json=_payload(_mid()), headers={"X-API-Key": "sk_bogus"}
        ).status_code
        == 401
    )


def test_ingest_with_jwt_is_idempotent(client):
    h = _signup(client)
    mid = _mid()

    first = client.post("/traces", json=_payload(mid), headers=h)
    assert first.status_code == 200, first.text
    body = first.json()
    assert body["created"] is True
    assert len(body["uuid"]) == 36
    assert body["message_id"] == mid
    assert body["conversation_id"] == "conv-1"
    assert body["created_at"].endswith("Z") and "T" in body["created_at"]

    retry = client.post("/traces", json=_payload(mid), headers=h)
    assert retry.status_code == 200
    assert retry.json()["created"] is False
    assert retry.json()["uuid"] == body["uuid"]


def test_ingest_with_api_key(client):
    h = _signup(client)
    key_headers = _api_key_headers(client, h)

    res = client.post("/traces", json=_payload(_mid()), headers=key_headers)
    assert res.status_code == 200, res.text
    assert res.json()["created"] is True


def test_ingest_validation(client):
    h = _signup(client)

    # output is required.
    bad = _payload(_mid())
    del bad["output"]
    assert client.post("/traces", json=bad, headers=h).status_code == 422

    # output needs a response or at least one tool call.
    empty_output = _payload(_mid(), output={"response": "  ", "tool_calls": None})
    assert client.post("/traces", json=empty_output, headers=h).status_code == 422

    # Tool-call-only turns are legal.
    tool_only = _payload(
        _mid(), output={"tool_calls": [{"tool": "get_schedule", "arguments": {}}]}
    )
    ok = client.post("/traces", json=tool_only, headers=h)
    assert ok.status_code == 200 and ok.json()["created"] is True

    # input must be non-empty.
    assert (
        client.post("/traces", json=_payload(_mid(), input=[]), headers=h).status_code
        == 422
    )

    # Unknown top-level keys are rejected; new needs belong in metadata.
    extra_top = _payload(_mid())
    extra_top["custom_fields"] = []
    assert client.post("/traces", json=extra_top, headers=h).status_code == 422

    # Metadata entries are strict {key, value} pairs.
    bad_meta = _payload(_mid(), metadata=[{"key": "k", "value": "v", "extra": 1}])
    assert client.post("/traces", json=bad_meta, headers=h).status_code == 422

    # OpenAI-format extras on input turns pass through (tool_calls, tool_call_id).
    openai_history = _payload(
        _mid(),
        input=[
            {"role": "user", "content": "check the schedule"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "get_schedule", "arguments": "{}"},
                    }
                ],
            },
            {"role": "tool", "content": "{\"weeks\": 14}", "tool_call_id": "call_1"},
        ],
    )
    ok = client.post("/traces", json=openai_history, headers=h)
    assert ok.status_code == 200 and ok.json()["created"] is True


def test_ingest_cap_returns_429_but_keeps_retries_idempotent(client, monkeypatch):
    from routers import org_limits as org_limits_mod

    h = _signup(client)
    monkeypatch.setattr(org_limits_mod, "DEFAULT_MAX_TRACES", 1)

    first_mid = _mid()
    assert client.post("/traces", json=_payload(first_mid), headers=h).status_code == 200

    capped = client.post("/traces", json=_payload(_mid()), headers=h)
    assert capped.status_code == 429
    detail = capped.json()["detail"]
    assert detail["current"] == 1
    assert detail["max_traces"] == 1
    assert "hint" in detail

    # A retry of an already-stored message_id still succeeds at the cap.
    retry = client.post("/traces", json=_payload(first_mid), headers=h)
    assert retry.status_code == 200
    assert retry.json()["created"] is False
