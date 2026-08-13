"""Fills coverage gaps in the evaluators, tests, agents, and auth routers."""

from __future__ import annotations

import uuid
from unittest.mock import AsyncMock, MagicMock, patch

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
            "first_name": "X",
            "last_name": "U",
            "email": f"xu-{suffix}@example.com",
            "password": "passw0rd",
        },
    ).json()
    return {
        "headers": {"Authorization": f"Bearer {body['access_token']}"},
        "user_uuid": body["user"]["uuid"],
    }


# ---------------------------------------------------------------------------
# Evaluators — create, version, set live, preview prompt, duplicate, delete
# ---------------------------------------------------------------------------


def test_evaluators_full_lifecycle(client):
    auth = _signup(client)
    h = auth["headers"]

    # Create a binary LLM evaluator
    name = f"ev-{uuid.uuid4().hex[:6]}"
    create = client.post(
        "/evaluators",
        json={
            "name": name,
            "description": "d",
            "evaluator_type": "llm",
            "data_type": "text",
            "kind": "single",
            "output_type": "binary",
            "version": {
                "judge_model": "openai/gpt-4",
                "system_prompt": "Judge {{x}}",
                "variables": [{"name": "x"}],
            },
        },
        headers=h,
    )
    assert create.status_code == 200
    ev_uuid = create.json()["uuid"]
    v_uuid = create.json()["version_uuid"]

    # Rating evaluator without scale → 422
    bad = client.post(
        "/evaluators",
        json={
            "name": f"bad-{uuid.uuid4().hex[:6]}",
            "evaluator_type": "llm",
            "data_type": "text",
            "kind": "single",
            "output_type": "rating",
            "version": {"judge_model": "openai/gpt-4", "system_prompt": "p"},
        },
        headers=h,
    )
    assert bad.status_code == 422

    # Duplicate name → 409
    dup = client.post(
        "/evaluators",
        json={
            "name": name,
            "evaluator_type": "llm",
            "data_type": "text",
            "kind": "single",
            "output_type": "binary",
            "version": {"judge_model": "x", "system_prompt": "p"},
        },
        headers=h,
    )
    assert dup.status_code == 409

    # GET detail
    detail = client.get(f"/evaluators/{ev_uuid}", headers=h)
    assert detail.status_code == 200

    # Versions list
    versions = client.get(f"/evaluators/{ev_uuid}/versions", headers=h)
    assert versions.status_code == 200

    # Add a new version
    new_ver = client.post(
        f"/evaluators/{ev_uuid}/versions",
        json={
            "judge_model": "openai/gpt-4o",
            "system_prompt": "v2 {{x}}",
            "variables": [{"name": "x"}],
            "make_live": True,
        },
        headers=h,
    )
    assert new_ver.status_code == 200
    v2_uuid = new_ver.json()["version_uuid"]

    # Set live
    live = client.post(
        f"/evaluators/{ev_uuid}/versions/live",
        json={"version_uuid": v_uuid},
        headers=h,
    )
    assert live.status_code == 200
    # Bad version → 404
    bad_live = client.post(
        f"/evaluators/{ev_uuid}/versions/live",
        json={"version_uuid": "00000000-0000-4000-8000-000000000001"},
        headers=h,
    )
    assert bad_live.status_code == 404

    # Preview prompt
    preview = client.post(
        f"/evaluators/{ev_uuid}/preview-prompt",
        json={"variables": {"x": "hello"}},
        headers=h,
    )
    assert preview.status_code == 200
    assert "hello" in preview.json()["rendered_system_prompt"]

    # Preview with specific version
    preview_v2 = client.post(
        f"/evaluators/{ev_uuid}/preview-prompt",
        json={"version_uuid": v2_uuid, "variables": {"x": "world"}},
        headers=h,
    )
    assert preview_v2.status_code == 200

    # Preview missing version
    bad_preview = client.post(
        f"/evaluators/{ev_uuid}/preview-prompt",
        json={"version_uuid": "00000000-0000-4000-8000-000000000001"},
        headers=h,
    )
    assert bad_preview.status_code == 404

    # Update
    upd = client.put(
        f"/evaluators/{ev_uuid}", json={"description": "new"}, headers=h
    )
    assert upd.status_code == 200

    # Update with no fields
    no_op = client.put(f"/evaluators/{ev_uuid}", json={}, headers=h)
    assert no_op.status_code == 400

    # The org's list never exposes a read-only seed template (those keep the
    # `slug`; forks and customs don't), so every listed row is editable.
    seeds = client.get("/evaluators", headers=h).json()["items"]
    assert all(e.get("slug") is None for e in seeds)

    # The seed templates themselves stay immutable (403 on org_uuid IS NULL). An
    # org can't discover a template uuid through the API anymore, so fetch one
    # directly to exercise the guard.
    from db import get_evaluator_by_slug

    template_uuid = get_evaluator_by_slug("default-safety")["uuid"]
    forbidden = client.put(
        f"/evaluators/{template_uuid}",
        json={"description": "x"},
        headers=h,
    )
    assert forbidden.status_code == 403

    cannot_delete = client.delete(f"/evaluators/{template_uuid}", headers=h)
    assert cannot_delete.status_code == 403

    cannot_version = client.post(
        f"/evaluators/{template_uuid}/versions",
        json={"judge_model": "x", "system_prompt": "p"},
        headers=h,
    )
    assert cannot_version.status_code == 403

    # Cross-user fetch / unknown → 404
    other = _signup(client)
    assert (
        client.get(f"/evaluators/{ev_uuid}", headers=other["headers"]).status_code
        == 404
    )
    assert (
        client.get("/evaluators/missing", headers=h).status_code == 404
    )

    # Duplicate
    name_dup = f"dup-{uuid.uuid4().hex[:6]}"
    dup_ev = client.post(
        f"/evaluators/{ev_uuid}/duplicate",
        json={"name": name_dup},
        headers=h,
    )
    assert dup_ev.status_code == 200

    # Duplicate with conflicting name → 409
    dup_conflict = client.post(
        f"/evaluators/{ev_uuid}/duplicate",
        json={"name": name_dup},
        headers=h,
    )
    assert dup_conflict.status_code == 409

    # Delete
    deleted = client.delete(f"/evaluators/{ev_uuid}", headers=h)
    assert deleted.status_code == 200


def test_list_evaluators_search_and_pagination(client):
    """GET /evaluators keeps its type filters and adds optional `?q=` name
    search + `?limit=&offset=` paging, returning the `{items, total, ...}`
    envelope. Scoped to the caller's own evaluators (`include_defaults=false`)
    so the seeded defaults don't perturb the counts."""
    h = _signup(client)["headers"]
    tag = uuid.uuid4().hex[:6]
    names = [f"alpha-{tag}", f"alpine-{tag}", f"beta-{tag}"]
    for n in names:
        r = client.post(
            "/evaluators",
            json={
                "name": n,
                "evaluator_type": "llm",
                "output_type": "binary",
                "version": {"judge_model": "openai/gpt-4", "system_prompt": "p"},
            },
            headers=h,
        )
        assert r.status_code == 200

    base = {"include_defaults": "false"}

    # q= narrows to the two "alp…" names; `total` is the pre-slice count.
    r = client.get("/evaluators", params={**base, "q": "ALP"}, headers=h)
    assert r.status_code == 200
    body = r.json()
    assert {e["name"] for e in body["items"]} == {f"alpha-{tag}", f"alpine-{tag}"}
    assert body["total"] == 2

    # limit slices; total stays the filtered pre-slice count.
    r = client.get(
        "/evaluators", params={**base, "q": "alp", "limit": 1}, headers=h
    )
    body = r.json()
    assert len(body["items"]) == 1 and body["total"] == 2


# ---------------------------------------------------------------------------
# Tests router — bulk upload
# ---------------------------------------------------------------------------


def test_bulk_test_upload(client):
    auth = _signup(client)
    h = auth["headers"]
    evaluators = client.get("/evaluators", headers=h).json()["items"]
    llm_ev = next(e for e in evaluators if e.get("evaluator_type") == "llm")

    # Empty tests → 422 (pydantic validator)
    empty = client.post(
        "/tests/bulk",
        json={"type": "response", "tests": []},
        headers=h,
    )
    assert empty.status_code == 422

    # response test missing evaluators → 422
    missing_ev = client.post(
        "/tests/bulk",
        json={
            "type": "response",
            "tests": [
                {
                    "name": "t1",
                    "conversation_history": [{"role": "user", "content": "hi"}],
                }
            ],
        },
        headers=h,
    )
    assert missing_ev.status_code == 422

    # duplicate names → 422
    dup = client.post(
        "/tests/bulk",
        json={
            "type": "response",
            "tests": [
                {
                    "name": "t1",
                    "conversation_history": [{"role": "user", "content": "hi"}],
                    "evaluators": [{"evaluator_uuid": llm_ev["uuid"]}],
                },
                {
                    "name": "t1",
                    "conversation_history": [{"role": "user", "content": "hi"}],
                    "evaluators": [{"evaluator_uuid": llm_ev["uuid"]}],
                },
            ],
        },
        headers=h,
    )
    assert dup.status_code == 422

    # tool_call missing tool_calls → 422
    bad_tc = client.post(
        "/tests/bulk",
        json={
            "type": "tool_call",
            "tests": [
                {
                    "name": "t1",
                    "conversation_history": [{"role": "user", "content": "hi"}],
                }
            ],
        },
        headers=h,
    )
    assert bad_tc.status_code == 422

    # Unknown agent_uuid → 404
    bad_agent = client.post(
        "/tests/bulk",
        json={
            "type": "response",
            "agent_uuids": ["00000000-0000-4000-8000-000000000001"],
            "tests": [
                {
                    "name": "t1",
                    "conversation_history": [{"role": "user", "content": "hi"}],
                    "evaluators": [{"evaluator_uuid": llm_ev["uuid"]}],
                }
            ],
        },
        headers=h,
    )
    assert bad_agent.status_code == 404

    # Cross-user agent → 403
    other = _signup(client)
    other_agent = client.post(
        "/agents",
        json={"name": f"oa-{uuid.uuid4().hex[:6]}", "type": "agent"},
        headers=other["headers"],
    ).json()
    cross = client.post(
        "/tests/bulk",
        json={
            "type": "response",
            "agent_uuids": [other_agent["uuid"]],
            "tests": [
                {
                    "name": "t1",
                    "conversation_history": [{"role": "user", "content": "hi"}],
                    "evaluators": [{"evaluator_uuid": llm_ev["uuid"]}],
                }
            ],
        },
        headers=h,
    )
    assert cross.status_code == 404

    # Bulk upload happy path
    own_agent = client.post(
        "/agents",
        json={"name": f"a-{uuid.uuid4().hex[:6]}", "type": "agent"},
        headers=h,
    ).json()
    ok = client.post(
        "/tests/bulk",
        json={
            "type": "response",
            "agent_uuids": [own_agent["uuid"]],
            "tests": [
                {
                    "name": f"t-{uuid.uuid4().hex[:6]}",
                    "conversation_history": [{"role": "user", "content": "hi"}],
                    "evaluators": [{"evaluator_uuid": llm_ev["uuid"]}],
                }
            ],
        },
        headers=h,
    )
    assert ok.status_code == 200
    assert ok.json()["count"] == 1

    # tool_call upload
    ok_tc = client.post(
        "/tests/bulk",
        json={
            "type": "tool_call",
            "tests": [
                {
                    "name": f"tc-{uuid.uuid4().hex[:6]}",
                    "conversation_history": [{"role": "user", "content": "hi"}],
                    "tool_calls": [{"tool": "search", "arguments": {"q": "x"}}],
                }
            ],
            "language": "english",
        },
        headers=h,
    )
    assert ok_tc.status_code == 200


def test_create_test_wrong_evaluator_type(client):
    """Tests with evaluator_type != 'llm' should be rejected."""
    auth = _signup(client)
    h = auth["headers"]
    evaluators = client.get("/evaluators", headers=h).json()["items"]
    # default-stt-transcription has evaluator_type=stt
    stt_ev = next(e for e in evaluators if e.get("evaluator_type") == "stt")
    bad = client.post(
        "/tests",
        json={
            "name": f"t-{uuid.uuid4().hex[:6]}",
            "type": "response",
            "config": {"history": [], "evaluation": {"type": "response"}},
            "evaluators": [{"evaluator_uuid": stt_ev["uuid"]}],
        },
        headers=h,
    )
    assert bad.status_code == 400


def test_update_test_with_evaluators(client):
    auth = _signup(client)
    h = auth["headers"]
    evaluators = client.get("/evaluators", headers=h).json()["items"]
    llm_ev = next(e for e in evaluators if e.get("evaluator_type") == "llm")
    create = client.post(
        "/tests",
        json={
            "name": f"t-{uuid.uuid4().hex[:6]}",
            "type": "response",
            "config": {"history": [], "evaluation": {"type": "response"}},
            "evaluators": [{"evaluator_uuid": llm_ev["uuid"]}],
        },
        headers=h,
    ).json()
    t_uuid = create["uuid"]

    # Update with evaluators-only
    upd = client.put(
        f"/tests/{t_uuid}",
        json={"evaluators": []},
        headers=h,
    )
    assert upd.status_code == 200


def _create_simulation_evaluator(client, h):
    return client.post(
        "/evaluators",
        json={
            "name": f"sim-ev-{uuid.uuid4().hex[:6]}",
            "evaluator_type": "conversation",
            "output_type": "binary",
            "version": {
                "judge_model": "openai/gpt-4.1",
                "system_prompt": "Judge the whole conversation.",
            },
        },
        headers=h,
    ).json()


def test_create_conversation_test_requires_evaluator(client):
    """A conversation test created without evaluators is rejected (it has no
    LLM fallback, so it would otherwise run with nothing to judge with)."""
    auth = _signup(client)
    h = auth["headers"]
    resp = client.post(
        "/tests",
        json={
            "name": f"conv-{uuid.uuid4().hex[:6]}",
            "type": "conversation",
            "config": {"history": [], "evaluation": {"type": "conversation"}},
        },
        headers=h,
    )
    assert resp.status_code == 400
    assert "evaluator" in resp.json()["detail"].lower()


def test_update_conversation_test_cannot_clear_evaluators(client):
    """Clearing all evaluators from a conversation test is rejected; a response
    test (which has a fallback) may still be cleared."""
    auth = _signup(client)
    h = auth["headers"]
    sim_ev = _create_simulation_evaluator(client, h)["uuid"]
    create = client.post(
        "/tests",
        json={
            "name": f"conv-{uuid.uuid4().hex[:6]}",
            "type": "conversation",
            "config": {"history": [], "evaluation": {"type": "conversation"}},
            "evaluators": [{"evaluator_uuid": sim_ev}],
        },
        headers=h,
    ).json()
    assert create.get("uuid"), create

    cleared = client.put(
        f"/tests/{create['uuid']}",
        json={"evaluators": []},
        headers=h,
    )
    assert cleared.status_code == 400
    assert "evaluator" in cleared.json()["detail"].lower()


# ---------------------------------------------------------------------------
# Agents router — verify-connection + duplicate flow
# ---------------------------------------------------------------------------


def test_agent_verify_connection_paths(client):
    auth = _signup(client)
    h = auth["headers"]

    # Pre-save verify with success → mock TextAgentConnection.verify
    fake_agent = MagicMock()
    fake_agent.verify = AsyncMock(
        return_value={"ok": True, "sample_output": {"text": "hi"}}
    )
    # We also need socket.getaddrinfo to return a non-private address
    fake_addr = [(0, 0, 0, "", ("93.184.216.34", 0))]
    with patch(
        "routers.agents.TextAgentConnection", return_value=fake_agent
    ), patch("routers.agents.socket.getaddrinfo", return_value=fake_addr):
        resp = client.post(
            "/agents/verify-connection",
            json={
                "agent_url": "https://example.com/agent",
                "agent_headers": {
                    "Authorization": "Bearer x",
                    "Host": "should-be-stripped",
                },
            },
            headers=h,
        )
    assert resp.status_code == 200
    assert resp.json()["success"] is True

    # Pre-save verify with failure
    fake_agent.verify = AsyncMock(
        return_value={"ok": False, "error": "bad", "sample_output": None}
    )
    with patch(
        "routers.agents.TextAgentConnection", return_value=fake_agent
    ), patch("routers.agents.socket.getaddrinfo", return_value=fake_addr):
        resp = client.post(
            "/agents/verify-connection",
            json={"agent_url": "https://example.com/agent"},
            headers=h,
        )
    assert resp.json()["success"] is False

    # Pre-save verify raises
    fake_agent.verify = AsyncMock(side_effect=RuntimeError("boom"))
    with patch(
        "routers.agents.TextAgentConnection", return_value=fake_agent
    ), patch("routers.agents.socket.getaddrinfo", return_value=fake_addr):
        resp = client.post(
            "/agents/verify-connection",
            json={"agent_url": "https://example.com/agent"},
            headers=h,
        )
    assert resp.json()["success"] is False

    # URL whose resolved addr is private
    fake_private_addr = [(0, 0, 0, "", ("10.0.0.1", 0))]
    with patch(
        "routers.agents.socket.getaddrinfo", return_value=fake_private_addr
    ):
        resp = client.post(
            "/agents/verify-connection",
            json={"agent_url": "https://example.com/agent"},
            headers=h,
        )
    assert resp.status_code == 400

    # Unresolvable hostname
    import socket as _socket

    with patch(
        "routers.agents.socket.getaddrinfo", side_effect=_socket.gaierror
    ):
        resp = client.post(
            "/agents/verify-connection",
            json={"agent_url": "https://bogus.example/"},
            headers=h,
        )
    assert resp.status_code == 400

    # URL missing hostname
    bad_url = client.post(
        "/agents/verify-connection",
        json={"agent_url": "http://"},
        headers=h,
    )
    assert bad_url.status_code == 400


def test_agent_verify_forwards_default_inputs(client):
    """Pre-save verify passes default_inputs to the connection and inputs to verify()."""
    auth = _signup(client)
    h = auth["headers"]

    fake_agent = MagicMock()
    fake_agent.verify = AsyncMock(
        return_value={"ok": True, "sample_output": {"text": "hi"}}
    )
    ctor = MagicMock(return_value=fake_agent)
    fake_addr = [(0, 0, 0, "", ("93.184.216.34", 0))]
    with patch("routers.agents.TextAgentConnection", ctor), patch(
        "routers.agents.socket.getaddrinfo", return_value=fake_addr
    ):
        resp = client.post(
            "/agents/verify-connection",
            json={
                "agent_url": "https://example.com/agent",
                "default_inputs": {"trimester": 2},
                "inputs": {"condition_area": "anc"},
            },
            headers=h,
        )
    assert resp.status_code == 200
    assert ctor.call_args.kwargs["default_inputs"] == {"trimester": 2}
    assert fake_agent.verify.await_args.kwargs["inputs"] == {"condition_area": "anc"}


def test_agent_verify_saved_with_model_persists(client):
    auth = _signup(client)
    h = auth["headers"]
    agent = client.post(
        "/agents",
        json={
            "name": f"a-{uuid.uuid4().hex[:6]}",
            "type": "connection",
            "config": {"agent_url": "https://example.com/agent"},
        },
        headers=h,
    ).json()

    fake_agent = MagicMock()
    fake_agent.verify = AsyncMock(
        return_value={"ok": True, "sample_output": {"text": "hi"}}
    )
    fake_addr = [(0, 0, 0, "", ("93.184.216.34", 0))]
    with patch(
        "routers.agents.TextAgentConnection", return_value=fake_agent
    ), patch("routers.agents.socket.getaddrinfo", return_value=fake_addr):
        ok = client.post(
            f"/agents/{agent['uuid']}/verify-connection",
            json={"model": "openai/gpt-4"},
            headers=h,
        )
    assert ok.status_code == 200

    # Verify connection without model (basic check)
    with patch(
        "routers.agents.TextAgentConnection", return_value=fake_agent
    ), patch("routers.agents.socket.getaddrinfo", return_value=fake_addr):
        ok2 = client.post(
            f"/agents/{agent['uuid']}/verify-connection",
            json={},
            headers=h,
        )
    assert ok2.status_code == 200


def test_agent_create_with_partial_config_merge(client):
    auth = _signup(client)
    h = auth["headers"]
    create = client.post(
        "/agents",
        json={
            "name": f"a-{uuid.uuid4().hex[:6]}",
            "type": "agent",
            "config": {"llm": {"model": "anthropic/claude-3.5"}},
        },
        headers=h,
    )
    assert create.status_code == 200


def test_agent_update_url_resets_verification(client):
    auth = _signup(client)
    h = auth["headers"]
    agent = client.post(
        "/agents",
        json={
            "name": f"a-{uuid.uuid4().hex[:6]}",
            "type": "connection",
            "config": {
                "agent_url": "https://example.com/x",
                "connection_verified": True,
            },
        },
        headers=h,
    ).json()
    upd = client.put(
        f"/agents/{agent['uuid']}",
        json={"config": {"agent_url": "https://different.example/x"}},
        headers=h,
    )
    assert upd.status_code == 200
    assert upd.json()["config"]["connection_verified"] is False


def test_agent_update_with_top_level_verification_fields(client):
    auth = _signup(client)
    h = auth["headers"]
    agent = client.post(
        "/agents",
        json={"name": f"a-{uuid.uuid4().hex[:6]}", "type": "agent"},
        headers=h,
    ).json()
    upd = client.put(
        f"/agents/{agent['uuid']}",
        json={
            "connection_verified": True,
            "benchmark_models_verified": {"x": True},
        },
        headers=h,
    )
    assert upd.status_code == 200


# ---------------------------------------------------------------------------
# Auth router — google login
# ---------------------------------------------------------------------------


def test_google_login_success(client):
    """Mock verify_google_token to return a synthetic profile and assert login."""

    async def fake_verify(_id_token):
        return {
            "email": f"goog-{uuid.uuid4().hex[:8]}@example.com",
            "given_name": "G",
            "family_name": "U",
        }

    with patch("routers.auth.verify_google_token", side_effect=fake_verify):
        resp = client.post(
            "/auth/google",
            json={"id_token": "fake"},
        )
    assert resp.status_code == 200
    body = resp.json()
    assert body["access_token"]


def test_google_login_no_email(client):
    async def fake_verify(_id_token):
        return {"given_name": "X"}

    with patch("routers.auth.verify_google_token", side_effect=fake_verify):
        resp = client.post(
            "/auth/google",
            json={"id_token": "fake"},
        )
    assert resp.status_code == 400


def test_google_login_token_verify_failure(client):
    import httpx

    # The verify_google_token coroutine raises an HTTPException
    async def fake_verify(_id_token):
        from fastapi import HTTPException

        raise HTTPException(status_code=401, detail="invalid")

    with patch("routers.auth.verify_google_token", side_effect=fake_verify):
        resp = client.post(
            "/auth/google",
            json={"id_token": "fake"},
        )
    assert resp.status_code == 401


def test_google_token_verify_pass():
    """Exercise verify_google_token directly for the httpx response path."""
    import asyncio

    class FakeResp:
        status_code = 200

        def json(self):
            return {"email": "x@example.com"}

        @property
        def text(self):
            return "ok"

    fake_client = MagicMock()
    fake_client.get = AsyncMock(return_value=FakeResp())
    fake_client.__aenter__ = AsyncMock(return_value=fake_client)
    fake_client.__aexit__ = AsyncMock(return_value=None)
    from routers import auth as auth_mod

    with patch("routers.auth.httpx.AsyncClient", return_value=fake_client):
        info = asyncio.run(auth_mod.verify_google_token("fake"))
    assert info["email"] == "x@example.com"


def test_google_token_verify_non_200():
    import asyncio

    class FakeResp:
        status_code = 400
        text = "invalid"

        def json(self):
            return {}

    fake_client = MagicMock()
    fake_client.get = AsyncMock(return_value=FakeResp())
    fake_client.__aenter__ = AsyncMock(return_value=fake_client)
    fake_client.__aexit__ = AsyncMock(return_value=None)
    from routers import auth as auth_mod
    from fastapi import HTTPException

    with patch("routers.auth.httpx.AsyncClient", return_value=fake_client):
        with pytest.raises(HTTPException):
            asyncio.run(auth_mod.verify_google_token("fake"))


def test_google_token_request_error():
    import asyncio
    import httpx

    fake_client = MagicMock()
    fake_client.get = AsyncMock(side_effect=httpx.RequestError("net error"))
    fake_client.__aenter__ = AsyncMock(return_value=fake_client)
    fake_client.__aexit__ = AsyncMock(return_value=None)
    from routers import auth as auth_mod
    from fastapi import HTTPException

    with patch("routers.auth.httpx.AsyncClient", return_value=fake_client):
        with pytest.raises(HTTPException):
            asyncio.run(auth_mod.verify_google_token("fake"))


# ---------------------------------------------------------------------------
# datasets — exercise file/upload validation branches
# ---------------------------------------------------------------------------


def test_datasets_validation_paths(client):
    auth = _signup(client)
    h = auth["headers"]

    # POST with invalid dataset_type (Pydantic validation passes since it's str, but
    # the router rejects via its own check)
    bad_type = client.post(
        "/datasets", json={"name": "x", "dataset_type": "bogus"}, headers=h
    )
    assert bad_type.status_code in (400, 422)


def test_datasets_duplicate_name_conflicts(client):
    auth = _signup(client)
    h = auth["headers"]
    name = f"ds-{uuid.uuid4().hex[:8]}"

    first = client.post(
        "/datasets", json={"name": name, "dataset_type": "stt"}, headers=h
    )
    assert first.status_code == 201, first.text

    # Same name in the same workspace conflicts.
    dup = client.post(
        "/datasets", json={"name": name, "dataset_type": "tts"}, headers=h
    )
    assert dup.status_code == 409, dup.text

    # Renaming another dataset onto the taken name also conflicts.
    other = client.post(
        "/datasets",
        json={"name": f"ds-{uuid.uuid4().hex[:8]}", "dataset_type": "stt"},
        headers=h,
    ).json()
    rename = client.patch(
        f"/datasets/{other['uuid']}", json={"name": name}, headers=h
    )
    assert rename.status_code == 409, rename.text

    # A different workspace can reuse the name.
    other_h = _signup(client)["headers"]
    ok = client.post(
        "/datasets", json={"name": name, "dataset_type": "stt"}, headers=other_h
    )
    assert ok.status_code == 201, ok.text
