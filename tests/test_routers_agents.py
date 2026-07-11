"""Integration tests for /agents, focused on the name→UUID resolve endpoint."""

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
            "first_name": "Res",
            "last_name": "Olve",
            "email": f"res-{suffix}@example.com",
            "password": "passw0rd",
        },
    ).json()
    return {"Authorization": f"Bearer {body['access_token']}"}


def _create_agent(client, h, name):
    return client.post(
        "/agents", json={"name": name, "type": "agent"}, headers=h
    ).json()


def _raw_key(client, h, name="ci"):
    return client.post("/api-keys", json={"name": name}, headers=h).json()["key"]


def test_resolve_agent_names_with_jwt(client):
    h = _signup(client)
    n1 = f"alpha-{uuid.uuid4().hex[:6]}"
    n2 = f"beta-{uuid.uuid4().hex[:6]}"
    a1 = _create_agent(client, h, n1)
    a2 = _create_agent(client, h, n2)
    missing = f"ghost-{uuid.uuid4().hex[:6]}"

    r = client.post(
        "/agents/resolve", json={"names": [n1, n2, missing]}, headers=h
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["resolved"] == {n1: a1["uuid"], n2: a2["uuid"]}
    assert body["not_found"] == [missing]


def test_resolve_agent_names_with_api_key(client):
    h = _signup(client)
    name = f"keyed-{uuid.uuid4().hex[:6]}"
    agent = _create_agent(client, h, name)
    raw = _raw_key(client, h)

    # X-API-Key header
    r1 = client.post(
        "/agents/resolve", json={"names": [name]}, headers={"X-API-Key": raw}
    )
    assert r1.status_code == 200, r1.text
    assert r1.json()["resolved"] == {name: agent["uuid"]}

    # Authorization: Bearer sk_…
    r2 = client.post(
        "/agents/resolve",
        json={"names": [name]},
        headers={"Authorization": f"Bearer {raw}"},
    )
    assert r2.status_code == 200, r2.text
    assert r2.json()["resolved"] == {name: agent["uuid"]}


def test_resolve_dedupes_not_found(client):
    h = _signup(client)
    missing = f"none-{uuid.uuid4().hex[:6]}"
    r = client.post(
        "/agents/resolve", json={"names": [missing, missing]}, headers=h
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["resolved"] == {}
    assert body["not_found"] == [missing]


def test_resolve_is_org_scoped(client):
    """An agent in org A must not resolve for a caller in org B."""
    ha = _signup(client)
    name = f"private-{uuid.uuid4().hex[:6]}"
    _create_agent(client, ha, name)

    hb = _signup(client)
    r = client.post("/agents/resolve", json={"names": [name]}, headers=hb)
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["resolved"] == {}
    assert body["not_found"] == [name]


def test_resolve_requires_auth(client):
    r = client.post("/agents/resolve", json={"names": ["whatever"]})
    assert r.status_code in (401, 403)

    bad = client.post(
        "/agents/resolve",
        json={"names": ["whatever"]},
        headers={"X-API-Key": "sk_not-a-real-key"},
    )
    assert bad.status_code == 401


def test_list_agents_with_api_key(client):
    """GET /agents accepts an sk_ API key and lists the caller's org agents."""
    h = _signup(client)
    n1 = f"list-a-{uuid.uuid4().hex[:6]}"
    n2 = f"list-b-{uuid.uuid4().hex[:6]}"
    a1 = _create_agent(client, h, n1)
    a2 = _create_agent(client, h, n2)
    raw = _raw_key(client, h)

    # X-API-Key header. Response is the paginated envelope: {items, total, ...}.
    r1 = client.get("/agents", headers={"X-API-Key": raw})
    assert r1.status_code == 200, r1.text
    uuids = {a["uuid"] for a in r1.json()["items"]}
    assert {a1["uuid"], a2["uuid"]} <= uuids

    # Authorization: Bearer sk_…
    r2 = client.get("/agents", headers={"Authorization": f"Bearer {raw}"})
    assert r2.status_code == 200, r2.text
    assert {a1["uuid"], a2["uuid"]} <= {a["uuid"] for a in r2.json()["items"]}


def test_list_agents_search_and_pagination(client):
    """GET /agents supports optional `?q=` name search and `?limit=&offset=`
    paging, returning the `{items, total, limit, offset}` envelope; `total` is
    the pre-slice count of the filtered set."""
    h = _signup(client)
    tag = uuid.uuid4().hex[:6]
    names = [f"zeta-{tag}", f"zebra-{tag}", f"other-{tag}"]
    created = {n: _create_agent(client, h, n)["uuid"] for n in names}

    # No params → all three present, echoed window is unbounded.
    r = client.get("/agents", headers=h)
    assert r.status_code == 200
    body = r.json()
    assert set(created.values()) <= {a["uuid"] for a in body["items"]}
    assert body["limit"] is None and body["offset"] == 0

    # q= narrows by case-insensitive substring; only the two "ze…" names match.
    r = client.get("/agents", params={"q": "ZE"}, headers=h)
    assert r.status_code == 200
    body = r.json()
    assert {a["name"] for a in body["items"]} == {f"zeta-{tag}", f"zebra-{tag}"}
    assert body["total"] == 2

    # limit slices the (searched) set; total is the pre-slice count.
    r = client.get("/agents", params={"q": "ze", "limit": 1, "offset": 0}, headers=h)
    b1 = r.json()
    assert len(b1["items"]) == 1 and b1["total"] == 2
    r2 = client.get("/agents", params={"q": "ze", "limit": 1, "offset": 1}, headers=h)
    b2 = r2.json()
    assert len(b2["items"]) == 1
    assert b1["items"][0]["uuid"] != b2["items"][0]["uuid"]


def test_list_agents_returns_trimmed_summary(client):
    """GET /agents returns a trimmed summary per agent, never the full config
    (which carries agent auth credentials in `agent_headers`)."""
    h = _signup(client)
    name = f"summary-{uuid.uuid4().hex[:6]}"
    agent = _create_agent(client, h, name)

    r = client.get("/agents", headers=h)
    assert r.status_code == 200, r.text
    item = next(a for a in r.json()["items"] if a["uuid"] == agent["uuid"])

    # Summary fields present.
    assert set(item.keys()) == {"uuid", "name", "type", "updated_at", "connection_verified"}
    assert item["name"] == name
    assert item["type"] == "agent"
    assert item["updated_at"]

    # Full config / credentials / created_at are NOT shipped in the list.
    assert "config" not in item
    assert "system_prompt" not in item
    assert "agent_headers" not in item
    assert "created_at" not in item


def test_list_agents_derives_connection_verified(client):
    """connection_verified in the summary is derived from config.connection_verified:
    None when absent, and the stored bool once set."""
    h = _signup(client)

    # Agent with no verification flag → connection_verified is None.
    plain = _create_agent(client, h, f"cv-none-{uuid.uuid4().hex[:6]}")

    # Connection agent, then flip verification true / false via JWT PUT.
    conn = client.post(
        "/agents",
        json={
            "name": f"cv-conn-{uuid.uuid4().hex[:6]}",
            "type": "connection",
            "config": {"agent_url": "https://example.com/agent"},
        },
        headers=h,
    ).json()

    def _cv(agent_uuid):
        r = client.get("/agents", headers=h)
        assert r.status_code == 200, r.text
        return next(a for a in r.json()["items"] if a["uuid"] == agent_uuid)["connection_verified"]

    assert _cv(plain["uuid"]) is None

    client.put(
        f"/agents/{conn['uuid']}", json={"connection_verified": True}, headers=h
    )
    assert _cv(conn["uuid"]) is True

    client.put(
        f"/agents/{conn['uuid']}", json={"connection_verified": False}, headers=h
    )
    assert _cv(conn["uuid"]) is False


def test_list_agents_is_org_scoped(client):
    """An API key for org A must not list agents from org B."""
    ha = _signup(client)
    name = f"scoped-{uuid.uuid4().hex[:6]}"
    a = _create_agent(client, ha, name)

    hb = _signup(client)
    raw_b = _raw_key(client, hb)
    r = client.get("/agents", headers={"X-API-Key": raw_b})
    assert r.status_code == 200, r.text
    assert a["uuid"] not in {x["uuid"] for x in r.json()["items"]}


def test_list_agents_requires_auth(client):
    r = client.get("/agents")
    assert r.status_code in (401, 403)

    bad = client.get("/agents", headers={"X-API-Key": "sk_not-a-real-key"})
    assert bad.status_code == 401


def test_create_agent_with_api_key(client):
    """POST /agents accepts an sk_ API key."""
    h = _signup(client)
    raw = _raw_key(client, h)
    name = f"key-create-{uuid.uuid4().hex[:6]}"
    r = client.post(
        "/agents", json={"name": name, "type": "agent"}, headers={"X-API-Key": raw}
    )
    assert r.status_code == 200, r.text
    assert r.json()["uuid"]


def test_get_agent_with_api_key(client):
    """GET /agents/{uuid} accepts an sk_ API key."""
    h = _signup(client)
    agent = _create_agent(client, h, f"key-get-{uuid.uuid4().hex[:6]}")
    raw = _raw_key(client, h)
    r = client.get(f"/agents/{agent['uuid']}", headers={"X-API-Key": raw})
    assert r.status_code == 200, r.text
    assert r.json()["uuid"] == agent["uuid"]


def test_update_agent_with_api_key(client):
    """PUT /agents/{uuid} accepts an sk_ API key."""
    h = _signup(client)
    agent = _create_agent(client, h, f"key-upd-{uuid.uuid4().hex[:6]}")
    raw = _raw_key(client, h)
    new_name = f"key-upd-new-{uuid.uuid4().hex[:6]}"
    r = client.put(
        f"/agents/{agent['uuid']}",
        json={"name": new_name},
        headers={"X-API-Key": raw},
    )
    assert r.status_code == 200, r.text
    assert r.json()["name"] == new_name


def test_create_agent_invalid_api_key(client):
    """POST /agents with a bogus key must 401."""
    r = client.post(
        "/agents",
        json={"name": f"bad-{uuid.uuid4().hex[:6]}", "type": "agent"},
        headers={"X-API-Key": "bad"},
    )
    assert r.status_code == 401


def test_get_agent_wrong_org_api_key(client):
    """A key from another org must not read an agent — 404 (existence-leak parity)."""
    ha = _signup(client)
    agent = _create_agent(client, ha, f"other-org-{uuid.uuid4().hex[:6]}")

    hb = _signup(client)
    raw_b = _raw_key(client, hb)
    r = client.get(f"/agents/{agent['uuid']}", headers={"X-API-Key": raw_b})
    assert r.status_code == 404


def test_create_agent_with_api_key_cannot_self_attest_verification(client):
    """An API key must not be able to flip connection_verified=true on create.

    Only POST /agents/{uuid}/verify-connection (JWT-only) may set this, since
    it's the sole path that runs the SSRF guard (_validate_agent_url) before
    ever contacting agent_url. Letting an API key smuggle
    connection_verified=true through config would let it point Calibrate's
    job runner at an unvalidated, arbitrary URL.
    """
    h = _signup(client)
    raw = _raw_key(client, h)
    r = client.post(
        "/agents",
        json={
            "name": f"key-ssrf-create-{uuid.uuid4().hex[:6]}",
            "type": "connection",
            "config": {
                "agent_url": "https://example.com/x",
                "connection_verified": True,
            },
        },
        headers={"X-API-Key": raw},
    )
    assert r.status_code == 200, r.text
    agent = client.get(f"/agents/{r.json()['uuid']}", headers={"X-API-Key": raw}).json()
    assert agent["config"].get("connection_verified") is not True


def test_update_agent_with_api_key_cannot_self_attest_verification(client):
    """An API key must not be able to flip connection_verified=true via PUT,
    whether through the dedicated field or smuggled inside `config`."""
    h = _signup(client)
    raw = _raw_key(client, h)
    agent = client.post(
        "/agents",
        json={
            "name": f"key-ssrf-update-{uuid.uuid4().hex[:6]}",
            "type": "connection",
            "config": {"agent_url": "https://example.com/x"},
        },
        headers={"X-API-Key": raw},
    ).json()

    # Paired with a real field change (name) so the request isn't a pure no-op
    # once the verification fields are stripped — isolates the strip behavior
    # rather than the separate "nothing to update" 400 path.
    r1 = client.put(
        f"/agents/{agent['uuid']}",
        json={
            "name": f"key-ssrf-update-renamed-{uuid.uuid4().hex[:6]}",
            "connection_verified": True,
            "benchmark_models_verified": {"x": True},
        },
        headers={"X-API-Key": raw},
    )
    assert r1.status_code == 200, r1.text
    assert r1.json()["config"].get("connection_verified") is not True
    assert not r1.json()["config"].get("benchmark_models_verified")

    r2 = client.put(
        f"/agents/{agent['uuid']}",
        json={"config": {"agent_url": "https://example.com/x", "connection_verified": True}},
        headers={"X-API-Key": raw},
    )
    assert r2.status_code == 200, r2.text
    assert r2.json()["config"].get("connection_verified") is not True


# ============ Agent <-> Evaluator association ============


def _create_evaluator(client, h, name=None):
    """Create a minimal LLM evaluator owned by the caller's org."""
    resp = client.post(
        "/evaluators",
        json={
            "name": name or f"ev-{uuid.uuid4().hex[:6]}",
            "evaluator_type": "llm",
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


def _default_evaluator_uuid(client, h):
    """A seeded default evaluator (owner_user_id IS NULL), visible to every org."""
    items = client.get("/evaluators", headers=h).json()["items"]
    defaults = [e for e in items if e["is_default"]]
    assert defaults, "expected at least one seeded default evaluator"
    return defaults[0]["uuid"]


def test_link_list_and_unlink_evaluator(client):
    h = _signup(client)
    agent = _create_agent(client, h, f"ev-agent-{uuid.uuid4().hex[:6]}")
    ev = _create_evaluator(client, h)

    # Initially none linked.
    r = client.get(f"/agents/{agent['uuid']}/evaluators", headers=h)
    assert r.status_code == 200, r.text
    assert r.json()["total"] == 0

    # Link.
    r = client.post(
        f"/agents/{agent['uuid']}/evaluators", json={"evaluator_ids": [ev]}, headers=h
    )
    assert r.status_code == 200, r.text

    r = client.get(f"/agents/{agent['uuid']}/evaluators", headers=h)
    assert r.status_code == 200
    body = r.json()
    assert body["total"] == 1
    assert body["items"][0]["uuid"] == ev
    # Slim list shape (mirrors GET /evaluators).
    assert "is_default" in body["items"][0]
    assert "live_version" in body["items"][0]

    # Unlink.
    r = client.delete(f"/agents/{agent['uuid']}/evaluators/{ev}", headers=h)
    assert r.status_code == 200, r.text
    assert client.get(f"/agents/{agent['uuid']}/evaluators", headers=h).json()["total"] == 0

    # Unlinking again is a 404 (link no longer present).
    r = client.delete(f"/agents/{agent['uuid']}/evaluators/{ev}", headers=h)
    assert r.status_code == 404


def test_link_multiple_evaluators_skips_already_linked(client):
    h = _signup(client)
    agent = _create_agent(client, h, f"ev-agent-{uuid.uuid4().hex[:6]}")
    a = _create_evaluator(client, h)
    b = _create_evaluator(client, h)
    c = _create_evaluator(client, h)

    # Link two at once.
    r = client.post(
        f"/agents/{agent['uuid']}/evaluators", json={"evaluator_ids": [a, b]}, headers=h
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert sorted(body["linked"]) == sorted([a, b])
    assert body["already_linked"] == []
    assert client.get(f"/agents/{agent['uuid']}/evaluators", headers=h).json()["total"] == 2

    # Link again with one existing + one new: only the new one is linked.
    r = client.post(
        f"/agents/{agent['uuid']}/evaluators", json={"evaluator_ids": [b, c]}, headers=h
    ).json()
    assert r["linked"] == [c]
    assert r["already_linked"] == [b]
    assert client.get(f"/agents/{agent['uuid']}/evaluators", headers=h).json()["total"] == 3

    # A bad id in the set links nothing (validated up front).
    other_org = _signup(client)
    foreign = _create_evaluator(client, other_org)
    d = _create_evaluator(client, h)
    r = client.post(
        f"/agents/{agent['uuid']}/evaluators",
        json={"evaluator_ids": [d, foreign]},
        headers=h,
    )
    assert r.status_code == 404, r.text
    assert client.get(f"/agents/{agent['uuid']}/evaluators", headers=h).json()["total"] == 3


def test_relink_evaluator_restores_link(client):
    h = _signup(client)
    agent = _create_agent(client, h, f"ev-agent-{uuid.uuid4().hex[:6]}")
    ev = _create_evaluator(client, h)

    client.post(f"/agents/{agent['uuid']}/evaluators", json={"evaluator_ids": [ev]}, headers=h)
    client.delete(f"/agents/{agent['uuid']}/evaluators/{ev}", headers=h)
    # Re-link restores the soft-deleted row rather than erroring on UNIQUE.
    r = client.post(
        f"/agents/{agent['uuid']}/evaluators", json={"evaluator_ids": [ev]}, headers=h
    )
    assert r.status_code == 200, r.text
    assert client.get(f"/agents/{agent['uuid']}/evaluators", headers=h).json()["total"] == 1


def test_link_evaluator_twice_is_idempotent(client):
    h = _signup(client)
    agent = _create_agent(client, h, f"ev-agent-{uuid.uuid4().hex[:6]}")
    ev = _create_evaluator(client, h)

    r1 = client.post(
        f"/agents/{agent['uuid']}/evaluators", json={"evaluator_ids": [ev]}, headers=h
    )
    assert r1.status_code == 200, r1.text
    r2 = client.post(
        f"/agents/{agent['uuid']}/evaluators", json={"evaluator_ids": [ev]}, headers=h
    )
    assert r2.status_code == 200, r2.text
    assert client.get(f"/agents/{agent['uuid']}/evaluators", headers=h).json()["total"] == 1


def test_link_default_evaluator_allowed(client):
    h = _signup(client)
    agent = _create_agent(client, h, f"ev-agent-{uuid.uuid4().hex[:6]}")
    ev = _default_evaluator_uuid(client, h)

    r = client.post(
        f"/agents/{agent['uuid']}/evaluators", json={"evaluator_ids": [ev]}, headers=h
    )
    assert r.status_code == 200, r.text
    listed = client.get(f"/agents/{agent['uuid']}/evaluators", headers=h).json()
    assert ev in [e["uuid"] for e in listed["items"]]


def test_link_evaluator_from_another_org_is_404(client):
    h1 = _signup(client)
    h2 = _signup(client)
    agent = _create_agent(client, h1, f"ev-agent-{uuid.uuid4().hex[:6]}")
    other_ev = _create_evaluator(client, h2)  # owned by org 2

    r = client.post(
        f"/agents/{agent['uuid']}/evaluators", json={"evaluator_ids": [other_ev]}, headers=h1
    )
    assert r.status_code == 404, r.text


def test_link_evaluator_to_other_org_agent_is_404(client):
    h1 = _signup(client)
    h2 = _signup(client)
    agent = _create_agent(client, h1, f"ev-agent-{uuid.uuid4().hex[:6]}")
    ev = _create_evaluator(client, h2)

    # org 2 cannot see org 1's agent.
    r = client.post(
        f"/agents/{agent['uuid']}/evaluators", json={"evaluator_ids": [ev]}, headers=h2
    )
    assert r.status_code == 404, r.text
    r = client.get(f"/agents/{agent['uuid']}/evaluators", headers=h2)
    assert r.status_code == 404, r.text


def test_evaluator_public_surface_with_api_key(client):
    """GET (list) and POST (link) are Public API; DELETE (unlink) is JWT-only,
    so an API key alone is rejected there."""
    h = _signup(client)
    agent = _create_agent(client, h, f"ev-agent-{uuid.uuid4().hex[:6]}")
    ev = _create_evaluator(client, h)
    raw = _raw_key(client, h)

    # POST (link) accepts an API key.
    r = client.post(
        f"/agents/{agent['uuid']}/evaluators",
        json={"evaluator_ids": [ev]},
        headers={"X-API-Key": raw},
    )
    assert r.status_code == 200, r.text

    # GET (list) accepts an API key.
    r = client.get(
        f"/agents/{agent['uuid']}/evaluators", headers={"X-API-Key": raw}
    )
    assert r.status_code == 200, r.text
    assert ev in [e["uuid"] for e in r.json()["items"]]

    # DELETE (unlink) is JWT-only — an API key alone is not accepted.
    r = client.delete(
        f"/agents/{agent['uuid']}/evaluators/{ev}",
        headers={"X-API-Key": raw},
    )
    assert r.status_code == 403, r.text


def test_link_evaluators_malformed_id_is_422(client):
    h = _signup(client)
    agent = _create_agent(client, h, f"ev-agent-{uuid.uuid4().hex[:6]}")
    r = client.post(
        f"/agents/{agent['uuid']}/evaluators",
        json={"evaluator_ids": ["not-a-uuid"]},
        headers=h,
    )
    assert r.status_code == 422, r.text


def test_duplicate_agent_copies_evaluator_links(client):
    h = _signup(client)
    agent = _create_agent(client, h, f"ev-agent-{uuid.uuid4().hex[:6]}")
    ev = _create_evaluator(client, h)
    client.post(f"/agents/{agent['uuid']}/evaluators", json={"evaluator_ids": [ev]}, headers=h)

    dup = client.post(
        f"/agents/{agent['uuid']}/duplicate",
        json={"name": f"dup-{uuid.uuid4().hex[:6]}"},
        headers=h,
    )
    assert dup.status_code == 200, dup.text
    dup_uuid = dup.json()["uuid"]
    listed = client.get(f"/agents/{dup_uuid}/evaluators", headers=h).json()
    assert ev in [e["uuid"] for e in listed["items"]]


def test_delete_agent_removes_evaluator_links(client):
    h = _signup(client)
    agent = _create_agent(client, h, f"ev-agent-{uuid.uuid4().hex[:6]}")
    ev = _create_evaluator(client, h)
    client.post(f"/agents/{agent['uuid']}/evaluators", json={"evaluator_ids": [ev]}, headers=h)

    assert client.delete(f"/agents/{agent['uuid']}", headers=h).status_code == 200
    # The agent is gone -> its evaluator listing 404s.
    assert client.get(f"/agents/{agent['uuid']}/evaluators", headers=h).status_code == 404
