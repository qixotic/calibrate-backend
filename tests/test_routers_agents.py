"""Integration tests for /agents, focused on the name→UUID resolve endpoint."""

from __future__ import annotations

import uuid
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

import db


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
    assert set(item.keys()) == {
        "uuid",
        "name",
        "type",
        "interaction_type",
        "created_at",
        "updated_at",
        "connection_verified",
        "has_default_inputs",
        "auto_score_traces",
    }
    assert item["name"] == name
    assert item["type"] == "agent"
    assert item["auto_score_traces"] is False
    assert item["created_at"]
    assert item["updated_at"]

    # Full config / credentials are NOT shipped in the list.
    assert "config" not in item
    assert "system_prompt" not in item
    assert "agent_headers" not in item


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


def test_list_agents_derives_has_default_inputs(client):
    """has_default_inputs is True only when config.default_inputs is a non-empty
    dict; absent or empty maps to False."""
    h = _signup(client)

    plain = _create_agent(client, h, f"di-none-{uuid.uuid4().hex[:6]}")

    empty = client.post(
        "/agents",
        json={
            "name": f"di-empty-{uuid.uuid4().hex[:6]}",
            "type": "connection",
            "config": {"agent_url": "https://example.com/a", "default_inputs": {}},
        },
        headers=h,
    ).json()

    withfields = client.post(
        "/agents",
        json={
            "name": f"di-set-{uuid.uuid4().hex[:6]}",
            "type": "connection",
            "config": {
                "agent_url": "https://example.com/a",
                "default_inputs": {"trimester": 2},
            },
        },
        headers=h,
    ).json()

    def _hdi(agent_uuid):
        r = client.get("/agents", headers=h)
        assert r.status_code == 200, r.text
        return next(
            a for a in r.json()["items"] if a["uuid"] == agent_uuid
        )["has_default_inputs"]

    assert _hdi(plain["uuid"]) is False
    assert _hdi(empty["uuid"]) is False
    assert _hdi(withfields["uuid"]) is True


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
    """A key from another org must not read an agent — 403, with no owning org named.

    A key is bound to one org, so there is no workspace for its holder to switch
    to; the response must not carry organization_uuid.
    """
    ha = _signup(client)
    agent = _create_agent(client, ha, f"other-org-{uuid.uuid4().hex[:6]}")

    hb = _signup(client)
    raw_b = _raw_key(client, hb)
    r = client.get(f"/agents/{agent['uuid']}", headers={"X-API-Key": raw_b})
    assert r.status_code == 403
    assert "organization_uuid" not in r.json()


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


def _create_evaluator(
    client, h, name=None, evaluator_type="llm", variables=None
):
    """Create a minimal evaluator owned by the caller's org."""
    version = {
        "judge_model": "openai/gpt-4.1",
        "system_prompt": (
            "Judge {{criteria}} carefully" if variables else "Judge the reply."
        ),
    }
    if variables is not None:
        version["variables"] = variables
    resp = client.post(
        "/evaluators",
        json={
            "name": name or f"ev-{uuid.uuid4().hex[:6]}",
            "evaluator_type": evaluator_type,
            "output_type": "binary",
            "version": version,
        },
        headers=h,
    )
    assert resp.status_code == 200, resp.text
    return resp.json()["uuid"]


def _default_evaluator_uuid(client, h):
    """The org's fork of a seeded default (provisioned at signup). Forks are
    editable rows that read as `is_default` True; the "Safety" default is always
    provisioned."""
    items = client.get("/evaluators", headers=h).json()["items"]
    fork = next((e for e in items if e.get("name") == "Safety"), None)
    assert fork is not None, "expected the org's forked Safety default"
    return fork["uuid"]


def test_link_list_and_unlink_evaluator(client):
    h = _signup(client)
    agent = _create_agent(client, h, f"ev-agent-{uuid.uuid4().hex[:6]}")
    ev = _create_evaluator(client, h)

    # Agent creation auto-links the default correctness evaluator.
    r = client.get(f"/agents/{agent['uuid']}/evaluators", headers=h)
    assert r.status_code == 200, r.text
    assert r.json()["total"] == 1

    # Link.
    r = client.post(
        f"/agents/{agent['uuid']}/evaluators", json={"evaluator_ids": [ev]}, headers=h
    )
    assert r.status_code == 200, r.text

    r = client.get(f"/agents/{agent['uuid']}/evaluators", headers=h)
    assert r.status_code == 200
    body = r.json()
    assert body["total"] == 2
    assert ev in [e["uuid"] for e in body["items"]]
    # Slim list shape (mirrors GET /evaluators).
    assert "is_default" in body["items"][0]
    assert "live_version" in body["items"][0]

    # Unlink.
    r = client.delete(f"/agents/{agent['uuid']}/evaluators/{ev}", headers=h)
    assert r.status_code == 200, r.text
    assert client.get(f"/agents/{agent['uuid']}/evaluators", headers=h).json()["total"] == 1

    # Unlinking again is a 404 (link no longer present).
    r = client.delete(f"/agents/{agent['uuid']}/evaluators/{ev}", headers=h)
    assert r.status_code == 404


def test_bulk_delete_agents(client):
    h = _signup(client)
    a1 = _create_agent(client, h, f"bd-{uuid.uuid4().hex[:6]}")["uuid"]
    a2 = _create_agent(client, h, f"bd-{uuid.uuid4().hex[:6]}")["uuid"]

    r = client.post("/agents/bulk-delete", json={"agent_uuids": [a1, a2]}, headers=h)
    assert r.status_code == 200, r.text
    assert r.json()["deleted_count"] == 2
    assert client.get(f"/agents/{a1}", headers=h).status_code == 404
    assert client.get(f"/agents/{a2}", headers=h).status_code == 404


def test_bulk_delete_agents_rejects_empty(client):
    h = _signup(client)
    r = client.post("/agents/bulk-delete", json={"agent_uuids": []}, headers=h)
    assert r.status_code == 400


def test_bulk_delete_agents_404_on_unknown_is_atomic(client):
    """An unknown id 404s and deletes nothing (all-or-nothing)."""
    h = _signup(client)
    a1 = _create_agent(client, h, f"bd-{uuid.uuid4().hex[:6]}")["uuid"]
    ghost = str(uuid.uuid4())

    r = client.post("/agents/bulk-delete", json={"agent_uuids": [a1, ghost]}, headers=h)
    assert r.status_code == 404, r.text
    assert r.json()["detail"]["not_found"] == [ghost]
    # a1 survives — nothing was deleted.
    assert client.get(f"/agents/{a1}", headers=h).status_code == 200


def test_bulk_delete_agents_is_org_scoped(client):
    """An agent in org A can't be bulk-deleted by a caller in org B."""
    ha = _signup(client)
    agent = _create_agent(client, ha, f"bd-{uuid.uuid4().hex[:6]}")["uuid"]

    hb = _signup(client)
    r = client.post("/agents/bulk-delete", json={"agent_uuids": [agent]}, headers=hb)
    assert r.status_code == 404, r.text
    assert client.get(f"/agents/{agent}", headers=ha).status_code == 200


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
    # +1 for the default correctness evaluator auto-linked on agent creation.
    assert client.get(f"/agents/{agent['uuid']}/evaluators", headers=h).json()["total"] == 3

    # Link again with one existing + one new: only the new one is linked.
    r = client.post(
        f"/agents/{agent['uuid']}/evaluators", json={"evaluator_ids": [b, c]}, headers=h
    ).json()
    assert r["linked"] == [c]
    assert r["already_linked"] == [b]
    assert client.get(f"/agents/{agent['uuid']}/evaluators", headers=h).json()["total"] == 4

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
    assert client.get(f"/agents/{agent['uuid']}/evaluators", headers=h).json()["total"] == 4


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
    # +1 for the default correctness evaluator auto-linked on agent creation.
    assert client.get(f"/agents/{agent['uuid']}/evaluators", headers=h).json()["total"] == 2


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
    # +1 for the default correctness evaluator auto-linked on agent creation.
    assert client.get(f"/agents/{agent['uuid']}/evaluators", headers=h).json()["total"] == 2


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


def test_link_evaluator_to_other_org_agent_is_denied(client):
    h1 = _signup(client)
    h2 = _signup(client)
    agent = _create_agent(client, h1, f"ev-agent-{uuid.uuid4().hex[:6]}")
    ev = _create_evaluator(client, h2)

    # org 2 cannot reach org 1's agent. It exists, so the answer is 403 — and
    # user 2 is not a member of org 1, so the owning org is not named.
    r = client.post(
        f"/agents/{agent['uuid']}/evaluators", json={"evaluator_ids": [ev]}, headers=h2
    )
    assert r.status_code == 403, r.text
    assert "organization_uuid" not in r.json()
    r = client.get(f"/agents/{agent['uuid']}/evaluators", headers=h2)
    assert r.status_code == 403, r.text
    assert "organization_uuid" not in r.json()


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


def test_duplicate_agent_does_not_resurrect_unlinked_default_evaluator(client):
    """A user who unlinked the auto-added correctness evaluator gets a
    duplicate that matches — not one where create_agent's auto-link comes
    back."""
    h = _signup(client)
    agent = _create_agent(client, h, f"ev-agent-{uuid.uuid4().hex[:6]}")
    default_ev = client.get(f"/agents/{agent['uuid']}/evaluators", headers=h).json()["items"][0]["uuid"]
    assert client.delete(f"/agents/{agent['uuid']}/evaluators/{default_ev}", headers=h).status_code == 200
    assert client.get(f"/agents/{agent['uuid']}/evaluators", headers=h).json()["total"] == 0

    dup = client.post(
        f"/agents/{agent['uuid']}/duplicate",
        json={"name": f"dup-{uuid.uuid4().hex[:6]}"},
        headers=h,
    )
    assert dup.status_code == 200, dup.text
    dup_uuid = dup.json()["uuid"]
    assert client.get(f"/agents/{dup_uuid}/evaluators", headers=h).json()["total"] == 0


def test_delete_agent_removes_evaluator_links(client):
    h = _signup(client)
    agent = _create_agent(client, h, f"ev-agent-{uuid.uuid4().hex[:6]}")
    ev = _create_evaluator(client, h)
    client.post(f"/agents/{agent['uuid']}/evaluators", json={"evaluator_ids": [ev]}, headers=h)

    assert client.delete(f"/agents/{agent['uuid']}", headers=h).status_code == 200
    # The agent is gone -> its evaluator listing 404s.
    assert client.get(f"/agents/{agent['uuid']}/evaluators", headers=h).status_code == 404


def test_create_agent_code_samples_match_examples(app):
    """POST /agents ships x-codeSamples generated from the request examples.

    Guards the one-source-of-truth contract: each named request example must
    appear as a full-body cURL snippet (Mintlify's schema-generated panel would
    otherwise show only `name`), and the snippet body must equal the example.
    """
    import json

    from routers.agents import _CREATE_AGENT_EXAMPLES

    op = app.openapi()["paths"]["/agents"]["post"]
    samples = op["x-codeSamples"]
    assert {s["label"] for s in samples} == {
        ex["summary"] for ex in _CREATE_AGENT_EXAMPLES.values()
    }
    by_label = {s["label"]: s for s in samples}
    for ex in _CREATE_AGENT_EXAMPLES.values():
        sample = by_label[ex["summary"]]
        assert sample["lang"] == "curl"
        # Full example body is embedded verbatim (pretty-printed for readability),
        # not a required-fields subset.
        assert json.dumps(ex["value"], indent=2) in sample["source"]


def test_public_spec_preserves_create_agent_code_samples(app):
    import main

    op = main._build_public_openapi()["paths"]["/agents"]["post"]
    assert len(op["x-codeSamples"]) == 2


# ============ interaction_type ============


def test_create_agent_defaults_interaction_type_to_conversation(client):
    h = _signup(client)
    agent = _create_agent(client, h, f"it-default-{uuid.uuid4().hex[:6]}")

    r = client.get(f"/agents/{agent['uuid']}", headers=h)
    assert r.status_code == 200, r.text
    assert r.json()["interaction_type"] == "conversation"


def test_create_agent_with_interaction_type_general(client):
    h = _signup(client)
    r = client.post(
        "/agents",
        json={
            "name": f"it-general-{uuid.uuid4().hex[:6]}",
            "type": "agent",
            "interaction_type": "general",
        },
        headers=h,
    )
    assert r.status_code == 200, r.text
    agent_uuid = r.json()["uuid"]

    r = client.get(f"/agents/{agent_uuid}", headers=h)
    assert r.status_code == 200, r.text
    assert r.json()["interaction_type"] == "general"


def test_update_agent_interaction_type_is_immutable(client):
    """It picks the request body the agent is sent, so changing it would strand
    every test already linked and leave the connection verified against a body
    the agent no longer receives. Ignored on update, exactly like `type`."""
    h = _signup(client)
    agent = _create_agent(client, h, f"it-immutable-{uuid.uuid4().hex[:6]}")

    r = client.put(
        f"/agents/{agent['uuid']}",
        json={"name": "renamed", "interaction_type": "general"},
        headers=h,
    )
    assert r.status_code == 200, r.text
    assert r.json()["interaction_type"] == "conversation"
    assert r.json()["name"] == "renamed"
    fetched = client.get(f"/agents/{agent['uuid']}", headers=h).json()
    assert fetched["interaction_type"] == "conversation"


def test_update_agent_cannot_flip_a_general_agent_either(client):
    h = _signup(client)
    created = client.post(
        "/agents",
        json={
            "name": f"it-gen-{uuid.uuid4().hex[:6]}",
            "type": "agent",
            "interaction_type": "general",
        },
        headers=h,
    ).json()

    r = client.put(
        f"/agents/{created['uuid']}",
        json={"name": "renamed", "interaction_type": "conversation"},
        headers=h,
    )
    assert r.status_code == 200, r.text
    assert r.json()["interaction_type"] == "general"

    # Sending it alone leaves nothing to update, so the caller gets an error
    # rather than a success that changed nothing.
    alone = client.put(
        f"/agents/{created['uuid']}",
        json={"interaction_type": "conversation"},
        headers=h,
    )
    assert alone.status_code == 400, alone.text


def test_update_agent_omitting_interaction_type_leaves_it_unchanged(client):
    h = _signup(client)
    r = client.post(
        "/agents",
        json={
            "name": f"it-noop-{uuid.uuid4().hex[:6]}",
            "type": "agent",
            "interaction_type": "general",
        },
        headers=h,
    )
    agent_uuid = r.json()["uuid"]

    # Update a different field only; interaction_type must stay "general".
    r = client.put(
        f"/agents/{agent_uuid}",
        json={"name": f"it-noop-renamed-{uuid.uuid4().hex[:6]}"},
        headers=h,
    )
    assert r.status_code == 200, r.text
    assert r.json()["interaction_type"] == "general"


def test_duplicate_agent_copies_interaction_type(client):
    h = _signup(client)
    r = client.post(
        "/agents",
        json={
            "name": f"it-dup-{uuid.uuid4().hex[:6]}",
            "type": "agent",
            "interaction_type": "general",
        },
        headers=h,
    )
    agent_uuid = r.json()["uuid"]

    dup = client.post(
        f"/agents/{agent_uuid}/duplicate",
        json={"name": f"it-dup-copy-{uuid.uuid4().hex[:6]}"},
        headers=h,
    )
    assert dup.status_code == 200, dup.text
    dup_uuid = dup.json()["uuid"]

    r = client.get(f"/agents/{dup_uuid}", headers=h)
    assert r.status_code == 200, r.text
    assert r.json()["interaction_type"] == "general"


def test_list_and_get_agents_include_interaction_type(client):
    h = _signup(client)
    agent = _create_agent(client, h, f"it-surface-{uuid.uuid4().hex[:6]}")

    r = client.get(f"/agents/{agent['uuid']}", headers=h)
    assert r.status_code == 200, r.text
    assert r.json()["interaction_type"] == "conversation"

    r = client.get("/agents", headers=h)
    assert r.status_code == 200, r.text
    item = next(a for a in r.json()["items"] if a["uuid"] == agent["uuid"])
    assert item["interaction_type"] == "conversation"


@pytest.mark.parametrize(
    "interaction_type,expected_body",
    [
        (
            "conversation",
            {"messages": [{"role": "user", "content": "Hello, are you there?"}]},
        ),
        ("general", {"input": "Hello, are you there?"}),
    ],
)
def test_verify_sends_the_body_the_agents_type_expects(
    client, monkeypatch, interaction_type, expected_body
):
    """A general agent takes one plain prompt, so verification must probe it that
    way rather than with a conversation it cannot read."""
    import httpx

    sent = {}

    class _Reply:
        status_code = 200
        text = ""

        @staticmethod
        def json():
            return {"response": "hello"}

    async def _fake_post(self, url, json=None, headers=None):
        sent["body"] = json
        return _Reply()

    monkeypatch.setattr(httpx.AsyncClient, "post", _fake_post)

    h = _signup(client)
    agent = client.post(
        "/agents",
        json={
            "name": f"a-{uuid.uuid4().hex[:6]}",
            "type": "connection",
            "interaction_type": interaction_type,
            "config": {"agent_url": "https://example.com/run"},
        },
        headers=h,
    ).json()

    res = client.post(
        f"/agents/{agent['uuid']}/verify-connection", json={}, headers=h
    )
    assert res.status_code == 200, res.text
    assert res.json()["success"] is True
    assert sent["body"] == expected_body


def test_presave_verify_sends_the_body_its_stated_type_expects(client, monkeypatch):
    """No agent exists yet, so the caller states the type on the request."""
    import httpx

    sent = {}

    class _Reply:
        status_code = 200
        text = ""

        @staticmethod
        def json():
            return {"response": "hello"}

    async def _fake_post(self, url, json=None, headers=None):
        sent["body"] = json
        return _Reply()

    monkeypatch.setattr(httpx.AsyncClient, "post", _fake_post)

    h = _signup(client)
    res = client.post(
        "/agents/verify-connection",
        json={
            "agent_url": "https://example.com/run",
            "interaction_type": "general",
        },
        headers=h,
    )
    assert res.status_code == 200, res.text
    assert sent["body"] == {"input": "Hello, are you there?"}

    # Omitting it keeps the conversation body.
    client.post(
        "/agents/verify-connection",
        json={"agent_url": "https://example.com/run"},
        headers=h,
    )
    assert sent["body"] == {
        "messages": [{"role": "user", "content": "Hello, are you there?"}]
    }


# ============ auto_score_traces + eligibility ============


def _unlink_all_evaluators(client, h, agent_uuid):
    items = client.get(f"/agents/{agent_uuid}/evaluators", headers=h).json()["items"]
    for ev in items:
        r = client.delete(f"/agents/{agent_uuid}/evaluators/{ev['uuid']}", headers=h)
        assert r.status_code == 200, r.text


def _link_evaluators(client, h, agent_uuid, *evaluator_ids):
    r = client.post(
        f"/agents/{agent_uuid}/evaluators",
        json={"evaluator_ids": list(evaluator_ids)},
        headers=h,
    )
    assert r.status_code == 200, r.text


def _insert_run(org, agent_id, trace_uuid, status, **overrides):
    row = {
        "uuid": str(uuid.uuid4()),
        "trace_uuid": trace_uuid,
        "org_uuid": org,
        "agent_id": agent_id,
        "status": status,
        "criteria": None,
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
            "INSERT INTO trace_evaluations "
            "(uuid, trace_uuid, org_uuid, agent_id, status, criteria, "
            "available_at, attempts, error, created_at, updated_at, completed_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                row["uuid"],
                row["trace_uuid"],
                row["org_uuid"],
                row["agent_id"],
                row["status"],
                row["criteria"],
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


def test_agent_reads_include_auto_score_traces_off_by_default(client):
    h = _signup(client)
    agent = _create_agent(client, h, f"flag-read-{uuid.uuid4().hex[:6]}")

    got = client.get(f"/agents/{agent['uuid']}", headers=h)
    assert got.status_code == 200, got.text
    assert got.json()["auto_score_traces"] is False

    listed = client.get("/agents", headers=h)
    item = next(a for a in listed.json()["items"] if a["uuid"] == agent["uuid"])
    assert item["auto_score_traces"] is False


def test_omitting_auto_score_traces_leaves_it_unchanged(client):
    h = _signup(client)
    agent = _create_agent(client, h, f"flag-omit-{uuid.uuid4().hex[:6]}")
    clean = _create_evaluator(client, h, name=f"omit-clean-{uuid.uuid4().hex[:6]}")
    _unlink_all_evaluators(client, h, agent["uuid"])
    _link_evaluators(client, h, agent["uuid"], clean)

    enabled = client.put(
        f"/agents/{agent['uuid']}",
        json={"auto_score_traces": True},
        headers=h,
    )
    assert enabled.status_code == 200, enabled.text
    assert enabled.json()["auto_score_traces"] is True

    renamed = f"flag-omit-renamed-{uuid.uuid4().hex[:6]}"
    r = client.put(
        f"/agents/{agent['uuid']}",
        json={"name": renamed},
        headers=h,
    )
    assert r.status_code == 200, r.text
    assert r.json()["name"] == renamed
    assert r.json()["auto_score_traces"] is True


def test_enable_auto_score_traces_conversation_with_eligible_llm(client):
    h = _signup(client)
    agent = _create_agent(client, h, f"flag-on-{uuid.uuid4().hex[:6]}")
    clean = _create_evaluator(client, h, name=f"conv-clean-{uuid.uuid4().hex[:6]}")
    _unlink_all_evaluators(client, h, agent["uuid"])
    _link_evaluators(client, h, agent["uuid"], clean)

    r = client.put(
        f"/agents/{agent['uuid']}",
        json={"auto_score_traces": True},
        headers=h,
    )
    assert r.status_code == 200, r.text
    assert r.json()["auto_score_traces"] is True
    fetched = client.get(f"/agents/{agent['uuid']}", headers=h).json()
    assert fetched["auto_score_traces"] is True


def test_enable_auto_score_traces_general_with_eligible_llm_general(client):
    h = _signup(client)
    created = client.post(
        "/agents",
        json={
            "name": f"flag-gen-{uuid.uuid4().hex[:6]}",
            "type": "agent",
            "interaction_type": "general",
        },
        headers=h,
    ).json()
    clean = _create_evaluator(
        client,
        h,
        name=f"gen-clean-{uuid.uuid4().hex[:6]}",
        evaluator_type="llm-general",
    )
    _unlink_all_evaluators(client, h, created["uuid"])
    _link_evaluators(client, h, created["uuid"], clean)

    r = client.put(
        f"/agents/{created['uuid']}",
        json={"auto_score_traces": True},
        headers=h,
    )
    assert r.status_code == 200, r.text
    assert r.json()["auto_score_traces"] is True
    assert r.json()["interaction_type"] == "general"


def test_enable_rejected_when_only_default_correctness_evaluator_is_linked(client):
    """The seeded correctness defaults declare `{{criteria}}`, so a new agent
    has zero eligible evaluators and cannot opt in."""
    h = _signup(client)
    agent = _create_agent(client, h, f"flag-block-{uuid.uuid4().hex[:6]}")

    r = client.put(
        f"/agents/{agent['uuid']}",
        json={"auto_score_traces": True},
        headers=h,
    )
    assert r.status_code == 422, r.text
    body = r.json()["detail"]
    assert body["error"] == (
        "There are no eligible evaluators configured for this agent"
    )
    assert body["ineligible"]
    assert {e["reason"] for e in body["ineligible"]} == {"declares_variables"}
    assert client.get(f"/agents/{agent['uuid']}", headers=h).json()[
        "auto_score_traces"
    ] is False


def test_enable_rejected_for_general_agent_with_only_default_evaluator(client):
    h = _signup(client)
    created = client.post(
        "/agents",
        json={
            "name": f"flag-gen-block-{uuid.uuid4().hex[:6]}",
            "type": "agent",
            "interaction_type": "general",
        },
        headers=h,
    ).json()

    r = client.put(
        f"/agents/{created['uuid']}",
        json={"auto_score_traces": True},
        headers=h,
    )
    assert r.status_code == 422, r.text
    assert {e["reason"] for e in r.json()["detail"]["ineligible"]} == {
        "declares_variables"
    }


def test_already_on_auto_score_true_survives_later_empty_eligibility(client):
    """The 422 gate is only off→on. An already-on agent can send true again
    after its linked set drifts to zero eligible evaluators."""
    h = _signup(client)
    agent = _create_agent(client, h, f"flag-drift-{uuid.uuid4().hex[:6]}")
    clean = _create_evaluator(client, h, name=f"drift-clean-{uuid.uuid4().hex[:6]}")
    _unlink_all_evaluators(client, h, agent["uuid"])
    _link_evaluators(client, h, agent["uuid"], clean)
    assert (
        client.put(
            f"/agents/{agent['uuid']}",
            json={"auto_score_traces": True},
            headers=h,
        ).status_code
        == 200
    )

    _unlink_all_evaluators(client, h, agent["uuid"])
    eligibility = client.get(
        f"/agents/{agent['uuid']}/trace-scoring-eligibility", headers=h
    ).json()
    assert eligibility["eligible"] == []

    r = client.put(
        f"/agents/{agent['uuid']}",
        json={"auto_score_traces": True},
        headers=h,
    )
    assert r.status_code == 200, r.text
    assert r.json()["auto_score_traces"] is True


def test_eligibility_endpoint_partitions_mixed_evaluator_types(client):
    h = _signup(client)
    agent = _create_agent(client, h, f"elig-mix-{uuid.uuid4().hex[:6]}")
    clean = _create_evaluator(client, h, name=f"mix-clean-{uuid.uuid4().hex[:6]}")
    general = _create_evaluator(
        client,
        h,
        name=f"mix-general-{uuid.uuid4().hex[:6]}",
        evaluator_type="llm-general",
    )
    stt = _create_evaluator(
        client, h, name=f"mix-stt-{uuid.uuid4().hex[:6]}", evaluator_type="stt"
    )
    _link_evaluators(client, h, agent["uuid"], clean, general, stt)

    r = client.get(
        f"/agents/{agent['uuid']}/trace-scoring-eligibility", headers=h
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert set(body.keys()) == {"eligible", "ineligible"}
    assert [e["evaluator_uuid"] for e in body["eligible"]] == [clean]
    assert body["eligible"][0]["name"]
    assert body["eligible"][0]["evaluator_version_id"]
    by_id = {e["evaluator_uuid"]: e["reason"] for e in body["ineligible"]}
    assert by_id[general] == "wrong_type_for_agent"
    assert by_id[stt] == "wrong_type_for_agent"
    assert "declares_variables" in by_id.values()


def test_eligibility_endpoint_reports_each_disqualification_reason(client):
    h = _signup(client)
    conv = _create_agent(client, h, f"elig-reasons-{uuid.uuid4().hex[:6]}")
    with_vars = _create_evaluator(
        client,
        h,
        name=f"reason-vars-{uuid.uuid4().hex[:6]}",
        variables=[{"name": "criteria"}],
    )
    wrong_type = _create_evaluator(
        client,
        h,
        name=f"reason-type-{uuid.uuid4().hex[:6]}",
        evaluator_type="llm-general",
    )
    agent_row = db.get_agent(conv["uuid"])
    no_live = db.create_evaluator(
        name=f"reason-nolive-{uuid.uuid4().hex[:6]}",
        evaluator_type="llm",
        org_uuid=agent_row["org_uuid"],
    )
    _unlink_all_evaluators(client, h, conv["uuid"])
    _link_evaluators(client, h, conv["uuid"], with_vars, wrong_type)
    db.add_evaluator_to_agent(conv["uuid"], no_live)

    r = client.get(
        f"/agents/{conv['uuid']}/trace-scoring-eligibility", headers=h
    )
    assert r.status_code == 200, r.text
    by_id = {e["evaluator_uuid"]: e["reason"] for e in r.json()["ineligible"]}
    assert by_id[with_vars] == "declares_variables"
    assert by_id[wrong_type] == "wrong_type_for_agent"
    assert by_id[no_live] == "no_live_version"
    assert r.json()["eligible"] == []

    blocked = client.put(
        f"/agents/{conv['uuid']}",
        json={"auto_score_traces": True},
        headers=h,
    )
    assert blocked.status_code == 422, blocked.text
    assert {e["reason"] for e in blocked.json()["detail"]["ineligible"]} == {
        "declares_variables",
        "wrong_type_for_agent",
        "no_live_version",
    }


def test_eligibility_endpoint_is_jwt_only_and_org_scoped(client):
    ha = _signup(client)
    agent = _create_agent(client, ha, f"elig-auth-{uuid.uuid4().hex[:6]}")
    raw = _raw_key(client, ha)

    keyed = client.get(
        f"/agents/{agent['uuid']}/trace-scoring-eligibility",
        headers={"X-API-Key": raw},
    )
    assert keyed.status_code == 403

    missing = client.get(
        f"/agents/{uuid.uuid4()}/trace-scoring-eligibility", headers=ha
    )
    assert missing.status_code == 404

    from main import _build_public_openapi

    assert (
        "/agents/{agent_uuid}/trace-scoring-eligibility"
        not in _build_public_openapi()["paths"]
    )

    hb = _signup(client)
    other = client.get(
        f"/agents/{agent['uuid']}/trace-scoring-eligibility", headers=hb
    )
    assert other.status_code == 403
    assert "organization_uuid" not in other.json()


def test_disable_auto_score_traces_deletes_pending_runs_only(client):
    h = _signup(client)
    agent = _create_agent(client, h, f"flag-off-{uuid.uuid4().hex[:6]}")
    clean = _create_evaluator(client, h, name=f"off-clean-{uuid.uuid4().hex[:6]}")
    _unlink_all_evaluators(client, h, agent["uuid"])
    _link_evaluators(client, h, agent["uuid"], clean)
    assert (
        client.put(
            f"/agents/{agent['uuid']}",
            json={"auto_score_traces": True},
            headers=h,
        ).status_code
        == 200
    )

    agent_row = db.get_agent(agent["uuid"])
    org = agent_row["org_uuid"]
    pending_trace = db.create_trace(
        org_uuid=org,
        agent_id=agent["uuid"],
        input=[{"role": "user", "content": "hi"}],
        output={"response": "hello", "tool_calls": None},
    )
    processing_trace = db.create_trace(
        org_uuid=org,
        agent_id=agent["uuid"],
        input=[{"role": "user", "content": "hi"}],
        output={"response": "hello", "tool_calls": None},
    )
    completed_trace = db.create_trace(
        org_uuid=org,
        agent_id=agent["uuid"],
        input=[{"role": "user", "content": "hi"}],
        output={"response": "hello", "tool_calls": None},
    )
    pending = _insert_run(org, agent["uuid"], pending_trace["uuid"], "pending")
    processing = _insert_run(
        org, agent["uuid"], processing_trace["uuid"], "processing"
    )
    completed = _insert_run(
        org,
        agent["uuid"],
        completed_trace["uuid"],
        "completed",
        completed_at=5,
    )

    r = client.put(
        f"/agents/{agent['uuid']}",
        json={"auto_score_traces": False},
        headers=h,
    )
    assert r.status_code == 200, r.text
    assert r.json()["auto_score_traces"] is False

    with db.get_db_connection() as conn:
        remaining = {
            row["uuid"]: row["status"]
            for row in conn.execute(
                "SELECT uuid, status FROM trace_evaluations "
                "WHERE uuid IN (?, ?, ?)",
                (pending, processing, completed),
            ).fetchall()
        }
    assert pending not in remaining
    assert remaining[processing] == "processing"
    assert remaining[completed] == "completed"


def test_omitting_auto_score_traces_does_not_delete_pending_runs(client):
    h = _signup(client)
    agent = _create_agent(client, h, f"flag-omit-runs-{uuid.uuid4().hex[:6]}")
    clean = _create_evaluator(client, h, name=f"omit-run-{uuid.uuid4().hex[:6]}")
    _unlink_all_evaluators(client, h, agent["uuid"])
    _link_evaluators(client, h, agent["uuid"], clean)
    client.put(
        f"/agents/{agent['uuid']}",
        json={"auto_score_traces": True},
        headers=h,
    )
    agent_row = db.get_agent(agent["uuid"])
    trace = db.create_trace(
        org_uuid=agent_row["org_uuid"],
        agent_id=agent["uuid"],
        input=[{"role": "user", "content": "hi"}],
        output={"response": "hello", "tool_calls": None},
    )
    pending = _insert_run(
        agent_row["org_uuid"], agent["uuid"], trace["uuid"], "pending"
    )

    r = client.put(
        f"/agents/{agent['uuid']}",
        json={"name": f"flag-omit-runs-renamed-{uuid.uuid4().hex[:6]}"},
        headers=h,
    )
    assert r.status_code == 200, r.text
    assert r.json()["auto_score_traces"] is True
    with db.get_db_connection() as conn:
        row = conn.execute(
            "SELECT status FROM trace_evaluations WHERE uuid = ?", (pending,)
        ).fetchone()
    assert row["status"] == "pending"


def test_enable_auto_score_traces_with_api_key(client):
    h = _signup(client)
    agent = _create_agent(client, h, f"flag-key-{uuid.uuid4().hex[:6]}")
    clean = _create_evaluator(client, h, name=f"key-clean-{uuid.uuid4().hex[:6]}")
    _unlink_all_evaluators(client, h, agent["uuid"])
    _link_evaluators(client, h, agent["uuid"], clean)
    raw = _raw_key(client, h)

    r = client.put(
        f"/agents/{agent['uuid']}",
        json={"auto_score_traces": True},
        headers={"X-API-Key": raw},
    )
    assert r.status_code == 200, r.text
    assert r.json()["auto_score_traces"] is True


def test_duplicate_agent_does_not_copy_auto_score_traces(client):
    h = _signup(client)
    agent = _create_agent(client, h, f"flag-dup-{uuid.uuid4().hex[:6]}")
    clean = _create_evaluator(client, h, name=f"dup-clean-{uuid.uuid4().hex[:6]}")
    _unlink_all_evaluators(client, h, agent["uuid"])
    _link_evaluators(client, h, agent["uuid"], clean)
    client.put(
        f"/agents/{agent['uuid']}",
        json={"auto_score_traces": True},
        headers=h,
    )

    dup = client.post(
        f"/agents/{agent['uuid']}/duplicate",
        json={"name": f"flag-dup-copy-{uuid.uuid4().hex[:6]}"},
        headers=h,
    )
    assert dup.status_code == 200, dup.text
    copied = client.get(f"/agents/{dup.json()['uuid']}", headers=h).json()
    assert copied["auto_score_traces"] is False
