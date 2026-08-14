"""Integration tests for the /trace-evals router.

`launch_trace_eval` is monkeypatched everywhere: the router's contract is which
runs it starts and with what, never that a judge thread really runs. The stand-in
still writes a real run row so the list and status endpoints are exercised
against stored data.
"""

from __future__ import annotations

import uuid
from typing import Any, Dict, List, Optional
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient


@pytest.fixture(scope="module")
def app():
    import main as main_mod
    from routers.trace_evals import router as trace_evals_router

    # The router is wired into main.py centrally; include it here so the tests
    # run against the real app either way.
    if not any(
        getattr(route, "path", "").startswith("/trace-evals")
        for route in main_mod.app.routes
    ):
        main_mod.app.include_router(trace_evals_router)
    return main_mod.app


@pytest.fixture(scope="module")
def client(app):
    with patch("main.recover_pending_jobs"):
        with TestClient(app) as c:
            yield c


@pytest.fixture
def launches(monkeypatch):
    """Record every `launch_trace_eval` call and create the run row it promises."""
    from traces import eval_store

    calls: List[Dict[str, Any]] = []

    def fake_launch(*, org_uuid, agent, inferred_type, traces, evaluators, trigger):
        run = eval_store.create_eval_run(
            org_uuid,
            agent["uuid"],
            trigger=trigger,
            inferred_type=inferred_type,
            status="queued",
            trace_count=len(traces),
            skipped_count=0,
        )
        calls.append(
            {
                "org_uuid": org_uuid,
                "agent_uuid": agent["uuid"],
                "inferred_type": inferred_type,
                "trace_uuids": [t["uuid"] for t in traces],
                "evaluator_types": sorted(
                    {e["evaluator_type"] for e in evaluators}
                ),
                "trigger": trigger,
                "task_id": run["uuid"],
            }
        )
        return run["uuid"], "queued"

    monkeypatch.setattr("routers.trace_evals.launch_trace_eval", fake_launch)
    return calls


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


def _signup(client):
    suffix = uuid.uuid4().hex[:8]
    body = client.post(
        "/auth/signup",
        json={
            "first_name": "Te",
            "last_name": "U",
            "email": f"te-{suffix}@example.com",
            "password": "passw0rd",
        },
    ).json()
    return {"Authorization": f"Bearer {body['access_token']}"}


def _api_key_headers(client, h):
    created = client.post("/api-keys", json={"name": "ci"}, headers=h)
    assert created.status_code == 201, created.text
    return {"X-API-Key": created.json()["key"]}


def _create_agent(client, h) -> str:
    created = client.post(
        "/agents",
        json={"name": f"agent-{uuid.uuid4().hex[:8]}", "type": "agent"},
        headers=h,
    )
    assert created.status_code in (200, 201), created.text
    return created.json()["uuid"]


def _create_evaluator(client, h, evaluator_type: str) -> str:
    created = client.post(
        "/evaluators",
        json={
            "name": f"ev-{uuid.uuid4().hex[:8]}",
            "evaluator_type": evaluator_type,
            "output_type": "binary",
            "version": {
                "judge_model": "openai/gpt-4.1",
                "system_prompt": "Judge it.",
            },
        },
        headers=h,
    )
    assert created.status_code == 200, created.text
    return created.json()["uuid"]


def _link_evaluator(client, h, agent_uuid: str, evaluator_uuid: str) -> None:
    linked = client.post(
        f"/agents/{agent_uuid}/evaluators",
        json={"evaluator_ids": [evaluator_uuid]},
        headers=h,
    )
    assert linked.status_code == 200, linked.text


def _ingest(
    client,
    h,
    agent_uuid: str,
    *,
    input: Optional[List[Dict[str, Any]]] = None,
    output: Optional[Dict[str, Any]] = None,
) -> str:
    res = client.post(
        "/traces",
        json={
            "agent_id": agent_uuid,
            "message_id": f"m-{uuid.uuid4().hex[:10]}",
            "conversation_id": "conv-1",
            "input": input or [{"role": "user", "content": "When is the next dose?"}],
            "output": output or {"response": "In 14 weeks."},
        },
        headers=h,
    )
    assert res.status_code == 200, res.text
    return res.json()["uuid"]


def _agent_with_evaluators(client, *evaluator_types: str):
    h = _signup(client)
    agent = _create_agent(client, h)
    for evaluator_type in evaluator_types:
        _link_evaluator(client, h, agent, _create_evaluator(client, h, evaluator_type))
    return h, agent


# ---------------------------------------------------------------------------
# Auth and workspace scoping
# ---------------------------------------------------------------------------


def test_every_route_requires_a_jwt(client):
    h, agent = _agent_with_evaluators(client, "llm")
    key_headers = _api_key_headers(client, h)
    task_id = str(uuid.uuid4())

    unauthenticated = [
        client.post(f"/trace-evals/agent/{agent}/run", json={"select_all": True}),
        client.get(f"/trace-evals/run/{task_id}"),
        client.get(f"/trace-evals/agent/{agent}/runs"),
        client.get(f"/trace-evals/agent/{agent}/settings"),
        client.patch(
            f"/trace-evals/agent/{agent}/settings", json={"auto_eval_enabled": True}
        ),
    ]
    for res in unauthenticated:
        assert res.status_code in (401, 403), res.text

    # Judging spends judge tokens, so an API key is not enough either.
    assert client.get(
        f"/trace-evals/agent/{agent}/runs", headers=key_headers
    ).status_code in (401, 403)
    assert client.post(
        f"/trace-evals/agent/{agent}/run",
        json={"select_all": True},
        headers=key_headers,
    ).status_code in (401, 403)


def test_another_workspaces_agent_is_a_404_everywhere(client, launches):
    _owner_h, agent = _agent_with_evaluators(client, "llm")
    intruder = _signup(client)

    assert (
        client.post(
            f"/trace-evals/agent/{agent}/run",
            json={"select_all": True},
            headers=intruder,
        ).status_code
        == 404
    )
    assert (
        client.get(f"/trace-evals/agent/{agent}/runs", headers=intruder).status_code
        == 404
    )
    assert (
        client.get(f"/trace-evals/agent/{agent}/settings", headers=intruder).status_code
        == 404
    )
    assert (
        client.patch(
            f"/trace-evals/agent/{agent}/settings",
            json={"auto_eval_enabled": True},
            headers=intruder,
        ).status_code
        == 404
    )


def test_another_workspaces_run_is_a_404(client, launches):
    h, agent = _agent_with_evaluators(client, "llm")
    _ingest(client, h, agent)
    launched = client.post(
        f"/trace-evals/agent/{agent}/run", json={"select_all": True}, headers=h
    ).json()
    task_id = launched["runs"][0]["task_id"]
    assert client.get(f"/trace-evals/run/{task_id}", headers=h).status_code == 200

    intruder = _signup(client)
    stranger = client.get(f"/trace-evals/run/{task_id}", headers=intruder)
    missing = client.get(f"/trace-evals/run/{uuid.uuid4()}", headers=intruder)
    # A run in another workspace must be indistinguishable from one that never
    # existed.
    assert stranger.status_code == missing.status_code == 404
    assert stranger.json()["detail"] == missing.json()["detail"]


# ---------------------------------------------------------------------------
# Launching runs
# ---------------------------------------------------------------------------


def test_run_requires_a_linked_evaluator(client, launches):
    h = _signup(client)
    agent = _create_agent(client, h)
    _ingest(client, h, agent)

    res = client.post(
        f"/trace-evals/agent/{agent}/run", json={"select_all": True}, headers=h
    )
    assert res.status_code == 400
    assert "evaluator" in res.json()["detail"]
    assert launches == []


def test_run_fans_out_one_run_for_each_inferred_type(client, launches):
    h, agent = _agent_with_evaluators(client, "llm", "conversation")

    conversational = _ingest(
        client,
        h,
        agent,
        input=[
            {"role": "system", "content": "You help with vaccinations."},
            {"role": "user", "content": "When is the next dose?"},
        ],
    )
    single_turn = _ingest(client, h, agent)
    # tool_call needs an `llm-general` evaluator, which this agent lacks.
    tool_only = _ingest(
        client,
        h,
        agent,
        output={"tool_calls": [{"tool": "get_schedule", "arguments": {}}]},
    )

    res = client.post(
        f"/trace-evals/agent/{agent}/run", json={"select_all": True}, headers=h
    )
    assert res.status_code == 200, res.text
    body = res.json()
    assert body["skipped_count"] == 1

    by_type = {run["inferred_type"]: run for run in body["runs"]}
    assert set(by_type) == {"conversation", "response"}
    assert by_type["conversation"]["trace_count"] == 1
    assert by_type["response"]["trace_count"] == 1
    assert all(run["status"] == "queued" for run in body["runs"])
    assert all(len(run["task_id"]) == 36 for run in body["runs"])

    launched = {call["inferred_type"]: call for call in launches}
    assert launched["conversation"]["trace_uuids"] == [conversational]
    assert launched["conversation"]["evaluator_types"] == ["conversation"]
    assert launched["response"]["trace_uuids"] == [single_turn]
    assert launched["response"]["evaluator_types"] == ["llm"]
    assert all(call["trigger"] == "manual" for call in launches)
    assert tool_only not in launched["response"]["trace_uuids"]


def test_run_with_explicit_trace_ids(client, launches):
    h, agent = _agent_with_evaluators(client, "llm")
    chosen = _ingest(client, h, agent)
    _ingest(client, h, agent)

    res = client.post(
        f"/trace-evals/agent/{agent}/run", json={"trace_ids": [chosen]}, headers=h
    )
    assert res.status_code == 200, res.text
    assert res.json()["runs"][0]["trace_count"] == 1
    assert launches[0]["trace_uuids"] == [chosen]


def test_run_rejects_a_trace_from_another_agent_or_workspace(client, launches):
    h, agent = _agent_with_evaluators(client, "llm")
    sibling = _create_agent(client, h)
    siblings_trace = _ingest(client, h, sibling)

    other_h, other_agent = _agent_with_evaluators(client, "llm")
    foreign_trace = _ingest(client, other_h, other_agent)

    for trace_uuid in (siblings_trace, foreign_trace, str(uuid.uuid4())):
        res = client.post(
            f"/trace-evals/agent/{agent}/run",
            json={"trace_ids": [trace_uuid]},
            headers=h,
        )
        assert res.status_code == 404, res.text
        assert trace_uuid in res.json()["detail"]
    assert launches == []


def test_run_needs_trace_ids_when_not_selecting_all(client, launches):
    h, agent = _agent_with_evaluators(client, "llm")
    _ingest(client, h, agent)

    assert (
        client.post(f"/trace-evals/agent/{agent}/run", json={}, headers=h).status_code
        == 400
    )
    assert (
        client.post(
            f"/trace-evals/agent/{agent}/run", json={"trace_ids": []}, headers=h
        ).status_code
        == 400
    )
    assert launches == []


def test_run_with_nothing_pending_launches_nothing(client, launches):
    h, agent = _agent_with_evaluators(client, "llm")

    res = client.post(
        f"/trace-evals/agent/{agent}/run", json={"select_all": True}, headers=h
    )
    assert res.status_code == 200
    assert res.json() == {"runs": [], "skipped_count": 0}
    assert launches == []


# ---------------------------------------------------------------------------
# Run status and listing
# ---------------------------------------------------------------------------


def test_run_status_shape(client, launches):
    h, agent = _agent_with_evaluators(client, "llm")
    _ingest(client, h, agent)
    task_id = client.post(
        f"/trace-evals/agent/{agent}/run", json={"select_all": True}, headers=h
    ).json()["runs"][0]["task_id"]

    res = client.get(f"/trace-evals/run/{task_id}", headers=h)
    assert res.status_code == 200, res.text
    body = res.json()
    assert body["task_id"] == task_id
    assert body["agent_id"] == agent
    assert body["status"] == "queued"
    assert body["inferred_type"] == "response"
    assert body["trace_count"] == 1
    assert body["skipped_count"] == 0
    assert body["error"] is None
    assert body["created_at"].endswith("Z")
    assert body["started_at"] is None and body["finished_at"] is None


def test_runs_list_is_a_paginated_envelope(client, launches):
    h, agent = _agent_with_evaluators(client, "llm")
    for _ in range(3):
        _ingest(client, h, agent)
        client.post(
            f"/trace-evals/agent/{agent}/run", json={"select_all": True}, headers=h
        )

    listed = client.get(f"/trace-evals/agent/{agent}/runs", headers=h)
    assert listed.status_code == 200, listed.text
    body = listed.json()
    assert set(body) == {"items", "total", "limit", "offset"}
    assert body["total"] == 3 and body["limit"] == 50 and body["offset"] == 0
    assert len(body["items"]) == 3
    assert {item["task_id"] for item in body["items"]} == {
        call["task_id"] for call in launches
    }
    assert all(item["agent_id"] == agent for item in body["items"])
    assert all(item["trigger"] == "manual" for item in body["items"])

    page = client.get(
        f"/trace-evals/agent/{agent}/runs", params={"limit": 2, "offset": 2}, headers=h
    ).json()
    assert page["total"] == 3
    assert len(page["items"]) == 1
    assert page["limit"] == 2 and page["offset"] == 2


def test_runs_list_is_scoped_to_one_agent(client, launches):
    h, agent = _agent_with_evaluators(client, "llm")
    sibling = _create_agent(client, h)
    _ingest(client, h, agent)
    client.post(
        f"/trace-evals/agent/{agent}/run", json={"select_all": True}, headers=h
    )

    assert client.get(f"/trace-evals/agent/{agent}/runs", headers=h).json()["total"] == 1
    assert (
        client.get(f"/trace-evals/agent/{sibling}/runs", headers=h).json()["total"] == 0
    )


# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------


def test_settings_default_to_off_and_round_trip(client):
    h = _signup(client)
    agent = _create_agent(client, h)

    # Ingesting traces must never start judge spend on its own.
    initial = client.get(f"/trace-evals/agent/{agent}/settings", headers=h)
    assert initial.status_code == 200, initial.text
    assert initial.json() == {"auto_eval_enabled": False}

    enabled = client.patch(
        f"/trace-evals/agent/{agent}/settings",
        json={"auto_eval_enabled": True},
        headers=h,
    )
    assert enabled.status_code == 200, enabled.text
    assert enabled.json() == {"auto_eval_enabled": True}
    assert client.get(f"/trace-evals/agent/{agent}/settings", headers=h).json() == {
        "auto_eval_enabled": True
    }

    disabled = client.patch(
        f"/trace-evals/agent/{agent}/settings",
        json={"auto_eval_enabled": False},
        headers=h,
    )
    assert disabled.json() == {"auto_eval_enabled": False}
    assert client.get(f"/trace-evals/agent/{agent}/settings", headers=h).json() == {
        "auto_eval_enabled": False
    }


def test_settings_update_keeps_the_rest_of_the_config(client):
    h = _signup(client)
    created = client.post(
        "/agents",
        json={
            "name": f"agent-{uuid.uuid4().hex[:8]}",
            "type": "connection",
            "config": {"agent_url": "https://example.test/agent"},
        },
        headers=h,
    )
    assert created.status_code in (200, 201), created.text
    agent = created.json()["uuid"]

    client.patch(
        f"/trace-evals/agent/{agent}/settings",
        json={"auto_eval_enabled": True},
        headers=h,
    )
    config = client.get(f"/agents/{agent}", headers=h).json()["config"]
    assert config["agent_url"] == "https://example.test/agent"
    assert config["auto_eval_enabled"] is True


def test_settings_patch_requires_the_flag(client):
    h = _signup(client)
    agent = _create_agent(client, h)

    assert (
        client.patch(
            f"/trace-evals/agent/{agent}/settings", json={}, headers=h
        ).status_code
        == 422
    )
    assert (
        client.patch(
            f"/trace-evals/agent/{agent}/settings",
            json={"auto_eval_enabled": True, "unknown": 1},
            headers=h,
        ).status_code
        == 422
    )
