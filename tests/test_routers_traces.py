"""Integration tests for the /traces router."""

from __future__ import annotations

import uuid
from typing import Optional
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


def _create_agent(client, h) -> str:
    created = client.post(
        "/agents",
        json={"name": f"agent-{uuid.uuid4().hex[:8]}", "type": "agent"},
        headers=h,
    )
    assert created.status_code in (200, 201), created.text
    return created.json()["uuid"]


def _signup_with_agent(client):
    h = _signup(client)
    return h, _create_agent(client, h)


def _mid() -> str:
    return f"m-{uuid.uuid4().hex[:10]}"


def _org_uuid(agent_id: str) -> str:
    from db import get_agent

    return get_agent(agent_id)["org_uuid"]


def _record_verdicts(agent_id: str, trace_uuid: str, verdicts) -> str:
    """Land one eval run's worth of verdicts on a trace, returning the run ID."""
    from traces import eval_store

    org_uuid = _org_uuid(agent_id)
    run = eval_store.create_eval_run(
        org_uuid,
        agent_id,
        trigger=eval_store.TRIGGER_MANUAL,
        inferred_type="response",
        status="done",
    )
    eval_store.record_results(
        org_uuid,
        run["uuid"],
        [dict(v, trace_uuid=trace_uuid) for v in verdicts],
    )
    return run["uuid"]


def _binary(name: str = "safety", passed: bool = True, **overrides):
    verdict = {
        "evaluator_uuid": str(uuid.uuid4()),
        "evaluator_name": name,
        "output_type": "binary",
        "passed": passed,
        "reasoning": f"{name} verdict",
    }
    verdict.update(overrides)
    return verdict


def _rating(name: str = "helpfulness", score: float = 4.0, **overrides):
    verdict = {
        "evaluator_uuid": str(uuid.uuid4()),
        "evaluator_name": name,
        "output_type": "rating",
        "score": score,
        "scale_min": 1.0,
        "scale_max": 5.0,
        "reasoning": f"{name} verdict",
    }
    verdict.update(overrides)
    return verdict


def _payload(
    agent_id: str,
    message_id: Optional[str] = None,
    conversation_id: str = "conv-1",
    **overrides,
):
    payload = {
        "agent_id": agent_id,
        "message_id": message_id or _mid(),
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


# ---------------------------------------------------------------------------
# Ingestion
# ---------------------------------------------------------------------------


def test_ingest_requires_auth(client):
    h, agent = _signup_with_agent(client)
    assert client.post("/traces", json=_payload(agent)).status_code in (401, 403)
    assert (
        client.post(
            "/traces", json=_payload(agent), headers={"X-API-Key": "sk_bogus"}
        ).status_code
        == 401
    )


def test_ingest_with_jwt_is_idempotent(client):
    h, agent = _signup_with_agent(client)
    mid = _mid()

    first = client.post("/traces", json=_payload(agent, mid), headers=h)
    assert first.status_code == 200, first.text
    body = first.json()
    assert body["created"] is True
    assert len(body["uuid"]) == 36
    assert body["agent_id"] == agent
    assert body["message_id"] == mid
    assert body["conversation_id"] == "conv-1"
    assert body["created_at"].endswith("Z") and "T" in body["created_at"]

    retry = client.post("/traces", json=_payload(agent, mid), headers=h)
    assert retry.status_code == 200
    assert retry.json()["created"] is False
    assert retry.json()["uuid"] == body["uuid"]


def test_ingest_with_api_key(client):
    h, agent = _signup_with_agent(client)
    key_headers = _api_key_headers(client, h)

    res = client.post("/traces", json=_payload(agent), headers=key_headers)
    assert res.status_code == 200, res.text
    assert res.json()["created"] is True


def test_ingest_requires_a_known_agent(client):
    h, agent = _signup_with_agent(client)

    # Missing entirely, or blank, is a schema error.
    no_agent = _payload(agent)
    del no_agent["agent_id"]
    assert client.post("/traces", json=no_agent, headers=h).status_code == 422
    assert client.post("/traces", json=_payload(""), headers=h).status_code == 422

    # A well-formed but unknown id is a 404, and nothing is stored.
    unknown = _payload("00000000-0000-4000-8000-000000000001")
    assert client.post("/traces", json=unknown, headers=h).status_code == 404
    assert client.get("/traces", headers=h).json()["total"] == 0

    # A malformed id lands on the same 404 rather than a third error shape.
    assert client.post("/traces", json=_payload("nope"), headers=h).status_code == 404


def test_ingest_rejects_another_workspaces_agent(client):
    h, _agent = _signup_with_agent(client)
    _other_h, other_agent = _signup_with_agent(client)

    # Cross-workspace must be indistinguishable from nonexistent.
    res = client.post("/traces", json=_payload(other_agent), headers=h)
    assert res.status_code == 404
    assert res.json()["detail"] == "Agent not found"
    assert client.get("/traces", headers=h).json()["total"] == 0


def test_ingest_agent_check_precedes_idempotency(client):
    """A retry naming an unknown agent must fail the same way as the first call."""
    h, agent = _signup_with_agent(client)
    mid = _mid()
    assert (
        client.post("/traces", json=_payload(agent, mid), headers=h).status_code == 200
    )

    retry = client.post(
        "/traces",
        json=_payload("00000000-0000-4000-8000-000000000002", mid),
        headers=h,
    )
    assert retry.status_code == 404


def test_reingest_under_a_different_agent_returns_the_stored_trace(client):
    """The idempotency key stays workspace-scoped, so the turn does not move."""
    h, agent = _signup_with_agent(client)
    second_agent = _create_agent(client, h)
    mid = _mid()

    first = client.post("/traces", json=_payload(agent, mid), headers=h).json()
    retry = client.post("/traces", json=_payload(second_agent, mid), headers=h)

    assert retry.status_code == 200
    body = retry.json()
    assert body["created"] is False
    assert body["uuid"] == first["uuid"]
    assert body["agent_id"] == agent


def test_ingest_validation(client):
    h, agent = _signup_with_agent(client)

    # output is required.
    bad = _payload(agent)
    del bad["output"]
    assert client.post("/traces", json=bad, headers=h).status_code == 422

    # output needs a response or at least one tool call.
    empty_output = _payload(agent, output={"response": "  ", "tool_calls": None})
    assert client.post("/traces", json=empty_output, headers=h).status_code == 422

    # Tool-call-only turns are legal.
    tool_only = _payload(
        agent, output={"tool_calls": [{"tool": "get_schedule", "arguments": {}}]}
    )
    ok = client.post("/traces", json=tool_only, headers=h)
    assert ok.status_code == 200 and ok.json()["created"] is True

    # input must be non-empty.
    assert (
        client.post("/traces", json=_payload(agent, input=[]), headers=h).status_code
        == 422
    )

    # Unknown top-level keys are rejected; new needs belong in metadata.
    extra_top = _payload(agent)
    extra_top["custom_fields"] = []
    assert client.post("/traces", json=extra_top, headers=h).status_code == 422

    # Metadata entries are strict {key, value} pairs.
    bad_meta = _payload(agent, metadata=[{"key": "k", "value": "v", "extra": 1}])
    assert client.post("/traces", json=bad_meta, headers=h).status_code == 422

    # OpenAI-format extras on input turns pass through (tool_calls, tool_call_id).
    openai_history = _payload(
        agent,
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

    h, agent = _signup_with_agent(client)
    monkeypatch.setattr(org_limits_mod, "DEFAULT_MAX_TRACES", 1)

    first_mid = _mid()
    assert (
        client.post("/traces", json=_payload(agent, first_mid), headers=h).status_code
        == 200
    )

    capped = client.post("/traces", json=_payload(agent), headers=h)
    assert capped.status_code == 429
    detail = capped.json()["detail"]
    assert detail["current"] == 1
    assert detail["max_traces"] == 1
    assert "hint" in detail

    # A retry of an already-stored message_id still succeeds at the cap.
    retry = client.post("/traces", json=_payload(agent, first_mid), headers=h)
    assert retry.status_code == 200
    assert retry.json()["created"] is False


def test_cap_counts_the_whole_workspace_not_one_agent(client, monkeypatch):
    from routers import org_limits as org_limits_mod

    h, agent = _signup_with_agent(client)
    second_agent = _create_agent(client, h)
    monkeypatch.setattr(org_limits_mod, "DEFAULT_MAX_TRACES", 1)

    assert client.post("/traces", json=_payload(agent), headers=h).status_code == 200
    capped = client.post("/traces", json=_payload(second_agent), headers=h)
    assert capped.status_code == 429


# ---------------------------------------------------------------------------
# List / detail / bulk delete (curation surface, JWT-only)
# ---------------------------------------------------------------------------


def test_curation_endpoints_are_jwt_only(client):
    h = _signup(client)
    key_headers = _api_key_headers(client, h)

    assert client.get("/traces").status_code in (401, 403)
    assert client.get("/traces", headers=key_headers).status_code in (401, 403)
    assert (
        client.post(
            "/traces/bulk-delete", json={"select_all": True}, headers=key_headers
        ).status_code
        in (401, 403)
    )


def test_list_and_detail_roundtrip(client):
    h, agent = _signup_with_agent(client)

    mid_a = _mid()
    client.post("/traces", json=_payload(agent, mid_a, "conv-a"), headers=h)
    mid_b = _mid()
    openai_extras = [
        {"role": "user", "content": "check the POLIO schedule"},
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
        {"role": "user", "content": "and in months?"},
    ]
    created_b = client.post(
        "/traces",
        json=_payload(agent, mid_b, "conv-b", input=openai_extras),
        headers=h,
    ).json()

    listed = client.get("/traces", headers=h)
    assert listed.status_code == 200
    body = listed.json()
    assert set(body) == {"items", "total", "limit", "offset"}
    assert body["total"] == 2 and body["limit"] == 50 and body["offset"] == 0
    # Newest first.
    assert [item["message_id"] for item in body["items"]] == [mid_b, mid_a]
    summary_b = body["items"][0]
    assert summary_b["agent_id"] == agent
    assert summary_b["turn_count"] == 4
    assert summary_b["tool_call_count"] == 1
    assert summary_b["tool_call_names"] == ["get_schedule"]
    assert summary_b["metadata_count"] == 1
    assert summary_b["input_preview"] == "and in months?"
    assert summary_b["response_preview"].startswith("Aapki beti")

    detail = client.get(f"/traces/{created_b['uuid']}", headers=h)
    assert detail.status_code == 200
    full = detail.json()
    assert full["agent_id"] == agent
    assert full["conversation_id"] == "conv-b"
    # OpenAI-format extras on history turns survive storage verbatim.
    assert full["input"][1]["tool_calls"][0]["function"]["name"] == "get_schedule"
    assert full["input"][2]["tool_call_id"] == "call_1"
    assert full["output"]["tool_calls"][0] == {
        "tool": "get_schedule",
        "arguments": {"child_age_weeks": 14},
    }
    assert full["metadata"] == [{"key": "gen_ai.request.model", "value": "gpt-4"}]

    assert (
        client.get(
            "/traces/00000000-0000-4000-8000-000000000001", headers=h
        ).status_code
        == 404
    )
    # Another workspace can't read this trace.
    other = _signup(client)
    assert client.get(f"/traces/{created_b['uuid']}", headers=other).status_code == 404


def test_summary_previews_distinct_tool_names(client):
    h, agent = _signup_with_agent(client)
    calls = [
        {"tool": "get_schedule", "arguments": {"weeks": 14}},
        # A repeat spends no preview slot — tool_call_count carries the volume.
        {"tool": "get_schedule", "arguments": {"weeks": 18}},
        {"tool": "send_reminder", "arguments": {}},
    ]
    client.post(
        "/traces",
        json=_payload(agent, output={"response": "done", "tool_calls": calls}),
        headers=h,
    )

    item = client.get("/traces", headers=h).json()["items"][0]
    assert item["tool_call_count"] == 3
    assert item["tool_call_names"] == ["get_schedule", "send_reminder"]


def test_summary_caps_the_tool_name_preview(client):
    h, agent = _signup_with_agent(client)
    calls = [{"tool": f"tool_{i}", "arguments": {}} for i in range(8)]
    client.post(
        "/traces",
        json=_payload(agent, output={"response": "done", "tool_calls": calls}),
        headers=h,
    )

    item = client.get("/traces", headers=h).json()["items"][0]
    assert item["tool_call_count"] == 8
    assert item["tool_call_names"] == [f"tool_{i}" for i in range(5)]


def test_summary_tool_names_empty_without_tool_calls(client):
    h, agent = _signup_with_agent(client)
    client.post(
        "/traces",
        json=_payload(agent, output={"response": "no tools here"}),
        headers=h,
    )

    item = client.get("/traces", headers=h).json()["items"][0]
    assert item["tool_call_count"] == 0
    assert item["tool_call_names"] == []


def test_list_filters_by_agent(client):
    h, agent = _signup_with_agent(client)
    other_agent = _create_agent(client, h)

    mine = _mid()
    client.post("/traces", json=_payload(agent, mine), headers=h)
    client.post("/traces", json=_payload(other_agent), headers=h)
    client.post("/traces", json=_payload(other_agent), headers=h)

    scoped = client.get("/traces", params={"agent_id": agent}, headers=h).json()
    assert scoped["total"] == 1
    assert scoped["items"][0]["message_id"] == mine

    assert (
        client.get("/traces", params={"agent_id": other_agent}, headers=h).json()[
            "total"
        ]
        == 2
    )
    # Omitting the filter still reads the whole workspace.
    assert client.get("/traces", headers=h).json()["total"] == 3


def test_list_rejects_another_workspaces_agent_filter(client):
    h, _agent = _signup_with_agent(client)
    _other, foreign_agent = _signup_with_agent(client)

    assert (
        client.get("/traces", params={"agent_id": foreign_agent}, headers=h).status_code
        == 404
    )


def test_list_search_filter_and_pagination(client):
    h, agent = _signup_with_agent(client)
    mid_polio = _mid()
    client.post(
        "/traces",
        json=_payload(
            agent,
            mid_polio,
            "conv-x",
            input=[{"role": "user", "content": "Tell me about POLIO boosters"}],
        ),
        headers=h,
    )
    client.post("/traces", json=_payload(agent, conversation_id="conv-y"), headers=h)
    client.post("/traces", json=_payload(agent, conversation_id="conv-y"), headers=h)

    hits = client.get("/traces", params={"q": "polio"}, headers=h).json()
    assert hits["total"] == 1
    assert hits["items"][0]["message_id"] == mid_polio

    conv = client.get("/traces", params={"conversation_id": "conv-y"}, headers=h).json()
    assert conv["total"] == 2

    page = client.get("/traces", params={"limit": 1, "offset": 1}, headers=h).json()
    assert page["total"] == 3
    assert len(page["items"]) == 1
    assert page["limit"] == 1 and page["offset"] == 1


def test_bulk_delete_router_contract(client):
    h, agent = _signup_with_agent(client)
    mid_keep = _mid()
    kept = client.post(
        "/traces", json=_payload(agent, mid_keep, "conv-keep"), headers=h
    ).json()
    mid_gone = _mid()
    client.post("/traces", json=_payload(agent, mid_gone, "conv-gone"), headers=h)
    client.post("/traces", json=_payload(agent, conversation_id="conv-gone"), headers=h)

    # Neither ids nor select_all is a 400.
    assert client.post("/traces/bulk-delete", json={}, headers=h).status_code == 400

    # select_all with a conversation filter deletes exactly that set.
    filtered = client.post(
        "/traces/bulk-delete",
        json={"select_all": True, "conversation_id": "conv-gone"},
        headers=h,
    )
    assert filtered.status_code == 200
    assert filtered.json() == {"deleted": 2}
    assert client.get("/traces", headers=h).json()["total"] == 1

    # Deleting frees the message_id: the same ID re-ingests as a new trace.
    by_ids = client.post(
        "/traces/bulk-delete", json={"trace_ids": [kept["uuid"]]}, headers=h
    )
    assert by_ids.status_code == 200 and by_ids.json() == {"deleted": 1}
    assert client.get(f"/traces/{kept['uuid']}", headers=h).status_code == 404

    reingested = client.post("/traces", json=_payload(agent, mid_keep), headers=h)
    assert reingested.status_code == 200
    assert reingested.json()["created"] is True
    assert reingested.json()["uuid"] != kept["uuid"]


def test_bulk_delete_select_all_is_bounded_by_agent(client):
    """The subtab's "delete all matching" must not reach another agent's traces."""
    h, agent = _signup_with_agent(client)
    other_agent = _create_agent(client, h)
    client.post("/traces", json=_payload(agent), headers=h)
    client.post("/traces", json=_payload(other_agent), headers=h)
    client.post("/traces", json=_payload(other_agent), headers=h)

    res = client.post(
        "/traces/bulk-delete",
        json={"select_all": True, "agent_id": agent},
        headers=h,
    )
    assert res.status_code == 200 and res.json() == {"deleted": 1}
    assert client.get("/traces", headers=h).json()["total"] == 2


def test_bulk_delete_by_ids_is_bounded_by_agent(client):
    """agent_id bounds an explicit id list too, not just select_all."""
    h, agent = _signup_with_agent(client)
    other_agent = _create_agent(client, h)
    mine = client.post("/traces", json=_payload(agent), headers=h).json()
    theirs = client.post("/traces", json=_payload(other_agent), headers=h).json()

    res = client.post(
        "/traces/bulk-delete",
        json={"trace_ids": [mine["uuid"], theirs["uuid"]], "agent_id": agent},
        headers=h,
    )
    assert res.status_code == 200 and res.json() == {"deleted": 1}
    assert client.get(f"/traces/{theirs['uuid']}", headers=h).status_code == 200


def test_bulk_delete_rejects_another_workspaces_agent(client):
    h, _agent = _signup_with_agent(client)
    _other, foreign_agent = _signup_with_agent(client)

    assert (
        client.post(
            "/traces/bulk-delete",
            json={"select_all": True, "agent_id": foreign_agent},
            headers=h,
        ).status_code
        == 404
    )


# ---------------------------------------------------------------------------
# Evaluation verdicts on the read surface
# ---------------------------------------------------------------------------


def test_list_eval_summary_is_absent_until_a_verdict_lands(client):
    h, agent = _signup_with_agent(client)
    trace = client.post("/traces", json=_payload(agent), headers=h).json()

    assert client.get("/traces", headers=h).json()["items"][0]["eval_summary"] is None

    _record_verdicts(
        agent,
        trace["uuid"],
        [_binary("safety", True), _binary("tone", False), _rating(score=5.0, passed=True)],
    )

    item = client.get("/traces", headers=h).json()["items"][0]
    assert item["eval_summary"] == {"passed": 2, "total": 3}


def test_list_eval_summary_separates_unevaluated_from_all_failed(client):
    """A badge must tell "not evaluated yet" from "evaluated and failed everything"."""
    h, agent = _signup_with_agent(client)
    unjudged = client.post("/traces", json=_payload(agent), headers=h).json()
    failed = client.post("/traces", json=_payload(agent), headers=h).json()
    _record_verdicts(
        agent, failed["uuid"], [_binary("safety", False), _binary("tone", False)]
    )

    by_uuid = {
        item["uuid"]: item["eval_summary"]
        for item in client.get("/traces", headers=h).json()["items"]
    }
    assert by_uuid[unjudged["uuid"]] is None
    assert by_uuid[failed["uuid"]] == {"passed": 0, "total": 2}


def test_list_eval_summary_is_looked_up_once_per_page(client, monkeypatch):
    from traces import eval_store as eval_store_mod

    h, agent = _signup_with_agent(client)
    traces = [
        client.post("/traces", json=_payload(agent), headers=h).json()
        for _ in range(3)
    ]
    _record_verdicts(agent, traces[0]["uuid"], [_binary("safety", True)])

    calls = []
    real = eval_store_mod.eval_summaries_for_traces

    def spy(org_uuid, trace_uuids):
        calls.append(list(trace_uuids))
        return real(org_uuid, trace_uuids)

    monkeypatch.setattr(eval_store_mod, "eval_summaries_for_traces", spy)

    body = client.get("/traces", headers=h).json()

    assert len(calls) == 1
    assert set(calls[0]) == {t["uuid"] for t in traces}
    assert sum(1 for item in body["items"] if item["eval_summary"]) == 1


def test_trace_evaluations_roundtrip(client):
    h, agent = _signup_with_agent(client)
    trace = client.post("/traces", json=_payload(agent), headers=h).json()
    first_run = _record_verdicts(
        agent, trace["uuid"], [_binary("safety", True, reasoning="No unsafe advice")]
    )
    second_run = _record_verdicts(
        agent,
        trace["uuid"],
        [_rating("helpfulness", 4.0, reasoning="Mostly actionable")],
    )

    res = client.get(f"/traces/{trace['uuid']}/evaluations", headers=h)
    assert res.status_code == 200, res.text
    body = res.json()
    assert body["trace_uuid"] == trace["uuid"]
    assert len(body["results"]) == 2

    # Newest run first.
    rating, binary = body["results"]
    assert rating["run_uuid"] == second_run
    assert binary["run_uuid"] == first_run

    assert rating["output_type"] == "rating"
    assert rating["evaluator_name"] == "helpfulness"
    assert rating["score"] == 4.0
    assert rating["scale_min"] == 1.0 and rating["scale_max"] == 5.0
    assert rating["passed"] is None
    assert rating["reasoning"] == "Mostly actionable"
    assert rating["created_at"].endswith("Z") and "T" in rating["created_at"]

    assert binary["output_type"] == "binary"
    assert binary["passed"] is True
    assert binary["score"] is None
    assert binary["scale_min"] is None and binary["scale_max"] is None
    assert binary["reasoning"] == "No unsafe advice"


def test_trace_evaluations_is_empty_before_any_run(client):
    h, agent = _signup_with_agent(client)
    trace = client.post("/traces", json=_payload(agent), headers=h).json()

    res = client.get(f"/traces/{trace['uuid']}/evaluations", headers=h)
    assert res.status_code == 200
    assert res.json() == {"trace_uuid": trace["uuid"], "results": []}


def test_trace_evaluations_404s_for_missing_and_cross_workspace_traces(client):
    h, agent = _signup_with_agent(client)
    trace = client.post("/traces", json=_payload(agent), headers=h).json()
    _record_verdicts(agent, trace["uuid"], [_binary("safety", True)])

    missing = client.get(
        "/traces/00000000-0000-4000-8000-000000000001/evaluations", headers=h
    )
    assert missing.status_code == 404
    assert missing.json()["detail"] == "Trace not found"

    # Cross-workspace must be indistinguishable from nonexistent.
    other = _signup(client)
    foreign = client.get(f"/traces/{trace['uuid']}/evaluations", headers=other)
    assert foreign.status_code == 404
    assert foreign.json()["detail"] == "Trace not found"


def test_trace_evaluations_is_jwt_only(client):
    h, agent = _signup_with_agent(client)
    trace = client.post("/traces", json=_payload(agent), headers=h).json()
    key_headers = _api_key_headers(client, h)

    assert client.get(f"/traces/{trace['uuid']}/evaluations").status_code in (401, 403)
    assert client.get(
        f"/traces/{trace['uuid']}/evaluations", headers=key_headers
    ).status_code in (401, 403)


def test_evaluations_path_does_not_collide_with_the_detail_route(client):
    """`/{trace_uuid}` must not swallow the more specific `/{trace_uuid}/evaluations`."""
    h, agent = _signup_with_agent(client)
    trace = client.post("/traces", json=_payload(agent), headers=h).json()

    detail = client.get(f"/traces/{trace['uuid']}", headers=h).json()
    evaluations = client.get(
        f"/traces/{trace['uuid']}/evaluations", headers=h
    ).json()

    assert "input" in detail and "results" not in detail
    assert set(evaluations) == {"trace_uuid", "results"}
    # A trace literally named "evaluations" would still be a plain 404, not a
    # route-shape error.
    assert client.get("/traces/evaluations", headers=h).status_code == 404
