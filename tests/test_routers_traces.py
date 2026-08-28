"""Integration tests for the /traces router."""

from __future__ import annotations

import json
import uuid
from unittest.mock import patch

import pytest

import db
from routers.traces import MAX_DELETE_IDS, MAX_LIST_LIMIT
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


def _create_agent(client, h, interaction_type=None):
    body = {"name": f"a-{uuid.uuid4().hex[:6]}", "type": "agent"}
    if interaction_type:
        body["interaction_type"] = interaction_type
    return client.post("/agents", json=body, headers=h).json()


def _signup_with_agent(client):
    """Sign up a fresh workspace and return its headers plus one agent's uuid."""
    h = _signup(client)
    return h, _create_agent(client, h)["uuid"]


def _payload(
    agent_id: str, message_id: str, conversation_id: str = "conv-1", **overrides
):
    payload = {
        "agent_id": agent_id,
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


def _general_payload(agent_id: str, message_id: str, **overrides):
    """Trace shaped for a `general` agent: a standalone prompt, not a conversation."""
    overrides.setdefault(
        "input", "Summarize the vaccination schedule for a 14-week-old."
    )
    return _payload(agent_id, message_id, **overrides)


def _signup_with_general_agent(client):
    h = _signup(client)
    return h, _create_agent(client, h, interaction_type="general")["uuid"]


def _mid() -> str:
    return f"m-{uuid.uuid4().hex[:10]}"


def _post_trace(client, h, payload):
    res = client.post("/traces", json=payload, headers=h)
    assert res.status_code == 200, res.text
    return res.json()


# ---------------------------------------------------------------------------
# Ingestion
# ---------------------------------------------------------------------------


def test_ingest_requires_auth(client):
    agent_id = "00000000-0000-4000-8000-000000000001"
    assert client.post("/traces", json=_payload(agent_id, _mid())).status_code in (
        401,
        403,
    )
    assert (
        client.post(
            "/traces",
            json=_payload(agent_id, _mid()),
            headers={"X-API-Key": "sk_bogus"},
        ).status_code
        == 401
    )


def test_ingest_with_jwt_always_inserts(client):
    """Reusing a message_id must keep both turns: matching on it once threw one
    of them away."""
    h, agent_id = _signup_with_agent(client)
    mid = _mid()

    first = _post_trace(
        client, h, _payload(agent_id, mid, output={"response": "first answer"})
    )
    assert len(first["uuid"]) == 36
    assert first["message_id"] == mid
    assert first["conversation_id"] == "conv-1"
    assert first["created_at"].endswith("Z") and "T" in first["created_at"]

    retry = _post_trace(
        client, h, _payload(agent_id, mid, output={"response": "second answer"})
    )
    assert retry["uuid"] != first["uuid"]

    listed = client.get("/traces", headers=h).json()
    assert listed["total"] == 2
    assert {item["response_preview"] for item in listed["items"]} == {
        "first answer",
        "second answer",
    }
    for created, expected in ((first, "first answer"), (retry, "second answer")):
        full = client.get(f"/traces/{created['uuid']}", headers=h).json()
        assert full["output"]["response"] == expected


def test_ingest_with_api_key(client):
    h, agent_id = _signup_with_agent(client)
    key_headers = _api_key_headers(client, h)

    body = _post_trace(client, key_headers, _payload(agent_id, _mid()))
    assert len(body["uuid"]) == 36


def test_ingest_validation(client):
    h, agent_id = _signup_with_agent(client)

    # output is required.
    bad = _payload(agent_id, _mid())
    del bad["output"]
    assert client.post("/traces", json=bad, headers=h).status_code == 422

    # output needs a response or at least one tool call.
    empty_output = _payload(
        agent_id, _mid(), output={"response": "  ", "tool_calls": None}
    )
    assert client.post("/traces", json=empty_output, headers=h).status_code == 422

    # Tool-call-only turns are legal.
    tool_only = _payload(
        agent_id,
        _mid(),
        output={"tool_calls": [{"tool": "get_schedule", "arguments": {}}]},
    )
    ok = _post_trace(client, h, tool_only)
    tool_only_row = next(
        item
        for item in client.get("/traces", headers=h).json()["items"]
        if item["uuid"] == ok["uuid"]
    )
    assert tool_only_row["response_preview"] is None
    assert tool_only_row["tool_names"] == ["get_schedule"]
    assert tool_only_row["tool_calls"] == [
        {"tool": "get_schedule", "arguments": {}, "output": None}
    ]

    # input must be non-empty.
    assert (
        client.post(
            "/traces", json=_payload(agent_id, _mid(), input=[]), headers=h
        ).status_code
        == 422
    )

    # Unknown top-level keys are rejected; new needs belong in metadata.
    extra_top = _payload(agent_id, _mid())
    extra_top["custom_fields"] = []
    assert client.post("/traces", json=extra_top, headers=h).status_code == 422

    # Metadata entries are strict {key, value} pairs.
    bad_meta = _payload(
        agent_id, _mid(), metadata=[{"key": "k", "value": "v", "extra": 1}]
    )
    assert client.post("/traces", json=bad_meta, headers=h).status_code == 422

    # OpenAI-format extras on input turns pass through (tool_calls, tool_call_id).
    openai_history = _payload(
        agent_id,
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
    ok = _post_trace(client, h, openai_history)


def test_ingest_rejects_oversized_payloads(client):
    h, agent_id = _signup_with_agent(client)

    too_many_turns = _payload(
        agent_id, _mid(), input=[{"role": "user", "content": "hi"}] * 501
    )
    assert client.post("/traces", json=too_many_turns, headers=h).status_code == 422

    long_turn = _payload(
        agent_id, _mid(), input=[{"role": "user", "content": "x" * 50_001}]
    )
    assert client.post("/traces", json=long_turn, headers=h).status_code == 422

    long_response = _payload(agent_id, _mid(), output={"response": "x" * 50_001})
    assert client.post("/traces", json=long_response, headers=h).status_code == 422

    too_many_calls = _payload(
        agent_id,
        _mid(),
        output={"response": "ok", "tool_calls": [{"tool": "get_schedule"}] * 51},
    )
    assert client.post("/traces", json=too_many_calls, headers=h).status_code == 422

    too_much_metadata = _payload(
        agent_id, _mid(), metadata=[{"key": "k", "value": "v"}] * 101
    )
    assert client.post("/traces", json=too_much_metadata, headers=h).status_code == 422


def test_ingest_cap_returns_429(client, monkeypatch):
    from routers import traces as traces_mod

    h, agent_id = _signup_with_agent(client)
    monkeypatch.setattr(traces_mod, "MAX_TRACES_PER_WORKSPACE", 1)

    _post_trace(client, h, _payload(agent_id, _mid()))

    capped = client.post("/traces", json=_payload(agent_id, _mid()), headers=h)
    assert capped.status_code == 429
    detail = capped.json()["detail"]
    assert detail["current"] == 1
    assert detail["max_traces"] == 1
    assert "hint" in detail


def test_ingest_without_ids_stores_null_labels(client):
    h, agent_id = _signup_with_agent(client)
    body = _payload(agent_id, _mid())
    del body["message_id"]
    del body["conversation_id"]

    first = _post_trace(client, h, body)
    assert first["message_id"] is None
    assert first["conversation_id"] is None

    summary = client.get("/traces", headers=h).json()["items"][0]
    assert summary["message_id"] is None and summary["conversation_id"] is None
    full = client.get(f"/traces/{first['uuid']}", headers=h).json()
    assert full["message_id"] is None and full["conversation_id"] is None

    second = _post_trace(client, h, body)
    assert second["uuid"] != first["uuid"]
    assert client.get("/traces", headers=h).json()["total"] == 2


def test_ingest_accepts_a_message_id_without_a_conversation_id(client):
    h, agent_id = _signup_with_agent(client)
    mid = _mid()
    body = _payload(agent_id, mid)
    del body["conversation_id"]

    created = _post_trace(client, h, body)
    assert created["message_id"] == mid
    assert created["conversation_id"] is None


def test_ingest_requires_a_known_agent(client):
    h, agent_id = _signup_with_agent(client)

    missing_agent = _payload(agent_id, _mid())
    del missing_agent["agent_id"]
    assert client.post("/traces", json=missing_agent, headers=h).status_code == 422

    for bad_agent_id in ("", "x" * 37):
        bad = client.post(
            "/traces", json=_payload(bad_agent_id, _mid()), headers=h
        )
        assert bad.status_code == 422, bad.text

    unknown = client.post(
        "/traces",
        json=_payload("00000000-0000-4000-8000-000000000002", _mid()),
        headers=h,
    )
    assert unknown.status_code == 404, unknown.text
    assert unknown.json()["detail"] == "Agent not found"


def test_ingest_rejects_agent_from_another_workspace(client):
    h, _ = _signup_with_agent(client)
    _, other_agent_id = _signup_with_agent(client)

    res = client.post("/traces", json=_payload(other_agent_id, _mid()), headers=h)
    assert res.status_code == 404, res.text
    assert res.json()["detail"] == "Agent not found"


def test_ingest_checks_agent_before_writing(client):
    h, agent_id = _signup_with_agent(client)
    mid = _mid()
    _post_trace(client, h, _payload(agent_id, mid))

    retry = client.post(
        "/traces",
        json=_payload("00000000-0000-4000-8000-000000000003", mid),
        headers=h,
    )
    assert retry.status_code == 404, retry.text


def test_same_message_id_on_two_agents_is_two_traces(client):
    h, agent_id = _signup_with_agent(client)
    other_agent_id = _create_agent(client, h)["uuid"]
    mid = _mid()

    first = _post_trace(client, h, _payload(agent_id, mid))
    second = _post_trace(client, h, _payload(other_agent_id, mid))
    assert second["uuid"] != first["uuid"]

    full = client.get(f"/traces/{first['uuid']}", headers=h).json()
    assert full["agent_id"] == agent_id
    other = client.get(f"/traces/{second['uuid']}", headers=h).json()
    assert other["agent_id"] == other_agent_id


# ---------------------------------------------------------------------------
# List / detail / bulk delete (curation surface, JWT-only)
# ---------------------------------------------------------------------------


def test_curation_endpoints_are_jwt_only(client):
    h = _signup(client)
    key_headers = _api_key_headers(client, h)

    trace_uuid = "00000000-0000-4000-8000-000000000001"
    assert client.get("/traces").status_code in (401, 403)
    assert client.get("/traces", headers=key_headers).status_code in (401, 403)
    assert (
        client.get(f"/traces/{trace_uuid}", headers=key_headers).status_code
        in (401, 403)
    )
    assert (
        client.post(
            "/traces/bulk-delete",
            json={"trace_ids": [trace_uuid]},
            headers=key_headers,
        ).status_code
        in (401, 403)
    )


def test_list_and_detail_roundtrip(client):
    h, agent_id = _signup_with_agent(client)

    mid_a = _mid()
    client.post(
        "/traces", json=_payload(agent_id, mid_a, conversation_id="conv-a"), headers=h
    )
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
    created_b = _post_trace(
        client,
        h,
        _payload(agent_id, mid_b, conversation_id="conv-b", input=openai_extras),
    )

    listed = client.get("/traces", headers=h)
    assert listed.status_code == 200
    body = listed.json()
    assert set(body) == {"items", "total", "limit", "offset"}
    assert body["total"] == 2 and body["limit"] == 50 and body["offset"] == 0
    # Newest first.
    assert [item["message_id"] for item in body["items"]] == [mid_b, mid_a]
    summary_b = body["items"][0]
    assert summary_b["turn_count"] == 4
    assert summary_b["tool_call_count"] == 1
    assert summary_b["tool_names"] == ["get_schedule"]
    assert summary_b["tool_calls"] == [
        {"tool": "get_schedule", "arguments": {"child_age_weeks": 14}, "output": None}
    ]
    assert summary_b["metadata_count"] == 1
    assert summary_b["input_preview"] == "and in months?"
    assert summary_b["response_preview"].startswith("Aapki beti")

    detail = client.get(f"/traces/{created_b['uuid']}", headers=h)
    assert detail.status_code == 200
    full = detail.json()
    assert full["conversation_id"] == "conv-b"
    # OpenAI-format extras on history turns survive storage verbatim.
    assert full["input"][1]["tool_calls"][0]["function"]["name"] == "get_schedule"
    assert full["input"][2]["tool_call_id"] == "call_1"
    assert full["output"]["tool_calls"][0] == {
        "tool": "get_schedule",
        "arguments": {"child_age_weeks": 14},
        "output": None,
    }
    assert full["metadata"] == [{"key": "gen_ai.request.model", "value": "gpt-4"}]

    assert (
        client.get(
            "/traces/00000000-0000-4000-8000-000000000001", headers=h
        ).status_code
        == 404
    )
    # Another workspace can't read this trace: it exists, so 403 not 404.
    other = _signup(client)
    denied = client.get(f"/traces/{created_b['uuid']}", headers=other)
    assert denied.status_code == 403
    assert denied.json()["detail"] == "This resource belongs to a different workspace"


def test_output_with_multiple_tool_calls_roundtrips(client):
    # A Responses turn can emit several tool calls (parallel, or accumulated
    # across round-trips) alongside its reply, so output.tool_calls is a list:
    # every entry, in order, must survive the count and the detail body.
    h, agent_id = _signup_with_agent(client)
    mid = _mid()
    output = {
        "response": "You're booked for Thursday at 4:30 PM.",
        "tool_calls": [
            {"tool": "check_availability", "arguments": {"date": "Thursday"}},
            {
                "tool": "book_appointment",
                "arguments": {
                    "patient_name": "Sam",
                    "date": "Thursday",
                    "time_slot": "4:30 PM",
                },
            },
        ],
    }
    created = _post_trace(
        client, h, _payload(agent_id, mid, output=output)
    )

    summary = client.get("/traces", headers=h).json()["items"][0]
    assert summary["tool_call_count"] == 2
    assert summary["tool_names"] == ["check_availability", "book_appointment"]
    assert summary["tool_calls"] == [
        {**call, "output": None} for call in output["tool_calls"]
    ]
    assert summary["response_preview"].startswith("You're booked")

    full = client.get(f"/traces/{created['uuid']}", headers=h).json()
    assert full["output"]["tool_calls"] == [
        {**call, "output": None} for call in output["tool_calls"]
    ]
    assert full["output"]["response"] == output["response"]


def test_previews_are_truncated_but_the_full_text_is_kept(client):
    h, agent_id = _signup_with_agent(client)
    question = "When is the next vaccination? " * 20
    answer = "Aapki beti ka agla vaccination 14 weeks pe hai. " * 20
    created = _post_trace(
        client,
        h,
        _payload(
            agent_id,
            _mid(),
            input=[{"role": "user", "content": question}],
            output={"response": answer},
        ),
    )

    summary = client.get("/traces", headers=h).json()["items"][0]
    for preview, full_text in (
        (summary["input_preview"], question),
        (summary["response_preview"], answer),
    ):
        assert len(preview) == 160
        assert preview.endswith("…")
        assert preview[:-1] == full_text.strip()[:159]

    full = client.get(f"/traces/{created['uuid']}", headers=h).json()
    assert full["input"][0]["content"] == question
    assert full["output"]["response"] == answer


def test_list_pagination(client):
    h, agent_id = _signup_with_agent(client)
    mids = [_mid() for _ in range(3)]
    for mid in mids:
        _post_trace(client, h, _payload(agent_id, mid))

    for offset, expected_mid in enumerate(reversed(mids)):
        page = client.get(
            "/traces", params={"limit": 1, "offset": offset}, headers=h
        ).json()
        assert page["total"] == 3
        assert page["limit"] == 1 and page["offset"] == offset
        assert [item["message_id"] for item in page["items"]] == [expected_mid]


def test_bulk_delete_router_contract(client):
    h, agent_id = _signup_with_agent(client)
    mid_keep = _mid()
    kept = _post_trace(client, h, _payload(agent_id, mid_keep))
    gone_a = _post_trace(client, h, _payload(agent_id, _mid()))
    gone_b = _post_trace(client, h, _payload(agent_id, _mid()))

    # trace_ids is required and must be non-empty.
    assert client.post("/traces/bulk-delete", json={}, headers=h).status_code == 422
    assert (
        client.post(
            "/traces/bulk-delete", json={"trace_ids": []}, headers=h
        ).status_code
        == 422
    )

    filtered = client.post(
        "/traces/bulk-delete",
        json={"trace_ids": [gone_a["uuid"], gone_b["uuid"]]},
        headers=h,
    )
    assert filtered.status_code == 200
    assert filtered.json() == {"deleted": 2}
    assert client.get("/traces", headers=h).json()["total"] == 1

    by_ids = client.post(
        "/traces/bulk-delete", json={"trace_ids": [kept["uuid"]]}, headers=h
    )
    assert by_ids.status_code == 200 and by_ids.json() == {"deleted": 1}
    assert client.get(f"/traces/{kept['uuid']}", headers=h).status_code == 404

    reingested = _post_trace(client, h, _payload(agent_id, mid_keep))
    assert reingested["uuid"] != kept["uuid"]


def test_agent_id_is_returned_on_list_and_detail(client):
    h, agent_id = _signup_with_agent(client)
    created = _post_trace(client, h, _payload(agent_id, _mid()))

    assert client.get("/traces", headers=h).json()["items"][0]["agent_id"] == agent_id

    detail = client.get(f"/traces/{created['uuid']}", headers=h)
    assert detail.status_code == 200, detail.text
    assert detail.json()["agent_id"] == agent_id


def test_list_filters_by_agent_id(client):
    h, agent_a = _signup_with_agent(client)
    agent_b = _create_agent(client, h)["uuid"]
    mid_a = _mid()
    client.post(
        "/traces",
        json=_payload(
            agent_a,
            mid_a,
            conversation_id="conv-a",
            input=[{"role": "user", "content": "POLIO booster for agent a"}],
        ),
        headers=h,
    )
    client.post(
        "/traces",
        json=_payload(agent_a, _mid(), conversation_id="conv-other"),
        headers=h,
    )
    client.post(
        "/traces",
        json=_payload(
            agent_b,
            _mid(),
            conversation_id="conv-a",
            input=[{"role": "user", "content": "POLIO booster for agent b"}],
        ),
        headers=h,
    )

    # `total` counts the filtered set, not every trace in the workspace.
    only_a = client.get("/traces", params={"agent_id": agent_a}, headers=h).json()
    assert only_a["total"] == 2
    assert {item["agent_id"] for item in only_a["items"]} == {agent_a}
    assert mid_a in {item["message_id"] for item in only_a["items"]}

    only_b = client.get("/traces", params={"agent_id": agent_b}, headers=h).json()
    assert only_b["total"] == 1
    assert only_b["items"][0]["agent_id"] == agent_b


def test_list_filters_by_output_type(client):
    h, agent_id = _signup_with_agent(client)
    reply_only = _post_trace(
        client, h, _payload(agent_id, _mid(), output={"response": "just a reply"})
    )
    tools_only = _post_trace(
        client,
        h,
        _payload(
            agent_id,
            _mid(),
            output={"tool_calls": [{"tool": "get_schedule", "arguments": {}}]},
        ),
    )
    # A trace carrying both counts as a reply, matching what the row shows.
    both = _post_trace(
        client,
        h,
        _payload(
            agent_id,
            _mid(),
            output={
                "response": "here you go",
                "tool_calls": [{"tool": "get_schedule", "arguments": {}}],
            },
        ),
    )

    assert client.get("/traces", headers=h).json()["total"] == 3

    replies = client.get(
        "/traces", params={"output_type": "response"}, headers=h
    ).json()
    assert replies["total"] == 2
    assert {item["uuid"] for item in replies["items"]} == {
        reply_only["uuid"],
        both["uuid"],
    }
    assert all(item["response_preview"] for item in replies["items"])

    calls = client.get(
        "/traces", params={"output_type": "tool_call"}, headers=h
    ).json()
    assert calls["total"] == 1
    assert calls["items"][0]["uuid"] == tools_only["uuid"]
    assert calls["items"][0]["tool_call_count"] == 1
    assert calls["items"][0]["response_preview"] is None

    # Combines with the other filters.
    other_agent = _create_agent(client, h)["uuid"]
    _post_trace(
        client, h, _payload(other_agent, _mid(), output={"response": "elsewhere"})
    )
    scoped = client.get(
        "/traces",
        params={"output_type": "response", "agent_id": other_agent},
        headers=h,
    ).json()
    assert scoped["total"] == 1

    assert (
        client.get("/traces", params={"output_type": "reply"}, headers=h).status_code
        == 422
    )


def test_list_searches_every_stored_text_field(client):
    h, agent_id = _signup_with_agent(client)
    target = _post_trace(
        client,
        h,
        _payload(
            agent_id,
            "m-needle-id",
            conversation_id="conv-needle",
            input=[{"role": "user", "content": "Where is the NEEDLE?"}],
            output={"response": "needle in the reply"},
            metadata=[{"key": "region", "value": "needle-district"}],
        ),
    )
    _post_trace(client, h, _payload(agent_id, _mid()))

    for term in [
        "needle-id",
        "conv-NEEDLE",
        "where is the needle",
        "in the reply",
        "needle-district",
    ]:
        found = client.get("/traces", params={"q": term}, headers=h).json()
        assert found["total"] == 1, term
        assert found["items"][0]["uuid"] == target["uuid"], term

    assert client.get("/traces", params={"q": "nothing here"}, headers=h).json()[
        "total"
    ] == 0
    # A blank query is a no-op, and `%`/`_` are literal, not wildcards.
    assert client.get("/traces", params={"q": "  "}, headers=h).json()["total"] == 2
    assert client.get("/traces", params={"q": "%"}, headers=h).json()["total"] == 0
    # Search narrows within the agent filter, not across it.
    other_agent = _create_agent(client, h)["uuid"]
    _post_trace(
        client,
        h,
        _payload(
            other_agent,
            _mid(),
            input=[{"role": "user", "content": "another NEEDLE"}],
        ),
    )
    assert (
        client.get(
            "/traces", params={"q": "needle", "agent_id": other_agent}, headers=h
        ).json()["total"]
        == 1
    )


def test_list_searches_non_english_and_accented_text(client):
    """Non-ASCII text is stored escaped inside the JSON columns, and SQLite's own
    LOWER folds only ASCII."""
    h, agent_id = _signup_with_agent(client)
    hindi = _post_trace(
        client,
        h,
        _payload(
            agent_id,
            _mid(),
            input=[{"role": "user", "content": "टीका कब है"}],
        ),
    )
    accented = _post_trace(client, h, _payload(agent_id, "CAFÉ-12"))

    hits = client.get("/traces", params={"q": "टीका"}, headers=h).json()
    assert hits["total"] == 1
    assert hits["items"][0]["uuid"] == hindi["uuid"]

    hits = client.get("/traces", params={"q": "café"}, headers=h).json()
    assert hits["total"] == 1
    assert hits["items"][0]["uuid"] == accented["uuid"]


def test_bulk_delete_ignores_another_workspaces_traces(client):
    h, agent_id = _signup_with_agent(client)
    mine = _post_trace(client, h, _payload(agent_id, _mid()))
    other = _signup(client)

    res = client.post(
        "/traces/bulk-delete", json={"trace_ids": [mine["uuid"]]}, headers=other
    )
    assert res.status_code == 200, res.text
    assert res.json() == {"deleted": 0}
    assert client.get(f"/traces/{mine['uuid']}", headers=h).status_code == 200


def test_trace_cap_comes_from_its_env_var():
    """Reload re-runs the module body, which is where the env var is read."""
    import importlib
    import os

    from routers import traces as traces_mod

    original = os.environ["DEFAULT_MAX_TRACES"]
    os.environ["DEFAULT_MAX_TRACES"] = "7"
    try:
        importlib.reload(traces_mod)
        assert traces_mod.MAX_TRACES_PER_WORKSPACE == 7
    finally:
        os.environ["DEFAULT_MAX_TRACES"] = original
        importlib.reload(traces_mod)
    assert traces_mod.MAX_TRACES_PER_WORKSPACE == int(original)


def test_list_rejects_an_oversized_page(client):
    """A list row parses each trace's whole conversation, so a huge page would
    load the workspace into memory."""
    h, _ = _signup_with_agent(client)
    over = client.get(
        "/traces", params={"limit": MAX_LIST_LIMIT + 1}, headers=h
    )
    assert over.status_code == 422, over.text
    assert client.get(
        "/traces", params={"limit": MAX_LIST_LIMIT}, headers=h
    ).status_code == 200


def test_bulk_delete_rejects_malformed_trace_ids(client):
    h, _ = _signup_with_agent(client)
    for bad in ("", "x" * 100_000, "too-short"):
        res = client.post("/traces/bulk-delete", json={"trace_ids": [bad]}, headers=h)
        assert res.status_code == 422, f"{bad[:20]!r} -> {res.text}"


def test_bulk_delete_rejects_an_unknown_key(client):
    """A misspelled key must 422 rather than look like it filtered something."""
    h, agent_id = _signup_with_agent(client)
    kept = _post_trace(client, h, _payload(agent_id, _mid()))
    _post_trace(client, h, _payload(agent_id, _mid()))

    res = client.post(
        "/traces/bulk-delete",
        json={"trace_ids": [kept["uuid"]], "selectall": True},
        headers=h,
    )
    assert res.status_code == 422, res.text
    assert client.get("/traces", headers=h).json()["total"] == 2


# ---------------------------------------------------------------------------
# Convert to tests (JWT-only)
# ---------------------------------------------------------------------------


def _create_evaluator(client, h, evaluator_type="llm", variables=None):
    name = f"ev-{uuid.uuid4().hex[:6]}"
    version = {
        "judge_model": "openai/gpt-4.1",
        "system_prompt": "Judge the reply.",
    }
    if variables:
        version["system_prompt"] = "Judge the reply against {{criteria}}."
        version["variables"] = variables
    res = client.post(
        "/evaluators",
        json={
            "name": name,
            "evaluator_type": evaluator_type,
            "output_type": "binary",
            "version": version,
        },
        headers=h,
    )
    assert res.status_code == 200, res.text
    return {**res.json(), "name": name}


def _test_names(client, h):
    return [t["name"] for t in client.get("/tests", headers=h).json()["items"]]


def _convert(client, h, **body):
    return client.post("/traces/convert-to-tests", json=body, headers=h)


def test_convert_is_jwt_only(client):
    h, agent_id = _signup_with_agent(client)
    key_headers = _api_key_headers(client, h)
    trace = _post_trace(client, h, _payload(agent_id, _mid()))

    assert (
        client.post(
            "/traces/convert-to-tests",
            json={"trace_ids": [trace["uuid"]], "type": "tool_call"},
        ).status_code
        in (401, 403)
    )
    assert (
        client.post(
            "/traces/convert-to-tests",
            json={"trace_ids": [trace["uuid"]], "type": "tool_call"},
            headers=key_headers,
        ).status_code
        in (401, 403)
    )
    assert client.get("/tests", headers=h).json()["total"] == 0


def test_convert_response_creates_and_links_tests(client):
    from db import get_evaluators_for_test, get_tests_for_agent

    h, agent_id = _signup_with_agent(client)
    evaluator = _create_evaluator(client, h)
    mid_a, mid_b = _mid(), _mid()
    trace_a = _post_trace(client, h, _payload(agent_id, mid_a))
    trace_b = _post_trace(client, h, _payload(agent_id, mid_b))

    res = _convert(
        client,
        h,
        trace_ids=[trace_b["uuid"], trace_a["uuid"]],
        type="response",
        evaluators=[evaluator["uuid"]],
    )
    assert res.status_code == 200, res.text
    body = res.json()
    assert body["created"] == 2
    assert len(body["test_uuids"]) == 2

    # Order follows the requested trace order.
    names = [
        client.get(f"/tests/{test_uuid}", headers=h).json()["name"]
        for test_uuid in body["test_uuids"]
    ]
    assert names == [mid_b, mid_a]

    first = client.get(f"/tests/{body['test_uuids'][0]}", headers=h).json()
    assert first["type"] == "response"
    assert first["config"]["evaluation"] == {"type": "response"}
    # config.history is the trace's `input` verbatim.
    assert first["config"]["history"] == _payload(agent_id, mid_b)["input"]

    # Every created test carries the evaluator, not just the first.
    for test_uuid in body["test_uuids"]:
        linked = get_evaluators_for_test(test_uuid)
        assert [e["uuid"] for e in linked] == [evaluator["uuid"]], test_uuid

    # Each test is linked to the agent that produced its trace.
    agent_test_uuids = {t["uuid"] for t in get_tests_for_agent(agent_id)}
    assert agent_test_uuids == set(body["test_uuids"])

    # Everything wired up, so there is nothing to warn about.
    assert body.get("warnings") is None


def test_convert_tool_call_captures_recorded_calls(client):
    h, agent_id = _signup_with_agent(client)
    payload = _payload(agent_id, _mid())
    trace = _post_trace(client, h, payload)

    res = _convert(client, h, trace_ids=[trace["uuid"]], type="tool_call")
    assert res.status_code == 200, res.text
    created = client.get(f"/tests/{res.json()['test_uuids'][0]}", headers=h).json()
    assert created["type"] == "tool_call"
    # config.history is the trace's `input` verbatim.
    assert created["config"]["history"] == payload["input"]
    assert created["config"]["evaluation"] == {
        "type": "tool_call",
        "tool_calls": [
            {
                "tool": "get_schedule",
                "arguments": {"child_age_weeks": 14},
                "accept_any_arguments": False,
            }
        ],
    }
    # No evaluators are needed or linked for a tool_call conversion.
    assert created["evaluators"] == []


def test_convert_tool_call_with_accept_any_arguments(client):
    h, agent_id = _signup_with_agent(client)
    trace = _post_trace(client, h, _payload(agent_id, _mid()))

    res = _convert(
        client,
        h,
        trace_ids=[trace["uuid"]],
        type="tool_call",
        accept_any_arguments=True,
    )
    assert res.status_code == 200, res.text
    created = client.get(f"/tests/{res.json()['test_uuids'][0]}", headers=h).json()
    call = created["config"]["evaluation"]["tool_calls"][0]
    assert call["accept_any_arguments"] is True
    assert call["arguments"] == {"child_age_weeks": 14}


def test_convert_names_a_trace_without_a_message_id_by_its_uuid(client):
    h, agent_id = _signup_with_agent(client)
    body = _payload(agent_id, _mid())
    del body["message_id"]
    trace = _post_trace(client, h, body)

    res = _convert(client, h, trace_ids=[trace["uuid"]], type="tool_call")
    assert res.status_code == 200, res.text
    created = client.get(f"/tests/{res.json()['test_uuids'][0]}", headers=h).json()
    assert created["name"] == trace["uuid"]


def test_convert_suffixes_colliding_test_names(client):
    h, agent_id = _signup_with_agent(client)
    mid = _mid()
    trace = _post_trace(client, h, _payload(agent_id, mid))
    existing = client.post(
        "/tests", json={"name": mid, "type": "response", "config": {}}, headers=h
    )
    assert existing.status_code == 200, existing.text

    for expected in (f"{mid} (2)", f"{mid} (3)"):
        res = _convert(client, h, trace_ids=[trace["uuid"]], type="tool_call")
        assert res.status_code == 200, res.text
        created = client.get(f"/tests/{res.json()['test_uuids'][0]}", headers=h).json()
        assert created["name"] == expected

    assert sorted(_test_names(client, h)) == sorted(
        [mid, f"{mid} (2)", f"{mid} (3)"]
    )


def test_convert_suffixes_collisions_within_one_batch(client):
    h, agent_id = _signup_with_agent(client)
    mid = _mid()
    first = _post_trace(client, h, _payload(agent_id, mid))
    second = _post_trace(client, h, _payload(agent_id, mid))

    res = _convert(
        client, h, trace_ids=[first["uuid"], second["uuid"]], type="tool_call"
    )
    assert res.status_code == 200, res.text
    assert sorted(_test_names(client, h)) == sorted([mid, f"{mid} (2)"])


def test_convert_response_requires_evaluators(client):
    h, agent_id = _signup_with_agent(client)
    trace = _post_trace(client, h, _payload(agent_id, _mid()))

    for body in (
        {"trace_ids": [trace["uuid"]], "type": "response"},
        {"trace_ids": [trace["uuid"]], "type": "response", "evaluators": []},
    ):
        res = client.post("/traces/convert-to-tests", json=body, headers=h)
        assert res.status_code == 400, res.text
        assert "evaluator" in res.json()["detail"]
    assert client.get("/tests", headers=h).json()["total"] == 0


def test_convert_rejects_an_evaluator_with_variables(client):
    h, agent_id = _signup_with_agent(client)
    evaluator = _create_evaluator(
        client, h, variables=[{"name": "criteria", "description": "What to judge"}]
    )
    trace = _post_trace(client, h, _payload(agent_id, _mid()))

    res = _convert(
        client,
        h,
        trace_ids=[trace["uuid"]],
        type="response",
        evaluators=[evaluator["uuid"]],
    )
    assert res.status_code == 400, res.text
    detail = res.json()["detail"]
    assert set(detail) == {"error", "evaluators"}
    assert len(detail["evaluators"]) == 1
    message = detail["evaluators"][0]
    # The parenthesised list names the actual variable, and the static tail
    # also contains the word "criteria", so match the parenthesised form.
    assert evaluator["name"] in message and "(criteria)" in message
    assert client.get("/tests", headers=h).json()["total"] == 0


def test_convert_rejects_an_evaluator_the_workspace_cannot_see(client):
    h, agent_id = _signup_with_agent(client)
    other = _signup(client)
    theirs = _create_evaluator(client, other)
    trace = _post_trace(client, h, _payload(agent_id, _mid()))

    res = _convert(
        client,
        h,
        trace_ids=[trace["uuid"]],
        type="response",
        evaluators=[theirs["uuid"]],
    )
    assert res.status_code == 404, res.text
    # The 404 names the evaluator, so a trace-not-found 404 cannot pass instead.
    assert res.json()["detail"] == f"Evaluator {theirs['uuid']} not found"
    assert client.get("/tests", headers=h).json()["total"] == 0


def test_convert_rejects_a_non_llm_evaluator(client):
    h, agent_id = _signup_with_agent(client)
    evaluator = _create_evaluator(client, h, evaluator_type="conversation")
    trace = _post_trace(client, h, _payload(agent_id, _mid()))

    res = _convert(
        client,
        h,
        trace_ids=[trace["uuid"]],
        type="response",
        evaluators=[evaluator["uuid"]],
    )
    assert res.status_code == 400, res.text
    assert res.json()["detail"] == (
        f"Evaluator {evaluator['uuid']} has evaluator_type='conversation'. "
        "Tests of type 'response' only accept 'llm' evaluators."
    )
    assert client.get("/tests", headers=h).json()["total"] == 0


def test_convert_reports_traces_it_could_not_find(client):
    h, agent_id = _signup_with_agent(client)
    live = _post_trace(client, h, _payload(agent_id, _mid()))
    deleted = _post_trace(client, h, _payload(agent_id, _mid()))
    client.post(
        "/traces/bulk-delete", json={"trace_ids": [deleted["uuid"]]}, headers=h
    )
    unknown = "00000000-0000-4000-8000-000000000009"
    other_h, other_agent_id = _signup_with_agent(client)
    theirs = _post_trace(client, other_h, _payload(other_agent_id, _mid()))

    res = _convert(
        client,
        h,
        trace_ids=[live["uuid"], deleted["uuid"], unknown, theirs["uuid"]],
        type="tool_call",
    )
    assert res.status_code == 404, res.text
    detail = res.json()["detail"]
    assert set(detail) == {"error", "trace_ids"}
    assert detail["trace_ids"] == [deleted["uuid"], unknown, theirs["uuid"]]
    assert client.get("/tests", headers=h).json()["total"] == 0


def test_convert_rejects_another_workspaces_traces(client):
    h, agent_id = _signup_with_agent(client)
    mine = _post_trace(client, h, _payload(agent_id, _mid()))
    other = _signup(client)

    res = _convert(client, other, trace_ids=[mine["uuid"]], type="tool_call")
    assert res.status_code == 404, res.text
    assert res.json()["detail"]["trace_ids"] == [mine["uuid"]]
    assert client.get("/tests", headers=other).json()["total"] == 0
    assert client.get("/tests", headers=h).json()["total"] == 0


def test_convert_tool_call_requires_recorded_tool_calls(client):
    h, agent_id = _signup_with_agent(client)
    with_calls = _post_trace(client, h, _payload(agent_id, _mid()))
    without = _post_trace(
        client, h, _payload(agent_id, _mid(), output={"response": "just text"})
    )

    res = _convert(
        client, h, trace_ids=[with_calls["uuid"], without["uuid"]], type="tool_call"
    )
    assert res.status_code == 400, res.text
    detail = res.json()["detail"]
    assert set(detail) == {"error", "trace_ids"}
    assert detail["trace_ids"] == [without["uuid"]]
    assert client.get("/tests", headers=h).json()["total"] == 0


def test_convert_dedupes_repeated_trace_ids(client):
    h, agent_id = _signup_with_agent(client)
    trace = _post_trace(client, h, _payload(agent_id, _mid()))

    res = _convert(
        client, h, trace_ids=[trace["uuid"], trace["uuid"]], type="tool_call"
    )
    assert res.status_code == 200, res.text
    assert res.json()["created"] == 1
    assert client.get("/tests", headers=h).json()["total"] == 1


def test_convert_rejects_too_many_traces(client):
    from routers.traces import MAX_CONVERT_TRACES

    h, agent_id = _signup_with_agent(client)
    trace = _post_trace(client, h, _payload(agent_id, _mid()))

    over = _convert(
        client,
        h,
        trace_ids=[trace["uuid"]] * (MAX_CONVERT_TRACES + 1),
        type="tool_call",
    )
    assert over.status_code == 422, over.text
    assert client.post(
        "/traces/convert-to-tests", json={"trace_ids": [], "type": "tool_call"},
        headers=h,
    ).status_code == 422
    assert client.get("/tests", headers=h).json()["total"] == 0


def test_convert_rejects_an_unknown_key(client):
    h, agent_id = _signup_with_agent(client)
    trace = _post_trace(client, h, _payload(agent_id, _mid()))

    res = client.post(
        "/traces/convert-to-tests",
        json={"trace_ids": [trace["uuid"]], "type": "tool_call", "evaluator": []},
        headers=h,
    )
    assert res.status_code == 422, res.text
    assert client.get("/tests", headers=h).json()["total"] == 0


def test_convert_still_creates_the_test_when_the_agent_is_gone(client):
    from db import get_tests_for_agent

    h, agent_id = _signup_with_agent(client)
    trace = _post_trace(client, h, _payload(agent_id, _mid()))
    assert client.delete(f"/agents/{agent_id}", headers=h).status_code == 200

    res = _convert(client, h, trace_ids=[trace["uuid"]], type="tool_call")
    assert res.status_code == 200, res.text
    body = res.json()
    assert body["created"] == 1
    assert client.get(f"/tests/{body['test_uuids'][0]}", headers=h).status_code == 200
    assert get_tests_for_agent(agent_id) == []
    # The response says what was created but not wired up.
    assert body["warnings"] == [
        "1 of 1 tests were not linked to an agent, "
        "so they will not appear on any agent's test list"
    ]


def test_convert_rejects_a_malformed_evaluator_id(client):
    h, agent_id = _signup_with_agent(client)
    trace = _post_trace(client, h, _payload(agent_id, _mid()))

    res = _convert(
        client, h, trace_ids=[trace["uuid"]], type="response", evaluators=["nope"]
    )
    assert res.status_code == 422, res.text
    assert client.get("/tests", headers=h).json()["total"] == 0


def test_convert_tool_call_rejects_evaluators(client):
    """A tool_call run only diffs the recorded calls, so a linked evaluator would
    never judge anything. Refuse it instead of storing a judge that never runs."""
    h, agent_id = _signup_with_agent(client)
    evaluator = _create_evaluator(client, h)
    trace = _post_trace(client, h, _payload(agent_id, _mid()))

    res = _convert(
        client,
        h,
        trace_ids=[trace["uuid"]],
        type="tool_call",
        evaluators=[evaluator["uuid"]],
    )
    assert res.status_code == 400, res.text
    assert res.json()["detail"] == (
        "tool_call tests compare the recorded tool calls and cannot take evaluators"
    )
    assert client.get("/tests", headers=h).json()["total"] == 0


def test_convert_rejects_an_evaluator_with_no_live_version(client):
    from db import get_db_connection

    h, agent_id = _signup_with_agent(client)
    evaluator = _create_evaluator(client, h)
    with get_db_connection() as conn:
        conn.execute(
            "UPDATE evaluators SET live_version_id = NULL WHERE uuid = ?",
            (evaluator["uuid"],),
        )
        conn.commit()
    trace = _post_trace(client, h, _payload(agent_id, _mid()))

    res = _convert(
        client,
        h,
        trace_ids=[trace["uuid"]],
        type="response",
        evaluators=[evaluator["uuid"]],
    )
    assert res.status_code == 400, res.text
    problems = res.json()["detail"]["evaluators"]
    assert problems == [f'Evaluator "{evaluator["name"]}" has no live version to run.']
    assert client.get("/tests", headers=h).json()["total"] == 0


def test_convert_dedupes_repeated_evaluator_ids(client):
    from db import get_evaluators_for_test

    h, agent_id = _signup_with_agent(client)
    evaluator = _create_evaluator(client, h)
    trace = _post_trace(client, h, _payload(agent_id, _mid()))

    res = _convert(
        client,
        h,
        trace_ids=[trace["uuid"]],
        type="response",
        evaluators=[evaluator["uuid"], evaluator["uuid"]],
    )
    assert res.status_code == 200, res.text
    linked = get_evaluators_for_test(res.json()["test_uuids"][0])
    assert [e["uuid"] for e in linked] == [evaluator["uuid"]]


def test_convert_links_each_test_to_its_own_traces_agent(client):
    from db import get_tests_for_agent

    h, agent_a = _signup_with_agent(client)
    agent_b = _create_agent(client, h)["uuid"]
    mid_a, mid_b = _mid(), _mid()
    trace_a = _post_trace(client, h, _payload(agent_a, mid_a))
    trace_b = _post_trace(client, h, _payload(agent_b, mid_b))

    res = _convert(
        client, h, trace_ids=[trace_a["uuid"], trace_b["uuid"]], type="tool_call"
    )
    assert res.status_code == 200, res.text
    body = res.json()
    assert body["created"] == 2
    assert body.get("warnings") is None

    names = {
        client.get(f"/tests/{test_uuid}", headers=h).json()["name"]: test_uuid
        for test_uuid in body["test_uuids"]
    }
    assert set(names) == {mid_a, mid_b}
    assert [t["uuid"] for t in get_tests_for_agent(agent_a)] == [names[mid_a]]
    assert [t["uuid"] for t in get_tests_for_agent(agent_b)] == [names[mid_b]]


def test_convert_tool_call_captures_every_recorded_call(client):
    """A turn can record several calls, and one may have carried no arguments."""
    h, agent_id = _signup_with_agent(client)
    tool_calls = [
        {"tool": "check_availability", "arguments": {"date": "Thursday"}},
        {"tool": "list_clinics"},
        {"tool": "book_appointment", "arguments": {"time_slot": "4:30 PM"}},
    ]
    trace = _post_trace(
        client,
        h,
        _payload(agent_id, _mid(), output={"response": "Booked.", "tool_calls": tool_calls}),
    )

    res = _convert(client, h, trace_ids=[trace["uuid"]], type="tool_call")
    assert res.status_code == 200, res.text
    created = client.get(f"/tests/{res.json()['test_uuids'][0]}", headers=h).json()
    assert created["config"]["evaluation"] == {
        "type": "tool_call",
        "tool_calls": [
            {
                "tool": "check_availability",
                "arguments": {"date": "Thursday"},
                "accept_any_arguments": False,
            },
            {
                "tool": "list_clinics",
                "arguments": None,
                "accept_any_arguments": False,
            },
            {
                "tool": "book_appointment",
                "arguments": {"time_slot": "4:30 PM"},
                "accept_any_arguments": False,
            },
        ],
    }


def test_convert_warns_when_evaluators_could_not_be_linked(client, monkeypatch):
    """The tests are already committed, so a failed evaluator link must warn
    rather than 500 and lose them."""
    from db import get_evaluators_for_test, get_tests_for_agent
    from routers import traces as traces_mod

    h, agent_id = _signup_with_agent(client)
    evaluator = _create_evaluator(client, h)
    trace_a = _post_trace(client, h, _payload(agent_id, _mid()))
    trace_b = _post_trace(client, h, _payload(agent_id, _mid()))

    def _boom(*args, **kwargs):
        raise RuntimeError("link failed")

    monkeypatch.setattr(traces_mod, "set_test_evaluators", _boom)
    # Deleting the agent makes the agent link fail too, so both warnings appear.
    assert client.delete(f"/agents/{agent_id}", headers=h).status_code == 200

    res = _convert(
        client,
        h,
        trace_ids=[trace_a["uuid"], trace_b["uuid"]],
        type="response",
        evaluators=[evaluator["uuid"]],
    )
    assert res.status_code == 200, res.text
    body = res.json()
    assert body["created"] == 2
    for test_uuid in body["test_uuids"]:
        assert client.get(f"/tests/{test_uuid}", headers=h).status_code == 200
        assert get_evaluators_for_test(test_uuid) == []
    assert get_tests_for_agent(agent_id) == []
    assert body["warnings"] == [
        "2 of 2 tests were created without evaluators "
        "and will not run until you attach one",
        "2 of 2 tests were not linked to an agent, "
        "so they will not appear on any agent's test list",
    ]


def test_convert_response_accepts_a_trace_with_no_tool_calls(client):
    h, agent_id = _signup_with_agent(client)
    evaluator = _create_evaluator(client, h)
    payload = _payload(agent_id, _mid(), output={"response": "just text"})
    trace = _post_trace(client, h, payload)

    res = _convert(
        client,
        h,
        trace_ids=[trace["uuid"]],
        type="response",
        evaluators=[evaluator["uuid"]],
    )
    assert res.status_code == 200, res.text
    created = client.get(f"/tests/{res.json()['test_uuids'][0]}", headers=h).json()
    assert created["type"] == "response"
    assert created["config"]["evaluation"] == {"type": "response"}
    assert created["config"]["history"] == payload["input"]


# ---------------------------------------------------------------------------
# Standalone-prompt traces (general agents)
# ---------------------------------------------------------------------------


def test_ingest_stores_a_standalone_prompt_for_a_general_agent(client):
    h, agent_id = _signup_with_general_agent(client)
    payload = _general_payload(agent_id, _mid())

    created = _post_trace(client, h, payload)

    detail = client.get(f"/traces/{created['uuid']}", headers=h).json()
    assert detail["input"] == payload["input"]
    assert detail["output"]["response"] == payload["output"]["response"]
    assert detail["output"]["tool_calls"] == [
        {**call, "output": None} for call in payload["output"]["tool_calls"]
    ]


def test_ingest_rejects_a_standalone_prompt_for_a_conversational_agent(client):
    h, agent_id = _signup_with_agent(client)

    res = client.post(
        "/traces", json=_general_payload(agent_id, _mid()), headers=h
    )
    assert res.status_code == 400, res.text
    assert "list of turns" in res.json()["detail"]


def test_ingest_rejects_a_conversation_for_a_general_agent(client):
    h, agent_id = _signup_with_general_agent(client)

    res = client.post("/traces", json=_payload(agent_id, _mid()), headers=h)
    assert res.status_code == 400, res.text
    assert "must be a string" in res.json()["detail"]


def test_ingest_rejects_a_blank_standalone_prompt(client):
    h, agent_id = _signup_with_general_agent(client)

    res = client.post(
        "/traces", json=_general_payload(agent_id, _mid(), input="   "), headers=h
    )
    assert res.status_code == 400, res.text
    assert res.json()["detail"] == "input must not be blank"


def test_ingest_rejects_an_empty_or_oversized_standalone_prompt(client):
    h, agent_id = _signup_with_general_agent(client)

    empty = client.post(
        "/traces", json=_general_payload(agent_id, _mid(), input=""), headers=h
    )
    assert empty.status_code == 422, empty.text

    oversized = client.post(
        "/traces",
        json=_general_payload(agent_id, _mid(), input="x" * 50_001),
        headers=h,
    )
    assert oversized.status_code == 422, oversized.text


def test_ingest_checks_the_agent_shape_before_writing(client):
    h, agent_id = _signup_with_general_agent(client)

    client.post("/traces", json=_payload(agent_id, _mid()), headers=h)

    assert client.get("/traces", headers=h).json()["total"] == 0


def test_a_standalone_prompt_trace_lists_and_searches(client):
    h, agent_id = _signup_with_general_agent(client)
    payload = _general_payload(agent_id, _mid())
    _post_trace(client, h, payload)

    row = client.get("/traces", headers=h).json()["items"][0]
    assert row["input_preview"] == payload["input"]
    assert row["turn_count"] == 1

    found = client.get("/traces?q=14-week-old", headers=h).json()
    assert found["total"] == 1


def test_a_long_standalone_prompt_preview_is_truncated(client):
    h, agent_id = _signup_with_general_agent(client)
    prompt = "s" * 400
    _post_trace(client, h, _general_payload(agent_id, _mid(), input=prompt))

    row = client.get("/traces", headers=h).json()["items"][0]
    assert row["input_preview"] == "s" * 159 + "…"
    assert client.get(f"/traces/{row['uuid']}", headers=h).json()["input"] == prompt


# ---------------------------------------------------------------------------
# Converting standalone-prompt traces
# ---------------------------------------------------------------------------


def test_convert_general_creates_and_links_tests(client):
    from db import get_evaluators_for_test, get_tests_for_agent

    h, agent_id = _signup_with_general_agent(client)
    evaluator = _create_evaluator(client, h, evaluator_type="llm-general")
    payload = _general_payload(agent_id, _mid())
    trace = _post_trace(client, h, payload)

    res = _convert(
        client,
        h,
        trace_ids=[trace["uuid"]],
        type="general",
        evaluators=[evaluator["uuid"]],
    )
    assert res.status_code == 200, res.text
    body = res.json()
    assert body["created"] == 1
    assert body.get("warnings") is None

    created = client.get(f"/tests/{body['test_uuids'][0]}", headers=h).json()
    assert created["type"] == "general"
    assert created["config"]["input"] == payload["input"]
    assert "history" not in created["config"]
    assert created["config"]["evaluation"] == {"type": "general"}

    linked = get_evaluators_for_test(body["test_uuids"][0])
    assert [e["uuid"] for e in linked] == [evaluator["uuid"]]
    assert {t["uuid"] for t in get_tests_for_agent(agent_id)} == set(body["test_uuids"])


def test_convert_general_rejects_a_conversation_trace(client):
    h, agent_id = _signup_with_agent(client)
    evaluator = _create_evaluator(client, h, evaluator_type="llm-general")
    trace = _post_trace(client, h, _payload(agent_id, _mid()))

    res = _convert(
        client,
        h,
        trace_ids=[trace["uuid"]],
        type="general",
        evaluators=[evaluator["uuid"]],
    )
    assert res.status_code == 400, res.text
    detail = res.json()["detail"]
    assert detail["trace_ids"] == [trace["uuid"]]
    assert "standalone prompt" in detail["error"]


def test_convert_response_rejects_a_standalone_prompt_trace(client):
    h, agent_id = _signup_with_general_agent(client)
    evaluator = _create_evaluator(client, h)
    trace = _post_trace(client, h, _general_payload(agent_id, _mid()))

    res = _convert(
        client,
        h,
        trace_ids=[trace["uuid"]],
        type="response",
        evaluators=[evaluator["uuid"]],
    )
    assert res.status_code == 400, res.text
    detail = res.json()["detail"]
    assert detail["trace_ids"] == [trace["uuid"]]
    assert "conversation" in detail["error"]


def test_convert_general_requires_evaluators(client):
    h, agent_id = _signup_with_general_agent(client)
    trace = _post_trace(client, h, _general_payload(agent_id, _mid()))

    res = _convert(client, h, trace_ids=[trace["uuid"]], type="general")
    assert res.status_code == 400, res.text
    assert res.json()["detail"] == "general tests require at least one evaluator"


def test_convert_general_rejects_an_llm_evaluator(client):
    h, agent_id = _signup_with_general_agent(client)
    evaluator = _create_evaluator(client, h, evaluator_type="llm")
    trace = _post_trace(client, h, _general_payload(agent_id, _mid()))

    res = _convert(
        client,
        h,
        trace_ids=[trace["uuid"]],
        type="general",
        evaluators=[evaluator["uuid"]],
    )
    assert res.status_code == 400, res.text
    assert "only accept 'llm-general' evaluators" in res.json()["detail"]


def test_convert_tool_call_keeps_a_standalone_prompt(client):
    from db import get_tests_for_agent

    h, agent_id = _signup_with_general_agent(client)
    payload = _general_payload(agent_id, _mid())
    trace = _post_trace(client, h, payload)

    res = _convert(client, h, trace_ids=[trace["uuid"]], type="tool_call")
    assert res.status_code == 200, res.text
    body = res.json()
    assert body.get("warnings") is None

    created = client.get(f"/tests/{body['test_uuids'][0]}", headers=h).json()
    assert created["type"] == "tool_call"
    assert created["config"]["input"] == payload["input"]
    assert "history" not in created["config"]
    assert created["config"]["evaluation"]["tool_calls"] == [
        {
            "tool": "get_schedule",
            "arguments": {"child_age_weeks": 14},
            "accept_any_arguments": False,
        }
    ]
    assert {t["uuid"] for t in get_tests_for_agent(agent_id)} == set(body["test_uuids"])


def test_convert_skips_the_link_when_the_agent_switched_interaction_type(client):
    from db import get_tests_for_agent

    h, agent_id = _signup_with_general_agent(client)
    evaluator = _create_evaluator(client, h, evaluator_type="llm-general")
    trace = _post_trace(client, h, _general_payload(agent_id, _mid()))

    # The API no longer allows this flip, so drive the column directly: the
    # link guard exists because `db.update_agent` still accepts the change.
    import db

    assert db.update_agent(agent_id, interaction_type="conversation")

    res = _convert(
        client,
        h,
        trace_ids=[trace["uuid"]],
        type="general",
        evaluators=[evaluator["uuid"]],
    )
    assert res.status_code == 200, res.text
    body = res.json()
    assert body["created"] == 1
    assert body["warnings"] == [
        "1 of 1 tests were not linked to an agent, "
        "so they will not appear on any agent's test list"
    ]
    assert get_tests_for_agent(agent_id) == []


def test_last_user_content_handles_both_input_shapes():
    from routers.traces import _last_user_content

    assert _last_user_content("just the prompt") == "just the prompt"
    assert _last_user_content([{"role": "user", "content": "hi"}]) == "hi"
    assert _last_user_content([{"role": "assistant", "content": "hi"}]) is None


def test_tool_call_output_is_stored_and_ignored_by_conversion(client):
    h, agent_id = _signup_with_agent(client)
    output = {
        "response": "Next visit is at 14 weeks.",
        "tool_calls": [
            {
                "tool": "get_schedule",
                "arguments": {"child_age_weeks": 14},
                "output": {"next_visit": "2026-09-01", "vaccines": ["OPV"]},
            }
        ],
    }
    created = _post_trace(client, h, _payload(agent_id, _mid(), output=output))

    summary = next(
        item
        for item in client.get("/traces", headers=h).json()["items"]
        if item["uuid"] == created["uuid"]
    )
    assert summary["tool_calls"] == output["tool_calls"]
    full = client.get(f"/traces/{created['uuid']}", headers=h).json()
    assert full["output"]["tool_calls"] == output["tool_calls"]

    # A tool result is not always an object: a list or a bare string is stored too.
    for value in ([{"date": "2026-09-01"}], "2026-09-01", 14, True):
        other = _post_trace(
            client,
            h,
            _payload(
                agent_id,
                _mid(),
                output={"tool_calls": [{"tool": "get_schedule", "output": value}]},
            ),
        )
        stored = client.get(f"/traces/{other['uuid']}", headers=h).json()
        assert stored["output"]["tool_calls"][0]["output"] == value

    res = _convert(client, h, trace_ids=[created["uuid"]], type="tool_call")
    assert res.status_code == 200, res.text
    test_uuid = res.json()["test_uuids"][0]
    assertion = client.get(f"/tests/{test_uuid}", headers=h).json()["config"][
        "evaluation"
    ]["tool_calls"][0]
    assert "output" not in assertion


# ---------------------------------------------------------------------------
# select_all
# ---------------------------------------------------------------------------


def test_bulk_delete_select_all_honours_the_list_filters(client):
    h, agent_id = _signup_with_agent(client)
    other_agent = _create_agent(client, h)["uuid"]
    text_only = _post_trace(
        client, h, _payload(agent_id, _mid(), output={"response": "plain reply"})
    )
    tool_only = _post_trace(
        client,
        h,
        _payload(
            agent_id,
            _mid(),
            output={"tool_calls": [{"tool": "get_schedule", "arguments": {}}]},
        ),
    )
    elsewhere = _post_trace(client, h, _payload(other_agent, _mid()))

    res = client.post(
        "/traces/bulk-delete",
        json={"select_all": True, "agent_id": agent_id, "output_type": "tool_call"},
        headers=h,
    )
    assert res.status_code == 200, res.text
    assert res.json() == {"deleted": 1}
    remaining = {
        item["uuid"] for item in client.get("/traces", headers=h).json()["items"]
    }
    assert remaining == {text_only["uuid"], elsewhere["uuid"]}
    assert tool_only["uuid"] not in remaining

    # q narrows the same way the list does.
    marked = _post_trace(
        client,
        h,
        _payload(agent_id, _mid(), output={"response": "needle in here"}),
    )
    by_q = client.post(
        "/traces/bulk-delete", json={"select_all": True, "q": "needle"}, headers=h
    )
    assert by_q.status_code == 200 and by_q.json() == {"deleted": 1}
    assert client.get(f"/traces/{marked['uuid']}", headers=h).status_code == 404


def test_bulk_delete_select_all_ignores_trace_ids_and_stays_in_the_workspace(client):
    h, agent_id = _signup_with_agent(client)
    mine = _post_trace(client, h, _payload(agent_id, _mid()))
    other_h, other_agent = _signup_with_agent(client)
    theirs = _post_trace(client, other_h, _payload(other_agent, _mid()))

    res = client.post(
        "/traces/bulk-delete",
        json={"select_all": True, "trace_ids": [theirs["uuid"]]},
        headers=h,
    )
    assert res.status_code == 200, res.text
    assert res.json() == {"deleted": 1}
    assert client.get(f"/traces/{mine['uuid']}", headers=h).status_code == 404
    assert client.get(f"/traces/{theirs['uuid']}", headers=other_h).status_code == 200


def test_bulk_delete_select_all_with_no_matches_deletes_nothing(client):
    h, agent_id = _signup_with_agent(client)
    _post_trace(client, h, _payload(agent_id, _mid()))

    res = client.post(
        "/traces/bulk-delete",
        json={"select_all": True, "q": "no-such-text-anywhere"},
        headers=h,
    )
    assert res.status_code == 200, res.text
    assert res.json() == {"deleted": 0}
    assert client.get("/traces", headers=h).json()["total"] == 1


def test_convert_select_all_honours_the_list_filters(client):
    h, agent_id = _signup_with_agent(client)
    other_agent = _create_agent(client, h)["uuid"]
    wanted = _post_trace(client, h, _payload(agent_id, _mid()))
    _post_trace(client, h, _payload(other_agent, _mid()))

    res = _convert(
        client, h, select_all=True, agent_id=agent_id, type="tool_call"
    )
    assert res.status_code == 200, res.text
    assert res.json()["created"] == 1
    tests = client.get("/tests", headers=h).json()
    assert tests["total"] == 1
    assert tests["items"][0]["name"] == wanted["message_id"]


def test_convert_select_all_ignores_trace_ids(client):
    h, agent_id = _signup_with_agent(client)
    a = _post_trace(client, h, _payload(agent_id, _mid()))
    _post_trace(client, h, _payload(agent_id, _mid()))

    res = _convert(
        client, h, select_all=True, trace_ids=[a["uuid"]], type="tool_call"
    )
    assert res.status_code == 200, res.text
    assert res.json()["created"] == 2


def test_convert_select_all_rejects_an_empty_match(client):
    h, agent_id = _signup_with_agent(client)
    _post_trace(client, h, _payload(agent_id, _mid()))

    res = _convert(
        client, h, select_all=True, q="no-such-text-anywhere", type="tool_call"
    )
    assert res.status_code == 400, res.text
    assert "No traces matched" in res.json()["detail"]
    assert client.get("/tests", headers=h).json()["total"] == 0


def test_convert_select_all_rejects_more_than_the_cap(client, monkeypatch):
    import routers.traces as traces_mod

    h, agent_id = _signup_with_agent(client)
    _post_trace(client, h, _payload(agent_id, _mid()))
    _post_trace(client, h, _payload(agent_id, _mid()))
    monkeypatch.setattr(traces_mod, "MAX_CONVERT_TRACES", 1)

    res = _convert(client, h, select_all=True, type="tool_call")
    assert res.status_code == 400, res.text
    assert "2 traces match" in res.json()["detail"]
    assert client.get("/tests", headers=h).json()["total"] == 0


def test_select_all_off_still_requires_trace_ids(client):
    h, _ = _signup_with_agent(client)

    assert (
        client.post(
            "/traces/bulk-delete", json={"select_all": False}, headers=h
        ).status_code
        == 422
    )
    assert _convert(client, h, select_all=False, type="tool_call").status_code == 422


def test_select_all_ignores_an_over_cap_trace_ids_list(client):
    """The cap belongs to the list the handler reads, not to one it ignores."""
    from routers.traces import MAX_CONVERT_TRACES

    h, agent_id = _signup_with_agent(client)
    trace = _post_trace(client, h, _payload(agent_id, _mid()))
    stale = ["0" * 36] * (MAX_CONVERT_TRACES + 1)

    converted = _convert(
        client, h, select_all=True, trace_ids=stale, type="tool_call"
    )
    assert converted.status_code == 200, converted.text
    assert converted.json()["created"] == 1

    deleted = client.post(
        "/traces/bulk-delete",
        json={"select_all": True, "trace_ids": ["0" * 36] * (MAX_DELETE_IDS + 1)},
        headers=h,
    )
    assert deleted.status_code == 200, deleted.text
    assert deleted.json() == {"deleted": 1}
    assert client.get(f"/traces/{trace['uuid']}", headers=h).status_code == 404


def test_convert_select_all_explains_a_shape_conflict(client):
    h, agent_id = _signup_with_agent(client)
    general_agent = _create_agent(client, h, interaction_type="general")["uuid"]
    _post_trace(client, h, _payload(agent_id, _mid()))
    odd = _post_trace(client, h, _general_payload(general_agent, _mid()))
    evaluator = _create_evaluator(client, h)

    res = _convert(
        client,
        h,
        select_all=True,
        type="response",
        evaluators=[evaluator["uuid"]],
    )
    assert res.status_code == 400, res.text
    detail = res.json()["detail"]
    assert detail["trace_ids"] == [odd["uuid"]]
    assert "1 of the 2 matching traces" in detail["error"]
    assert "nothing was created" in detail["error"]
    assert "agent_id" in detail["hint"]
    assert "trace_ids_truncated" not in detail
    assert client.get("/tests", headers=h).json()["total"] == 0

    # Narrowing to one agent is what the hint tells you to do, and it works.
    ok = _convert(client, h, select_all=True, agent_id=agent_id, type="tool_call")
    assert ok.status_code == 200, ok.text
    assert ok.json()["created"] == 1


def test_convert_select_all_truncates_a_long_conflict_list(client, monkeypatch):
    import routers.traces as traces_mod

    h, agent_id = _signup_with_agent(client)
    monkeypatch.setattr(traces_mod, "_MAX_REPORTED_CONFLICTS", 1)
    for _ in range(2):
        _post_trace(
            client, h, _payload(agent_id, _mid(), output={"response": "text only"})
        )

    res = _convert(client, h, select_all=True, type="tool_call")
    assert res.status_code == 400, res.text
    detail = res.json()["detail"]
    assert len(detail["trace_ids"]) == 1
    assert detail["trace_ids_truncated"] is True
    assert "output_type=tool_call" in detail["hint"]


def test_convert_by_ids_keeps_the_plain_conflict_error(client):
    h, agent_id = _signup_with_agent(client)
    odd = _post_trace(
        client, h, _payload(agent_id, _mid(), output={"response": "text only"})
    )

    res = _convert(client, h, trace_ids=[odd["uuid"]], type="tool_call")
    assert res.status_code == 400, res.text
    detail = res.json()["detail"]
    assert set(detail) == {"error", "trace_ids"}
    assert detail["error"] == "Some traces recorded no tool calls to assert"


# ---------------------------------------------------------------------------
# Auto-score ingest (run creation, public contract unchanged)
# ---------------------------------------------------------------------------


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
    body = resp.json()
    return body["uuid"], body["version_uuid"]


def _unlink_all_evaluators(client, h, agent_uuid):
    items = client.get(f"/agents/{agent_uuid}/evaluators", headers=h).json()["items"]
    for ev in items:
        r = client.delete(f"/agents/{agent_uuid}/evaluators/{ev['uuid']}", headers=h)
        assert r.status_code == 200, r.text


def _enable_auto_score(client, h, agent_uuid, evaluator_type="llm"):
    ev_uuid, version_id = _create_clean_evaluator(client, h, evaluator_type)
    _unlink_all_evaluators(client, h, agent_uuid)
    r = client.post(
        f"/agents/{agent_uuid}/evaluators",
        json={"evaluator_ids": [ev_uuid]},
        headers=h,
    )
    assert r.status_code == 200, r.text
    enabled = client.put(
        f"/agents/{agent_uuid}",
        json={"auto_score_traces": True},
        headers=h,
    )
    assert enabled.status_code == 200, enabled.text
    assert enabled.json()["auto_score_traces"] is True
    return ev_uuid, version_id


def _runs_for_trace(trace_uuid: str):
    with db.get_db_connection() as conn:
        return conn.execute(
            "SELECT * FROM trace_evaluations WHERE trace_uuid = ?",
            (trace_uuid,),
        ).fetchall()


def test_ingest_opted_out_creates_no_run_and_keeps_response_contract(client):
    h, agent_id = _signup_with_agent(client)
    mid = _mid()
    body = _post_trace(client, h, _payload(agent_id, mid))

    assert set(body) == {"uuid", "message_id", "conversation_id", "created_at"}
    assert body["message_id"] == mid
    assert body["conversation_id"] == "conv-1"
    assert len(body["uuid"]) == 36
    assert _runs_for_trace(body["uuid"]) == []


def test_ingest_opted_in_conversation_creates_pending_response_run(client):
    h, agent_id = _signup_with_agent(client)
    ev_uuid, version_id = _enable_auto_score(client, h, agent_id)

    body = _post_trace(client, h, _payload(agent_id, _mid()))
    assert set(body) == {"uuid", "message_id", "conversation_id", "created_at"}

    rows = _runs_for_trace(body["uuid"])
    assert len(rows) == 1
    assert rows[0]["status"] == "pending"
    assert rows[0]["criteria"] is not None
    snapshot = json.loads(rows[0]["criteria"])
    assert snapshot == {
        "type": "response",
        "evaluators": [
            {"evaluator_uuid": ev_uuid, "evaluator_version_id": version_id},
        ],
    }


def test_ingest_opted_in_general_creates_pending_general_run(client):
    h, agent_id = _signup_with_general_agent(client)
    ev_uuid, version_id = _enable_auto_score(
        client, h, agent_id, evaluator_type="llm-general"
    )

    body = _post_trace(client, h, _general_payload(agent_id, _mid()))
    assert set(body) == {"uuid", "message_id", "conversation_id", "created_at"}

    rows = _runs_for_trace(body["uuid"])
    assert len(rows) == 1
    snapshot = json.loads(rows[0]["criteria"])
    assert snapshot["type"] == "general"
    assert snapshot["evaluators"] == [
        {"evaluator_uuid": ev_uuid, "evaluator_version_id": version_id},
    ]
    assert rows[0]["status"] == "pending"


def test_ingest_eligibility_drift_creates_skipped_run(client):
    h, agent_id = _signup_with_agent(client)
    ev_uuid, _ = _enable_auto_score(client, h, agent_id)
    r = client.delete(f"/agents/{agent_id}/evaluators/{ev_uuid}", headers=h)
    assert r.status_code == 200, r.text
    assert client.get(f"/agents/{agent_id}", headers=h).json()["auto_score_traces"] is True

    body = _post_trace(client, h, _payload(agent_id, _mid()))
    rows = _runs_for_trace(body["uuid"])
    assert len(rows) == 1
    assert rows[0]["status"] == "skipped"
    assert rows[0]["error"] == "no_usable_evaluators"
    assert rows[0]["criteria"] is None


def test_ingest_opted_in_still_scopes_to_the_caller_org(client):
    h_a, agent_a = _signup_with_agent(client)
    h_b, _agent_b = _signup_with_agent(client)
    _enable_auto_score(client, h_a, agent_a)

    body = _post_trace(client, h_a, _payload(agent_a, _mid()))
    denied = client.get(f"/traces/{body['uuid']}", headers=h_b)
    assert denied.status_code == 403
    assert denied.json()["detail"] == "This resource belongs to a different workspace"
    assert client.get(f"/traces/{body['uuid']}", headers=h_a).status_code == 200
    assert client.get("/traces", headers=h_b).json()["total"] == 0

    agent = db.get_agent(agent_a)
    rows = _runs_for_trace(body["uuid"])
    assert len(rows) == 1
    assert rows[0]["org_uuid"] == agent["org_uuid"]


def test_ingest_opted_in_cap_still_returns_429(client, monkeypatch):
    from routers import traces as traces_mod

    h, agent_id = _signup_with_agent(client)
    _enable_auto_score(client, h, agent_id)
    monkeypatch.setattr(traces_mod, "MAX_TRACES_PER_WORKSPACE", 1)

    first = _post_trace(client, h, _payload(agent_id, _mid()))
    assert _runs_for_trace(first["uuid"])

    capped = client.post("/traces", json=_payload(agent_id, _mid()), headers=h)
    assert capped.status_code == 429
    detail = capped.json()["detail"]
    assert detail["current"] == 1
    assert detail["max_traces"] == 1

    with db.get_db_connection() as conn:
        run_count = conn.execute(
            "SELECT COUNT(*) c FROM trace_evaluations WHERE agent_id = ?",
            (agent_id,),
        ).fetchone()["c"]
    assert run_count == 1


def test_ingest_opted_in_with_api_key_creates_a_run(client):
    h, agent_id = _signup_with_agent(client)
    _enable_auto_score(client, h, agent_id)
    key_headers = _api_key_headers(client, h)

    body = _post_trace(client, key_headers, _payload(agent_id, _mid()))
    assert set(body) == {"uuid", "message_id", "conversation_id", "created_at"}
    assert len(_runs_for_trace(body["uuid"])) == 1
