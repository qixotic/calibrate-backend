"""Unit tests for the traces store (src/traces/)."""

from __future__ import annotations

import uuid

from traces import store


def _org() -> str:
    return str(uuid.uuid4())


def _ingest(org: str, message_id: str, conversation_id: str = "conv-1", **overrides):
    payload = {
        "input": [
            {"role": "system", "content": "You are a vaccination assistant."},
            {"role": "user", "content": "When is the next vaccination?"},
        ],
        "output": {
            "response": "At 14 weeks, for OPV and DPT.",
            "tool_calls": [{"tool": "get_schedule", "arguments": {"child_age_weeks": 14}}],
        },
        "metadata": [{"key": "gen_ai.request.model", "value": "gpt-4"}],
    }
    payload.update(overrides)
    return store.create_trace(
        org_uuid=org,
        message_id=message_id,
        conversation_id=conversation_id,
        **payload,
    )


def test_create_and_get_roundtrip():
    org = _org()
    row, created = _ingest(org, "m-1")
    assert created is True
    assert len(row["uuid"]) == 36
    assert row["message_id"] == "m-1"
    assert row["conversation_id"] == "conv-1"
    assert row["input"][0]["role"] == "system"
    assert row["output"]["tool_calls"][0]["tool"] == "get_schedule"
    assert row["metadata"][0]["key"] == "gen_ai.request.model"
    assert row["created_at"].endswith("Z") and "T" in row["created_at"]

    by_uuid = store.get_trace(org, row["uuid"])
    assert by_uuid is not None and by_uuid["uuid"] == row["uuid"]
    by_mid = store.get_trace_by_message_id(org, "m-1")
    assert by_mid is not None and by_mid["uuid"] == row["uuid"]


def test_create_is_idempotent_on_message_id():
    org = _org()
    first, created_first = _ingest(org, "m-dup")
    second, created_second = _ingest(
        org, "m-dup", output={"response": "different retry body", "tool_calls": None}
    )
    assert created_first is True
    assert created_second is False
    assert second["uuid"] == first["uuid"]
    # The original stored row wins; the retry payload is not an upsert.
    assert second["output"]["response"] == "At 14 weeks, for OPV and DPT."
    assert store.count_live_traces(org) == 1


def test_soft_delete_frees_message_id_for_reingestion():
    org = _org()
    row, _ = _ingest(org, "m-free")
    assert store.soft_delete_traces(org, trace_ids=[row["uuid"]]) == 1
    assert store.get_trace(org, row["uuid"]) is None
    assert store.count_live_traces(org) == 0

    again, created = _ingest(org, "m-free")
    assert created is True
    assert again["uuid"] != row["uuid"]


def test_list_filters_search_and_pagination():
    org = _org()
    _ingest(org, "m-a", conversation_id="conv-a")
    _ingest(
        org,
        "m-b",
        conversation_id="conv-b",
        input=[{"role": "user", "content": "Tell me about POLIO boosters"}],
        output={"response": "Polio boosters are due at 16 months.", "tool_calls": None},
    )
    _ingest(org, "m-c", conversation_id="conv-b")

    rows, total = store.list_traces(org, limit=50, offset=0)
    assert total == 3
    # Newest first: same-second timestamps fall back to id descending.
    assert [r["message_id"] for r in rows] == ["m-c", "m-b", "m-a"]

    page, total = store.list_traces(org, limit=1, offset=1)
    assert total == 3
    assert [r["message_id"] for r in page] == ["m-b"]

    conv, total = store.list_traces(org, limit=50, offset=0, conversation_id="conv-b")
    assert total == 2
    assert {r["message_id"] for r in conv} == {"m-b", "m-c"}

    # q matches message content (case-insensitive) inside the stored JSON.
    hits, total = store.list_traces(org, limit=50, offset=0, q="polio")
    assert total == 1 and hits[0]["message_id"] == "m-b"
    # q matches identifiers too.
    hits, total = store.list_traces(org, limit=50, offset=0, q="m-a")
    assert total == 1 and hits[0]["message_id"] == "m-a"
    # Filters combine.
    hits, total = store.list_traces(
        org, limit=50, offset=0, q="polio", conversation_id="conv-a"
    )
    assert total == 0 and hits == []


def test_search_escapes_like_wildcards():
    org = _org()
    _ingest(
        org,
        "m-pct",
        output={"response": "Coverage reached 100% this quarter", "tool_calls": None},
    )
    _ingest(org, "m-plain")

    hits, total = store.list_traces(org, limit=50, offset=0, q="100%")
    assert total == 1 and hits[0]["message_id"] == "m-pct"
    # A bare % must match only literal percent signs, not act as a wildcard.
    hits, total = store.list_traces(org, limit=50, offset=0, q="%")
    assert total == 1 and hits[0]["message_id"] == "m-pct"


def test_bulk_delete_contract():
    org = _org()
    a, _ = _ingest(org, "m-1", conversation_id="conv-x")
    _ingest(org, "m-2", conversation_id="conv-y")
    _ingest(org, "m-3", conversation_id="conv-y")

    # No ids and no select_all deletes nothing.
    assert store.soft_delete_traces(org) == 0
    # select_all with a filter deletes exactly the matching set.
    assert store.soft_delete_traces(org, select_all=True, conversation_id="conv-y") == 2
    assert store.count_live_traces(org) == 1
    # Explicit ids path; already-deleted rows don't count again.
    assert store.soft_delete_traces(org, trace_ids=[a["uuid"], "not-a-real-uuid"]) == 1
    assert store.count_live_traces(org) == 0


def test_org_isolation():
    org_a, org_b = _org(), _org()
    row_a, _ = _ingest(org_a, "m-shared")
    row_b, created_b = _ingest(org_b, "m-shared")

    # Same message_id in two workspaces is two independent traces.
    assert created_b is True
    assert row_a["uuid"] != row_b["uuid"]

    assert store.get_trace(org_a, row_b["uuid"]) is None
    rows, total = store.list_traces(org_a, limit=50, offset=0)
    assert total == 1 and rows[0]["uuid"] == row_a["uuid"]
    # Deletes never cross workspaces even with explicit foreign ids.
    assert store.soft_delete_traces(org_a, trace_ids=[row_b["uuid"]]) == 0
    assert store.count_live_traces(org_b) == 1
