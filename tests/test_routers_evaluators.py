"""Tests for the evaluators router, focused on the list vs. detail payload shapes.

`GET /evaluators` returns a slimmed `live_version` (only `variables` /
`version_number` / `judge_model` / `uuid`) so bulky `system_prompt` and
`output_config` rubrics don't ship on the list. `GET /evaluators/{uuid}` still
returns the full version history with those fields.
"""

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
            "first_name": "X",
            "last_name": "U",
            "email": f"ev-{suffix}@example.com",
            "password": "passw0rd",
        },
    ).json()
    return {"Authorization": f"Bearer {body['access_token']}"}


def _create_rating_evaluator(client, headers):
    name = f"ev-{uuid.uuid4().hex[:6]}"
    resp = client.post(
        "/evaluators",
        json={
            "name": name,
            "description": "d",
            "evaluator_type": "llm",
            "data_type": "text",
            "kind": "single",
            "output_type": "rating",
            "version": {
                "judge_model": "openai/gpt-4",
                "system_prompt": "Judge {{criteria}} carefully",
                "variables": [{"name": "criteria"}],
                "output_config": {
                    "scale": [
                        {"value": 1, "name": "Bad"},
                        {"value": 2, "name": "Good"},
                    ]
                },
            },
        },
        headers=headers,
    )
    assert resp.status_code == 200, resp.text
    return name, resp.json()["uuid"], resp.json()["version_uuid"]


def test_list_evaluators_live_version_is_slim(client):
    h = _signup(client)
    name, ev_uuid, v_uuid = _create_rating_evaluator(client, h)

    resp = client.get("/evaluators?include_defaults=false", headers=h)
    assert resp.status_code == 200
    items = resp.json()["items"]
    mine = [e for e in items if e["uuid"] == ev_uuid]
    assert len(mine) == 1
    item = mine[0]

    # Top-level fields are unchanged and still present.
    for field in (
        "uuid",
        "name",
        "description",
        "evaluator_type",
        "data_type",
        "kind",
        "output_type",
        "is_default",
        "slug",
        "live_version_id",
        "created_at",
        "updated_at",
    ):
        assert field in item, f"missing top-level field {field}"
    assert item["is_default"] is False

    lv = item["live_version"]
    assert lv is not None
    # The slim summary keeps identity + variables (read by the test dialogs)...
    assert lv["uuid"] == v_uuid
    assert lv["version_number"] == 1
    assert lv["judge_model"] == "openai/gpt-4"
    assert [v["name"] for v in lv["variables"]] == ["criteria"]
    # ...but drops the heavy prompt text and rubric.
    assert "system_prompt" not in lv
    assert "output_config" not in lv


def test_list_evaluators_batched_live_version_matches_detail(client):
    """GET /evaluators resolves every row's live version from one batched query;
    across a multi-evaluator list, each row's live_version must match the id it
    was created with and the value the single-evaluator path resolves — no
    cross-contamination between evaluators."""
    from routers.evaluators import _evaluator_response
    from db import get_all_evaluators

    h = _signup(client)
    created = {}  # ev_uuid -> version_uuid
    for _ in range(3):
        _, ev_uuid, v_uuid = _create_rating_evaluator(client, h)
        created[ev_uuid] = v_uuid

    items = client.get("/evaluators?include_defaults=false", headers=h).json()["items"]
    by_uuid = {e["uuid"]: e for e in items}
    for ev_uuid, v_uuid in created.items():
        assert ev_uuid in by_uuid
        # The batched map picked the right live version for THIS evaluator.
        assert by_uuid[ev_uuid]["live_version"]["uuid"] == v_uuid
        # ...and it equals the un-batched single-evaluator shaping (no map).
        row = next(e for e in get_all_evaluators() if e["uuid"] == ev_uuid)
        assert (
            by_uuid[ev_uuid]["live_version"]
            == _evaluator_response(row).model_dump()["live_version"]
        )


def test_get_evaluator_detail_returns_full_versions(client):
    h = _signup(client)
    name, ev_uuid, v_uuid = _create_rating_evaluator(client, h)

    resp = client.get(f"/evaluators/{ev_uuid}", headers=h)
    assert resp.status_code == 200
    body = resp.json()

    # Detail shape keeps the full version history, not the slim live_version.
    assert "live_version" not in body
    versions = body["versions"]
    assert len(versions) == 1
    ver = versions[0]
    assert ver["uuid"] == v_uuid
    assert ver["system_prompt"] == "Judge {{criteria}} carefully"
    assert ver["output_config"]["scale"][0]["name"] == "Bad"
    assert [v["name"] for v in ver["variables"]] == ["criteria"]
    assert body["live_version_index"] == 0


def test_get_evaluator_detail_default_matches_no_compact(client):
    """Omitting `compact` must be byte-for-byte identical to `?compact=false`."""
    h = _signup(client)
    _, ev_uuid, _ = _create_rating_evaluator(client, h)

    default = client.get(f"/evaluators/{ev_uuid}", headers=h)
    explicit = client.get(f"/evaluators/{ev_uuid}?compact=false", headers=h)
    assert default.status_code == 200
    assert explicit.status_code == 200
    assert default.json() == explicit.json()
    # Heavy per-version fields are populated in the default response.
    ver = default.json()["versions"][0]
    assert ver["system_prompt"] == "Judge {{criteria}} carefully"
    assert ver["output_config"] is not None
    assert ver["variables"] is not None


def test_get_evaluator_detail_compact_nulls_heavy_version_fields(client):
    h = _signup(client)
    _, ev_uuid, v_uuid = _create_rating_evaluator(client, h)

    resp = client.get(f"/evaluators/{ev_uuid}?compact=true", headers=h)
    assert resp.status_code == 200
    body = resp.json()

    # Top-level evaluator identity is untouched.
    assert body["uuid"] == ev_uuid
    assert body["evaluator_type"] == "llm"
    assert body["data_type"] == "text"
    assert body["output_type"] == "rating"
    assert body["live_version_index"] == 0

    versions = body["versions"]
    assert len(versions) == 1
    ver = versions[0]
    # Heavy fields are nulled in place (key kept, value None).
    assert ver["system_prompt"] is None
    assert ver["output_config"] is None
    assert ver["variables"] is None
    # Lightweight version fields survive.
    assert ver["uuid"] == v_uuid
    assert ver["version_number"] == 1
    assert ver["judge_model"] == "openai/gpt-4"
