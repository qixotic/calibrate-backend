"""Integration tests for /agent-tests."""

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
            "first_name": "AT",
            "last_name": "U",
            "email": f"at-{suffix}@example.com",
            "password": "passw0rd",
        },
    ).json()
    return {
        "headers": {"Authorization": f"Bearer {body['access_token']}"},
        "user_uuid": body["user"]["uuid"],
    }


def _create_agent(client, h, name=None):
    return client.post(
        "/agents",
        json={"name": name or f"a-{uuid.uuid4().hex[:6]}", "type": "agent"},
        headers=h,
    ).json()


def _create_test(client, h, name=None):
    evaluators = client.get("/evaluators", headers=h).json()
    llm_ev = next(e for e in evaluators if e.get("evaluator_type") == "llm")
    return client.post(
        "/tests",
        json={
            "name": name or f"t-{uuid.uuid4().hex[:6]}",
            "type": "response",
            "config": {"history": [], "evaluation": {"type": "response"}},
            "evaluators": [{"evaluator_uuid": llm_ev["uuid"]}],
        },
        headers=h,
    ).json()


def _create_simulation_evaluator(client, h, name=None):
    """Create a simulation-type evaluator (no default is seeded for this type)."""
    return client.post(
        "/evaluators",
        json={
            "name": name or f"sim-ev-{uuid.uuid4().hex[:6]}",
            "evaluator_type": "conversation",
            "output_type": "binary",
            "version": {
                "judge_model": "openai/gpt-4.1",
                "system_prompt": "Judge the whole conversation.",
            },
        },
        headers=h,
    ).json()


def _create_conversation_test(client, h, name=None, sim_ev_uuid=None):
    if sim_ev_uuid is None:
        sim_ev_uuid = _create_simulation_evaluator(client, h)["uuid"]
    return client.post(
        "/tests",
        json={
            "name": name or f"conv-{uuid.uuid4().hex[:6]}",
            "type": "conversation",
            "config": {
                "history": [
                    {"role": "user", "content": "hi"},
                    {"role": "assistant", "content": "hello"},
                ],
                "evaluation": {"type": "conversation"},
            },
            "evaluators": [{"evaluator_uuid": sim_ev_uuid}],
        },
        headers=h,
    ).json()


# ---------------------------------------------------------------------------
# Link CRUD
# ---------------------------------------------------------------------------


def test_agent_tests_link_crud(client):
    auth = _signup(client)
    h = auth["headers"]
    agent = _create_agent(client, h)
    test_a = _create_test(client, h)
    test_b = _create_test(client, h)

    link = client.post(
        "/agent-tests",
        json={"agent_uuid": agent["uuid"], "test_uuids": [test_a["uuid"]]},
    )
    assert link.status_code == 200
    # Re-link (idempotent — skip already linked)
    again = client.post(
        "/agent-tests",
        json={"agent_uuid": agent["uuid"], "test_uuids": [test_a["uuid"], test_b["uuid"]]},
    )
    assert again.status_code == 200

    # List
    assert client.get("/agent-tests").status_code == 200
    assert (
        client.get(f"/agent-tests/agent/{agent['uuid']}/tests").status_code == 200
    )
    assert (
        client.get(f"/agent-tests/test/{test_a['uuid']}/agents").status_code == 200
    )
    assert client.get("/agent-tests/test/missing/agents").status_code == 404
    assert client.get("/agent-tests/agent/missing/tests").status_code == 404
    assert client.get("/agent-tests/agent/missing/runs").status_code == 404

    # Runs list (no runs yet)
    runs = client.get(f"/agent-tests/agent/{agent['uuid']}/runs")
    assert runs.status_code == 200
    assert runs.json()["runs"] == []

    # Global runs list (auth required)
    global_runs = client.get("/agent-tests/runs", headers=h)
    assert global_runs.status_code == 200

    # Filtered
    global_runs2 = client.get(
        "/agent-tests/runs", params={"type": "llm-unit-test"}, headers=h
    )
    assert global_runs2.status_code == 200

    # Bulk-unlink validation
    empty = client.post(
        "/agent-tests/bulk-unlink",
        json={"agent_uuid": agent["uuid"], "test_uuids": []},
    )
    assert empty.status_code == 400
    bulk_unlink = client.post(
        "/agent-tests/bulk-unlink",
        json={"agent_uuid": agent["uuid"], "test_uuids": [test_a["uuid"]]},
    )
    assert bulk_unlink.status_code == 200

    # Bulk-unlink missing agent
    missing = client.post(
        "/agent-tests/bulk-unlink",
        json={"agent_uuid": "missing", "test_uuids": [test_b["uuid"]]},
    )
    assert missing.status_code == 404

    # Bulk-delete-tests
    bulk_del = client.post(
        "/agent-tests/bulk-delete-tests",
        json={"agent_uuid": agent["uuid"], "test_uuids": [test_b["uuid"]]},
        headers=h,
    )
    assert bulk_del.status_code == 200

    # Bulk-delete with empty
    empty_del = client.post(
        "/agent-tests/bulk-delete-tests",
        json={"agent_uuid": agent["uuid"], "test_uuids": []},
        headers=h,
    )
    assert empty_del.status_code == 400

    # Bulk-delete with missing agent
    missing_del = client.post(
        "/agent-tests/bulk-delete-tests",
        json={"agent_uuid": "missing", "test_uuids": ["x"]},
        headers=h,
    )
    assert missing_del.status_code == 404

    # Bulk-delete with foreign agent → 404
    other = _signup(client)
    foreign = client.post(
        "/agent-tests/bulk-delete-tests",
        json={"agent_uuid": agent["uuid"], "test_uuids": [test_b["uuid"]]},
        headers=other["headers"],
    )
    assert foreign.status_code == 404


def test_agent_tests_link_with_missing(client):
    auth = _signup(client)
    h = auth["headers"]
    # Missing agent
    resp = client.post(
        "/agent-tests",
        json={"agent_uuid": "missing-agent", "test_uuids": []},
    )
    assert resp.status_code == 404

    agent = _create_agent(client, h)
    # Missing test
    bad = client.post(
        "/agent-tests",
        json={"agent_uuid": agent["uuid"], "test_uuids": ["missing-test"]},
    )
    assert bad.status_code == 404


def test_agent_tests_delete_link_not_found(client):
    auth = _signup(client)
    h = auth["headers"]
    resp = client.request(
        "DELETE",
        "/agent-tests",
        json={"agent_uuid": "x", "test_uuid": "y"},
    )
    assert resp.status_code == 404


# ---------------------------------------------------------------------------
# Run + benchmark validations (queue path, no thread)
# ---------------------------------------------------------------------------


def test_run_agent_test_validation(client, monkeypatch):
    auth = _signup(client)
    h = auth["headers"]

    # Unauthenticated → 403 (HTTPBearer rejects the missing header)
    assert client.post("/agent-tests/agent/missing/run", json={}).status_code == 403

    # Missing agent
    resp = client.post("/agent-tests/agent/missing/run", json={}, headers=h)
    assert resp.status_code == 404

    # Agent with no linked tests
    agent = _create_agent(client, h)
    no_tests = client.post(
        f"/agent-tests/agent/{agent['uuid']}/run", json={}, headers=h
    )
    assert no_tests.status_code == 400

    # Provide bogus test_uuids
    bad = client.post(
        f"/agent-tests/agent/{agent['uuid']}/run",
        json={"test_uuids": ["missing"]},
        headers=h,
    )
    assert bad.status_code == 404

    # Another org's user cannot run tests on this agent → 404 (existence parity)
    other = _signup(client)
    cross = client.post(
        f"/agent-tests/agent/{agent['uuid']}/run",
        json={},
        headers=other["headers"],
    )
    assert cross.status_code == 404


def test_run_agent_test_queued_path(client, monkeypatch):
    auth = _signup(client)
    h = auth["headers"]
    agent = _create_agent(client, h)
    test = _create_test(client, h)
    client.post(
        "/agent-tests",
        json={"agent_uuid": agent["uuid"], "test_uuids": [test["uuid"]]},
    )

    monkeypatch.setenv("S3_OUTPUT_BUCKET", "test-bucket")
    with patch("routers.agent_tests.can_start_agent_test_job", return_value=False), patch(
        "threading.Thread"
    ):
        resp = client.post(
            f"/agent-tests/agent/{agent['uuid']}/run", json={}, headers=h
        )
    assert resp.status_code == 200
    task_id = resp.json()["task_id"]
    assert resp.json()["status"] == "queued"

    # Status requires auth
    assert client.get(f"/agent-tests/run/{task_id}").status_code == 403
    got = client.get(f"/agent-tests/run/{task_id}", headers=h)
    assert got.status_code == 200

    # Another org's user cannot poll this run → 404
    other_poll = _signup(client)
    assert (
        client.get(
            f"/agent-tests/run/{task_id}", headers=other_poll["headers"]
        ).status_code
        == 404
    )

    # 404 unknown run
    assert client.get("/agent-tests/run/missing", headers=h).status_code == 404

    # Visibility toggle
    on = client.patch(
        f"/agent-tests/run/{task_id}/visibility",
        json={"is_public": True},
        headers=h,
    )
    assert on.status_code == 200
    off = client.patch(
        f"/agent-tests/run/{task_id}/visibility",
        json={"is_public": False},
        headers=h,
    )
    assert off.status_code == 200
    other = _signup(client)
    assert (
        client.patch(
            f"/agent-tests/run/{task_id}/visibility",
            json={"is_public": True},
            headers=other["headers"],
        ).status_code
        == 404
    )
    assert (
        client.patch(
            "/agent-tests/run/missing/visibility",
            json={"is_public": True},
            headers=h,
        ).status_code
        == 404
    )

    # Delete
    deleted = client.delete(f"/agent-tests/job/{task_id}", headers=h)
    assert deleted.status_code == 200
    # already gone
    assert (
        client.delete(f"/agent-tests/job/{task_id}", headers=h).status_code == 404
    )
    assert client.delete("/agent-tests/job/missing", headers=h).status_code == 404


def test_run_conversation_test_queued_path(client, monkeypatch):
    auth = _signup(client)
    h = auth["headers"]
    agent = _create_agent(client, h)
    conv = _create_conversation_test(client, h)
    assert conv.get("uuid"), conv
    client.post(
        "/agent-tests",
        json={"agent_uuid": agent["uuid"], "test_uuids": [conv["uuid"]]},
    )

    monkeypatch.setenv("S3_OUTPUT_BUCKET", "test-bucket")
    with patch(
        "routers.agent_tests.can_start_agent_test_job", return_value=False
    ), patch("threading.Thread"):
        resp = client.post(
            f"/agent-tests/agent/{agent['uuid']}/run", json={}, headers=h
        )
    assert resp.status_code == 200, resp.text
    assert resp.json()["status"] == "queued"
    task_id = resp.json()["task_id"]

    # Conversation tests flow through the normal calibrate-llm config: the
    # frozen calibrate_config carries the conversation test case (with its
    # evaluation.type + criteria) and the top-level evaluators list.
    import db

    job = db.get_agent_test_job(task_id)
    details = job["details"]
    cfg = details["calibrate_config"]
    assert cfg.get("evaluators"), cfg
    case = next(c for c in cfg["test_cases"] if c["id"] == conv["uuid"])
    assert case["evaluation"]["type"] == "conversation"
    assert case["evaluation"]["criteria"]
    assert conv["uuid"] in details["evaluators_by_test_id"]

    # Status endpoint serializes fine for a conversation run.
    got = client.get(f"/agent-tests/run/{task_id}", headers=h)
    assert got.status_code == 200

    # Clean up the queued job so it doesn't pollute the shared session DB
    # (try_start_queued_agent_test_job picks the oldest queued job).
    assert client.delete(f"/agent-tests/job/{task_id}", headers=h).status_code == 200


def test_run_mixed_conversation_and_response_allowed(client, monkeypatch):
    """The calibrate CLI dispatches per test case on evaluation.type, so a run
    may mix conversation with response tests in a single job."""
    auth = _signup(client)
    h = auth["headers"]
    agent = _create_agent(client, h)
    response_test = _create_test(client, h)
    conv = _create_conversation_test(client, h)

    monkeypatch.setenv("S3_OUTPUT_BUCKET", "test-bucket")
    with patch(
        "routers.agent_tests.can_start_agent_test_job", return_value=False
    ), patch("threading.Thread"):
        resp = client.post(
            f"/agent-tests/agent/{agent['uuid']}/run",
            json={"test_uuids": [response_test["uuid"], conv["uuid"]]},
            headers=h,
        )
    assert resp.status_code == 200, resp.text
    task_id = resp.json()["task_id"]

    import db

    cfg = db.get_agent_test_job(task_id)["details"]["calibrate_config"]
    types = {c["evaluation"]["type"] for c in cfg["test_cases"]}
    assert types == {"response", "conversation"}

    assert client.delete(f"/agent-tests/job/{task_id}", headers=h).status_code == 200


def test_benchmark_allows_conversation_tests(client, monkeypatch):
    """Conversation rows ignore the benchmarked model, but a benchmark that
    includes them is still accepted (handled by the same calibrate-llm path)."""
    auth = _signup(client)
    h = auth["headers"]
    agent = _create_agent(client, h)
    conv = _create_conversation_test(client, h)
    client.post(
        "/agent-tests",
        json={"agent_uuid": agent["uuid"], "test_uuids": [conv["uuid"]]},
    )

    monkeypatch.setenv("S3_OUTPUT_BUCKET", "test-bucket")
    with patch(
        "routers.agent_tests.can_start_agent_test_job", return_value=False
    ), patch("threading.Thread"):
        resp = client.post(
            f"/agent-tests/agent/{agent['uuid']}/benchmark",
            json={"models": ["openai/gpt-4"]},
            headers=h,
        )
    assert resp.status_code == 200, resp.text
    task_id = resp.json()["task_id"]
    assert client.delete(f"/agent-tests/job/{task_id}", headers=h).status_code == 200


def test_unverified_connection_blocks_all_test_types(client, monkeypatch):
    """Every test type runs the agent (conversation tests are live too), so an
    unverified agent-connection agent blocks response AND conversation runs."""
    import db

    auth = _signup(client)
    h = auth["headers"]
    agent = _create_agent(client, h)
    # Agent-connection agent that hasn't been verified.
    db.update_agent(
        agent["uuid"],
        config={"agent_url": "http://agent.local/run", "connection_verified": False},
    )

    response_test = _create_test(client, h)
    conv = _create_conversation_test(client, h)
    monkeypatch.setenv("S3_OUTPUT_BUCKET", "test-bucket")

    for test_uuid in (response_test["uuid"], conv["uuid"]):
        blocked = client.post(
            f"/agent-tests/agent/{agent['uuid']}/run",
            json={"test_uuids": [test_uuid]},
            headers=h,
        )
        assert blocked.status_code == 400, blocked.text
        assert "not verified" in blocked.json()["detail"].lower()


def test_drifted_config_eval_type_follows_immutable_row_type(client, monkeypatch):
    """The immutable row `type` is authoritative: a conversation-typed test whose
    stored config.evaluation.type has drifted to "response" is normalized back to
    conversation at CLI handoff, so calibrate dispatches it as a conversation."""
    import db

    auth = _signup(client)
    h = auth["headers"]
    # Plain (calibrate-agent-mode) agent — no agent_url, so the connection guard
    # doesn't apply and we can inspect the built config.
    agent = _create_agent(client, h)
    sim_ev = _create_simulation_evaluator(client, h)["uuid"]
    # Schema lets config be arbitrary while row `type` is immutable, so this
    # divergent state is reachable via the API.
    drifted = client.post(
        "/tests",
        json={
            "name": f"mm-{uuid.uuid4().hex[:6]}",
            "type": "conversation",
            "config": {"history": [], "evaluation": {"type": "response"}},
            "evaluators": [{"evaluator_uuid": sim_ev}],
        },
        headers=h,
    ).json()

    monkeypatch.setenv("S3_OUTPUT_BUCKET", "test-bucket")
    with patch(
        "routers.agent_tests.can_start_agent_test_job", return_value=False
    ), patch("threading.Thread"):
        resp = client.post(
            f"/agent-tests/agent/{agent['uuid']}/run",
            json={"test_uuids": [drifted["uuid"]]},
            headers=h,
        )
    assert resp.status_code == 200, resp.text
    task_id = resp.json()["task_id"]
    # The built calibrate config normalizes evaluation.type to the row type.
    cfg = db.get_agent_test_job(task_id)["details"]["calibrate_config"]
    case = next(c for c in cfg["test_cases"] if c["id"] == drifted["uuid"])
    assert case["evaluation"]["type"] == "conversation"
    assert client.delete(f"/agent-tests/job/{task_id}", headers=h).status_code == 200


def test_run_agent_test_missing_s3_config_500(client, monkeypatch):
    auth = _signup(client)
    h = auth["headers"]
    agent = _create_agent(client, h)
    conv = _create_conversation_test(client, h)

    with patch(
        "routers.agent_tests.get_s3_output_config",
        side_effect=ValueError("no bucket configured"),
    ):
        resp = client.post(
            f"/agent-tests/agent/{agent['uuid']}/run",
            json={"test_uuids": [conv["uuid"]]},
            headers=h,
        )
    assert resp.status_code == 500


def test_run_agent_benchmark_validation(client):
    auth = _signup(client)
    h = auth["headers"]
    # Unauthenticated → 403 (HTTPBearer rejects the missing header)
    assert (
        client.post(
            "/agent-tests/agent/missing/benchmark", json={"models": ["x"]}
        ).status_code
        == 403
    )

    # Missing agent
    resp = client.post(
        "/agent-tests/agent/missing/benchmark", json={"models": ["x"]}, headers=h
    )
    assert resp.status_code == 404

    # No models
    agent = _create_agent(client, h)
    bad = client.post(
        f"/agent-tests/agent/{agent['uuid']}/benchmark",
        json={"models": []},
        headers=h,
    )
    assert bad.status_code == 400

    # No tests linked
    no_tests = client.post(
        f"/agent-tests/agent/{agent['uuid']}/benchmark",
        json={"models": ["openai/gpt-4"]},
        headers=h,
    )
    assert no_tests.status_code == 400


def test_run_agent_benchmark_queued_path(client, monkeypatch):
    auth = _signup(client)
    h = auth["headers"]
    agent = _create_agent(client, h)
    test = _create_test(client, h)
    client.post(
        "/agent-tests",
        json={"agent_uuid": agent["uuid"], "test_uuids": [test["uuid"]]},
    )

    monkeypatch.setenv("S3_OUTPUT_BUCKET", "test-bucket")
    with patch("routers.agent_tests.can_start_agent_test_job", return_value=False), patch(
        "threading.Thread"
    ):
        resp = client.post(
            f"/agent-tests/agent/{agent['uuid']}/benchmark",
            json={"models": ["openai/gpt-4"]},
            headers=h,
        )
    assert resp.status_code == 200
    task_id = resp.json()["task_id"]

    # Status requires auth
    assert client.get(f"/agent-tests/benchmark/{task_id}").status_code == 403
    got = client.get(f"/agent-tests/benchmark/{task_id}", headers=h)
    assert got.status_code == 200
    # `evaluators[]` block is now exposed on the benchmark status response
    # the same way it is on the unit-test status — confirm the field is
    # at least present (may be empty for a queued/never-run job).
    assert "evaluators" in got.json()

    # Another org's user cannot poll this benchmark → 404
    other_poll = _signup(client)
    assert (
        client.get(
            f"/agent-tests/benchmark/{task_id}", headers=other_poll["headers"]
        ).status_code
        == 404
    )
    assert (
        client.get("/agent-tests/benchmark/missing", headers=h).status_code == 404
    )

    # Visibility toggle
    on = client.patch(
        f"/agent-tests/benchmark/{task_id}/visibility",
        json={"is_public": True},
        headers=h,
    )
    assert on.status_code == 200
    off = client.patch(
        f"/agent-tests/benchmark/{task_id}/visibility",
        json={"is_public": False},
        headers=h,
    )
    assert off.status_code == 200
    assert (
        client.patch(
            "/agent-tests/benchmark/missing/visibility",
            json={"is_public": True},
            headers=h,
        ).status_code
        == 404
    )


def test_agent_test_inflight(client, monkeypatch):
    """Cover the can_start=True branch where the thread is spawned."""
    auth = _signup(client)
    h = auth["headers"]
    agent = _create_agent(client, h)
    test = _create_test(client, h)
    client.post(
        "/agent-tests",
        json={"agent_uuid": agent["uuid"], "test_uuids": [test["uuid"]]},
    )

    monkeypatch.setenv("S3_OUTPUT_BUCKET", "test-bucket")
    with patch("routers.agent_tests.can_start_agent_test_job", return_value=True), patch(
        "routers.agent_tests.threading.Thread"
    ) as thread_mock:
        resp = client.post(
            f"/agent-tests/agent/{agent['uuid']}/run", json={}, headers=h
        )
        assert resp.status_code == 200
        thread_mock.return_value.start.assert_called_once()
