"""Integration tests for /simulations.

CRUD on simulations, run-flow with queue-only path (threading.Thread mocked),
visibility, abort.
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
            "first_name": "S",
            "last_name": "U",
            "email": f"sim-{suffix}@example.com",
            "password": "passw0rd",
        },
    ).json()
    return {
        "headers": {"Authorization": f"Bearer {body['access_token']}"},
        "user_uuid": body["user"]["uuid"],
    }


def _create_persona(client, h, name=None):
    return client.post(
        "/personas",
        json={"name": name or f"p-{uuid.uuid4().hex[:6]}", "description": "d"},
        headers=h,
    ).json()


def _create_scenario(client, h, name=None):
    return client.post(
        "/scenarios",
        json={"name": name or f"s-{uuid.uuid4().hex[:6]}", "description": "d"},
        headers=h,
    ).json()


def _create_agent(client, h, name=None):
    return client.post(
        "/agents",
        json={"name": name or f"a-{uuid.uuid4().hex[:6]}", "type": "agent"},
        headers=h,
    ).json()


# ---------------------------------------------------------------------------
# CRUD
# ---------------------------------------------------------------------------


def test_create_simulation_with_all_links(client):
    auth = _signup(client)
    h = auth["headers"]
    agent = _create_agent(client, h)
    persona = _create_persona(client, h)
    scenario = _create_scenario(client, h)

    create = client.post(
        "/simulations",
        json={
            "name": f"sim-{uuid.uuid4().hex[:6]}",
            "agent_uuid": agent["uuid"],
            "persona_uuids": [persona["uuid"]],
            "scenario_uuids": [scenario["uuid"]],
        },
        headers=h,
    )
    assert create.status_code == 200
    sim_uuid = create.json()["uuid"]

    # GET detail
    detail = client.get(f"/simulations/{sim_uuid}", headers=h)
    assert detail.status_code == 200
    body = detail.json()
    assert body["agent"]["uuid"] == agent["uuid"]
    assert len(body["personas"]) == 1
    assert len(body["scenarios"]) == 1

    # List
    listing = client.get("/simulations", headers=h)
    assert listing.status_code == 200
    assert any(s["uuid"] == sim_uuid for s in listing.json())

    # Get other-user denied
    other = _signup(client)
    assert (
        client.get(f"/simulations/{sim_uuid}", headers=other["headers"]).status_code
        == 404
    )


def test_list_simulations_batched_agent_hydration(client):
    """GET /simulations hydrates each row's agent from one batched query; a
    multi-item list must carry the same per-row agent summary as the detail
    endpoint, and distinct agents must not cross-contaminate."""
    auth = _signup(client)
    h = auth["headers"]
    agent_a = _create_agent(client, h, name=f"agent-a-{uuid.uuid4().hex[:6]}")
    agent_b = _create_agent(client, h, name=f"agent-b-{uuid.uuid4().hex[:6]}")

    sims = {}
    for agent in (agent_a, agent_b, agent_a):
        created = client.post(
            "/simulations",
            json={"name": f"sim-{uuid.uuid4().hex[:6]}", "agent_uuid": agent["uuid"]},
            headers=h,
        )
        assert created.status_code == 200
        sims[created.json()["uuid"]] = agent["uuid"]

    listing = client.get("/simulations", headers=h).json()
    by_uuid = {s["uuid"]: s for s in listing}
    for sim_uuid, agent_uuid in sims.items():
        assert sim_uuid in by_uuid
        # Each row's agent matches the one it was created with (correct bucketing)
        # and matches the detail endpoint's agent block (identical hydration).
        assert by_uuid[sim_uuid]["agent"]["uuid"] == agent_uuid
        detail = client.get(f"/simulations/{sim_uuid}", headers=h).json()
        assert by_uuid[sim_uuid]["agent"] == {
            k: detail["agent"][k] for k in by_uuid[sim_uuid]["agent"]
        }


def test_create_simulation_unknown_agent_404(client):
    auth = _signup(client)
    resp = client.post(
        "/simulations",
        json={"name": f"sim-{uuid.uuid4().hex[:6]}", "agent_uuid": "00000000-0000-4000-8000-000000000001"},
        headers=auth["headers"],
    )
    assert resp.status_code == 404


def test_create_simulation_unknown_persona_404(client):
    auth = _signup(client)
    resp = client.post(
        "/simulations",
        json={
            "name": f"sim-{uuid.uuid4().hex[:6]}",
            "persona_uuids": ["missing-p"],
        },
        headers=auth["headers"],
    )
    assert resp.status_code == 404


def test_create_simulation_unknown_scenario_404(client):
    auth = _signup(client)
    resp = client.post(
        "/simulations",
        json={
            "name": f"sim-{uuid.uuid4().hex[:6]}",
            "scenario_uuids": ["missing-s"],
        },
        headers=auth["headers"],
    )
    assert resp.status_code == 404


def test_create_simulation_with_invalid_evaluator(client):
    auth = _signup(client)
    resp = client.post(
        "/simulations",
        json={
            "name": f"sim-{uuid.uuid4().hex[:6]}",
            "evaluators": [{"evaluator_uuid": "00000000-0000-4000-8000-000000000001"}],
        },
        headers=auth["headers"],
    )
    assert resp.status_code == 404


def test_create_simulation_rejects_non_conversation_evaluator(client):
    # Simulations only accept `conversation` evaluators; linking an `llm` one → 400.
    auth = _signup(client)
    h = auth["headers"]
    evaluators = client.get("/evaluators", headers=h).json()["items"]
    llm_ev = next(e for e in evaluators if e.get("evaluator_type") == "llm")
    resp = client.post(
        "/simulations",
        json={
            "name": f"sim-{uuid.uuid4().hex[:6]}",
            "evaluators": [{"evaluator_uuid": llm_ev["uuid"]}],
        },
        headers=h,
    )
    assert resp.status_code == 400
    assert "conversation" in resp.json()["detail"]


def test_update_simulation_basic(client):
    auth = _signup(client)
    h = auth["headers"]
    create = client.post(
        "/simulations",
        json={"name": f"sim-{uuid.uuid4().hex[:6]}"},
        headers=h,
    )
    sim_uuid = create.json()["uuid"]

    new_persona = _create_persona(client, h)
    new_scenario = _create_scenario(client, h)
    new_agent = _create_agent(client, h)
    upd = client.put(
        f"/simulations/{sim_uuid}",
        json={
            "name": f"renamed-{uuid.uuid4().hex[:6]}",
            "agent_uuid": new_agent["uuid"],
            "persona_uuids": [new_persona["uuid"]],
            "scenario_uuids": [new_scenario["uuid"]],
        },
        headers=h,
    )
    assert upd.status_code == 200
    assert len(upd.json()["personas"]) == 1

    # Update with unknown agent → 404
    bad = client.put(
        f"/simulations/{sim_uuid}",
        json={"agent_uuid": "00000000-0000-4000-8000-000000000001"},
        headers=h,
    )
    assert bad.status_code == 404

    # Clear agent with empty string
    clear = client.put(
        f"/simulations/{sim_uuid}",
        json={"agent_uuid": ""},
        headers=h,
    )
    assert clear.status_code == 200

    # Update with bogus persona / scenario uuids
    bad_p = client.put(
        f"/simulations/{sim_uuid}", json={"persona_uuids": ["00000000-0000-4000-8000-000000000001"]}, headers=h
    )
    assert bad_p.status_code == 404
    bad_s = client.put(
        f"/simulations/{sim_uuid}", json={"scenario_uuids": ["00000000-0000-4000-8000-000000000001"]}, headers=h
    )
    assert bad_s.status_code == 404
    bad_e = client.put(
        f"/simulations/{sim_uuid}",
        json={"evaluators": [{"evaluator_uuid": "00000000-0000-4000-8000-000000000001"}]},
        headers=h,
    )
    assert bad_e.status_code == 404

    # 404 / 403 paths
    other = _signup(client)
    assert (
        client.put(
            f"/simulations/{sim_uuid}",
            json={"name": "x"},
            headers=other["headers"],
        ).status_code
        == 404
    )
    assert (
        client.put(
            "/simulations/missing", json={"name": "x"}, headers=h
        ).status_code
        == 404
    )


def test_simulation_runs_listing(client):
    auth = _signup(client)
    h = auth["headers"]
    create = client.post(
        "/simulations",
        json={"name": f"sim-{uuid.uuid4().hex[:6]}"},
        headers=h,
    )
    sim_uuid = create.json()["uuid"]

    # No runs yet → empty list
    runs = client.get(f"/simulations/{sim_uuid}/runs", headers=h)
    assert runs.status_code == 200
    assert runs.json()["runs"] == []

    # 404 / 403
    assert client.get("/simulations/missing/runs", headers=h).status_code == 404
    other = _signup(client)
    assert (
        client.get(
            f"/simulations/{sim_uuid}/runs", headers=other["headers"]
        ).status_code
        == 404
    )

    # delete
    assert client.delete(f"/simulations/{sim_uuid}", headers=h).status_code == 200
    # again 404
    assert client.delete(f"/simulations/{sim_uuid}", headers=h).status_code == 404
    # missing
    assert client.delete("/simulations/missing", headers=h).status_code == 404
    # other-user delete
    create2 = client.post(
        "/simulations", json={"name": f"sim-{uuid.uuid4().hex[:6]}"}, headers=h
    ).json()
    assert (
        client.delete(
            f"/simulations/{create2['uuid']}", headers=other["headers"]
        ).status_code
        == 404
    )


# ---------------------------------------------------------------------------
# Run a simulation (queued path, no thread)
# ---------------------------------------------------------------------------


def test_run_simulation_validation_errors(client, monkeypatch):
    auth = _signup(client)
    h = auth["headers"]

    # No simulation
    assert (
        client.post(
            "/simulations/missing/run", json={"type": "text"}, headers=h
        ).status_code
        == 404
    )

    # Simulation has no agent → 400
    create = client.post(
        "/simulations",
        json={"name": f"sim-{uuid.uuid4().hex[:6]}"},
        headers=h,
    ).json()
    no_agent = client.post(
        f"/simulations/{create['uuid']}/run", json={"type": "text"}, headers=h
    )
    assert no_agent.status_code == 400

    # Other-user denied
    other = _signup(client)
    assert (
        client.post(
            f"/simulations/{create['uuid']}/run",
            json={"type": "text"},
            headers=other["headers"],
        ).status_code
        == 404
    )

    # Wire up an agent + verify "no personas" branch
    agent = _create_agent(client, h)
    client.put(
        f"/simulations/{create['uuid']}",
        json={"agent_uuid": agent["uuid"]},
        headers=h,
    )
    monkeypatch.setenv("S3_OUTPUT_BUCKET", "test-bucket")
    no_personas = client.post(
        f"/simulations/{create['uuid']}/run", json={"type": "text"}, headers=h
    )
    assert no_personas.status_code == 400

    # Wire up a persona + verify "no scenarios"
    persona = _create_persona(client, h)
    client.put(
        f"/simulations/{create['uuid']}",
        json={"persona_uuids": [persona["uuid"]]},
        headers=h,
    )
    no_scenarios = client.post(
        f"/simulations/{create['uuid']}/run", json={"type": "text"}, headers=h
    )
    assert no_scenarios.status_code == 400


def test_run_simulation_queued_path(client, monkeypatch):
    auth = _signup(client)
    h = auth["headers"]
    agent = _create_agent(client, h)
    persona = _create_persona(client, h)
    scenario = _create_scenario(client, h)
    create = client.post(
        "/simulations",
        json={
            "name": f"sim-{uuid.uuid4().hex[:6]}",
            "agent_uuid": agent["uuid"],
            "persona_uuids": [persona["uuid"]],
            "scenario_uuids": [scenario["uuid"]],
        },
        headers=h,
    ).json()

    monkeypatch.setenv("S3_OUTPUT_BUCKET", "test-bucket")
    with patch("routers.simulations.can_start_simulation_job", return_value=False), patch(
        "threading.Thread"
    ):
        resp = client.post(
            f"/simulations/{create['uuid']}/run",
            json={"type": "text"},
            headers=h,
        )
    assert resp.status_code == 200
    job_uuid = resp.json()["task_id"]
    assert resp.json()["status"] == "queued"

    # GET run status
    got = client.get(f"/simulations/run/{job_uuid}", headers=h)
    assert got.status_code == 200
    assert got.json()["status"] == "queued"

    # GET unknown run
    assert (
        client.get("/simulations/run/missing", headers=h).status_code == 404
    )

    # Visibility toggle
    on = client.patch(
        f"/simulations/run/{job_uuid}/visibility",
        json={"is_public": True},
        headers=h,
    )
    assert on.status_code == 200
    off = client.patch(
        f"/simulations/run/{job_uuid}/visibility",
        json={"is_public": False},
        headers=h,
    )
    assert off.status_code == 200
    assert (
        client.patch(
            "/simulations/run/missing/visibility",
            json={"is_public": True},
            headers=h,
        ).status_code
        == 404
    )

    # Abort a queued (not in-progress) job — should 400
    aborted = client.post(
        f"/simulations/run/{job_uuid}/abort", headers=h
    )
    assert aborted.status_code == 400

    # Delete run
    deleted = client.delete(f"/simulations/run/{job_uuid}", headers=h)
    assert deleted.status_code == 200
    # Already gone
    assert client.delete(f"/simulations/run/{job_uuid}", headers=h).status_code == 404
    assert client.delete("/simulations/run/missing", headers=h).status_code == 404


def _set_job_timestamps(job_uuid, created_at, updated_at):
    import db

    with db.get_db_connection() as conn:
        conn.execute(
            "UPDATE simulation_jobs SET created_at = ?, updated_at = ? WHERE uuid = ?",
            (created_at, updated_at, job_uuid),
        )
        conn.commit()


def test_simulation_runs_slim_ordering_and_naming(client):
    """Runs list numbers by creation order (Run 1 = oldest) yet returns
    most-recently-updated first, so the two orderings can diverge."""
    import db

    auth = _signup(client)
    h = auth["headers"]
    sim_uuid = client.post(
        "/simulations", json={"name": f"sim-{uuid.uuid4().hex[:6]}"}, headers=h
    ).json()["uuid"]

    # Created oldest→newest: A, B, C. Updated so C is freshest, then A, then B.
    job_a = db.create_simulation_job(sim_uuid, "text", status="done")
    job_b = db.create_simulation_job(sim_uuid, "voice", status="failed")
    job_c = db.create_simulation_job(sim_uuid, "text", status="in_progress")
    _set_job_timestamps(job_a, "2024-01-01 00:00:00", "2024-01-02 00:00:00")
    _set_job_timestamps(job_b, "2024-01-01 00:00:01", "2024-01-01 12:00:00")
    _set_job_timestamps(job_c, "2024-01-01 00:00:02", "2024-01-03 00:00:00")

    resp = client.get(f"/simulations/{sim_uuid}/runs", headers=h)
    assert resp.status_code == 200
    runs = resp.json()["runs"]
    assert len(runs) == 3

    by_uuid = {r["uuid"]: r for r in runs}
    # Names follow creation order regardless of position in the list.
    assert by_uuid[job_a]["name"] == "Run 1"
    assert by_uuid[job_b]["name"] == "Run 2"
    assert by_uuid[job_c]["name"] == "Run 3"

    # List order is most-recently-updated first: C, A, B.
    assert [r["uuid"] for r in runs] == [job_c, job_a, job_b]
    assert [r["name"] for r in runs] == ["Run 3", "Run 1", "Run 2"]

    assert by_uuid[job_a]["status"] == "done"
    assert by_uuid[job_a]["type"] == "text"
    assert by_uuid[job_b]["status"] == "failed"
    assert by_uuid[job_b]["type"] == "voice"
    assert by_uuid[job_c]["status"] == "in_progress"
    assert by_uuid[job_c]["updated_at"] == "2024-01-03 00:00:00"

    # Slim contract: exactly these keys, no heavy blobs.
    for r in runs:
        assert set(r.keys()) == {"uuid", "name", "status", "type", "updated_at"}
        assert "results" not in r
        assert "details" not in r


def test_simulation_runs_does_not_ship_heavy_blobs(client):
    """The slim summary path must never serialize a job's transcript results
    or config details into the list response."""
    import db

    auth = _signup(client)
    h = auth["headers"]
    sim_uuid = client.post(
        "/simulations", json={"name": f"sim-{uuid.uuid4().hex[:6]}"}, headers=h
    ).json()["uuid"]

    transcript_sentinel = "HEAVY_TRANSCRIPT_SENTINEL_" + uuid.uuid4().hex
    config_sentinel = "HEAVY_CONFIG_SENTINEL_" + uuid.uuid4().hex
    heavy_results = {
        "conversation": [
            {"role": "user", "content": transcript_sentinel} for _ in range(50)
        ],
        "judge_output": {"reasoning": transcript_sentinel},
    }
    heavy_details = {
        "config": {"system_prompt": config_sentinel, "pid": 1234},
        "blob": [config_sentinel] * 50,
    }
    for _ in range(3):
        db.create_simulation_job(
            sim_uuid, "text", status="done",
            details=heavy_details, results=heavy_results,
        )

    resp = client.get(f"/simulations/{sim_uuid}/runs", headers=h)
    assert resp.status_code == 200
    body = resp.text
    assert transcript_sentinel not in body
    assert config_sentinel not in body
    assert len(resp.json()["runs"]) == 3


def test_run_simulation_voice_connection_blocked(client, monkeypatch):
    """An agent_url-mode (connection) agent cannot run voice sims."""
    auth = _signup(client)
    h = auth["headers"]
    # Create connection-mode agent with verified flag and agent_url
    agent = client.post(
        "/agents",
        json={
            "name": f"conn-{uuid.uuid4().hex[:6]}",
            "type": "connection",
            "config": {
                "agent_url": "https://example.com/agent",
                "connection_verified": True,
            },
        },
        headers=h,
    ).json()
    persona = _create_persona(client, h)
    scenario = _create_scenario(client, h)
    create = client.post(
        "/simulations",
        json={
            "name": f"sim-{uuid.uuid4().hex[:6]}",
            "agent_uuid": agent["uuid"],
            "persona_uuids": [persona["uuid"]],
            "scenario_uuids": [scenario["uuid"]],
        },
        headers=h,
    ).json()

    monkeypatch.setenv("S3_OUTPUT_BUCKET", "test-bucket")
    resp = client.post(
        f"/simulations/{create['uuid']}/run",
        json={"type": "voice"},
        headers=h,
    )
    assert resp.status_code == 400
