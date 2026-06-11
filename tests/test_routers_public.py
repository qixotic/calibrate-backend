"""Integration tests for /public routes.

The /public routes do not require auth. They resolve share_tokens against
the various job tables. We seed jobs directly via db.* and toggle visibility
via update_job_visibility / update_agent_test_job_visibility / etc.
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


def _user(client):
    """Sign up a fresh user and return `(user_uuid, org_uuid)`.

    Most tests don't care about distinguishing the two, but the multi-tenant
    DB layer requires `org_uuid` on every `create_*` call. Returning both keeps
    each test self-contained.
    """
    import db as _db

    suffix = uuid.uuid4().hex[:8]
    body = client.post(
        "/auth/signup",
        json={
            "first_name": "P",
            "last_name": "U",
            "email": f"pub-{suffix}@example.com",
            "password": "passw0rd",
        },
    ).json()
    user_uuid = body["user"]["uuid"]
    org = _db.get_personal_org_for_user(user_uuid)
    return {"user_uuid": user_uuid, "org_uuid": org["uuid"]}


# ---------------------------------------------------------------------------
# Defaults / share token gate
# ---------------------------------------------------------------------------


def test_public_evaluators_defaults_requires_valid_token(client):
    resp = client.get(
        "/public/evaluators/defaults", params={"share_token": "missing"}
    )
    assert resp.status_code == 404


def test_public_evaluators_defaults_token_validation(client):
    """When the token is valid we should get the seed list. Use an STT job we
    just made public."""
    import db as db_mod

    auth = _user(client)
    user_id = auth["user_uuid"]
    org_uuid = auth["org_uuid"]
    job_uuid = db_mod.create_job(
        job_type="stt-eval",
        org_uuid=org_uuid,
        user_id=user_id,
        status="done",
        details={"providers": ["openai"], "language": "en"},
        results={"provider_results": []},
    )
    token = uuid.uuid4().hex
    db_mod.update_job_visibility(job_uuid, True, token)

    resp = client.get(
        "/public/evaluators/defaults", params={"share_token": token}
    )
    assert resp.status_code == 200
    body = resp.json()
    assert isinstance(body, list) and body

    # Filtered by types
    filtered = client.get(
        "/public/evaluators/defaults",
        params={"share_token": token, "types": "stt,llm"},
    )
    assert filtered.status_code == 200

    # `llm-general` is an accepted filter value and returns the seeded
    # default-llm-general evaluator.
    general = client.get(
        "/public/evaluators/defaults",
        params={"share_token": token, "types": "llm-general"},
    )
    assert general.status_code == 200
    general_body = general.json()
    assert general_body
    assert all(e["evaluator_type"] == "llm-general" for e in general_body)
    assert any(e["name"] == "Output correctness" for e in general_body)

    # Invalid types value → 400
    bad = client.get(
        "/public/evaluators/defaults",
        params={"share_token": token, "types": "bogus"},
    )
    assert bad.status_code == 400

    # Empty types value passes (returns full list)
    empty = client.get(
        "/public/evaluators/defaults",
        params={"share_token": token, "types": ""},
    )
    assert empty.status_code == 200


# ---------------------------------------------------------------------------
# Public STT / TTS / annotation-eval / sim / agent-test / benchmark — token unknown
# ---------------------------------------------------------------------------


def test_public_unknown_token_404(client):
    for path in [
        "/public/stt/none",
        "/public/tts/none",
        "/public/test-run/none",
        "/public/benchmark/none",
        "/public/simulation-run/none",
        "/public/annotation-eval/none",
        "/public/annotation-jobs/none",
        "/public/annotation-jobs/view/none",
    ]:
        r = client.get(path)
        assert r.status_code == 404, path


# ---------------------------------------------------------------------------
# Public STT / TTS with valid token
# ---------------------------------------------------------------------------


def test_public_stt_valid_token(client):
    import db as db_mod

    auth = _user(client)
    user_id = auth["user_uuid"]
    org_uuid = auth["org_uuid"]
    job_uuid = db_mod.create_job(
        job_type="stt-eval",
        org_uuid=org_uuid,
        user_id=user_id,
        status="done",
        details={
            "providers": ["openai"],
            "language": "en",
            "audio_paths": [],
        },
        results={"provider_results": [], "leaderboard_summary": None},
    )
    token = uuid.uuid4().hex
    db_mod.update_job_visibility(job_uuid, True, token)
    resp = client.get(f"/public/stt/{token}")
    assert resp.status_code == 200
    body = resp.json()
    assert body["task_id"] == job_uuid


def test_public_tts_valid_token(client):
    import db as db_mod

    auth = _user(client)
    user_id = auth["user_uuid"]
    org_uuid = auth["org_uuid"]
    job_uuid = db_mod.create_job(
        job_type="tts-eval",
        org_uuid=org_uuid,
        user_id=user_id,
        status="done",
        details={"providers": ["openai"], "language": "en"},
        results={"provider_results": [], "leaderboard_summary": None},
    )
    token = uuid.uuid4().hex
    db_mod.update_job_visibility(job_uuid, True, token)
    resp = client.get(f"/public/tts/{token}")
    assert resp.status_code == 200


def test_public_test_run_valid_token(client):
    import db as db_mod

    auth = _user(client)
    user_id = auth["user_uuid"]
    org_uuid = auth["org_uuid"]
    agent_uuid = db_mod.create_agent(
        name=f"a-{uuid.uuid4().hex[:6]}", org_uuid=org_uuid, user_id=user_id
    )
    job_uuid = db_mod.create_agent_test_job(
        agent_id=agent_uuid, job_type="llm-unit-test", status="done"
    )
    token = uuid.uuid4().hex
    db_mod.update_agent_test_job_visibility(job_uuid, True, token)
    resp = client.get(f"/public/test-run/{token}")
    assert resp.status_code == 200


def test_public_benchmark_valid_token(client):
    import db as db_mod

    auth = _user(client)
    user_id = auth["user_uuid"]
    org_uuid = auth["org_uuid"]
    agent_uuid = db_mod.create_agent(
        name=f"a-{uuid.uuid4().hex[:6]}", org_uuid=org_uuid, user_id=user_id
    )
    job_uuid = db_mod.create_agent_test_job(
        agent_id=agent_uuid, job_type="llm-benchmark", status="done"
    )
    token = uuid.uuid4().hex
    db_mod.update_agent_test_job_visibility(job_uuid, True, token)
    resp = client.get(f"/public/benchmark/{token}")
    assert resp.status_code == 200


def test_public_simulation_run_valid_token(client):
    import db as db_mod

    auth = _user(client)
    user_id = auth["user_uuid"]
    org_uuid = auth["org_uuid"]
    sim_uuid = db_mod.create_simulation(
        name=f"sim-{uuid.uuid4().hex[:6]}", org_uuid=org_uuid, user_id=user_id
    )
    job_uuid = db_mod.create_simulation_job(
        simulation_id=sim_uuid, job_type="text", status="done"
    )
    token = uuid.uuid4().hex
    db_mod.update_simulation_job_visibility(job_uuid, True, token)
    resp = client.get(f"/public/simulation-run/{token}")
    assert resp.status_code == 200


def test_public_annotation_eval_must_be_done(client):
    """The endpoint refuses to serve in-progress or failed annotation-eval jobs."""
    import db as db_mod
    from annotation_eval_runner import ANNOTATION_EVAL_JOB_TYPE

    auth = _user(client)
    user_id = auth["user_uuid"]
    org_uuid = auth["org_uuid"]
    # Create a task to host the job
    task_uuid = db_mod.create_annotation_task(
        name=f"t-{uuid.uuid4().hex[:6]}",
        type="llm",
        org_uuid=org_uuid,
        user_id=user_id,
    )
    job_uuid = db_mod.create_job(
        job_type=ANNOTATION_EVAL_JOB_TYPE,
        org_uuid=org_uuid,
        user_id=user_id,
        status="in_progress",
        details={"task_id": task_uuid, "evaluators": []},
    )
    token = uuid.uuid4().hex
    db_mod.update_job_visibility(job_uuid, True, token)
    # Not done → 404
    resp = client.get(f"/public/annotation-eval/{token}")
    assert resp.status_code == 404

    # Mark done → 200
    db_mod.update_job(job_uuid, status="done")
    resp = client.get(f"/public/annotation-eval/{token}")
    assert resp.status_code == 200
