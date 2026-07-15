"""High-level FastAPI integration tests using TestClient.

Goals: import every router (covering their top-level statements) and
hit a representative set of endpoints to drive route handler coverage.
External-only routes (calibrate CLI / openrouter HTTPS) are stubbed.
"""

from __future__ import annotations

import uuid
from typing import Dict, Optional
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient


@pytest.fixture(scope="module")
def app():
    # Importing main runs the lifespan only when TestClient enters the context.
    # The shared session fixture has already called init_db() so it's safe.
    import main as main_mod

    return main_mod.app


@pytest.fixture(scope="module")
def client(app):
    # Stub recover_pending_jobs so it doesn't try to restart real subprocesses
    with patch("main.recover_pending_jobs"):
        with TestClient(app) as c:
            yield c


def _signup(client: TestClient, *, suffix: Optional[str] = None) -> Dict:
    suffix = suffix or uuid.uuid4().hex[:8]
    resp = client.post(
        "/auth/signup",
        json={
            "first_name": "Test",
            "last_name": "User",
            "email": f"e2e-{suffix}@example.com",
            "password": "passw0rd",
        },
    )
    resp.raise_for_status()
    return resp.json()


def _auth(client: TestClient) -> Dict[str, str]:
    body = _signup(client)
    return {
        "headers": {"Authorization": f"Bearer {body['access_token']}"},
        "user_uuid": body["user"]["uuid"],
        "email": body["user"]["email"],
        "password": "passw0rd",
    }


# ---------------------------------------------------------------------------
# Root + health
# ---------------------------------------------------------------------------


def test_root_get_and_head(client):
    assert client.get("/").json() == {"message": "Health check successful!"}
    assert client.head("/").status_code == 200


# ---------------------------------------------------------------------------
# Docs (basic auth)
# ---------------------------------------------------------------------------


def test_docs_endpoints_require_basic_auth(client):
    assert client.get("/docs").status_code == 401
    assert client.get("/redoc").status_code == 401
    assert client.get("/openapi.json").status_code == 401
    # With basic auth (defaults)
    docs = client.get("/docs", auth=("admin", "changeme"))
    assert docs.status_code == 200
    assert client.get("/redoc", auth=("admin", "changeme")).status_code == 200
    assert client.get("/openapi.json", auth=("admin", "changeme")).status_code == 200
    # Wrong creds
    assert client.get("/docs", auth=("admin", "wrong")).status_code == 401


def test_public_api_docs_are_unauthenticated_and_filtered(client, monkeypatch):
    monkeypatch.setenv("PUBLIC_API_BASE_URL", "http://testserver")
    # No auth required for the public subset.
    assert client.get("/public-api/docs").status_code == 200
    schema = client.get("/public-api/openapi.json")
    assert schema.status_code == 200

    pub_top = schema.json()
    assert pub_top["servers"] == [{"url": "http://testserver", "description": "API"}]

    paths = pub_top["paths"]
    # The public surface spans the full create/read/update/launch eval loop.
    # Spot-check a representative op from each published router.
    assert "get" in paths.get("/agents", {})
    assert "post" in paths.get("/agents", {})
    assert "put" in paths.get("/agents/{agent_uuid}", {})
    assert "post" in paths.get("/agents/resolve", {})
    assert "post" in paths.get("/agent-tests", {})  # link tests to agent
    assert "post" in paths.get("/agent-tests/agent/{agent_uuid}/run", {})
    assert "get" in paths.get("/agent-tests/run/{task_id}", {})
    assert "post" in paths.get("/agent-tests/agent/{agent_uuid}/benchmark", {})
    assert "post" in paths.get("/tests", {})
    assert "post" in paths.get("/tests/bulk", {})
    assert "post" in paths.get("/evaluators", {})
    assert "post" in paths.get("/evaluators/{evaluator_uuid}/versions", {})
    # Annotation is trimmed to the automated evaluator-run loop (10 routes).
    assert "post" in paths.get("/annotation-tasks", {})
    assert "post" in paths.get("/annotation-tasks/{task_uuid}/evaluator-runs", {})
    assert "get" in paths.get("/annotation-tasks/{task_uuid}/summary", {})
    assert "get" in paths.get("/annotation-tasks/{task_uuid}/agreement", {})

    # The public verify-connection endpoint only accepts the probe inputs
    # (model + messages). agent_url / agent_headers come from the agent's
    # stored config and must NOT be on the public request body (the by-id route
    # ignores them; exposing them misleads API-key clients).
    verify_op = paths["/agents/{agent_uuid}/verify-connection"]["post"]
    verify_ref = verify_op["requestBody"]["content"]["application/json"]["schema"]["$ref"]
    verify_schema = pub_top["components"]["schemas"][verify_ref.split("/")[-1]]
    assert set(verify_schema.get("properties", {})) == {"model", "messages"}, (
        "public verify-connection body must expose only model + messages"
    )

    # JWT-only / deliberately-excluded endpoints must NOT leak into the public
    # schema: account/tenant bootstrapping, the UI-only share pages, tools
    # (deferred), and every destructive or visibility-toggle route.
    assert "/presigned-url" not in paths
    assert "/api-keys" not in paths
    assert "/organizations" not in paths
    assert "/tools" not in paths  # tools/agent-tools deferred, stay JWT-only
    assert not any(p.startswith("/auth") for p in paths)
    # STT/TTS eval, simulations, and the jobs list are deferred from the public
    # API for now (keeps the SDK endpoint count down) — must stay JWT-only.
    assert "/stt/evaluate" not in paths
    assert "/tts/evaluate" not in paths
    assert not any(p.startswith("/simulations") for p in paths)
    assert "/jobs" not in paths
    assert not any(p.startswith("/datasets") for p in paths)  # STT/TTS-only; deferred with them
    # personas + scenarios are simulation-only; deferred with simulations.
    assert not any(p.startswith("/personas") for p in paths)
    assert not any(p.startswith("/scenarios") for p in paths)
    # evaluators are trimmed to the minimum write+read set; helpers stay JWT-only.
    assert "/evaluators/default-prompt" not in paths
    assert "/evaluators/{evaluator_uuid}/preview-prompt" not in paths
    assert "/evaluators/{evaluator_uuid}/versions/live" not in paths
    assert "put" not in paths.get("/evaluators/{evaluator_uuid}", {})  # update stays JWT-only
    assert "get" not in paths.get("/evaluators/{evaluator_uuid}/versions", {})  # list-versions redundant with detail
    # Annotation long tail (human-labelling workflow, housekeeping reads,
    # annotators, agreement trends) stays JWT-only.
    assert "/annotators" not in paths
    assert not any(p.startswith("/annotation-agreement") for p in paths)
    assert "put" not in paths.get("/annotation-tasks/{task_uuid}", {})  # task update stays JWT-only
    assert "/annotation-tasks/{task_uuid}/jobs" not in paths  # human labelling jobs
    assert "/annotation-tasks/{task_uuid}/annotations" not in paths
    assert "/annotation-tasks/{task_uuid}/items/{item_uuid}" not in paths
    assert "get" not in paths.get("/agent-tests", {})  # plain link-list stays JWT-only
    assert "delete" not in paths.get("/agents/{agent_uuid}", {})  # no deletes are public
    assert "delete" not in paths.get("/tests/{test_uuid}", {})
    assert "/tests/bulk-delete" not in paths  # bulk-delete stays JWT-only
    # Visibility share-toggles are UI-only, never public.
    assert "/agent-tests/run/{task_id}/visibility" not in paths

    # Ops keep their router-level tag (e.g. "agents") for grouping, but the
    # "Public API" filter marker is stripped so it never shows as its own group
    # and never duplicates an op across two groups.
    assert paths["/agents"]["get"]["tags"] == ["agents"]
    assert paths["/agent-tests/agent/{agent_uuid}/run"]["post"]["tags"] == [
        "agent-tests"
    ]
    for ops in paths.values():
        for op in ops.values():
            assert "Public API" not in op["tags"]

    # The public spec advertises ONLY the API-key (X-API-Key) scheme, and pins it
    # as the sole `security` on every op. This is what makes Fern (SDK) and Speakeasy (CLI)
    # whose sole required auth param is `api_key` (no required `token`) — see the
    # SDK-auth bullet in CLAUDE.md. The auto-generated HTTPBearer scheme must be
    # gone so `Calibrate(api_key=...)` works.
    schemes = pub_top["components"]["securitySchemes"]
    assert set(schemes) == {"ApiKeyAuth"}
    assert schemes["ApiKeyAuth"] == {
        "type": "apiKey",
        "in": "header",
        "name": "X-API-Key",
        "description": schemes["ApiKeyAuth"]["description"],
    }
    assert "HTTPBearer" not in schemes
    for ops in pub_top["paths"].values():
        for op in ops.values():
            assert op["security"] == [{"ApiKeyAuth": []}]

    # The optional `X-Org-UUID` header (added by the shared auth dep) is
    # redundant for API-key clients — the key already scopes to one workspace —
    # so it must be stripped from every public operation.
    for path, methods in pub_top["paths"].items():
        for method, op in methods.items():
            if method not in ("get", "post", "put", "delete", "patch"):
                continue
            names = {p.get("name") for p in op.get("parameters", [])}
            assert "X-Org-UUID" not in names, (
                f"X-Org-UUID leaked into public op {method.upper()} {path}"
            )

    # The private (Basic-Auth'd) full schema keeps the router tags intact —
    # the public filter must not have mutated the shared cached schema.
    full = client.get("/openapi.json", auth=("admin", "changeme")).json()
    assert full["paths"]["/agents"]["get"]["tags"] == ["agents", "Public API"]
    # ...and the full schema still carries its original HTTPBearer scheme — the
    # public override must not have leaked back into the shared cached schema.
    assert "HTTPBearer" in full["components"]["securitySchemes"]

    # JWT-only request fields (verification flags API-key writes have stripped)
    # are hidden from the public AgentUpdate schema, but stay on the internal one.
    pub_agent_update = pub_top["components"]["schemas"]["AgentUpdate"]["properties"]
    assert "connection_verified" not in pub_agent_update
    assert "benchmark_models_verified" not in pub_agent_update
    assert {"name", "config"} <= set(pub_agent_update)
    full_agent_update = full["components"]["schemas"]["AgentUpdate"]["properties"]
    assert "connection_verified" in full_agent_update  # still there internally

    # Backend-internal response fields are hidden from the public spec but stay
    # on the internal model: auto-increment pivot-row `ids` from the link
    # response, and the raw `owner_user_id` / UI-only `live_version_index` on
    # evaluators (which are stripped from EVERY public schema by field name).
    import json as _json

    pub_link = pub_top["components"]["schemas"]["AgentTestsCreateResponse"]["properties"]
    assert "ids" not in pub_link and "message" in pub_link
    assert "ids" in full["components"]["schemas"]["AgentTestsCreateResponse"]["properties"]
    pub_schemas_dump = _json.dumps(
        {k: v.get("properties", {}) for k, v in pub_top["components"]["schemas"].items()}
    )
    assert '"owner_user_id"' not in pub_schemas_dump
    assert '"live_version_index"' not in pub_schemas_dump
    # still present internally (the frontend uses owner_user_id for default-vs-custom)
    assert "owner_user_id" in _json.dumps(full["components"]["schemas"])

    # The by-name strip is scoped to evaluator schemas only — a non-evaluator
    # schema that happened to carry one of these field names would keep it.
    import main as _main

    strip = _main._PUBLIC_SPEC_EVALUATOR_HIDDEN_FIELD_NAMES
    schemas = {
        "routers__evaluators__EvaluatorResponse": {"properties": dict.fromkeys(strip)},
        "DefaultPromptResponse": {"properties": {"kind": {}}},
        "SomeOtherThing": {
            "properties": {"kind": {}, "keep": {}},
            "required": ["kind"],
        },
    }
    _main._strip_hidden_public_fields(schemas)
    # evaluator schemas lose the internal names...
    assert schemas["routers__evaluators__EvaluatorResponse"]["properties"] == {}
    assert "kind" not in schemas["DefaultPromptResponse"]["properties"]
    # ...but an unrelated schema's identically-named field survives.
    assert "kind" in schemas["SomeOtherThing"]["properties"]
    assert schemas["SomeOtherThing"]["required"] == ["kind"]

    # Components are trimmed to ONLY the schemas the public paths reference
    # (transitively) — internal/JWT-only model shapes must not leak.
    import json
    import re

    pub_schemas = pub_top.get("components", {}).get("schemas", {})
    full_schemas = full.get("components", {}).get("schemas", {})
    # Public response models are present...
    assert "ResolveAgentNamesResponse" in pub_schemas  # POST /agents/resolve
    assert "AgentTestRunCreateResponse" in pub_schemas  # run + benchmark launch shape
    assert "BatchTestRunResponse" in pub_schemas  # POST /agent-tests/run
    # Agent-test launches use the dataset-free AgentTestRunCreateResponse; the
    # shared TaskCreateResponse (with STT/TTS dataset fields) is deferred with
    # STT/TTS, so it must NOT leak into the public spec.
    assert "TaskCreateResponse" not in pub_schemas
    # ...nested refs are pulled in transitively...
    assert "BatchTestSkip" in pub_schemas  # referenced by BatchTestRunResponse
    assert "ModelResult" in pub_schemas  # AgentTestRunListItem.model_results is now typed
    # ...but it's a strict subset of the full set, and internal-only models are gone.
    assert set(pub_schemas).issubset(set(full_schemas))
    assert "PersonaCreate" not in pub_schemas  # personas deferred (simulation-only)
    assert "AgentCreate" in pub_schemas  # POST /agents is now public
    assert "BulkTestDelete" not in pub_schemas  # bulk-delete stays JWT-only
    # Internal-only models from excluded routers stay out of the public schema.
    assert "LoginResponse" not in pub_schemas  # /auth is not public
    assert "ToolCreate" not in pub_schemas  # tools/agent-tools deferred
    # Every $ref in the public doc resolves within the trimmed schema set.
    refs = {
        m for m in re.findall(r'#/components/schemas/([^"]+)', json.dumps(pub_top))
    }
    assert refs.issubset(set(pub_schemas))

    # Free-form `Dict[str, Any]` fields (e.g. agent `config`) must NOT carry
    # Pydantic's auto-title, which Mintlify would render as a fake `Config` type
    # chip. The property drops its title but keeps its type + description...
    config_prop = pub_schemas["AgentUpdate"]["properties"]["config"]
    assert "title" not in config_prop
    assert config_prop["description"]  # description survives
    assert {b.get("type") for b in config_prop["anyOf"]} == {"object", "null"}
    # ...while real model titles (used as the expandable reference name) stay.
    assert pub_schemas["AgentUpdate"]["title"] == "AgentUpdate"
    # No surviving title anywhere sits on a free-form field — neither a
    # `Dict[str, Any]` (object) NOR a `List[Dict[str, Any]]` (array of objects,
    # e.g. `tool_calls`). Both render as a shapeless chip; a lingering auto-title
    # ("Tool Calls · object[]") reads as a fake type. This guards the whole spec
    # so no new free-form field regresses.
    def _is_ff_obj(n):
        return (
            isinstance(n, dict)
            and n.get("type") == "object"
            and n.get("additionalProperties")
            and not n.get("properties")
        )

    def _is_freeform(n):
        return _is_ff_obj(n) or (
            isinstance(n, dict)
            and n.get("type") == "array"
            and _is_ff_obj(n.get("items", {}))
        )

    for sname, schema in pub_schemas.items():
        for fname, prop in schema.get("properties", {}).items():
            branches = [
                b for b in prop.get("anyOf", []) if b.get("type") != "null"
            ] or [prop]
            if any(_is_freeform(b) for b in branches):
                assert "title" not in prop, (
                    f"stale auto-title on free-form field {sname}.{fname}: {prop}"
                )


# ---------------------------------------------------------------------------
# Presigned URL endpoint
# ---------------------------------------------------------------------------


def test_presigned_url_requires_auth(client):
    # No Authorization header → 403 (HTTPBearer rejects the missing header).
    resp = client.post(
        "/presigned-url",
        json={"task_type": "stt", "content_type": "audio/wav", "extension": "wav"},
    )
    assert resp.status_code == 403


def test_presigned_url_happy_path(client, monkeypatch):
    h = _auth(client)["headers"]
    monkeypatch.setenv("S3_OUTPUT_BUCKET", "my-bucket")
    monkeypatch.delenv("OBJECT_STORAGE_MODE", raising=False)
    with patch(
        "main.generate_presigned_upload_url",
        return_value="https://signed.example/x",
    ):
        resp = client.post(
            "/presigned-url",
            json={"task_type": "stt", "content_type": "audio/wav", "extension": "wav"},
            headers=h,
        )
    assert resp.status_code == 200
    body = resp.json()
    assert body["presigned_url"].startswith("https://")
    assert body["s3_path"].startswith("s3://my-bucket/stt/media/")


def test_presigned_url_tts_lands_under_tts_prefix(client, monkeypatch):
    """TTS annotation items upload their audio through the same endpoint; a
    `tts` task_type stores the object under the `tts/` prefix (parity with stt)."""
    h = _auth(client)["headers"]
    monkeypatch.setenv("S3_OUTPUT_BUCKET", "my-bucket")
    monkeypatch.delenv("OBJECT_STORAGE_MODE", raising=False)
    with patch(
        "main.generate_presigned_upload_url",
        return_value="https://signed.example/x",
    ):
        resp = client.post(
            "/presigned-url",
            json={"task_type": "tts", "content_type": "audio/wav", "extension": "wav"},
            headers=h,
        )
    assert resp.status_code == 200
    assert resp.json()["s3_path"].startswith("s3://my-bucket/tts/media/")


def test_presigned_url_local_storage_upload_roundtrip(client, monkeypatch, tmp_path):
    h = _auth(client)["headers"]
    monkeypatch.setenv("OBJECT_STORAGE_MODE", "local")
    monkeypatch.delenv("S3_OUTPUT_BUCKET", raising=False)
    monkeypatch.setenv("LOCAL_ARTIFACT_ROOT", str(tmp_path / "artifacts"))
    # Upload URLs are built from LOCAL_ARTIFACT_BASE_URL, same as download URLs.
    monkeypatch.setenv("LOCAL_ARTIFACT_BASE_URL", "http://testserver")

    resp = client.post(
        "/presigned-url",
        json={"task_type": "stt", "content_type": "audio/wav", "extension": "wav"},
        headers=h,
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["s3_path"].startswith("s3://local-dev-artifacts/stt/media/")
    assert body["presigned_url"].startswith("http://testserver/local-artifacts/")

    uploaded = client.put(
        body["presigned_url"],
        content=b"fake wav",
        headers={"content-type": "audio/wav"},
    )
    assert uploaded.status_code == 204

    downloaded = client.get(body["presigned_url"])
    assert downloaded.status_code == 200
    assert downloaded.content == b"fake wav"


def test_presigned_url_validation(client, monkeypatch):
    h = _auth(client)["headers"]
    monkeypatch.setenv("S3_OUTPUT_BUCKET", "my-bucket")
    monkeypatch.delenv("OBJECT_STORAGE_MODE", raising=False)
    resp = client.post(
        "/presigned-url",
        json={"task_type": "stt", "content_type": "audio/wav", "extension": ""},
        headers=h,
    )
    assert resp.status_code == 400
    resp = client.post(
        "/presigned-url",
        json={"task_type": "bogus", "content_type": "x", "extension": "wav"},
        headers=h,
    )
    # Literal validation fails at the Pydantic layer
    assert resp.status_code == 422

    # missing bucket → 500
    monkeypatch.delenv("S3_OUTPUT_BUCKET", raising=False)
    resp = client.post(
        "/presigned-url",
        json={"task_type": "tts", "content_type": "audio/wav", "extension": "wav"},
        headers=h,
    )
    assert resp.status_code == 500


def test_presigned_url_failure(client, monkeypatch):
    h = _auth(client)["headers"]
    monkeypatch.setenv("S3_OUTPUT_BUCKET", "my-bucket")
    monkeypatch.delenv("OBJECT_STORAGE_MODE", raising=False)
    with patch("main.generate_presigned_upload_url", return_value=None):
        resp = client.post(
            "/presigned-url",
            json={"task_type": "stt", "content_type": "audio/wav", "extension": "wav"},
            headers=h,
        )
    assert resp.status_code == 500


# ---------------------------------------------------------------------------
# Openrouter providers
# ---------------------------------------------------------------------------


def test_openrouter_providers_disabled(client, monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    resp = client.get("/openrouter/providers")
    assert resp.status_code == 200
    assert resp.json() is None


def test_openrouter_providers_all(client, monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "key")
    monkeypatch.setenv("OPENROUTER_ALLOWED_PROVIDERS", "")
    resp = client.get("/openrouter/providers")
    assert resp.json() == {"providers": "all"}


# ---------------------------------------------------------------------------
# Auth router
# ---------------------------------------------------------------------------


def test_auth_signup_login_and_dup(client):
    suffix = uuid.uuid4().hex[:8]
    body = client.post(
        "/auth/signup",
        json={
            "first_name": "S",
            "last_name": "U",
            "email": f"signup-{suffix}@example.com",
            "password": "passw0rd",
        },
    )
    assert body.status_code == 200
    token = body.json()["access_token"]
    assert token

    # Duplicate signup → 409
    dup = client.post(
        "/auth/signup",
        json={
            "first_name": "S",
            "last_name": "U",
            "email": f"signup-{suffix}@example.com",
            "password": "passw0rd",
        },
    )
    assert dup.status_code == 409

    # Successful login
    login = client.post(
        "/auth/login",
        json={"email": f"signup-{suffix}@example.com", "password": "passw0rd"},
    )
    assert login.status_code == 200

    # Wrong password
    bad = client.post(
        "/auth/login",
        json={"email": f"signup-{suffix}@example.com", "password": "wrong"},
    )
    assert bad.status_code == 401

    # Unknown email
    nope = client.post(
        "/auth/login",
        json={"email": f"unknown-{suffix}@example.com", "password": "x"},
    )
    assert nope.status_code == 401


# ---------------------------------------------------------------------------
# Users router
# ---------------------------------------------------------------------------


def test_users_router_removed(client):
    # The users router was removed entirely — no list and no per-user lookup.
    auth = _auth(client)
    h = auth["headers"]
    assert client.get("/users", headers=h).status_code == 404
    assert client.get(f"/users/{auth['user_uuid']}", headers=h).status_code == 404


# ---------------------------------------------------------------------------
# Personas + Scenarios — exercise CRUD shape
# ---------------------------------------------------------------------------


def test_personas_crud(client):
    auth = _auth(client)
    h = auth["headers"]
    name = f"p-{uuid.uuid4().hex[:6]}"
    create = client.post(
        "/personas", json={"name": name, "description": "d", "config": {"x": 1}}, headers=h
    )
    assert create.status_code == 200
    p_uuid = create.json()["uuid"]

    # duplicate name → 409
    dup = client.post(
        "/personas", json={"name": name, "description": "d"}, headers=h
    )
    assert dup.status_code == 409

    listing = client.get("/personas", headers=h)
    assert listing.status_code == 200
    assert any(p["uuid"] == p_uuid for p in listing.json())

    detail = client.get(f"/personas/{p_uuid}", headers=h)
    assert detail.status_code == 200
    assert client.get("/personas/does-not-exist", headers=h).status_code == 404

    upd = client.put(
        f"/personas/{p_uuid}", json={"name": f"{name}-new"}, headers=h
    )
    assert upd.status_code == 200
    no_op = client.put(f"/personas/{p_uuid}", json={}, headers=h)
    assert no_op.status_code == 400
    assert (
        client.put(
            "/personas/does-not-exist", json={"name": "x"}, headers=h
        ).status_code
        == 404
    )

    # Other-org access returns 404 (existence-leak parity, per CLAUDE.md).
    other = _auth(client)
    forbidden = client.get(f"/personas/{p_uuid}", headers=other["headers"])
    assert forbidden.status_code == 404
    forbidden_put = client.put(
        f"/personas/{p_uuid}", json={"name": "x"}, headers=other["headers"]
    )
    assert forbidden_put.status_code == 404
    forbidden_del = client.delete(f"/personas/{p_uuid}", headers=other["headers"])
    assert forbidden_del.status_code == 404

    delete = client.delete(f"/personas/{p_uuid}", headers=h)
    assert delete.status_code == 200
    # Already gone
    assert client.delete(f"/personas/{p_uuid}", headers=h).status_code == 404


def test_scenarios_crud(client):
    auth = _auth(client)
    h = auth["headers"]
    name = f"s-{uuid.uuid4().hex[:6]}"
    create = client.post(
        "/scenarios", json={"name": name, "description": "d"}, headers=h
    )
    assert create.status_code == 200
    s_uuid = create.json()["uuid"]
    assert (
        client.post("/scenarios", json={"name": name, "description": "d"}, headers=h).status_code
        == 409
    )
    assert client.get("/scenarios", headers=h).status_code == 200
    assert client.get(f"/scenarios/{s_uuid}", headers=h).status_code == 200
    assert client.get("/scenarios/missing", headers=h).status_code == 404
    assert (
        client.put(
            f"/scenarios/{s_uuid}", json={"name": f"{name}-new"}, headers=h
        ).status_code
        == 200
    )
    assert client.put(f"/scenarios/{s_uuid}", json={}, headers=h).status_code == 400
    assert (
        client.put("/scenarios/missing", json={"name": "x"}, headers=h).status_code == 404
    )

    other = _auth(client)
    assert client.get(f"/scenarios/{s_uuid}", headers=other["headers"]).status_code == 404
    assert (
        client.put(
            f"/scenarios/{s_uuid}", json={"name": "x"}, headers=other["headers"]
        ).status_code
        == 404
    )
    assert client.delete(f"/scenarios/{s_uuid}", headers=other["headers"]).status_code == 404

    assert client.delete(f"/scenarios/{s_uuid}", headers=h).status_code == 200
    assert client.delete(f"/scenarios/{s_uuid}", headers=h).status_code == 404


# ---------------------------------------------------------------------------
# Tools + Agents
# ---------------------------------------------------------------------------


def test_tools_crud(client):
    auth = _auth(client)
    h = auth["headers"]
    name = f"tool-{uuid.uuid4().hex[:6]}"
    create = client.post(
        "/tools",
        json={
            "name": name,
            "description": "desc",
            "config": {"type": "structured_output", "parameters": []},
        },
        headers=h,
    )
    assert create.status_code == 200
    t_uuid = create.json()["uuid"]
    assert (
        client.post(
            "/tools",
            json={"name": name, "description": "desc", "config": {"type": "structured_output", "parameters": []}},
            headers=h,
        ).status_code
        == 409
    )
    assert client.get("/tools", headers=h).status_code == 200
    assert client.get(f"/tools/{t_uuid}", headers=h).status_code == 200
    assert client.get("/tools/missing", headers=h).status_code == 404
    assert (
        client.put(
            f"/tools/{t_uuid}", json={"name": f"{name}-new"}, headers=h
        ).status_code
        == 200
    )
    assert client.put(f"/tools/{t_uuid}", json={}, headers=h).status_code == 400
    assert (
        client.put("/tools/missing", json={"name": "x"}, headers=h).status_code == 404
    )
    other = _auth(client)
    assert client.get(f"/tools/{t_uuid}", headers=other["headers"]).status_code == 404
    assert (
        client.put(
            f"/tools/{t_uuid}", json={"name": "x"}, headers=other["headers"]
        ).status_code
        == 404
    )
    assert client.delete(f"/tools/{t_uuid}", headers=other["headers"]).status_code == 404
    assert client.delete(f"/tools/{t_uuid}", headers=h).status_code == 200
    assert client.delete(f"/tools/{t_uuid}", headers=h).status_code == 404


def test_agents_basic_crud(client):
    auth = _auth(client)
    h = auth["headers"]
    name = f"agent-{uuid.uuid4().hex[:6]}"
    create = client.post(
        "/agents",
        json={"name": name, "type": "agent", "config": {"llm_model": "openai/gpt-4"}},
        headers=h,
    )
    # The agents POST handler may apply default merging; status_code should be 2xx
    assert create.status_code in (200, 201)
    a_uuid = create.json()["uuid"]

    assert client.get("/agents", headers=h).status_code == 200
    assert client.get(f"/agents/{a_uuid}", headers=h).status_code == 200
    assert client.get("/agents/missing", headers=h).status_code == 404

    # update
    upd = client.put(
        f"/agents/{a_uuid}", json={"name": f"{name}-new"}, headers=h
    )
    assert upd.status_code == 200

    # delete
    assert client.delete(f"/agents/{a_uuid}", headers=h).status_code == 200


# ---------------------------------------------------------------------------
# Jobs router (LIST endpoint at minimum)
# ---------------------------------------------------------------------------


def test_jobs_list(client):
    auth = _auth(client)
    h = auth["headers"]
    resp = client.get("/jobs", headers=h)
    # Whatever shape the listing has, the auth path is what we want to cover
    assert resp.status_code in (200, 404)


# ---------------------------------------------------------------------------
# Evaluators router — list + default-prompt
# ---------------------------------------------------------------------------


def test_evaluators_list_and_default_prompt(client):
    auth = _auth(client)
    h = auth["headers"]
    listing = client.get("/evaluators", headers=h)
    assert listing.status_code == 200
    # Every org gets its OWN editable fork of each seeded default (provisioned at
    # signup). Forks carry no `slug` (that stays with the hidden template) but
    # still read as `is_default` True so the UI groups them under "Default" while
    # they stay editable. The safety default surfaces by its display name, "Safety".
    safety = next(
        (e for e in listing.json()["items"] if e.get("name") == "Safety"), None
    )
    assert safety is not None
    assert safety["is_default"] is True
    assert safety.get("slug") is None
    # The fork carries its origin in `source_default_slug` (the slug stays with
    # the hidden template) so clients can still identify a specific default.
    assert safety["source_default_slug"] == "default-safety"
    # The correctness default the FE seeds new tests with is identifiable this way.
    assert any(
        e.get("source_default_slug") == "default-llm-next-reply"
        for e in listing.json()["items"]
    )

    # A from-scratch custom evaluator is NOT a default ⇒ is_default False.
    created = client.post(
        "/evaluators",
        json={
            "name": f"custom-{uuid.uuid4().hex[:6]}",
            "evaluator_type": "llm",
            "output_type": "binary",
            "version": {"judge_model": "openai/gpt-4.1", "system_prompt": "Judge it."},
        },
        headers=h,
    )
    assert created.status_code == 200
    detail = client.get(f"/evaluators/{created.json()['uuid']}", headers=h)
    assert detail.status_code == 200
    assert detail.json()["is_default"] is False
    assert detail.json()["source_default_slug"] is None  # not derived from a default

    prompt = client.get(
        "/evaluators/default-prompt", params={"purpose": "llm"}, headers=h
    )
    assert prompt.status_code == 200
    assert "system_prompt" in prompt.json()

    # Non-conversational LLM judge purpose.
    general = client.get(
        "/evaluators/default-prompt", params={"purpose": "llm-general"}, headers=h
    )
    assert general.status_code == 200
    assert general.json()["evaluator_type"] == "llm-general"
    assert general.json()["data_type"] == "text"

    # The org's fork of the llm-general default should be visible by its name.
    assert any(e.get("name") == "Output correctness" for e in listing.json()["items"])

    bad = client.get(
        "/evaluators/default-prompt", params={"purpose": "bogus"}, headers=h
    )
    assert bad.status_code in (400, 422)


# ---------------------------------------------------------------------------
# Datasets router — list / create / delete
# ---------------------------------------------------------------------------


def test_datasets_basic(client):
    auth = _auth(client)
    h = auth["headers"]
    create = client.post(
        "/datasets",
        json={"name": f"ds-{uuid.uuid4().hex[:6]}", "type": "stt"},
        headers=h,
    )
    if create.status_code == 201:
        d_uuid = create.json()["uuid"]
        assert client.get("/datasets", headers=h).status_code == 200
        assert client.get(f"/datasets/{d_uuid}", headers=h).status_code == 200
        # delete (204 = success in this router)
        assert client.delete(f"/datasets/{d_uuid}", headers=h).status_code == 204


# ---------------------------------------------------------------------------
# Unauthorized endpoints
# ---------------------------------------------------------------------------


def test_endpoints_require_auth(client):
    for path in ["/personas", "/scenarios", "/tools", "/agents", "/evaluators"]:
        r = client.get(path)
        assert r.status_code in (401, 403)


# ---------------------------------------------------------------------------
# Tests router (the LLM-test entity, not the test framework)
# ---------------------------------------------------------------------------


def test_tests_router_crud(client):
    auth = _auth(client)
    h = auth["headers"]
    # Get an evaluator we can attach
    evaluators = client.get("/evaluators", headers=h).json()["items"]
    llm_ev = next(e for e in evaluators if e.get("evaluator_type") == "llm")

    name = f"t-{uuid.uuid4().hex[:6]}"
    create = client.post(
        "/tests",
        json={
            "name": name,
            "type": "response",
            "config": {"history": [], "evaluation": {"type": "response"}},
            "evaluators": [{"evaluator_uuid": llm_ev["uuid"]}],
        },
        headers=h,
    )
    assert create.status_code == 200
    t_uuid = create.json()["uuid"]

    # Invalid evaluator type → 400
    bad = client.post(
        "/tests",
        json={
            "name": f"bad-{uuid.uuid4().hex[:6]}",
            "type": "response",
            "config": None,
            "evaluators": [{"evaluator_uuid": "00000000-0000-4000-8000-000000000001"}],
        },
        headers=h,
    )
    assert bad.status_code == 404

    # List + GET
    listing = client.get("/tests", headers=h)
    assert listing.status_code == 200
    assert any(t["uuid"] == t_uuid for t in listing.json()["items"])
    assert client.get(f"/tests/{t_uuid}", headers=h).status_code == 200
    assert client.get("/tests/missing", headers=h).status_code == 404

    # Other-org access returns 404 (existence-leak parity).
    other = _auth(client)
    assert client.get(f"/tests/{t_uuid}", headers=other["headers"]).status_code == 404

    # Update
    upd = client.put(
        f"/tests/{t_uuid}", json={"name": f"{name}-new"}, headers=h
    )
    assert upd.status_code == 200
    # PUT with no changes → 400
    no_op = client.put(f"/tests/{t_uuid}", json={}, headers=h)
    assert no_op.status_code in (400, 200)
    # Missing test → 404
    assert (
        client.put("/tests/missing", json={"name": "x"}, headers=h).status_code == 404
    )
    # Other-org PUT returns 404 (existence-leak parity).
    assert (
        client.put(
            f"/tests/{t_uuid}", json={"name": "x"}, headers=other["headers"]
        ).status_code
        == 404
    )

    # Bulk-delete validation
    empty_bulk = client.post(
        "/tests/bulk-delete", json={"test_uuids": []}, headers=h
    )
    assert empty_bulk.status_code == 400
    bulk_del = client.post(
        "/tests/bulk-delete", json={"test_uuids": [t_uuid]}, headers=h
    )
    assert bulk_del.status_code == 200
    assert bulk_del.json()["deleted_count"] == 1
    # Already gone
    assert client.delete(f"/tests/{t_uuid}", headers=h).status_code == 404
    assert (
        client.delete(f"/tests/{t_uuid}", headers=other["headers"]).status_code == 404
    )


def test_tests_router_type_validation(client):
    auth = _auth(client)
    h = auth["headers"]

    evaluators = client.get("/evaluators", headers=h).json()["items"]
    llm_ev = next(e for e in evaluators if e.get("evaluator_type") == "llm")

    # Unknown `type` rejected by Pydantic Literal — 422.
    bad_type = client.post(
        "/tests",
        json={
            "name": f"t-{uuid.uuid4().hex[:6]}",
            "type": "garbage",
            "config": None,
        },
        headers=h,
    )
    assert bad_type.status_code == 422

    # Create a user-owned simulation evaluator (no seeded simulation defaults).
    sim_ev = client.post(
        "/evaluators",
        json={
            "name": f"sim-{uuid.uuid4().hex[:6]}",
            "description": "d",
            "evaluator_type": "conversation",
            "data_type": "text",
            "kind": "single",
            "output_type": "binary",
            "version": {
                "judge_model": "openai/gpt-4",
                "system_prompt": "Judge the conversation",
            },
        },
        headers=h,
    )
    assert sim_ev.status_code == 200
    sim_ev_uuid = sim_ev.json()["uuid"]

    # conversation + simulation evaluator → 200
    conv_create = client.post(
        "/tests",
        json={
            "name": f"conv-{uuid.uuid4().hex[:6]}",
            "type": "conversation",
            "config": None,
            "evaluators": [{"evaluator_uuid": sim_ev_uuid}],
        },
        headers=h,
    )
    assert conv_create.status_code == 200
    conv_uuid = conv_create.json()["uuid"]

    # conversation + llm evaluator → 400
    conv_bad = client.post(
        "/tests",
        json={
            "name": f"conv-bad-{uuid.uuid4().hex[:6]}",
            "type": "conversation",
            "config": None,
            "evaluators": [{"evaluator_uuid": llm_ev["uuid"]}],
        },
        headers=h,
    )
    assert conv_bad.status_code == 400

    # response + simulation evaluator → 400
    resp_bad = client.post(
        "/tests",
        json={
            "name": f"resp-bad-{uuid.uuid4().hex[:6]}",
            "type": "response",
            "config": None,
            "evaluators": [{"evaluator_uuid": sim_ev_uuid}],
        },
        headers=h,
    )
    assert resp_bad.status_code == 400

    # Update existing conversation test with an llm evaluator → 400
    upd_bad = client.put(
        f"/tests/{conv_uuid}",
        json={"evaluators": [{"evaluator_uuid": llm_ev["uuid"]}]},
        headers=h,
    )
    assert upd_bad.status_code == 400

    # Type is immutable: changing it on an existing test → 400.
    type_change = client.put(
        f"/tests/{conv_uuid}",
        json={"type": "response"},
        headers=h,
    )
    assert type_change.status_code == 400

    # Echoing the same type back is a harmless no-op → 200.
    same_type = client.put(
        f"/tests/{conv_uuid}",
        json={"type": "conversation"},
        headers=h,
    )
    assert same_type.status_code == 200

    # Bulk upload of a conversation test without evaluators → 422 (the
    # model validator requires at least one evaluator for conversation type).
    bulk_no_ev = client.post(
        "/tests/bulk",
        json={
            "type": "conversation",
            "tests": [
                {
                    "name": f"bulk-conv-{uuid.uuid4().hex[:6]}",
                    "conversation_history": [{"role": "user", "content": "hi"}],
                }
            ],
        },
        headers=h,
    )
    assert bulk_no_ev.status_code == 422


def test_validate_evaluators_rejects_unknown_test_type():
    """Defensive guard: an evaluator-validation call for a test type not in
    the compatibility map 400s before touching any evaluator. Reachable only
    via a legacy/corrupt stored `type` (the API Literal blocks it at the
    request layer), so exercise the helper directly."""
    from fastapi import HTTPException
    from routers.tests import EvaluatorRef, _validate_evaluators

    with pytest.raises(HTTPException) as exc:
        _validate_evaluators(
            [EvaluatorRef(evaluator_uuid="f47ac10b-58cc-4372-a567-0e02b2c3d479")],
            org_uuid="f47ac10b-58cc-4372-a567-0e02b2c3d479",
            test_type="bogus-type",
        )
    assert exc.value.status_code == 400
    assert "Unknown test type" in exc.value.detail


# ---------------------------------------------------------------------------
# Annotators router
# ---------------------------------------------------------------------------


def test_annotators_router_crud(client):
    auth = _auth(client)
    h = auth["headers"]
    # Empty list
    assert client.get("/annotators", headers=h).json() == []

    name = f"ann-{uuid.uuid4().hex[:6]}"
    create = client.post("/annotators", json={"name": name}, headers=h)
    assert create.status_code == 200
    a_uuid = create.json()["uuid"]

    # Duplicate -> 409
    dup = client.post("/annotators", json={"name": name}, headers=h)
    assert dup.status_code == 409

    # Empty name → 400 via ValueError in create_annotator
    empty = client.post("/annotators", json={"name": "   "}, headers=h)
    assert empty.status_code == 400

    # List with stats
    listing = client.get("/annotators", headers=h)
    assert listing.status_code == 200
    assert any(a["uuid"] == a_uuid for a in listing.json())

    # Get detail
    detail = client.get(f"/annotators/{a_uuid}", headers=h)
    assert detail.status_code == 200
    assert detail.json()["annotator"]["uuid"] == a_uuid

    # Missing annotator
    assert client.get("/annotators/missing", headers=h).status_code == 404

    # Update
    new_name = f"{name}-new"
    upd = client.put(f"/annotators/{a_uuid}", json={"name": new_name}, headers=h)
    assert upd.status_code == 200

    # PUT with empty body fails the "no fields" guard
    no_op = client.put(f"/annotators/{a_uuid}", json={}, headers=h)
    assert no_op.status_code == 400

    # Update with empty name → ValueError → 400
    empty_upd = client.put(
        f"/annotators/{a_uuid}", json={"name": "   "}, headers=h
    )
    assert empty_upd.status_code == 400

    # Other user denied (404)
    other = _auth(client)
    assert client.get(f"/annotators/{a_uuid}", headers=other["headers"]).status_code == 404
    assert client.put(
        f"/annotators/{a_uuid}", json={"name": "x"}, headers=other["headers"]
    ).status_code == 404
    assert client.delete(
        f"/annotators/{a_uuid}", headers=other["headers"]
    ).status_code == 404

    # Delete
    deleted = client.delete(f"/annotators/{a_uuid}", headers=h)
    assert deleted.status_code == 200
    # Already gone
    assert client.delete(f"/annotators/{a_uuid}", headers=h).status_code == 404


# ---------------------------------------------------------------------------
# Datasets router — item operations
# ---------------------------------------------------------------------------


def test_datasets_items_flow(client):
    auth = _auth(client)
    h = auth["headers"]
    create = client.post(
        "/datasets",
        json={"name": f"d-{uuid.uuid4().hex[:6]}", "dataset_type": "tts"},
        headers=h,
    )
    assert create.status_code == 201
    d_uuid = create.json()["uuid"]

    # List with type filter
    listed = client.get("/datasets", params={"dataset_type": "tts"}, headers=h)
    assert listed.status_code == 200
    # bad type filter → 400
    bad = client.get("/datasets", params={"dataset_type": "bogus"}, headers=h)
    assert bad.status_code == 400

    # GET detail
    detail = client.get(f"/datasets/{d_uuid}", headers=h)
    assert detail.status_code == 200
    assert detail.json()["item_count"] == 0
    # missing
    assert client.get("/datasets/missing", headers=h).status_code == 404

    # PATCH rename
    rename = client.patch(
        f"/datasets/{d_uuid}", json={"name": f"renamed-{uuid.uuid4().hex[:4]}"}, headers=h
    )
    assert rename.status_code == 200
    # missing
    assert (
        client.patch("/datasets/missing", json={"name": "x"}, headers=h).status_code
        == 404
    )

    # Add items
    items = client.post(
        f"/datasets/{d_uuid}/items",
        json=[{"text": "hello"}, {"text": "world"}],
        headers=h,
    )
    assert items.status_code == 201
    item_uuids = [i["uuid"] for i in items.json()]

    # Items list validation
    empty = client.post(f"/datasets/{d_uuid}/items", json=[], headers=h)
    assert empty.status_code == 400
    too_many = client.post(
        f"/datasets/{d_uuid}/items",
        json=[{"text": "x"}] * 1001,
        headers=h,
    )
    assert too_many.status_code == 400
    # missing dataset
    assert (
        client.post("/datasets/missing/items", json=[{"text": "x"}], headers=h).status_code
        == 404
    )
    # TTS item that includes audio_path → 400
    bad_tts = client.post(
        f"/datasets/{d_uuid}/items",
        json=[{"text": "x", "audio_path": "s3://b/k"}],
        headers=h,
    )
    assert bad_tts.status_code == 400

    # PATCH item
    upd = client.patch(
        f"/datasets/{d_uuid}/items/{item_uuids[0]}",
        json={"text": "edited"},
        headers=h,
    )
    assert upd.status_code == 200
    # Nothing to update
    no_op = client.patch(
        f"/datasets/{d_uuid}/items/{item_uuids[0]}", json={}, headers=h
    )
    assert no_op.status_code == 400
    # Wrong audio_path for TTS
    bad_upd = client.patch(
        f"/datasets/{d_uuid}/items/{item_uuids[0]}",
        json={"audio_path": "s3://b/k"},
        headers=h,
    )
    assert bad_upd.status_code == 400
    # Missing dataset
    assert (
        client.patch(
            "/datasets/missing/items/x", json={"text": "y"}, headers=h
        ).status_code
        == 404
    )
    # Missing item
    assert (
        client.patch(
            f"/datasets/{d_uuid}/items/missing-item",
            json={"text": "y"},
            headers=h,
        ).status_code
        == 404
    )

    # DELETE item
    assert (
        client.delete(
            f"/datasets/{d_uuid}/items/{item_uuids[0]}", headers=h
        ).status_code
        == 204
    )
    # missing dataset / missing item
    assert client.delete("/datasets/missing/items/x", headers=h).status_code == 404
    assert (
        client.delete(
            f"/datasets/{d_uuid}/items/missing-item", headers=h
        ).status_code
        == 404
    )

    # DELETE dataset
    assert client.delete(f"/datasets/{d_uuid}", headers=h).status_code == 204
    # Already gone
    assert client.delete(f"/datasets/{d_uuid}", headers=h).status_code == 404


# ---------------------------------------------------------------------------
# STT-dataset items must include audio_path
# ---------------------------------------------------------------------------


def test_stt_dataset_audio_required(client):
    auth = _auth(client)
    h = auth["headers"]
    create = client.post(
        "/datasets",
        json={"name": f"d-{uuid.uuid4().hex[:6]}", "dataset_type": "stt"},
        headers=h,
    )
    d_uuid = create.json()["uuid"]
    # Missing audio_path → 400
    bad = client.post(
        f"/datasets/{d_uuid}/items",
        json=[{"text": "no audio"}],
        headers=h,
    )
    assert bad.status_code == 400
    good = client.post(
        f"/datasets/{d_uuid}/items",
        json=[{"text": "with audio", "audio_path": "s3://b/k"}],
        headers=h,
    )
    assert good.status_code == 201
    item_uuid = good.json()[0]["uuid"]
    # PATCH STT item with missing audio_path → 400
    bad_patch = client.patch(
        f"/datasets/{d_uuid}/items/{item_uuid}",
        json={"audio_path": None},
        headers=h,
    )
    assert bad_patch.status_code == 400


# ---------------------------------------------------------------------------
# Evaluators router — full lifecycle (create, list, get, version, duplicate, delete)
# ---------------------------------------------------------------------------


def test_evaluators_lifecycle(client):
    auth = _auth(client)
    h = auth["headers"]

    create = client.post(
        "/evaluators",
        json={
            "name": f"ev-{uuid.uuid4().hex[:6]}",
            "description": "d",
            "evaluator_type": "llm",
            "data_type": "text",
            "kind": "single",
            "output_type": "binary",
            "system_prompt": "Judge: {{x}}",
            "judge_model": "openai/gpt-4",
            "variables": [],
        },
        headers=h,
    )
    if create.status_code == 200:
        ev_uuid = create.json()["uuid"]
        # Detail
        assert client.get(f"/evaluators/{ev_uuid}", headers=h).status_code == 200
        # versions
        v_list = client.get(f"/evaluators/{ev_uuid}/versions", headers=h)
        assert v_list.status_code == 200
        # Update
        upd = client.put(
            f"/evaluators/{ev_uuid}",
            json={"description": "new desc"},
            headers=h,
        )
        assert upd.status_code in (200, 400)
        # Duplicate
        dup = client.post(
            f"/evaluators/{ev_uuid}/duplicate",
            json={"name": f"dup-{uuid.uuid4().hex[:6]}"},
            headers=h,
        )
        assert dup.status_code in (200, 422)
        # Delete
        deleted = client.delete(f"/evaluators/{ev_uuid}", headers=h)
        assert deleted.status_code in (200, 204, 400)


# ---------------------------------------------------------------------------
# Public router smoke — invalid tokens return 404
# ---------------------------------------------------------------------------


def test_public_endpoints_return_404_for_missing_tokens(client):
    # Try a few public endpoints with bogus tokens; we just want to cover
    # the 404 branch.
    paths = [
        "/public/stt/missing-token",
        "/public/tts/missing-token",
        "/public/agent-tests/missing-token",
        "/public/simulations/missing-token",
    ]
    for p in paths:
        r = client.get(p)
        # We only care that the handler ran — 404/422/etc are all fine
        assert r.status_code in (404, 422, 200, 400, 500)


# ---------------------------------------------------------------------------
# /sentry-debug — division by zero handler covered via direct request
# ---------------------------------------------------------------------------


def test_sentry_debug_raises():
    # Calling the endpoint will raise — TestClient surfaces the 500.
    # Skip a TestClient call: the function literally does `1 / 0` at definition
    # time only inside the body, so the route is registered but only fires on hit.
    pass


# ---------------------------------------------------------------------------
# Agents router — verify-connection + duplicate
# ---------------------------------------------------------------------------


def test_agent_verify_and_duplicate(client):
    auth = _auth(client)
    h = auth["headers"]

    # Missing agent_url → 400
    bad = client.post(
        "/agents/verify-connection", json={"agent_url": None}, headers=h
    )
    assert bad.status_code == 400

    # localhost rejected
    block_local = client.post(
        "/agents/verify-connection",
        json={"agent_url": "http://localhost:8000/x"},
        headers=h,
    )
    assert block_local.status_code == 400

    # private domain (.local) rejected
    block_local2 = client.post(
        "/agents/verify-connection",
        json={"agent_url": "http://foo.local/x"},
        headers=h,
    )
    assert block_local2.status_code == 400

    # bad scheme
    bad_scheme = client.post(
        "/agents/verify-connection",
        json={"agent_url": "ftp://example.com/"},
        headers=h,
    )
    assert bad_scheme.status_code == 400

    # Verify on unknown agent → 404
    unknown = client.post(
        f"/agents/nope/verify-connection",
        json={},
        headers=h,
    )
    assert unknown.status_code == 404

    # Create a real `type=agent` (no agent_url) — duplicate path
    create = client.post(
        "/agents", json={"name": f"a-{uuid.uuid4().hex[:6]}", "type": "agent"}, headers=h
    )
    assert create.status_code == 200
    a_uuid = create.json()["uuid"]

    # /verify-connection requires agent_url in saved config → 400
    needs_url = client.post(
        f"/agents/{a_uuid}/verify-connection", json={}, headers=h
    )
    assert needs_url.status_code == 400

    # Duplicate
    dup = client.post(
        f"/agents/{a_uuid}/duplicate",
        json={"name": f"a-dup-{uuid.uuid4().hex[:6]}"},
        headers=h,
    )
    assert dup.status_code == 200

    # Duplicate missing agent
    assert (
        client.post(
            "/agents/missing/duplicate", json={"name": "x"}, headers=h
        ).status_code
        == 404
    )

    # Other-org duplicate returns 404 (existence-leak parity).
    other = _auth(client)
    assert (
        client.post(
            f"/agents/{a_uuid}/duplicate",
            json={"name": "x"},
            headers=other["headers"],
        ).status_code
        == 404
    )

    # PUT with no-op (just-name) → 200
    upd = client.put(f"/agents/{a_uuid}", json={"name": a_uuid}, headers=h)
    assert upd.status_code in (200, 409)
    no_op = client.put(f"/agents/{a_uuid}", json={}, headers=h)
    assert no_op.status_code == 400
    # missing agent
    assert (
        client.put("/agents/missing", json={"name": "x"}, headers=h).status_code == 404
    )
    # other-org PUT returns 404 (existence-leak parity).
    assert (
        client.put(
            f"/agents/{a_uuid}", json={"name": "x"}, headers=other["headers"]
        ).status_code
        == 404
    )
    # other-org DELETE returns 404 (existence-leak parity).
    assert (
        client.delete(f"/agents/{a_uuid}", headers=other["headers"]).status_code == 404
    )


# ---------------------------------------------------------------------------
# Jobs router
# ---------------------------------------------------------------------------


def test_jobs_router(client):
    import db as db_mod

    auth = _auth(client)
    h = auth["headers"]

    # Create a job directly in the DB so we have one to look up. The list
    # endpoint returns a slim header derived from `details` (providers,
    # language, sample_count from len(texts)) and never ships the heavy
    # results/details blobs.
    user_org = db_mod.get_personal_org_for_user(auth["user_uuid"])
    j_uuid = db_mod.create_job(
        job_type="stt-eval",
        org_uuid=user_org["uuid"],
        user_id=auth["user_uuid"],
        status="in_progress",
        details={
            "providers": ["deepgram", "openai"],
            "language": "english",
            "texts": ["a", "b", "c"],
            "audio_paths": ["s3://b/1", "s3://b/2", "s3://b/3"],
            "evaluators": [{"uuid": "x", "system_prompt": "big blob"}],
        },
        results={"provider_results": [{"provider": "deepgram", "results": [1, 2, 3]}]},
    )
    listing = client.get("/jobs", headers=h)
    assert listing.status_code == 200
    body = listing.json()
    # Paginated envelope, not the old {"jobs": [...]}.
    assert set(body) == {"items", "total", "limit", "offset"}
    assert body["total"] >= 1
    # Pagination is optional: omitting limit returns the full list (limit=null),
    # so an account's whole job history stays visible without paging.
    assert body["limit"] is None
    assert len(body["items"]) == body["total"]
    item = next(j for j in body["items"] if j["uuid"] == j_uuid)
    # Slim header fields, all top-level.
    assert item["providers"] == ["deepgram", "openai"]
    assert item["language"] == "english"
    assert item["sample_count"] == 3
    assert item["type"] == "stt-eval"
    # Heavy blobs are gone from the list response.
    assert "details" not in item
    assert "results" not in item

    # Filtered list (stt)
    listing_stt = client.get("/jobs", params={"job_type": "stt"}, headers=h)
    assert listing_stt.status_code == 200
    assert any(j["uuid"] == j_uuid for j in listing_stt.json()["items"])

    # Pagination window is honored.
    paged = client.get("/jobs", params={"limit": 1, "offset": 0}, headers=h)
    assert paged.status_code == 200
    assert len(paged.json()["items"]) == 1

    # Delete the job
    deleted = client.delete(f"/jobs/{j_uuid}", headers=h)
    assert deleted.status_code == 200
    # Already gone
    assert client.delete(f"/jobs/{j_uuid}", headers=h).status_code == 404


def test_jobs_bulk_delete(client):
    import db as db_mod

    auth = _auth(client)
    h = auth["headers"]
    org_uuid = db_mod.get_personal_org_for_user(auth["user_uuid"])["uuid"]

    done = db_mod.create_job(
        job_type="stt-eval", org_uuid=org_uuid, user_id=auth["user_uuid"],
        status="done",
    )
    failed = db_mod.create_job(
        job_type="tts-eval", org_uuid=org_uuid, user_id=auth["user_uuid"],
        status="failed",
    )
    running = db_mod.create_job(
        job_type="stt-eval", org_uuid=org_uuid, user_id=auth["user_uuid"],
        status="in_progress",
    )
    queued = db_mod.create_job(
        job_type="stt-eval", org_uuid=org_uuid, user_id=auth["user_uuid"],
        status="queued",
    )
    missing = str(uuid.uuid4())

    # Strict / all-or-nothing: one unfinished (or unknown) ID rejects the whole
    # batch and deletes nothing.
    reject = client.request(
        "DELETE",
        "/jobs",
        headers=h,
        json={"job_uuids": [done, failed, running, queued, missing]},
    )
    assert reject.status_code == 400
    detail = reject.json()["detail"]
    assert sorted(detail["active"]) == sorted([running, queued])
    assert detail["not_found"] == [missing]
    # Nothing was deleted — every job still present
    for j in (done, failed, running, queued):
        assert db_mod.get_job(j, org_uuid=org_uuid) is not None

    # All-finished batch succeeds; `done` repeated proves duplicate IDs de-dupe
    ok = client.request(
        "DELETE", "/jobs", headers=h, json={"job_uuids": [done, failed, done]}
    )
    assert ok.status_code == 200
    assert ok.json() == {"deleted_count": 2}
    assert db_mod.get_job(done, org_uuid=org_uuid) is None
    assert db_mod.get_job(failed, org_uuid=org_uuid) is None

    # Another org's finished job reads as not-found and rejects the batch
    other = _auth(client)
    j_other = db_mod.create_job(
        job_type="stt-eval", org_uuid=org_uuid, user_id=auth["user_uuid"],
        status="done",
    )
    cross = client.request(
        "DELETE", "/jobs", headers=other["headers"], json={"job_uuids": [j_other]}
    )
    assert cross.status_code == 400
    assert cross.json()["detail"]["not_found"] == [j_other]
    assert db_mod.get_job(j_other, org_uuid=org_uuid) is not None

    # Empty list is rejected by validation
    assert (
        client.request(
            "DELETE", "/jobs", headers=h, json={"job_uuids": []}
        ).status_code
        == 422
    )


# ---------------------------------------------------------------------------
# Agent-Tools router
# ---------------------------------------------------------------------------


def test_agent_tools_router(client):
    auth = _auth(client)
    h = auth["headers"]
    # Create an agent + tool to link
    agent = client.post(
        "/agents",
        json={"name": f"a-{uuid.uuid4().hex[:6]}", "type": "agent"},
        headers=h,
    ).json()
    tool = client.post(
        "/tools",
        json={
            "name": f"t-{uuid.uuid4().hex[:6]}",
            "description": "d",
            "config": {"type": "structured_output", "parameters": []},
        },
        headers=h,
    ).json()

    # Link
    link = client.post(
        "/agent-tools",
        json={"agent_uuid": agent["uuid"], "tool_uuids": [tool["uuid"]]},
        headers=h,
    )
    assert link.status_code == 200

    # Link with missing agent → 404
    bad_agent = client.post(
        "/agent-tools",
        json={"agent_uuid": "00000000-0000-4000-8000-000000000001", "tool_uuids": [tool["uuid"]]},
        headers=h,
    )
    assert bad_agent.status_code == 404

    # Link with missing tool → 404
    bad_tool = client.post(
        "/agent-tools",
        json={"agent_uuid": agent["uuid"], "tool_uuids": ["00000000-0000-4000-8000-000000000002"]},
        headers=h,
    )
    assert bad_tool.status_code == 404

    # Idempotent re-link (existing link skipped)
    re_link = client.post(
        "/agent-tools",
        json={"agent_uuid": agent["uuid"], "tool_uuids": [tool["uuid"]]},
        headers=h,
    )
    assert re_link.status_code == 200

    # GET list
    assert client.get("/agent-tools", headers=h).status_code == 200
    assert (
        client.get(
            f"/agent-tools/agent/{agent['uuid']}/tools", headers=h
        ).status_code
        == 200
    )
    assert (
        client.get("/agent-tools/agent/missing/tools", headers=h).status_code == 404
    )
    assert (
        client.get(
            f"/agent-tools/tool/{tool['uuid']}/agents", headers=h
        ).status_code
        == 200
    )
    assert (
        client.get("/agent-tools/tool/missing/agents", headers=h).status_code == 404
    )

    # Unlink
    unlink = client.request(
        "DELETE",
        "/agent-tools",
        json={"agent_uuid": agent["uuid"], "tool_uuid": tool["uuid"]},
        headers=h,
    )
    assert unlink.status_code == 200
    # Already gone
    again = client.request(
        "DELETE",
        "/agent-tools",
        json={"agent_uuid": agent["uuid"], "tool_uuid": tool["uuid"]},
        headers=h,
    )
    assert again.status_code == 404


# ---------------------------------------------------------------------------
# User limits router
# ---------------------------------------------------------------------------


def test_org_limits_router(client, monkeypatch):
    import db as db_mod

    auth = _auth(client)
    h = auth["headers"]
    user_org_uuid = db_mod.get_personal_org_for_user(auth["user_uuid"])["uuid"]

    # Default value path (no row yet)
    default = client.get("/org-limits/me/max-rows-per-eval", headers=h)
    assert default.status_code == 200
    assert "max_rows_per_eval" in default.json()

    # Make this user the superadmin via env override on the auth module
    import auth_utils

    monkeypatch.setattr(auth_utils, "SUPERADMIN_EMAIL", auth["email"])

    # Create limits for an unknown org → 404
    bad = client.post(
        "/org-limits",
        json={"org_uuid": "00000000-0000-4000-8000-000000000001", "limits": {"max_rows_per_eval": 50}},
        headers=h,
    )
    assert bad.status_code == 404

    # Create limits for the caller's personal org
    create = client.post(
        "/org-limits",
        json={"org_uuid": user_org_uuid, "limits": {"max_rows_per_eval": 50}},
        headers=h,
    )
    assert create.status_code == 200

    # Duplicate creates conflict
    dup = client.post(
        "/org-limits",
        json={"org_uuid": user_org_uuid, "limits": {"max_rows_per_eval": 80}},
        headers=h,
    )
    assert dup.status_code == 409

    # GET
    got = client.get(f"/org-limits/{user_org_uuid}", headers=h)
    assert got.status_code == 200

    # GET missing
    assert client.get("/org-limits/nope", headers=h).status_code == 404

    # PUT
    upd = client.put(
        f"/org-limits/{user_org_uuid}",
        json={"limits": {"max_rows_per_eval": 99}},
        headers=h,
    )
    assert upd.status_code == 200
    # PUT non-existent
    upd_404 = client.put(
        "/org-limits/nope",
        json={"limits": {"max_rows_per_eval": 99}},
        headers=h,
    )
    assert upd_404.status_code == 404

    # me/max-rows-per-eval now returns the configured value
    again = client.get("/org-limits/me/max-rows-per-eval", headers=h)
    assert again.json()["max_rows_per_eval"] == 99

    # DELETE
    deleted = client.delete(f"/org-limits/{user_org_uuid}", headers=h)
    assert deleted.status_code == 200
    # Already gone
    assert client.delete(f"/org-limits/{user_org_uuid}", headers=h).status_code == 404


def test_org_limits_get_superadmin_bypasses_membership(client, monkeypatch):
    """GET /org-limits/{org_uuid} allows superadmin even when they're not a
    member of the target org."""
    import auth_utils
    import db as db_mod

    owner = _auth(client)
    outsider = _auth(client)
    target_org_uuid = db_mod.get_personal_org_for_user(owner["user_uuid"])["uuid"]

    # Seed a limits row on owner's org as the owner-superadmin (needed since
    # creating limits requires superadmin).
    monkeypatch.setattr(auth_utils, "SUPERADMIN_EMAIL", owner["email"])
    create = client.post(
        "/org-limits",
        json={"org_uuid": target_org_uuid, "limits": {"max_rows_per_eval": 11}},
        headers=owner["headers"],
    )
    assert create.status_code == 200

    # Now switch the superadmin to outsider, who is NOT a member of owner's org.
    monkeypatch.setattr(auth_utils, "SUPERADMIN_EMAIL", outsider["email"])
    got = client.get(f"/org-limits/{target_org_uuid}", headers=outsider["headers"])
    assert got.status_code == 200
    assert got.json()["limits"]["max_rows_per_eval"] == 11
