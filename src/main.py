import os
import copy
import uuid
import asyncio
import logging
from typing import Literal, List, Optional, Dict, Any
from contextlib import asynccontextmanager

import httpx
from dotenv import load_dotenv

load_dotenv()

import sentry_sdk

# Initialize Sentry if DSN is configured
sentry_dsn = os.getenv("SENTRY_DSN")
if sentry_dsn:
    sentry_sdk.init(
        dsn=sentry_dsn,
        enable_logs=True,
        environment=os.getenv("SENTRY_ENVIRONMENT", "development"),
        traces_sample_rate=float(os.getenv("SENTRY_TRACES_SAMPLE_RATE", "1.0")),
        profiles_sample_rate=float(os.getenv("SENTRY_PROFILES_SAMPLE_RATE", "1.0")),
    )

import secrets

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.openapi.docs import get_swagger_ui_html, get_redoc_html
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

from db import init_db, NameAlreadyExistsError
from auth_utils import get_current_user_id
from traces.migrate import run_traces_migrations
from routers.auth import router as auth_router
from routers.agents import router as agents_router
from routers.tools import router as tools_router
from routers.agent_tools import router as agent_tools_router
from routers.stt import router as stt_router
from routers.tts import router as tts_router
from routers.tests import router as tests_router
from routers.agent_tests import router as agent_tests_router
from routers.personas import router as personas_router
from routers.scenarios import router as scenarios_router
from routers.evaluators import router as evaluators_router
from routers.simulations import router as simulations_router
from routers.jobs import router as jobs_router
from routers.datasets import router as datasets_router
from routers.org_limits import router as org_limits_router
from routers.public import router as public_router
from routers.annotation_tasks import router as annotation_tasks_router
from routers.annotators import router as annotators_router
from routers.annotation_agreement import router as annotation_agreement_router
from routers.organizations import router as organizations_router
from routers.api_keys import router as api_keys_router
from routers.traces import router as traces_router
from utils import (
    LOCAL_ARTIFACTS_URL_PREFIX,
    generate_presigned_upload_url,
    get_local_artifact_path,
    get_s3_output_config,
    is_local_object_storage,
    PRESIGNED_URL_EXPIRY_SECONDS,
)
from job_recovery import recover_pending_jobs
from provider_status import available_provider_names, provider_status_monitor


# Set up logger
logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup and shutdown events."""
    # Startup: Initialize database and recover in_progress jobs
    init_db()
    run_traces_migrations()
    logger.info("Checking for in_progress jobs to recover...")
    recover_pending_jobs()
    provider_status_task = asyncio.create_task(provider_status_monitor.refresh_loop())
    try:
        yield
    finally:
        provider_status_task.cancel()
        try:
            await provider_status_task
        except asyncio.CancelledError:
            pass
        logger.info("Application shutting down")


DOCS_USERNAME = os.getenv("DOCS_USERNAME", "admin")
DOCS_PASSWORD = os.getenv("DOCS_PASSWORD", "changeme")

docs_basic_auth = HTTPBasic()


def _verify_docs_access(
    credentials: HTTPBasicCredentials = Depends(docs_basic_auth),
):
    correct_username = secrets.compare_digest(credentials.username, DOCS_USERNAME)
    correct_password = secrets.compare_digest(credentials.password, DOCS_PASSWORD)
    if not (correct_username and correct_password):
        raise HTTPException(
            status_code=401,
            detail="Unauthorized",
            headers={"WWW-Authenticate": "Basic"},
        )


app = FastAPI(docs_url=None, redoc_url=None, openapi_url=None, lifespan=lifespan)


@app.exception_handler(NameAlreadyExistsError)
async def _name_already_exists_handler(_: Request, exc: NameAlreadyExistsError):
    """Convert DB-layer unique-name violations to a friendly 409.

    Routers wrap their write call with `db.name_uniqueness_guard("Test")`
    so the rare TOCTOU race past the API-layer `is_name_taken` pre-check
    still returns the same 409 shape the FE expects, instead of an
    uncaught IntegrityError surfacing as 500.
    """
    return JSONResponse(
        status_code=409,
        content={"detail": f"{exc.entity_label} name already exists"},
    )


@app.get("/docs", include_in_schema=False)
def custom_swagger_ui(_: HTTPBasicCredentials = Depends(_verify_docs_access)):
    return get_swagger_ui_html(openapi_url="/openapi.json", title="API Docs")


@app.get("/redoc", include_in_schema=False)
def custom_redoc(_: HTTPBasicCredentials = Depends(_verify_docs_access)):
    return get_redoc_html(openapi_url="/openapi.json", title="API ReDoc")


@app.get("/openapi.json", include_in_schema=False)
def custom_openapi(_: HTTPBasicCredentials = Depends(_verify_docs_access)):
    return app.openapi()


# --- Public API docs ------------------------------------------------------
# A no-auth subset of the docs covering ONLY endpoints that accept an `sk_`
# API key (those tagged "Public API"). Everything else stays behind the
# Basic-Auth'd /docs above. The schema is filtered from the full app schema,
# so it stays in sync automatically as routes change.
PUBLIC_API_TAG = "Public API"
# Name of the apiKey security scheme published on the public spec. Fern (Python SDK)
# and Speakeasy (CLI) both derive the required api_key auth param from this being
# the sole scheme.
PUBLIC_API_KEY_SCHEME = "ApiKeyAuth"


def _public_api_base_url() -> str:
    return os.getenv("PUBLIC_API_BASE_URL", "http://localhost:8000").rstrip("/")


def _collect_schema_refs(node: Any, acc: set) -> None:
    """Walk an OpenAPI fragment, collecting every `#/components/schemas/<Name>`
    schema name referenced (transitively) into ``acc``."""
    if isinstance(node, dict):
        ref = node.get("$ref")
        if isinstance(ref, str) and ref.startswith("#/components/schemas/"):
            acc.add(ref.rsplit("/", 1)[-1])
        for value in node.values():
            _collect_schema_refs(value, acc)
    elif isinstance(node, list):
        for item in node:
            _collect_schema_refs(item, acc)


def _is_freeform_object_schema(node: Any) -> bool:
    """True if ``node`` is a free-form object schema — ``type: object`` with
    ``additionalProperties`` and no declared ``properties`` (i.e. a
    ``Dict[str, Any]``). These carry no sub-fields to document."""
    return (
        isinstance(node, dict)
        and node.get("type") == "object"
        and bool(node.get("additionalProperties"))
        and not node.get("properties")
    )


def _is_freeform_schema(node: Any) -> bool:
    """True if ``node`` is a free-form object (`Dict[str, Any]`) OR an array of
    them (`List[Dict[str, Any]]`). Both render as a shapeless `object`/`object[]`
    chip with nothing to expand, so both should shed their auto-title."""
    if _is_freeform_object_schema(node):
        return True
    return (
        isinstance(node, dict)
        and node.get("type") == "array"
        and _is_freeform_object_schema(node.get("items", {}))
    )


def _strip_freeform_titles(node: Any) -> None:
    """Recursively delete Pydantic's auto-generated ``title`` from free-form
    fields — `Dict[str, Any]` and `List[Dict[str, Any]]`. Pydantic title-cases
    the field name (``config`` → ``"Config"``, ``tool_calls`` → ``"Tool Calls"``),
    which Mintlify surfaces as a fake type chip (`Config · object`,
    `Tool Calls · object[]`) even though no such named type exists and there's
    nothing to expand. Real component/model titles are untouched — only inline
    free-form blobs (including the `anyOf: [<freeform>, null]` wrapper that
    `Optional[...]` produces) lose their noise title."""
    if isinstance(node, dict):
        if "title" in node:
            branches = node.get("anyOf") or node.get("oneOf") or []
            non_null = [
                b
                for b in branches
                if not (isinstance(b, dict) and b.get("type") == "null")
            ]
            wraps_freeform = bool(non_null) and all(
                _is_freeform_schema(b) for b in non_null
            )
            if _is_freeform_schema(node) or wraps_freeform:
                node.pop("title", None)
        for value in node.values():
            _strip_freeform_titles(value)
    elif isinstance(node, list):
        for item in node:
            _strip_freeform_titles(item)


# Fields kept on the model (the frontend uses them) but hidden from the public
# spec and its generated SDK/code samples. Two kinds:
#   1. JWT-only request fields — API-key writes strip them server-side (see
#      agents._strip_verification_fields), so they're inert and misleading.
#   2. Backend-internal response fields — auto-increment link/pivot IDs and other
#      storage/tenant plumbing that a public API consumer (who works in UUIDs)
#      has no use for. Not part of the public contract.
# Keyed by component-schema name.
_PUBLIC_SPEC_HIDDEN_FIELDS: Dict[str, tuple] = {
    "AgentUpdate": ("connection_verified", "benchmark_models_verified"),
    # Auto-increment pivot-row IDs from the link endpoints — internal DB keys a
    # UUID-based public client has no use for. `ids` is too generic to strip
    # globally, so it's pinned to the link responses that return it.
    "AgentTestsCreateResponse": ("ids",),
    "AgentToolsCreateResponse": ("ids",),
}

# Distinctive internal field names stripped from every *evaluator* public
# component schema. These names only appear on evaluator-shaped models today, but
# matching them globally by name would silently drop any unrelated future field
# that happened to reuse one (e.g. a `kind` on some other resource). So the strip
# is scoped to evaluator schemas via `_is_evaluator_schema` — the concrete schema
# name is sometimes module-namespaced (e.g. `routers__evaluators__EvaluatorResponse`),
# which is why we match by schema name rather than pinning an exact list.
# `owner_user_id` is a raw tenant user ID; `live_version_index` is a UI-only array
# position (the live version is already identified by the `live_version_id` UUID);
# `kind` (single vs side_by_side) is a niche scoring mode not on the public surface.
_PUBLIC_SPEC_EVALUATOR_HIDDEN_FIELD_NAMES: frozenset = frozenset(
    {"owner_user_id", "live_version_index", "kind"}
)


def _is_evaluator_schema(name: str) -> bool:
    """True for evaluator-shaped component schemas, incl. module-namespaced ones
    (`routers__evaluators__EvaluatorResponse`) and the evaluator default-prompt
    response (`DefaultPromptResponse`)."""
    lowered = name.lower()
    return "evaluator" in lowered or lowered == "defaultpromptresponse"


def _drop_fields(schema: Dict[str, Any], fields) -> None:
    props = schema.get("properties")
    if props:
        for f in fields:
            props.pop(f, None)
    req = schema.get("required")
    if req:
        schema["required"] = [r for r in req if r not in fields]


def _strip_hidden_public_fields(schemas: Dict[str, Any]) -> None:
    """Hide backend-internal / JWT-only fields from the public component schemas."""
    for model, fields in _PUBLIC_SPEC_HIDDEN_FIELDS.items():
        schema = schemas.get(model)
        if schema:
            _drop_fields(schema, fields)
    if _PUBLIC_SPEC_EVALUATOR_HIDDEN_FIELD_NAMES:
        for name, schema in schemas.items():
            if isinstance(schema, dict) and _is_evaluator_schema(name):
                _drop_fields(schema, _PUBLIC_SPEC_EVALUATOR_HIDDEN_FIELD_NAMES)


def _build_public_openapi() -> Dict[str, Any]:
    full = app.openapi()
    public_paths: Dict[str, Any] = {}
    for path, ops in full.get("paths", {}).items():
        kept = {
            # PUBLIC_API_TAG is only a filter marker, not a display group — drop
            # it so each op renders under its router-level tag ("agents",
            # "agent-tests") instead of duplicating across both that tag AND a
            # "Public API" group. Copy the op so we don't mutate the cached full
            # schema shared with /docs.
            #
            # Force the API-key scheme as the SOLE auth on every public op. The
            # underlying dep (`get_org_jwt_or_api_key`) also accepts a JWT bearer,
            # but FastAPI's auto-generated `HTTPBearer` scheme would make Fern/Speakeasy emit
            # a REQUIRED `token` constructor arg in the SDK (with `api_key`
            # optional) — so `Calibrate(api_key=…)` would TypeError. Pinning
            # `PUBLIC_API_KEY_SCHEME` here makes `api_key` the one required auth
            # param and drops `token`. (JWT-bearer callers are the frontend, which
            # never uses the SDK, so nothing is lost.)
            method: {
                **op,
                "tags": [t for t in op["tags"] if t != PUBLIC_API_TAG],
                "security": [{PUBLIC_API_KEY_SCHEME: []}],
            }
            for method, op in ops.items()
            if isinstance(op, dict) and PUBLIC_API_TAG in op.get("tags", [])
        }
        if kept:
            public_paths[path] = kept

    # Include ONLY the component schemas the public paths actually reference, so
    # the anonymous page doesn't expose internal (JWT-only) model shapes. Resolve
    # transitively: a kept schema may $ref further schemas.
    all_schemas = full.get("components", {}).get("schemas", {})
    needed: set = set()
    _collect_schema_refs(public_paths, needed)
    queue = list(needed)
    while queue:
        name = queue.pop()
        nested: set = set()
        _collect_schema_refs(all_schemas.get(name, {}), nested)
        for dep in nested - needed:
            needed.add(dep)
            queue.append(dep)

    components: Dict[str, Any] = dict(full.get("components", {}))
    if all_schemas:
        # Deep-copy: the strip passes below mutate these schemas, and the source
        # objects are shared with the cached `app.openapi()` (internal /docs).
        components["schemas"] = copy.deepcopy(
            {name: schema for name, schema in all_schemas.items() if name in needed}
        )
    # Expose ONLY the API-key scheme on the public spec — see the per-op
    # `security` override above for why the auto-generated bearer scheme is
    # dropped. `X-API-Key: sk_…` is the SDK's auth path.
    components["securitySchemes"] = {
        PUBLIC_API_KEY_SCHEME: {
            "type": "apiKey",
            "in": "header",
            "name": "X-API-Key",
            "description": "Workspace API key. Create one under Workspace settings → API keys.",
        }
    }

    # Drop Pydantic's auto-titles on free-form `Dict[str, Any]` fields so
    # Mintlify renders them as plain `object | null` instead of a misleading
    # `Config`-style type chip. Real model titles in `components.schemas` stay.
    _strip_freeform_titles(public_paths)
    _strip_freeform_titles(components.get("schemas", {}))
    _strip_hidden_public_fields(components.get("schemas", {}))

    # Drop the optional `X-Org-UUID` header the shared `get_org_jwt_or_api_key`
    # dep adds. An API-key client is already scoped to one workspace by the key
    # itself, so the header is redundant and confusing on the anonymous public
    # page. `X-API-Key` (and every other param) stays.
    for ops in public_paths.values():
        for op in ops.values():
            params = op.get("parameters")
            if params:
                op["parameters"] = [
                    p for p in params if p.get("name") != "X-Org-UUID"
                ]

    return {
        "openapi": full.get("openapi", "3.1.0"),
        "info": {
            "title": "Calibrate Public API",
            "version": full.get("info", {}).get("version", "1.0.0"),
            "description": (
                "Programmatic API for CI/automation. Pass your key in the `X-API-Key` "
                "header."
            ),
        },
        "servers": [
            {
                "url": _public_api_base_url(),
                "description": "API",
            }
        ],
        "components": components,
        "paths": public_paths,
    }


@app.get("/public-api/openapi.json", include_in_schema=False)
def public_openapi():
    return _build_public_openapi()


@app.get("/public-api/docs", include_in_schema=False)
def public_swagger_ui():
    return get_swagger_ui_html(
        openapi_url="/public-api/openapi.json", title="Calibrate Public API"
    )


# Include routers
app.include_router(auth_router)
app.include_router(agents_router)
app.include_router(tools_router)
app.include_router(agent_tools_router)
app.include_router(stt_router)
app.include_router(tts_router)
app.include_router(tests_router)
app.include_router(agent_tests_router)
app.include_router(personas_router)
app.include_router(scenarios_router)
app.include_router(evaluators_router)
app.include_router(simulations_router)
app.include_router(jobs_router)
app.include_router(datasets_router)
app.include_router(org_limits_router)
app.include_router(annotation_tasks_router)
app.include_router(annotators_router)
app.include_router(annotation_agreement_router)
app.include_router(organizations_router)
app.include_router(api_keys_router)
app.include_router(traces_router)
# Public (no-auth) sharing endpoints — must be registered without any auth dependency
app.include_router(public_router)

# Configure CORS allowed origins from environment variable
# CORS_ALLOWED_ORIGINS can be comma-separated list (e.g., "http://localhost:3000,https://app.example.com")
# Defaults to ["*"] if not set
cors_origins_env = os.getenv("CORS_ALLOWED_ORIGINS", "*")
cors_allowed_origins = [origin.strip() for origin in cors_origins_env.split(",")]

app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class PresignedURLRequest(BaseModel):
    task_type: Literal["stt", "tts", "agent"]
    content_type: str  # e.g., "audio/wav", "text/csv"
    extension: str  # e.g., "wav", "csv"


class PresignedURLResponse(BaseModel):
    presigned_url: str
    s3_path: str
    expires_in: int  # expiration time in seconds


@app.get("/")
@app.head("/")
def read_root():
    return {"message": "Health check successful!"}


@app.api_route(
    f"{LOCAL_ARTIFACTS_URL_PREFIX}{{artifact_path:path}}",
    methods=["GET", "PUT"],
    include_in_schema=False,
)
async def local_artifact(artifact_path: str, request: Request):
    """Development-only local stand-in for S3 presigned upload/download URLs."""
    if not is_local_object_storage():
        raise HTTPException(status_code=404, detail="Not found")

    try:
        path = get_local_artifact_path(artifact_path)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid artifact path")

    if request.method == "PUT":
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(await request.body())
        return Response(status_code=204)

    if not path.is_file():
        raise HTTPException(status_code=404, detail="Artifact not found")
    return FileResponse(path)


@app.post("/presigned-url", response_model=PresignedURLResponse)
async def get_presigned_url(
    request: PresignedURLRequest,
    user_id: str = Depends(get_current_user_id),
):
    """
    Generate a presigned URL for uploading files to S3.

    Requires a valid JWT.

    The file will be stored at: bucket/task_type/media/UUID.extension

    Args:
        request: Contains task_type (stt, tts, agent) and file_extension

    Returns:
        Presigned URL, S3 path, and expiration time
    """
    # Validate file extension (remove leading dot if present)
    file_extension = request.extension
    if not file_extension:
        raise HTTPException(
            status_code=400,
            detail="File extension cannot be empty",
        )

    # Validate task type
    if request.task_type not in ["stt", "tts", "agent"]:
        raise HTTPException(
            status_code=400,
            detail="task_type must be one of: stt, tts, agent",
        )

    # Get S3 bucket from environment
    try:
        s3_bucket = get_s3_output_config()
    except ValueError as e:
        raise HTTPException(status_code=500, detail=str(e))

    # Generate UUID for unique file name
    file_uuid = str(uuid.uuid4())

    # Construct S3 key: task_type/media/UUID.extension (no prefix)
    s3_key = f"{request.task_type}/media/{file_uuid}.{file_extension}"

    # Generate presigned URL (expires in 1 hour)
    expiration = PRESIGNED_URL_EXPIRY_SECONDS  # 1 hour in seconds

    presigned_url = generate_presigned_upload_url(
        s3_key,
        request.content_type,
        expiration=expiration,
    )
    if not presigned_url:
        raise HTTPException(
            status_code=500,
            detail="Failed to generate presigned URL",
        )

    return PresignedURLResponse(
        presigned_url=presigned_url,
        s3_path=f"s3://{s3_bucket}/{s3_key}",
        expires_in=expiration,
    )


@app.get("/provider-status")
@app.head("/provider-status")
async def get_provider_status(request: Request, refresh: bool = False):
    """
    Return the latest cached status for all configured providers.

    A background task refreshes the cache by running `calibrate status`.
    Pass ``?refresh=true`` on GET to ignore the cache and run a fresh check
    synchronously (may take up to ``PROVIDER_STATUS_CHECK_TIMEOUT_SECONDS``).
    Returns 200 if all providers pass and 503 if any provider failed or the
    cached result is missing/stale.
    """
    force_refresh = refresh and request.method == "GET"
    return await provider_status_monitor.response(force_refresh=force_refresh)


@app.get("/providers")
async def list_available_providers() -> Dict[str, Any]:
    """
    List providers enabled by the current environment's API keys.

    A provider is available when every environment variable it requires is set,
    so the frontend can show only the providers it can actually run. This is a
    cheap config check — it does not verify the keys work (see `/provider-status`
    for live reachability).
    """
    return {"providers": available_provider_names()}


@app.get("/openrouter/providers")
async def list_openrouter_providers() -> Optional[Dict[str, Any]]:
    """
    List model authors available on OpenRouter.

    If `OPENROUTER_API_KEY` is not set, returns `null` (OpenRouter is disabled).

    Otherwise, if `OPENROUTER_ALLOWED_PROVIDERS` is set (comma-separated author
    slugs), fetches `https://openrouter.ai/api/v1/models`, derives each author from
    the model-id prefix (`google/gemini-2.5-pro` → `google`), filters to the allowed
    subset, and returns `{"providers": [{slug, name}, ...]}`.

    The slug is the model-id author prefix — the same key the model dropdown groups
    by — NOT an OpenRouter serving-provider slug (`google`, not `google-ai-studio`).

    If `OPENROUTER_ALLOWED_PROVIDERS` is empty/unset, all authors are supported —
    returns `{"providers": "all"}`.
    """
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        return None

    allowed_env = os.getenv("OPENROUTER_ALLOWED_PROVIDERS", "")
    allowed = {s.strip() for s in allowed_env.split(",") if s.strip()}

    if not allowed:
        return {"providers": "all"}

    try:
        async with httpx.AsyncClient(timeout=15) as client:
            response = await client.get(
                "https://openrouter.ai/api/v1/models",
                headers={"Authorization": f"Bearer {api_key}"},
            )
            response.raise_for_status()
            payload = response.json()
    except httpx.HTTPError as exc:
        raise HTTPException(
            status_code=502,
            detail=f"Failed to fetch OpenRouter models: {exc}",
        )

    raw_models = payload.get("data", []) if isinstance(payload, dict) else []
    providers: List[Dict[str, str]] = []
    seen: set[str] = set()
    for model in raw_models:
        model_id = model.get("id") or ""
        slug = model_id.split("/", 1)[0]
        if slug not in allowed or slug in seen:
            continue
        seen.add(slug)
        # Model names are "Author: Model" — the author label is display-only.
        display = (model.get("name") or "").split(":", 1)[0].strip() or slug
        providers.append({"slug": slug, "name": display})

    return {"providers": providers}


@app.get("/sentry-debug")
async def trigger_error():
    division_by_zero = 1 / 0
