import os
import uuid
import asyncio
import logging
from typing import Literal, Optional, Dict, Any
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
from fastapi.responses import JSONResponse
from fastapi.openapi.docs import get_swagger_ui_html, get_redoc_html
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

from db import init_db, NameAlreadyExistsError
from auth_utils import get_current_user_id
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
from utils import (
    generate_presigned_upload_url,
    get_s3_output_config,
    PRESIGNED_URL_EXPIRY_SECONDS,
)
from job_recovery import recover_pending_jobs
from provider_status import provider_status_monitor


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
        s3_key, request.content_type, expiration=expiration
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


@app.api_route("/provider-status", methods=["GET", "HEAD"])
async def get_provider_status():
    """
    Return the latest cached status for all configured providers.

    A background task refreshes the cache by running `calibrate status`.
    Returns 200 if all providers pass and 503 if any provider failed or the
    cached result is missing/stale.
    """
    return await provider_status_monitor.response()


@app.get("/openrouter/providers")
async def list_openrouter_providers() -> Optional[Dict[str, Any]]:
    """
    List providers available on OpenRouter.

    If `OPENROUTER_API_KEY` is not set, returns `null` (OpenRouter is disabled).

    Otherwise, if `OPENROUTER_ALLOWED_PROVIDERS` is set (comma-separated provider
    slugs), fetches the canonical list from `https://openrouter.ai/api/v1/providers`,
    filters to that subset, and returns `{"providers": [{slug, name}, ...]}`.

    If `OPENROUTER_ALLOWED_PROVIDERS` is empty/unset, all providers are supported —
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
                "https://openrouter.ai/api/v1/providers",
                headers={"Authorization": f"Bearer {api_key}"},
            )
            response.raise_for_status()
            payload = response.json()
    except httpx.HTTPError as exc:
        raise HTTPException(
            status_code=502,
            detail=f"Failed to fetch OpenRouter providers: {exc}",
        )

    raw_providers = payload.get("data", []) if isinstance(payload, dict) else []
    providers = [
        {"slug": p.get("slug"), "name": p.get("name")}
        for p in raw_providers
        if p.get("slug") in allowed
    ]

    return {"providers": providers}


@app.get("/sentry-debug")
async def trigger_error():
    division_by_zero = 1 / 0
