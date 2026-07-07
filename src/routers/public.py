"""Public endpoints for shared eval and run results.

View STT/TTS runs, agent tests, benchmarks, simulations, and annotation
results via share links — no sign-in required. Annotator labelling jobs use
a separate token for read-write access.
"""

import logging
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, Query, Path
from pydantic import BaseModel, Field

from db import (
    DEFAULT_EVALUATORS_SEED,
    get_evaluator_by_slug,
    get_evaluator_version,
    get_job_by_share_token,
    get_agent_test_job_by_share_token,
    get_simulation_job_by_share_token,
    get_simulation_jobs_for_simulation,
    get_annotation_job_by_token,
    get_annotation_job_by_view_token,
    get_annotation_task,
    get_evaluator_ids_for_job,
    get_evaluators_for_job,
    get_annotator,
    get_job_items,
    get_annotations_for_job,
    upsert_annotation,
    update_annotation_job_status,
    get_evaluator_runs_for_job,
    get_eval_job_items,
)
from utils import (
    TaskStatus,
    AnnotationStatus,
    SimulationRunType,
    AnnotationTaskTypeLiteral,
    EvaluatorTypeLiteral,
    OutputTypeLiteral,
    ProviderResult,
    enrich_evaluator_runs_with_current_names,
    generate_presigned_download_url,
    get_s3_output_config,
    load_evaluator_metric_key_map,
    normalize_metrics,
    post_process_provider_results,
    presign_audio_path,
)

# Re-use the audio URL helper from simulations (no circular import risk)
from routers.simulations import (
    SimulationEvaluatorRef,
    apply_simulation_job_evaluator_enrichment,
    _get_audio_urls_from_s3_key,
)
from routers.agent_tests import (
    _enrich_test_results_with_evaluators,
    _enrich_model_results_with_evaluators,
    _build_evaluators_block_for_test_run,
)
from routers.annotation_tasks import (
    _build_evaluators_block_for_eval_job,
    _strip_details_evaluators,
    _strip_run_evaluator_blocks,
    _human_agreement_for_run,
    _shape_eval_job_for_response,
)
from annotation_eval_runner import ANNOTATION_EVAL_JOB_TYPE

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/public", tags=["public"])

_EXAMPLE_TASK_UUID = "a3b2c1d0-e5f4-3210-abcd-ef1234567890"
_EXAMPLE_DATASET_UUID = "f47ac10b-58cc-4372-a567-0e02b2c3d479"
_EXAMPLE_EVALUATOR_UUID = "6ba7b810-9dad-11d1-80b4-00c04fd430c8"
_EXAMPLE_JOB_UUID = "b1c2d3e4-f5a6-7890-bcde-f12345678901"
_EXAMPLE_ITEM_UUID = "c1d2e3f4-a5b6-4789-abcd-ef0123456789"
_EXAMPLE_ANNOTATION_TASK_UUID = "d4e5f6a7-b8c9-4012-def0-234567890abc"


# ---------------------------------------------------------------------------
# Response models
# ---------------------------------------------------------------------------


class PublicSTTResponse(BaseModel):
    task_id: str = Field(
        min_length=36,
        max_length=36,
        description="STT eval job ID",
        examples=[_EXAMPLE_TASK_UUID],
    )
    status: TaskStatus = Field(description="Job status, e.g. `in_progress`, `done`, `failed`")
    language: Optional[str] = Field(None, description="Evaluated language code; `null` if unset")
    dataset_id: Optional[str] = Field(
        None,
        min_length=36,
        max_length=36,
        description="Source dataset ID; `null` if unavailable",
        examples=[_EXAMPLE_DATASET_UUID],
    )
    dataset_name: Optional[str] = Field(None, description="Source dataset name; `null` if unavailable")
    provider_results: Optional[List[ProviderResult]] = Field(
        None, description="Per-provider transcription results and metrics; `null` until available"
    )
    leaderboard_summary: Optional[List[Dict[str, Any]]] = Field(
        None, description="Ranked provider comparison; `null` if not yet computed"
    )
    error: Optional[str] = Field(None, description="Failure message; `null` on success")


class PublicTTSResponse(BaseModel):
    task_id: str = Field(
        min_length=36,
        max_length=36,
        description="TTS eval job ID",
        examples=[_EXAMPLE_TASK_UUID],
    )
    status: TaskStatus = Field(description="Job status, e.g. `in_progress`, `done`, `failed`")
    language: Optional[str] = Field(None, description="Evaluated language code; `null` if unset")
    dataset_id: Optional[str] = Field(
        None,
        min_length=36,
        max_length=36,
        description="Source dataset ID; `null` if unavailable",
        examples=[_EXAMPLE_DATASET_UUID],
    )
    dataset_name: Optional[str] = Field(None, description="Source dataset name; `null` if unavailable")
    provider_results: Optional[List[ProviderResult]] = Field(
        None, description="Per-provider synthesis results and metrics; `null` until available"
    )
    leaderboard_summary: Optional[List[Dict[str, Any]]] = Field(
        None, description="Ranked provider comparison; `null` if not yet computed"
    )
    error: Optional[str] = Field(None, description="Failure message; `null` on success")


class PublicTestRunResponse(BaseModel):
    task_id: str = Field(
        min_length=36,
        max_length=36,
        description="LLM test run job ID",
        examples=[_EXAMPLE_TASK_UUID],
    )
    status: TaskStatus = Field(description="Run status, e.g. `in_progress`, `done`, `failed`")
    total_tests: Optional[int] = Field(None, description="Total test cases in the run; `null` until known")
    passed: Optional[int] = Field(None, description="Test cases that passed; `null` until computed")
    failed: Optional[int] = Field(None, description="Test cases that failed; `null` until computed")
    # Top-level evaluator block — name/description/output_type/rubric
    # shared across every judge_results row. Rows reference back via
    # `evaluator_uuid` so the rubric isn't duplicated per test case.
    evaluators: Optional[List[Dict[str, Any]]] = Field(
        None,
        description="Shared evaluator definitions (name, description, output type, rubric); rows reference these by evaluator ID",
    )
    results: Optional[List[Dict[str, Any]]] = Field(
        None, description="Per-test-case results; `null` until the run produces them"
    )
    # Aggregated latency/cost/total_tokens: {mean, min, max, count}. Values are
    # `Any` — don't assume int: total_tokens is per-run an int but its aggregate
    # `mean` can be fractional. None when calibrate omits it (eval-only / no cost
    # or token usage reported).
    latency_ms: Optional[Dict[str, Any]] = Field(
        None,
        description="Aggregated latency (`{p50, p95, p99, count}`); `null` for eval-only runs or when not reported",
    )
    cost: Optional[Dict[str, Any]] = Field(
        None,
        description="Aggregated cost in USD (`{mean, min, max, count}`); `null` when no cost is reported",
    )
    total_tokens: Optional[Dict[str, Any]] = Field(
        None,
        description="Aggregated token usage (`{mean, min, max, count}`); `null` when not reported",
    )
    error: bool = Field(False, description="`true` if the run failed")


class PublicBenchmarkResponse(BaseModel):
    task_id: str = Field(
        min_length=36,
        max_length=36,
        description="LLM benchmark job ID",
        examples=[_EXAMPLE_TASK_UUID],
    )
    status: TaskStatus = Field(description="Run status, e.g. `in_progress`, `done`, `failed`")
    # Same as PublicTestRunResponse.evaluators — shared by every model's
    # test_results inside model_results[] (all models run the same suite).
    evaluators: Optional[List[Dict[str, Any]]] = Field(
        None,
        description="Shared evaluator definitions referenced by every model's results; `null` until available",
    )
    model_results: Optional[List[Dict[str, Any]]] = Field(
        None, description="Per-model test results (all models run the same suite); `null` until available"
    )
    leaderboard_summary: Optional[List[Dict[str, Any]]] = Field(
        None, description="Ranked model comparison; `null` if not yet computed"
    )
    error: bool = Field(False, description="`true` if the run failed")


class PublicSimulationRunResponse(BaseModel):
    task_id: str = Field(
        min_length=36,
        max_length=36,
        description="Simulation run job ID",
        examples=[_EXAMPLE_TASK_UUID],
    )
    name: str = Field(description="Display name for the run, e.g. `Run 1`")
    status: TaskStatus = Field(description="Run status, e.g. `in_progress`, `done`, `failed`")
    type: SimulationRunType = Field(description="Simulation type (`text` | `voice`)")
    updated_at: str = Field(description="When the run was last updated (ISO 8601 UTC)")
    total_simulations: Optional[int] = Field(None, description="Number of simulations in the run; `null` until known")
    metrics: Optional[Dict[str, Any]] = Field(None, description="Aggregated run metrics; `null` until computed")
    simulation_results: Optional[List[Dict[str, Any]]] = Field(
        None,
        description="Per-simulation results; voice runs include freshly presigned audio URLs. `null` until available",
    )
    evaluators: Optional[List[SimulationEvaluatorRef]] = Field(
        None, description="Evaluators applied to the run; `null` if none"
    )
    error: Optional[str] = Field(None, description="Failure message; `null` on success")


class PublicAnnotationEvalTaskRef(BaseModel):
    uuid: str = Field(
        min_length=36,
        max_length=36,
        description="Annotation task ID",
        examples=[_EXAMPLE_ANNOTATION_TASK_UUID],
    )
    name: str = Field(description="Annotation task name")
    type: AnnotationTaskTypeLiteral = Field(
        description="Annotation task type (e.g. `stt`, `llm`, `conversation`)"
    )
    description: Optional[str] = Field(None, description="Task description; `null` if unset")


class PublicAnnotationEvalResponse(BaseModel):
    task_id: str = Field(
        min_length=36,
        max_length=36,
        description="Parent annotation task ID",
        examples=[_EXAMPLE_ANNOTATION_TASK_UUID],
    )
    job_uuid: str = Field(
        min_length=36,
        max_length=36,
        description="Annotation evaluator-run job ID",
        examples=[_EXAMPLE_JOB_UUID],
    )
    status: AnnotationStatus = Field(description="Run status; share links only expose completed runs")
    created_at: Optional[str] = Field(None, description="When the run was created (ISO 8601 UTC); `null` if unset")
    completed_at: Optional[str] = Field(None, description="When the run finished (ISO 8601 UTC); `null` if unset")
    updated_at: Optional[str] = Field(None, description="When the run was last updated (ISO 8601 UTC); `null` if unset")
    task: PublicAnnotationEvalTaskRef = Field(description="Parent annotation task summary")
    # `details` mirrors the authenticated GET shape so a shared FE component
    # can read `job.details?.evaluators` against either endpoint without
    # branching. Only the safe-to-share keys are forwarded — operational
    # fields (pid, pgid, s3_prefix, user_id) are intentionally stripped.
    details: Optional[Dict[str, Any]] = Field(
        None,
        description="Safe-to-share job metadata; operational fields are stripped",
    )
    # Top-level mirrors retained for the existing public consumers that
    # were already reading them. Both shapes carry the same data.
    evaluators: Optional[List[Dict[str, Any]]] = Field(
        None, description="Evaluator definitions applied in the run; `null` if none"
    )
    item_count: Optional[int] = Field(None, description="Number of items evaluated; `null` if unknown")
    items: Optional[List[Dict[str, Any]]] = Field(None, description="Evaluated items; `null` if none")
    runs: Optional[List[Dict[str, Any]]] = Field(
        None,
        description="Per-item evaluator run rows; each keys back into `evaluators[]` by evaluator and version ID",
    )
    human_agreement: Optional[Dict[str, Any]] = Field(
        None, description="Human-vs-evaluator agreement metrics; `null` if unavailable"
    )
    error: Optional[str] = Field(None, description="Failure message; `null` on success")


class PublicDefaultEvaluatorVersionResponse(BaseModel):
    output_config: Optional[Dict[str, Any]] = Field(
        None, description="Rubric config (scale values/labels/descriptions/colors); `null` if unset"
    )


class PublicDefaultEvaluatorResponse(BaseModel):
    uuid: str = Field(
        min_length=36,
        max_length=36,
        description="Evaluator ID",
        examples=[_EXAMPLE_EVALUATOR_UUID],
    )
    name: str = Field(description="Evaluator display name")
    description: Optional[str] = Field(None, description="Evaluator description; `null` if unset")
    evaluator_type: EvaluatorTypeLiteral = Field(
        description="Semantic category (`stt`, `tts`, `llm`, `llm-general`, `conversation`)"
    )
    output_type: OutputTypeLiteral = Field(description="Output shape (`binary` | `rating`)")
    live_version: Optional[PublicDefaultEvaluatorVersionResponse] = Field(
        None, description="Public-safe fields of the live version; `null` if none is set"
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _build_tts_provider_results_with_presigned_urls(
    provider_results: List[Dict[str, Any]],
    status: str,
) -> List[Dict[str, Any]]:
    """
    For DONE/FAILED TTS jobs regenerate presigned download URLs for audio
    entries whose audio_path is an S3 key rather than an http URL.
    Mirrors the logic in routers/tts.py::get_tts_evaluation_status.
    """
    if status not in (TaskStatus.DONE.value, TaskStatus.FAILED.value):
        return provider_results

    for provider_result in provider_results:
        if provider_result.get("results"):
            for result_row in provider_result["results"]:
                if "audio_path" in result_row and result_row["audio_path"]:
                    audio_s3_key = result_row["audio_path"]
                    if audio_s3_key.startswith("http") or audio_s3_key.startswith(
                        "s3://"
                    ):
                        continue
                    presigned_url = generate_presigned_download_url(audio_s3_key)
                    if presigned_url:
                        result_row["audio_path"] = presigned_url

    return provider_results


def _build_simulation_results_with_presigned_urls(
    job: Dict[str, Any],
    simulation_results: List[Dict[str, Any]],
    status: str,
) -> List[Dict[str, Any]]:
    """
    For DONE voice simulations regenerate presigned audio URLs on-the-fly.
    Mirrors the logic in routers/simulations.py::get_simulation_run_status.
    """
    if job.get("type") != "voice" or not simulation_results:
        return simulation_results

    if status == TaskStatus.DONE.value:
        try:
            s3_bucket = get_s3_output_config()
            for sim_result in simulation_results:
                audios_s3_key_prefix = sim_result.get("audios_s3_path")
                if audios_s3_key_prefix:
                    audio_urls = _get_audio_urls_from_s3_key(
                        audios_s3_key_prefix,
                        s3_bucket,
                        transcript=sim_result.get("transcript"),
                    )
                    sim_result["audio_urls"] = audio_urls

                conversation_wav_s3_key = sim_result.get("conversation_wav_s3_key")
                if conversation_wav_s3_key:
                    conversation_wav_url = generate_presigned_download_url(
                        conversation_wav_s3_key, bucket=s3_bucket
                    )
                    sim_result["conversation_wav_url"] = conversation_wav_url or ""
                else:
                    sim_result["conversation_wav_url"] = ""
        except Exception as exc:
            logger.warning(f"Failed to generate audio URLs for public endpoint: {exc}")

    return simulation_results


def _get_simulation_run_name(job: Dict[str, Any]) -> str:
    """Return 'Run N' by looking at the job's position among sibling jobs."""
    simulation_id = job.get("simulation_id")
    if not simulation_id:
        return "Run 1"
    all_jobs = get_simulation_jobs_for_simulation(simulation_id)
    sorted_jobs = sorted(all_jobs, key=lambda j: j.get("created_at", ""))
    for idx, j in enumerate(sorted_jobs, start=1):
        if j["uuid"] == job["uuid"]:
            return f"Run {idx}"
    return "Run 1"


def _ensure_valid_public_share_token(share_token: str) -> None:
    """Allow metadata reads only for callers that already have a valid public share token.

    `get_job_by_share_token` covers stt-eval, tts-eval, and annotation-eval —
    they all live in the generic `jobs` table."""
    if (
        get_job_by_share_token(share_token)
        or get_agent_test_job_by_share_token(share_token)
        or get_simulation_job_by_share_token(share_token)
    ):
        return
    raise HTTPException(status_code=404, detail="Not found")


def _parse_evaluator_types(types: Optional[str]) -> Optional[set[str]]:
    if types is None or not types.strip():
        return None

    allowed = {"stt", "tts", "llm", "llm-general", "conversation"}
    parsed = {item.strip() for item in types.split(",") if item.strip()}
    invalid = parsed - allowed
    if invalid:
        raise HTTPException(
            status_code=400,
            detail=f"types must contain only: {', '.join(sorted(allowed))}",
        )
    return parsed


def _public_default_evaluator_response(
    evaluator: Dict[str, Any],
) -> PublicDefaultEvaluatorResponse:
    live_version = None
    if evaluator.get("live_version_id"):
        version = get_evaluator_version(evaluator["live_version_id"])
        if version and version.get("evaluator_id") == evaluator["uuid"]:
            live_version = PublicDefaultEvaluatorVersionResponse(
                output_config=version.get("output_config")
            )

    return PublicDefaultEvaluatorResponse(
        uuid=evaluator["uuid"],
        name=evaluator["name"],
        description=evaluator.get("description"),
        evaluator_type=evaluator.get("evaluator_type", "llm"),
        output_type=evaluator.get("output_type", "binary"),
        live_version=live_version,
    )


# ---------------------------------------------------------------------------
# Public GET endpoints (no auth)
# ---------------------------------------------------------------------------


@router.get(
    "/evaluators/defaults",
    response_model=List[PublicDefaultEvaluatorResponse],
    summary="List public default evaluators",
)
async def get_public_default_evaluators(
    share_token: str = Query(..., min_length=1, description="Share token that grants access to the linked run"),
    types: Optional[str] = Query(
        None,
        description="Evaluator types to include, comma-separated (`stt`, `tts`, `llm`, `llm-general`, `conversation`); omit for all",
    ),
):
    """List default evaluator metadata when you have a valid share token."""
    _ensure_valid_public_share_token(share_token)
    requested_types = _parse_evaluator_types(types)

    defaults: List[PublicDefaultEvaluatorResponse] = []
    for seed in DEFAULT_EVALUATORS_SEED:
        if requested_types is not None and seed["evaluator_type"] not in requested_types:
            continue
        evaluator = get_evaluator_by_slug(seed["slug"])
        if evaluator and evaluator.get("org_uuid") is None:
            defaults.append(_public_default_evaluator_response(evaluator))
    return defaults


@router.get("/stt/{share_token}", response_model=PublicSTTResponse, summary="Get shared STT run")
async def get_public_stt(
    share_token: str = Path(description="Share token for the STT run"),
):
    """Get a shared STT evaluation result."""
    job = get_job_by_share_token(share_token, job_type="stt-eval")
    if not job:
        raise HTTPException(status_code=404, detail="Not found")

    task_id = job["uuid"]
    status = job["status"]
    results = job.get("results") or {}
    details = job.get("details") or {}

    provider_results = results.get("provider_results") or []

    # Normalize metrics format for backward compatibility (list -> dict)
    for pr in provider_results:
        if pr.get("metrics"):
            pr["metrics"] = normalize_metrics(pr["metrics"])

    enrich_evaluator_runs_with_current_names(
        provider_results, details.get("evaluators") or []
    )

    # Same canonical post-processing as the authenticated endpoints — public
    # callers see the same shape (evaluator_outputs[uuid], typed values,
    # evaluator_runs[].aggregate) instead of the legacy flat-keyed fallback.
    # `evaluator_id_by_metric_key` (read from on-disk config.json) is what
    # makes the per-row lift safe against reserved-column collisions —
    # without it the helper falls back to snapshot display names and an
    # evaluator named e.g. `wer` would lift the built-in WER value into
    # `evaluator_outputs[uuid].value`. Authenticated handlers already pass
    # this; the public mirrors must too or they'll disagree on the same run.
    post_process_provider_results(
        provider_results,
        evaluator_snapshots=details.get("evaluators") or [],
        evaluator_id_by_metric_key=load_evaluator_metric_key_map(details),
    )

    # Enrich result rows with presigned audio URLs from the dataset
    audio_paths = details.get("audio_paths", [])
    if audio_paths:
        # Collect IDs actually present in results to avoid unnecessary presigning
        needed_ids: set[str] = set()
        for pr in provider_results:
            for row in pr.get("results") or []:
                if row.get("id"):
                    needed_ids.add(row["id"])

        audio_url_map = {}
        for idx, path in enumerate(audio_paths):
            audio_id = f"audio_{idx + 1}"
            if audio_id in needed_ids:
                audio_url_map[audio_id] = presign_audio_path(path)

        for pr in provider_results:
            for row in pr.get("results") or []:
                row["audio_url"] = audio_url_map.get(row.get("id", ""))

    return PublicSTTResponse(
        task_id=task_id,
        status=status,
        language=details.get("language"),
        dataset_id=details.get("dataset_id"),
        dataset_name=details.get("dataset_name"),
        provider_results=provider_results or None,
        leaderboard_summary=results.get("leaderboard_summary"),
        error=results.get("error"),
    )


@router.get("/tts/{share_token}", response_model=PublicTTSResponse, summary="Get shared TTS run")
async def get_public_tts(
    share_token: str = Path(description="Share token for the TTS run"),
):
    """Get a shared TTS evaluation result."""
    job = get_job_by_share_token(share_token, job_type="tts-eval")
    if not job:
        raise HTTPException(status_code=404, detail="Not found")

    task_id = job["uuid"]
    status = job["status"]
    results = job.get("results") or {}
    details = job.get("details") or {}

    provider_results = results.get("provider_results") or []

    # Normalize metrics format for backward compatibility (list -> dict)
    for pr in provider_results:
        if pr.get("metrics"):
            pr["metrics"] = normalize_metrics(pr["metrics"])

    enrich_evaluator_runs_with_current_names(
        provider_results, details.get("evaluators") or []
    )

    # Same canonical post-processing as the authenticated endpoints — public
    # callers see the same shape (evaluator_outputs[uuid], typed values,
    # evaluator_runs[].aggregate) instead of the legacy flat-keyed fallback.
    # `evaluator_id_by_metric_key` (read from on-disk config.json) is what
    # makes the per-row lift safe against reserved-column collisions —
    # without it the helper falls back to snapshot display names and an
    # evaluator named e.g. `wer` would lift the built-in WER value into
    # `evaluator_outputs[uuid].value`. Authenticated handlers already pass
    # this; the public mirrors must too or they'll disagree on the same run.
    post_process_provider_results(
        provider_results,
        evaluator_snapshots=details.get("evaluators") or [],
        evaluator_id_by_metric_key=load_evaluator_metric_key_map(details),
    )

    # Regenerate presigned audio URLs for completed/failed jobs
    provider_results = _build_tts_provider_results_with_presigned_urls(
        provider_results, status
    )

    return PublicTTSResponse(
        task_id=task_id,
        status=status,
        language=details.get("language"),
        dataset_id=details.get("dataset_id"),
        dataset_name=details.get("dataset_name"),
        provider_results=provider_results or None,
        leaderboard_summary=results.get("leaderboard_summary"),
        error=results.get("error"),
    )


@router.get("/test-run/{share_token}", response_model=PublicTestRunResponse, summary="Get shared test run")
async def get_public_test_run(
    share_token: str = Path(description="Share token for the LLM test run"),
):
    """Get a shared LLM test run result."""
    job = get_agent_test_job_by_share_token(share_token, job_type="llm-unit-test")
    if not job:
        raise HTTPException(status_code=404, detail="Not found")

    task_id = job["uuid"]
    status = job["status"]
    results = job.get("results") or {}
    details = job.get("details") or {}

    evaluators_snapshot = details.get("evaluators_by_test_id") or {}
    evaluator_cache: Dict[str, Optional[Dict[str, Any]]] = {}
    _enrich_test_results_with_evaluators(
        results.get("test_results"), evaluators_snapshot, evaluator_cache
    )
    evaluators_block = _build_evaluators_block_for_test_run(
        evaluators_snapshot,
        test_results=results.get("test_results"),
        evaluator_cache=evaluator_cache,
    )

    return PublicTestRunResponse(
        task_id=task_id,
        status=status,
        total_tests=results.get("total_tests"),
        passed=results.get("passed"),
        failed=results.get("failed"),
        evaluators=evaluators_block or None,
        results=results.get("test_results"),
        latency_ms=results.get("latency_ms"),
        cost=results.get("cost"),
        total_tokens=results.get("total_tokens"),
        error=bool(results.get("error")),
    )


@router.get("/benchmark/{share_token}", response_model=PublicBenchmarkResponse, summary="Get shared benchmark")
async def get_public_benchmark(
    share_token: str = Path(description="Share token for the LLM benchmark run"),
):
    """Get a shared LLM benchmark result."""
    job = get_agent_test_job_by_share_token(share_token, job_type="llm-benchmark")
    if not job:
        raise HTTPException(status_code=404, detail="Not found")

    task_id = job["uuid"]
    status = job["status"]
    results = job.get("results") or {}
    details = job.get("details") or {}

    evaluators_snapshot = details.get("evaluators_by_test_id") or {}
    evaluator_cache: Dict[str, Optional[Dict[str, Any]]] = {}
    _enrich_model_results_with_evaluators(
        results.get("model_results"), evaluators_snapshot, evaluator_cache
    )
    evaluators_block = _build_evaluators_block_for_test_run(
        evaluators_snapshot,
        model_results=results.get("model_results"),
        evaluator_cache=evaluator_cache,
    )

    return PublicBenchmarkResponse(
        task_id=task_id,
        status=status,
        evaluators=evaluators_block or None,
        model_results=results.get("model_results"),
        leaderboard_summary=results.get("leaderboard_summary"),
        error=bool(results.get("error")),
    )


@router.get("/simulation-run/{share_token}", response_model=PublicSimulationRunResponse, summary="Get shared simulation run")
async def get_public_simulation_run(
    share_token: str = Path(description="Share token for the simulation run"),
):
    """Get a shared simulation run result."""
    job = get_simulation_job_by_share_token(share_token)
    if not job:
        raise HTTPException(status_code=404, detail="Not found")

    task_id = job["uuid"]
    status = job["status"]
    results = job.get("results") or {}

    simulation_results = results.get("simulation_results") or []
    simulation_results = _build_simulation_results_with_presigned_urls(
        job, simulation_results, status
    )

    evaluators_out, simulation_results = apply_simulation_job_evaluator_enrichment(
        job.get("details") or {}, simulation_results
    )

    run_name = _get_simulation_run_name(job)

    return PublicSimulationRunResponse(
        task_id=task_id,
        name=run_name,
        status=status,
        type=job["type"],
        updated_at=job["updated_at"],
        total_simulations=results.get("total_simulations"),
        metrics=results.get("metrics"),
        simulation_results=simulation_results or None,
        evaluators=evaluators_out,
        error=results.get("error"),
    )


# ---------------------------------------------------------------------------
# Annotation evaluator-run jobs (public, share_token toggled by owner)
# ---------------------------------------------------------------------------


@router.get(
    "/annotation-eval/{share_token}",
    response_model=PublicAnnotationEvalResponse,
    summary="Get shared annotation eval run",
)
async def get_public_annotation_eval(
    share_token: str = Path(description="Share token for the annotation evaluator-run job"),
):
    """Get a shared annotation evaluator-run result."""
    # Only `done` runs are exposed; in-flight or failed runs return 404.
    job = get_job_by_share_token(share_token, job_type=ANNOTATION_EVAL_JOB_TYPE)
    if not job:
        raise HTTPException(status_code=404, detail="Not found")

    # Defense-in-depth: never expose an in-flight or failed annotation-eval
    # run via the public link, even if some other code path managed to flip
    # `is_public` before the job reached `done`. The owner PATCH route at
    # `/annotation-tasks/{task_uuid}/evaluator-runs/{job_uuid}/visibility`
    # already enforces this gate, but other generic-jobs visibility routes
    # (stt/tts) wouldn't catch a wrong-type UUID — so we re-assert here.
    if job.get("status") != "done":
        raise HTTPException(status_code=404, detail="Not found")

    shaped = _shape_eval_job_for_response(job)
    task_uuid = shaped.get("task_id")
    task = get_annotation_task(task_uuid) if task_uuid else None
    if not task:
        # The parent task was deleted out from under the share link.
        raise HTTPException(status_code=404, detail="Not found")

    raw_runs = get_evaluator_runs_for_job(job["uuid"])
    details = job.get("details") or {}

    # Forward only the safe-to-share keys from the raw `details` blob. Strip
    # operational/identifying ones — `pid`, `pgid`, `s3_prefix`, `user_id`,
    # `output_dir`, etc. — that the auth view exposes but a public viewer
    # has no business seeing. Whitelist (not blacklist) so any future
    # addition to the runner's details dict stays private until explicitly
    # opted in here.
    public_details_whitelist = {
        "task_id",
        "evaluators",
        "item_count",
        "item_ids",
        "metrics",
        "completed_at",
    }
    public_details = {
        k: v for k, v in details.items() if k in public_details_whitelist
    }
    # `evaluators` is promoted to the response's top-level field — drop the
    # slim snapshot from the forwarded `details` to keep the auth and
    # public responses shaped the same way.
    public_details.pop("evaluators", None)

    evaluators_block = _build_evaluators_block_for_eval_job(details, raw_runs)

    return PublicAnnotationEvalResponse(
        task_id=task_uuid,
        job_uuid=job["uuid"],
        status=shaped["status"],
        created_at=job.get("created_at"),
        completed_at=shaped.get("completed_at"),
        updated_at=job.get("updated_at"),
        task=PublicAnnotationEvalTaskRef(
            uuid=task["uuid"],
            name=task["name"],
            type=task["type"],
            description=task.get("description"),
        ),
        details=public_details,
        evaluators=evaluators_block,
        item_count=details.get("item_count"),
        items=get_eval_job_items(job["uuid"]),
        # Per-run `evaluator` / `evaluator_version` blobs are intentionally
        # not surfaced — `(evaluator_id, evaluator_version_id)` on each run
        # keys back into the top-level evaluators[] block.
        runs=_strip_run_evaluator_blocks(raw_runs),
        human_agreement=_human_agreement_for_run(task_uuid, raw_runs),
        error=shaped.get("error"),
    )


# ---------------------------------------------------------------------------
# Annotation jobs (public, token-only)
# ---------------------------------------------------------------------------


def _resolve_public_annotation_job(token: str) -> Dict[str, Any]:
    """Fetch a job by its public_token, ensuring it's a real shareable link
    (not a CSV-import sentinel). Raises 404 otherwise."""
    job = get_annotation_job_by_token(token)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return job


def _build_annotation_job_payload(
    job: Dict[str, Any], read_only: bool
) -> Dict[str, Any]:
    """Shared response shape for both the annotator route (read+write) and
    the viewer route (read-only). When `read_only=True`, the response carries
    a `read_only` flag so the FE can disable form inputs. The annotator
    identity is included in both modes — the viewer is meant to see *whose*
    labels they're looking at."""
    task = get_annotation_task(job["task_id"])
    if not task:
        raise HTTPException(status_code=404, detail="Job not found")
    annotator = get_annotator(job["annotator_id"])
    # Read the SNAPSHOTTED evaluator set so the labelling form columns match
    # exactly what was assigned at job creation, regardless of subsequent
    # link/unlink on the parent task.
    evaluators = get_evaluators_for_job(job["uuid"])
    # Enrich each evaluator with the live version's rubric, variable specs,
    # and derived scale bounds so the FE has everything it needs to render
    # the labelling form without a second roundtrip.
    from llm_judge import _scale_bounds  # local to avoid module-load cycle

    for ev in evaluators:
        version = (
            get_evaluator_version(ev["live_version_id"])
            if ev.get("live_version_id")
            else None
        )
        output_config = version.get("output_config") if version else None
        scale_min, scale_max = _scale_bounds(output_config)
        ev["output_config"] = output_config
        ev["scale_min"] = scale_min
        ev["scale_max"] = scale_max
        ev["variables"] = version.get("variables") if version else None
    items = get_job_items(job["uuid"])
    annotations = get_annotations_for_job(job["uuid"])

    return {
        "job": {
            "uuid": job["uuid"],
            "status": job["status"],
            "created_at": job["created_at"],
            "completed_at": job.get("completed_at"),
        },
        "annotator": {
            "uuid": annotator["uuid"] if annotator else None,
            "name": annotator["name"] if annotator else None,
        },
        "task": {
            "uuid": task["uuid"],
            "name": task["name"],
            "type": task["type"],
            "description": task.get("description"),
        },
        "evaluators": evaluators,
        "items": items,
        "annotations": annotations,
        "read_only": read_only,
    }


@router.get("/annotation-jobs/view/{view_token}", summary="Get shared annotation job (read-only)")
def get_public_annotation_job_view(
    view_token: str = Path(description="Read-only view token for the annotation job"),
):
    """Get a read-only view of an annotator's labelling job."""
    job = get_annotation_job_by_view_token(view_token)
    if not job:
        raise HTTPException(status_code=404, detail="Not found")
    return _build_annotation_job_payload(job, read_only=True)


@router.get("/annotation-jobs/{token}", summary="Get annotation job for annotator")
def get_public_annotation_job(
    token: str = Path(description="Annotator token for the labelling job"),
):
    """Get everything you need to render an annotator's labelling job page."""
    job = _resolve_public_annotation_job(token)
    return _build_annotation_job_payload(job, read_only=False)


class PublicAnnotationEntry(BaseModel):
    evaluator_id: Optional[str] = Field(
        None,
        min_length=36,
        max_length=36,
        description="Evaluator ID this judgement is for; `null` marks a row-level overall annotation",
        examples=[_EXAMPLE_EVALUATOR_UUID],
    )
    value: Optional[Dict[str, Any]] = Field(
        None, description="Judgement payload (shape depends on the evaluator's output type); `null` to clear"
    )


class PublicAnnotationUpsertRequest(BaseModel):
    item_id: str = Field(
        min_length=36,
        max_length=36,
        description="Job item you are annotating",
        examples=[_EXAMPLE_ITEM_UUID],
    )
    annotations: List[PublicAnnotationEntry] = Field(
        description="Judgements to save — one entry per evaluator, plus optionally one row-level entry"
    )


@router.post("/annotation-jobs/{token}/annotations", summary="Upsert annotations for item")
def upsert_public_annotations(
    token: str = Path(description="Annotator token for the labelling job"),
    payload: PublicAnnotationUpsertRequest = ...,
):
    """Save all judgements for one item in a single request."""
    # First save moves the job from `pending` to `in_progress`; all slots filled moves it to `completed`.
    job = _resolve_public_annotation_job(token)
    if not payload.annotations:
        raise HTTPException(
            status_code=400, detail="annotations must be non-empty"
        )

    # Validate against the job's snapshotted items (annotation_job_items), not
    # the source annotation_items row — the source may have been edited or
    # soft-deleted after the job was created, but the snapshot is what the
    # annotator is actually labeling.
    job_items = get_job_items(job["uuid"])
    if not any(it["uuid"] == payload.item_id for it in job_items):
        raise HTTPException(status_code=404, detail="Item not found in this job")

    # Validate that every non-null `evaluator_id` is in the job's snapshotted
    # evaluator set (NOT the task's current linked set — the contract is
    # frozen at job creation). Without this, a token holder could upsert
    # annotations against any evaluator UUID — polluting downstream agreement
    # aggregates that read `annotations` joined through `annotation_jobs`.
    # `evaluator_id IS NULL` is the row-level overall annotation case and is
    # always allowed.
    linked_evaluator_ids = set(get_evaluator_ids_for_job(job["uuid"]))
    invalid = [
        entry.evaluator_id
        for entry in payload.annotations
        if entry.evaluator_id is not None
        and entry.evaluator_id not in linked_evaluator_ids
    ]
    if invalid:
        raise HTTPException(
            status_code=400,
            detail=f"Evaluator(s) not linked to this task: {invalid}",
        )

    saved_uuids: List[str] = []
    for entry in payload.annotations:
        annotation_uuid = upsert_annotation(
            job_id=job["uuid"],
            item_id=payload.item_id,
            evaluator_id=entry.evaluator_id,
            value=entry.value,
        )
        saved_uuids.append(annotation_uuid)

    if job["status"] == "pending":
        update_annotation_job_status(job["uuid"], "in_progress")

    # Auto-complete: every (item, evaluator) slot in this job must have a row.
    # We re-check on every save (including post-completion edits) so the
    # status remains accurate. `completed_at` is preserved on subsequent
    # edits — it marks the first time the job was fully filled.
    # Both the items AND the evaluator set are read from the job's snapshot
    # so post-creation link/unlink on the parent task can't shift the
    # completion bar under the annotator.
    job_items = get_job_items(job["uuid"])
    evaluator_ids = get_evaluator_ids_for_job(job["uuid"])
    annotated_pairs = {
        (a["item_id"], a.get("evaluator_id"))
        for a in get_annotations_for_job(job["uuid"])
        if a.get("evaluator_id") is not None
    }
    expected_pairs = {
        (it["uuid"], ev_id) for it in job_items for ev_id in evaluator_ids
    }
    completed = bool(expected_pairs) and expected_pairs.issubset(annotated_pairs)
    if completed and job["status"] != "completed":
        update_annotation_job_status(
            job["uuid"], "completed", set_completed_at=True
        )

    final_status = "completed" if completed else "in_progress"
    return {
        "saved": saved_uuids,
        "count": len(saved_uuids),
        "status": final_status,
    }
