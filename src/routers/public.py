"""
Public read-only endpoints for shared eval/run results.

These routes bypass authentication entirely — they are excluded from the
auth middleware by design so that anyone with a valid share_token can view
the results without logging in.

URL scheme:
  GET /public/evaluators/defaults?share_token=...&types=stt,tts
  GET /public/stt/{share_token}
  GET /public/tts/{share_token}
  GET /public/test-run/{share_token}
  GET /public/benchmark/{share_token}
  GET /public/simulation-run/{share_token}
  GET /public/annotation-eval/{share_token}
  GET /public/annotation-jobs/{public_token}             (annotator: read+write)
  POST /public/annotation-jobs/{public_token}/annotations (annotator: upsert)
  GET /public/annotation-jobs/view/{view_token}          (viewer: read-only)
"""

import logging
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

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
)
from routers.annotation_tasks import (
    _enrich_runs_with_live_evaluator,
    _human_agreement_for_run,
    _shape_eval_job_for_response,
)
from annotation_eval_runner import ANNOTATION_EVAL_JOB_TYPE

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/public", tags=["public"])


# ---------------------------------------------------------------------------
# Response models
# ---------------------------------------------------------------------------


class PublicSTTResponse(BaseModel):
    task_id: str
    status: str
    language: Optional[str] = None
    dataset_id: Optional[str] = None
    dataset_name: Optional[str] = None
    provider_results: Optional[List[ProviderResult]] = None
    leaderboard_summary: Optional[List[Dict[str, Any]]] = None
    error: Optional[str] = None


class PublicTTSResponse(BaseModel):
    task_id: str
    status: str
    language: Optional[str] = None
    dataset_id: Optional[str] = None
    dataset_name: Optional[str] = None
    provider_results: Optional[List[ProviderResult]] = None
    leaderboard_summary: Optional[List[Dict[str, Any]]] = None
    error: Optional[str] = None


class PublicTestRunResponse(BaseModel):
    task_id: str
    status: str
    total_tests: Optional[int] = None
    passed: Optional[int] = None
    failed: Optional[int] = None
    results: Optional[List[Dict[str, Any]]] = None
    error: bool = False


class PublicBenchmarkResponse(BaseModel):
    task_id: str
    status: str
    model_results: Optional[List[Dict[str, Any]]] = None
    leaderboard_summary: Optional[List[Dict[str, Any]]] = None
    error: bool = False


class PublicSimulationRunResponse(BaseModel):
    task_id: str
    name: str
    status: str
    type: str
    updated_at: str
    total_simulations: Optional[int] = None
    metrics: Optional[Dict[str, Any]] = None
    simulation_results: Optional[List[Dict[str, Any]]] = None
    evaluators: Optional[List[SimulationEvaluatorRef]] = None
    error: Optional[str] = None


class PublicAnnotationEvalTaskRef(BaseModel):
    uuid: str
    name: str
    type: str
    description: Optional[str] = None


class PublicAnnotationEvalResponse(BaseModel):
    task_id: str
    job_uuid: str
    status: str
    created_at: Optional[str] = None
    completed_at: Optional[str] = None
    updated_at: Optional[str] = None
    task: PublicAnnotationEvalTaskRef
    # `details` mirrors the authenticated GET shape so a shared FE component
    # can read `job.details?.evaluators` against either endpoint without
    # branching. Only the safe-to-share keys are forwarded — operational
    # fields (pid, pgid, s3_prefix, user_id) are intentionally stripped.
    details: Optional[Dict[str, Any]] = None
    # Top-level mirrors retained for the existing public consumers that
    # were already reading them. Both shapes carry the same data.
    evaluators: Optional[List[Dict[str, Any]]] = None
    item_count: Optional[int] = None
    items: Optional[List[Dict[str, Any]]] = None
    runs: Optional[List[Dict[str, Any]]] = None
    human_agreement: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


class PublicDefaultEvaluatorVersionResponse(BaseModel):
    output_config: Optional[Dict[str, Any]] = None


class PublicDefaultEvaluatorResponse(BaseModel):
    uuid: str
    name: str
    description: Optional[str] = None
    evaluator_type: str
    output_type: str
    live_version: Optional[PublicDefaultEvaluatorVersionResponse] = None


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

    allowed = {"stt", "tts", "llm", "simulation"}
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
    "/evaluators/defaults", response_model=List[PublicDefaultEvaluatorResponse]
)
async def get_public_default_evaluators(
    share_token: str = Query(..., min_length=1),
    types: Optional[str] = Query(
        None,
        description="Comma-separated evaluator types: stt,tts,llm,simulation",
    ),
):
    """
    Return public-safe default evaluator metadata for callers with a valid public share token.

    This intentionally omits prompts, judge models, owner metadata, and custom/private evaluators.
    """
    _ensure_valid_public_share_token(share_token)
    requested_types = _parse_evaluator_types(types)

    defaults: List[PublicDefaultEvaluatorResponse] = []
    for seed in DEFAULT_EVALUATORS_SEED:
        if requested_types is not None and seed["evaluator_type"] not in requested_types:
            continue
        evaluator = get_evaluator_by_slug(seed["slug"])
        if evaluator and evaluator.get("owner_user_id") is None:
            defaults.append(_public_default_evaluator_response(evaluator))
    return defaults


@router.get("/stt/{share_token}", response_model=PublicSTTResponse)
async def get_public_stt(share_token: str):
    """
    Return a publicly shared STT evaluation result.
    No authentication required — accessible to anyone with the share_token.
    Returns 404 if the token is unknown or the run has been made private again.
    """
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


@router.get("/tts/{share_token}", response_model=PublicTTSResponse)
async def get_public_tts(share_token: str):
    """
    Return a publicly shared TTS evaluation result.
    No authentication required — accessible to anyone with the share_token.
    Returns 404 if the token is unknown or the run has been made private again.
    """
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


@router.get("/test-run/{share_token}", response_model=PublicTestRunResponse)
async def get_public_test_run(share_token: str):
    """
    Return a publicly shared LLM test run result.
    No authentication required — accessible to anyone with the share_token.
    Returns 404 if the token is unknown or the run has been made private again.
    """
    job = get_agent_test_job_by_share_token(share_token, job_type="llm-unit-test")
    if not job:
        raise HTTPException(status_code=404, detail="Not found")

    task_id = job["uuid"]
    status = job["status"]
    results = job.get("results") or {}
    details = job.get("details") or {}

    _enrich_test_results_with_evaluators(
        results.get("test_results"),
        details.get("evaluators_by_test_id") or {},
    )

    return PublicTestRunResponse(
        task_id=task_id,
        status=status,
        total_tests=results.get("total_tests"),
        passed=results.get("passed"),
        failed=results.get("failed"),
        results=results.get("test_results"),
        error=bool(results.get("error")),
    )


@router.get("/benchmark/{share_token}", response_model=PublicBenchmarkResponse)
async def get_public_benchmark(share_token: str):
    """
    Return a publicly shared LLM benchmark result.
    No authentication required — accessible to anyone with the share_token.
    Returns 404 if the token is unknown or the run has been made private again.
    """
    job = get_agent_test_job_by_share_token(share_token, job_type="llm-benchmark")
    if not job:
        raise HTTPException(status_code=404, detail="Not found")

    task_id = job["uuid"]
    status = job["status"]
    results = job.get("results") or {}
    details = job.get("details") or {}

    _enrich_model_results_with_evaluators(
        results.get("model_results"),
        details.get("evaluators_by_test_id") or {},
    )

    return PublicBenchmarkResponse(
        task_id=task_id,
        status=status,
        model_results=results.get("model_results"),
        leaderboard_summary=results.get("leaderboard_summary"),
        error=bool(results.get("error")),
    )


@router.get("/simulation-run/{share_token}", response_model=PublicSimulationRunResponse)
async def get_public_simulation_run(share_token: str):
    """
    Return a publicly shared simulation run result.
    No authentication required — accessible to anyone with the share_token.
    Returns 404 if the token is unknown or the run has been made private again.
    Presigned audio URLs are regenerated on-the-fly for voice simulations.
    """
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
    "/annotation-eval/{share_token}", response_model=PublicAnnotationEvalResponse
)
async def get_public_annotation_eval(share_token: str):
    """Return a publicly shared annotation evaluator-run job result.

    No authentication required. Returns 404 if the token is unknown or the
    run has been made private again. Mirrors the authenticated
    `GET /annotation-tasks/{task_uuid}/evaluator-runs/{job_uuid}` shape so the
    frontend can render the same view.
    """
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
        evaluators=details.get("evaluators"),
        item_count=details.get("item_count"),
        items=get_eval_job_items(job["uuid"]),
        runs=_enrich_runs_with_live_evaluator(raw_runs),
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


@router.get("/annotation-jobs/view/{view_token}")
def get_public_annotation_job_view(view_token: str):
    """Read-only public view of one annotator's completed labelling job,
    served behind a separate `view_token` toggled on by the owner via
    `PATCH /annotation-tasks/{task_uuid}/jobs/{job_uuid}/visibility`.

    Returns the same shape as the annotator route (`/public/annotation-jobs/
    {public_token}`) but with `read_only: true`. There is intentionally NO
    `POST /annotation-jobs/view/{view_token}/annotations` companion — a leaked
    view_token cannot be coerced into writing labels because the upsert path
    only accepts the annotator's `public_token`."""
    job = get_annotation_job_by_view_token(view_token)
    if not job:
        raise HTTPException(status_code=404, detail="Not found")
    return _build_annotation_job_payload(job, read_only=True)


@router.get("/annotation-jobs/{token}")
def get_public_annotation_job(token: str):
    """Everything an annotator needs to render their job page:

    - Job + status
    - Annotator's name (so the page can greet them)
    - Task type + linked evaluators (drives form rendering: binary toggle vs
      rating scale, plus per-evaluator name/description/output_config)
    - Items (with their parsed `payload`)
    - Existing annotations on this job (so the page can resume in-progress work)
    """
    job = _resolve_public_annotation_job(token)
    return _build_annotation_job_payload(job, read_only=False)


class PublicAnnotationEntry(BaseModel):
    evaluator_id: Optional[str] = None  # None = row-level overall annotation
    value: Optional[Dict[str, Any]] = None


class PublicAnnotationUpsertRequest(BaseModel):
    item_id: str
    annotations: List[PublicAnnotationEntry]


@router.post("/annotation-jobs/{token}/annotations")
def upsert_public_annotations(token: str, payload: PublicAnnotationUpsertRequest):
    """Upsert all judgements for one item in one call. Pass one entry per
    evaluator (plus optionally one with `evaluator_id = null` for a row-level
    overall annotation).

    Side effects:
      - The job auto-flips from `pending` -> `in_progress` on the first save.
      - After saving, the job auto-flips to `completed` (with `completed_at`)
        when every item in the job has annotations for every evaluator linked
        to the task. Row-level annotations are NOT required.
    """
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
