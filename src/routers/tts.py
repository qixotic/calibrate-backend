import os
import csv
import json
import subprocess
import tempfile
import time
import traceback
import threading
import logging
from pathlib import Path
from typing import List, Optional

from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel, ConfigDict

from db import (
    create_job,
    get_job,
    get_evaluator,
    get_evaluator_by_slug,
    get_evaluator_version,
    update_job,
    update_job_visibility,
)
from dataset_utils import resolve_dataset_inputs
from auth_utils import get_current_user_id
from llm_judge import build_evaluator_cli_payload
from utils import (
    TaskStatus,
    ProviderResult,
    TaskCreateResponse,
    TaskStatusResponse,
    build_evaluator_runs_for_eval_job,
    compute_share_token_toggle,
    enrich_evaluator_runs_with_current_names,
    load_evaluator_metric_key_map,
    post_process_provider_results,
    get_s3_client,
    get_s3_output_config,
    can_start_job,
    try_start_queued_job,
    register_job_starter,
    generate_presigned_download_url,
    is_job_timed_out,
    kill_process_group,
    capture_exception_to_sentry,
    normalize_metrics,
    read_leaderboard_xlsx,
    read_evaluators_map_from_config,
    upload_directory_tree_to_s3,
    upload_file_to_s3,
    upload_top_level_files_to_s3,
)

# Job types that share the same queue
EVAL_JOB_TYPES = ["stt-eval", "tts-eval", "annotation-eval"]


def _start_tts_job_from_queue(job: dict) -> bool:
    """Start a TTS evaluation job from the queue.

    This is called by the job queue manager when there's capacity to run a new job.
    """
    job_id = job["uuid"]
    details = job.get("details", {})

    # Reconstruct request from job details
    request = TTSEvaluationRequest(
        texts=details.get("texts", []),
        providers=details.get("providers", []),
        language=details.get("language", ""),
    )
    s3_bucket = details.get("s3_bucket", "")

    # Start background task in a separate thread
    thread = threading.Thread(
        target=run_tts_evaluation_task,
        args=(job_id, request, s3_bucket),
        daemon=True,
    )
    thread.start()

    return True


# Register the job starter for TTS evaluation jobs
register_job_starter("tts-eval", _start_tts_job_from_queue)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/tts", tags=["tts"])


def _collect_tts_intermediate_results(
    output_dir: Path,
    providers: list,
    task_id: str,
    s3_bucket: str,
    expected_total: int,
) -> list:
    """Read whatever intermediate results are available from disk for each provider.

    Uploads audio files to S3 and replaces local paths with S3 keys.
    Returns a list of ProviderResult objects preserving any partial results.
    A provider is only marked ``success=True`` when it has BOTH a complete
    row count (``>= expected_total``) AND an aggregate ``metrics.json`` on
    disk — same contract as the in-progress GET reader. A non-empty but
    incomplete `results.csv` (e.g. calibrate crashed mid-run) is preserved
    as ``success=False`` so the FE doesn't show a half-finished provider as
    successful.
    """
    evaluator_id_by_metric_key = read_evaluators_map_from_config(output_dir)
    s3 = get_s3_client()
    provider_results = []
    for provider in providers:
        provider_output_dir = _find_tts_provider_output_dir(output_dir, provider)
        results_data = _read_tts_results_csv(provider_output_dir)
        metrics_data = _read_tts_metrics_json(provider_output_dir)
        if results_data and provider_output_dir:
            # Upload audio files to S3 and map local paths to S3 keys
            results_prefix = f"tts/evals/{task_id}/outputs/{provider}"
            audio_path_to_s3_key = {}

            for root, dirs, files in os.walk(provider_output_dir):
                for file in files:
                    local_file_path = Path(root) / file
                    relative_path = local_file_path.relative_to(provider_output_dir)
                    s3_key = f"{results_prefix}/{relative_path}"
                    try:
                        upload_file_to_s3(s3, local_file_path, s3_bucket, s3_key)
                        if file.endswith((".wav", ".mp3", ".ogg")):
                            audio_path_to_s3_key[str(local_file_path)] = s3_key
                    except Exception:
                        pass

            # Replace local audio paths with S3 keys in results
            for result_row in results_data:
                if "audio_path" in result_row and result_row["audio_path"]:
                    local_audio_path = result_row["audio_path"]
                    audio_s3_key = audio_path_to_s3_key.get(local_audio_path)
                    if audio_s3_key:
                        result_row["audio_path"] = audio_s3_key

            runs = (
                build_evaluator_runs_for_eval_job(
                    metrics_data, evaluator_id_by_metric_key
                )
                if metrics_data is not None
                else []
            )
            provider_done = (
                metrics_data is not None
                and len(results_data) >= expected_total
            )
            provider_results.append(
                ProviderResult(
                    provider=provider,
                    success=True if provider_done else False,
                    metrics=metrics_data,
                    results=results_data,
                    evaluator_runs=runs or None,
                )
            )
        else:
            provider_results.append(
                ProviderResult(
                    provider=provider,
                    success=False,
                )
            )
    return provider_results


class TTSEvaluationRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    # Option 1: reuse an existing dataset
    dataset_id: Optional[str] = None
    # Option 2: inline texts (legacy / new inputs)
    texts: Optional[List[str]] = None  # List of texts to synthesize
    # When providing inline data, name for the new dataset to save (ignored when dataset_id is set)
    dataset_name: Optional[str] = None
    providers: List[
        str
    ]  # List of TTS providers (e.g., ["smallest", "cartesia", "openai"])
    language: str  # Language (e.g., "english", "hindi")
    # Optional list of evaluator UUIDs to score this run. If omitted, the seeded TTS default
    # evaluator is used. Each evaluator must have `evaluator_type == "tts"` (validated at
    # submission time). At submission each UUID is hydrated against its then-live version
    # and pinned into job details.
    evaluator_uuids: Optional[List[str]] = None


def _resolve_evaluators_for_tts_job(
    uuids: Optional[List[str]],
    user_id: str,
    default_slug: str,
    expected_evaluator_type: str,
) -> List[dict]:
    """Resolve TTS evaluator UUIDs into hydrated dicts for the background task to write
    into the calibrate CLI config. Pins each to its current live version at submission
    time and enforces `evaluator_type == expected_evaluator_type`.
    """
    resolved: List[dict] = []
    effective_refs: List[dict] = [
        {"evaluator_uuid": uid, "version_uuid": None, "variable_values": None}
        for uid in (uuids or [])
    ]

    if not effective_refs:
        default = get_evaluator_by_slug(default_slug)
        if default and default.get("live_version_id"):
            effective_refs = [
                {
                    "evaluator_uuid": default["uuid"],
                    "version_uuid": default["live_version_id"],
                    "variable_values": None,
                }
            ]

    for ref in effective_refs:
        evaluator = get_evaluator(ref["evaluator_uuid"])
        if not evaluator:
            raise HTTPException(
                status_code=404, detail=f"Evaluator {ref['evaluator_uuid']} not found"
            )
        if (
            evaluator.get("owner_user_id") is not None
            and evaluator["owner_user_id"] != user_id
        ):
            raise HTTPException(
                status_code=404, detail=f"Evaluator {ref['evaluator_uuid']} not found"
            )
        if evaluator.get("evaluator_type") != expected_evaluator_type:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"Evaluator {ref['evaluator_uuid']} has evaluator_type="
                    f"'{evaluator.get('evaluator_type')}' but this job requires "
                    f"'{expected_evaluator_type}' evaluators."
                ),
            )
        version_uuid = ref["version_uuid"] or evaluator.get("live_version_id")
        if not version_uuid:
            raise HTTPException(
                status_code=400,
                detail=f"Evaluator {ref['evaluator_uuid']} has no live version",
            )
        version = get_evaluator_version(version_uuid)
        if not version or version["evaluator_id"] != evaluator["uuid"]:
            raise HTTPException(
                status_code=400,
                detail=f"Evaluator version {version_uuid} not found for evaluator {ref['evaluator_uuid']}",
            )
        resolved.append(
            {
                "uuid": evaluator["uuid"],
                "name": evaluator["name"],
                "evaluator_type": evaluator.get(
                    "evaluator_type", expected_evaluator_type
                ),
                "data_type": evaluator.get("data_type", "audio"),
                "kind": evaluator.get("kind", "single"),
                "output_type": evaluator.get("output_type", "binary"),
                "evaluator_version_id": version["uuid"],
                "judge_model": version["judge_model"],
                "system_prompt": version["system_prompt"],
                "output_config": version.get("output_config"),
                "variables": version.get("variables"),
                "variable_values": ref.get("variable_values") or {},
            }
        )
    return resolved


def _find_tts_provider_output_dir(output_dir: Path, provider: str) -> Optional[Path]:
    """Find the provider-specific output directory."""
    if not output_dir.exists():
        return None
    for item in output_dir.iterdir():
        if item.is_dir() and provider in item.name.lower():
            return item
    return None


def _read_tts_results_csv(provider_output_dir: Path) -> Optional[List[dict]]:
    """Read results.csv from provider output directory if it exists."""
    if not provider_output_dir:
        return None
    results_file = provider_output_dir / "results.csv"
    if not results_file.exists():
        return None
    try:
        results_data = []
        with open(results_file, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                results_data.append(dict(row))
        return results_data
    except Exception:
        return None


def _read_tts_metrics_json(provider_output_dir: Path) -> Optional[dict]:
    """Read metrics.json from provider output directory if it exists.

    Handles both new format (dict) and old format (list of dicts) for backward compatibility.
    """
    if not provider_output_dir:
        return None
    metrics_file = provider_output_dir / "metrics.json"
    if not metrics_file.exists():
        return None
    try:
        with open(metrics_file, "r", encoding="utf-8") as f:
            data = json.load(f)
            return data
    except Exception:
        return None


def run_tts_evaluation_task(
    task_id: str,
    request: TTSEvaluationRequest,
    s3_bucket: str,
):
    """Run the TTS evaluation in the background."""
    try:
        logger.info(
            f"Running TTS evaluation task {task_id} with {len(request.providers)} providers"
        )
        update_job(task_id, status=TaskStatus.IN_PROGRESS.value)

        s3 = get_s3_client()

        # Create temporary directory for processing
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            try:
                # Create input CSV file
                input_csv = temp_path / "input.csv"
                with open(input_csv, "w", newline="", encoding="utf-8") as csvfile:
                    writer = csv.writer(csvfile)
                    writer.writerow(["id", "text"])
                    for idx, text in enumerate(request.texts):
                        writer.writerow([idx, text])

                # Create output directory
                output_dir = temp_path / "output"
                output_dir.mkdir()

                # Run calibrate tts command with all providers at once
                # The CLI now handles parallelization internally and generates leaderboard
                eval_cmd = (
                    [
                        "calibrate",
                        "tts",
                        "-p",
                    ]
                    + request.providers
                    + [
                        "-l",
                        request.language,
                        "-i",
                        str(input_csv),
                        "-o",
                        str(output_dir),
                    ]
                )

                # Calibrate: --config <path> with {evaluators: [...]}
                job_details = (get_job(task_id) or {}).get("details", {}) or {}
                raw_evaluators = job_details.get("evaluators") or []
                if raw_evaluators:
                    evaluator_payload = build_evaluator_cli_payload(raw_evaluators)
                    config_path = temp_path / "config.json"
                    with open(config_path, "w", encoding="utf-8") as f:
                        json.dump(
                            {"evaluators": evaluator_payload}, f, ensure_ascii=False
                        )
                    eval_cmd.extend(["--config", str(config_path)])

                logger.info(f"Running TTS eval command: {' '.join(eval_cmd)}")

                # Create temp files for stdout/stderr
                stdout_path = output_dir / "stdout.log"
                stderr_path = output_dir / "stderr.log"

                with (
                    open(stdout_path, "w") as stdout_f,
                    open(stderr_path, "w") as stderr_f,
                ):
                    process = subprocess.Popen(
                        eval_cmd,
                        stdout=stdout_f,
                        stderr=stderr_f,
                        text=True,
                        start_new_session=True,
                        cwd=str(temp_path),
                    )

                    # Store PID and output dir for cleanup and intermediate results
                    update_job(
                        task_id,
                        details={
                            "pid": process.pid,
                            "pgid": process.pid,
                            "output_dir": str(output_dir),
                        },
                    )

                    # Poll for process completion with heartbeat to keep updated_at fresh
                    # This prevents the job from being marked as timed out during long runs
                    HEARTBEAT_INTERVAL = 2  # seconds
                    while process.poll() is None:
                        time.sleep(HEARTBEAT_INTERVAL)
                        if process.poll() is None:
                            # Process still running, send heartbeat to refresh updated_at
                            update_job(task_id)

                # Read stdout/stderr
                with open(stdout_path, "r") as f:
                    stdout = f.read()
                with open(stderr_path, "r") as f:
                    stderr = f.read()

                if process.returncode != 0:
                    logger.error(f"TTS eval failed with code {process.returncode}")
                    logger.error(f"stderr: {stderr}")
                    raise subprocess.CalledProcessError(
                        process.returncode, eval_cmd, stdout, stderr
                    )

                logger.info("TTS eval command completed successfully")

                # Read results for each provider
                provider_results = []
                evaluator_id_by_metric_key = read_evaluators_map_from_config(output_dir)
                for provider in request.providers:
                    provider_output_dir = _find_tts_provider_output_dir(
                        output_dir, provider
                    )
                    if provider_output_dir:
                        metrics_data = _read_tts_metrics_json(provider_output_dir)
                        results_data = _read_tts_results_csv(provider_output_dir)

                        # Upload provider results to S3 and map audio paths
                        results_prefix = f"tts/evals/{task_id}/outputs/{provider}"
                        audio_path_to_s3_key = {}

                        for root, dirs, files in os.walk(provider_output_dir):
                            for file in files:
                                local_file_path = Path(root) / file
                                relative_path = local_file_path.relative_to(
                                    provider_output_dir
                                )
                                s3_key = f"{results_prefix}/{relative_path}"
                                upload_file_to_s3(
                                    s3, local_file_path, s3_bucket, s3_key
                                )

                                # Track audio files for path mapping
                                if file.endswith((".wav", ".mp3", ".ogg")):
                                    audio_path_to_s3_key[str(local_file_path)] = s3_key

                        # Replace local audio paths with S3 keys in results
                        successful_count = 0
                        if results_data:
                            for result_row in results_data:
                                if (
                                    "audio_path" in result_row
                                    and result_row["audio_path"]
                                ):
                                    local_audio_path = result_row["audio_path"]
                                    audio_s3_key = audio_path_to_s3_key.get(
                                        local_audio_path
                                    )
                                    if audio_s3_key:
                                        result_row["audio_path"] = audio_s3_key
                                        successful_count += 1

                        eruns = (
                            build_evaluator_runs_for_eval_job(
                                metrics_data,
                                evaluator_id_by_metric_key,
                            )
                            if metrics_data is not None
                            else []
                        )
                        if successful_count > 0:
                            provider_results.append(
                                ProviderResult(
                                    provider=provider,
                                    success=True,
                                    metrics=metrics_data,
                                    results=results_data,
                                    evaluator_runs=eruns or None,
                                )
                            )
                        else:
                            provider_results.append(
                                ProviderResult(
                                    provider=provider,
                                    success=False,
                                    metrics=metrics_data,
                                    results=results_data,
                                    evaluator_runs=eruns or None,
                                )
                            )
                    else:
                        provider_results.append(
                            ProviderResult(
                                provider=provider,
                                success=False,
                                message=f"No output found for provider {provider}",
                            )
                        )

                upload_top_level_files_to_s3(
                    s3,
                    output_dir,
                    s3_bucket,
                    f"tts/evals/{task_id}/outputs",
                )

                # Read leaderboard from output directory
                leaderboard_dir = output_dir / "leaderboard"
                leaderboard_summary = None

                # Log what's in output_dir for debugging
                logger.info(
                    f"Output directory contents: {[f.name for f in output_dir.iterdir()]}"
                )

                if leaderboard_dir.exists():
                    logger.info(f"Leaderboard directory exists: {leaderboard_dir}")
                    leaderboard_summary = read_leaderboard_xlsx(leaderboard_dir)

                    # Upload leaderboard to S3
                    leaderboard_prefix = f"tts/evals/{task_id}/leaderboard"
                    for root, dirs, files in os.walk(leaderboard_dir):
                        for file in files:
                            local_file_path = Path(root) / file
                            relative_path = local_file_path.relative_to(leaderboard_dir)
                            s3_key = f"{leaderboard_prefix}/{relative_path}"
                            upload_file_to_s3(s3, local_file_path, s3_bucket, s3_key)
                else:
                    logger.warning(
                        f"Leaderboard directory does not exist: {leaderboard_dir}"
                    )

                # Prefer calibrate's run-root config.json because it contains evaluator IDs/maps.
                config_file = output_dir / "config.json"
                if not config_file.exists():
                    config_data = {
                        "providers": request.providers,
                        "language": request.language,
                        "text_count": len(request.texts),
                    }
                    config_file = temp_path / "config.json"
                    with open(config_file, "w", encoding="utf-8") as f:
                        json.dump(config_data, f, indent=2)
                config_s3_key = f"tts/evals/{task_id}/config.json"
                upload_file_to_s3(s3, config_file, s3_bucket, config_s3_key)
                logger.info(f"Uploaded config file to S3: {config_s3_key}")

                # Check if all providers succeeded
                all_succeeded = all(r.success for r in provider_results)
                final_status = (
                    TaskStatus.DONE.value if all_succeeded else TaskStatus.FAILED.value
                )

                error_msg = None
                if not all_succeeded:
                    failed = [r.provider for r in provider_results if not r.success]
                    error_msg = f"Some providers failed: {', '.join(failed)}"

                # Update job with results
                update_job(
                    task_id,
                    status=final_status,
                    results={
                        "provider_results": [r.model_dump() for r in provider_results],
                        "leaderboard_summary": leaderboard_summary,
                        "error": error_msg,
                    },
                )

            except subprocess.CalledProcessError as e:
                traceback.print_exc()
                capture_exception_to_sentry(e)
                error_results = {
                    "error": f"TTS evaluation failed: {e.stderr if hasattr(e, 'stderr') else str(e)}",
                }
                # Preserve any intermediate results already written to disk
                try:
                    if output_dir.exists():
                        intermediate = _collect_tts_intermediate_results(
                            output_dir,
                            request.providers,
                            task_id,
                            s3_bucket,
                            len(request.texts),
                        )
                        if intermediate:
                            error_results["provider_results"] = [
                                r.model_dump() for r in intermediate
                            ]
                        if output_dir.exists():
                            upload_directory_tree_to_s3(
                                s3,
                                output_dir,
                                s3_bucket,
                                f"tts/evals/{task_id}/outputs",
                            )
                except Exception:
                    pass
                update_job(
                    task_id,
                    status=TaskStatus.FAILED.value,
                    results=error_results,
                )
            except Exception as e:
                traceback.print_exc()
                capture_exception_to_sentry(e)
                error_results = {
                    "error": f"Unexpected error during TTS evaluation: {str(e)}",
                }
                # Preserve any intermediate results already written to disk
                try:
                    if output_dir.exists():
                        intermediate = _collect_tts_intermediate_results(
                            output_dir,
                            request.providers,
                            task_id,
                            s3_bucket,
                            len(request.texts),
                        )
                        if intermediate:
                            error_results["provider_results"] = [
                                r.model_dump() for r in intermediate
                            ]
                        if output_dir.exists():
                            upload_directory_tree_to_s3(
                                s3,
                                output_dir,
                                s3_bucket,
                                f"tts/evals/{task_id}/outputs",
                            )
                except Exception:
                    pass
                update_job(
                    task_id,
                    status=TaskStatus.FAILED.value,
                    results=error_results,
                )

    except Exception as e:
        traceback.print_exc()
        capture_exception_to_sentry(e)
        update_job(
            task_id,
            status=TaskStatus.FAILED.value,
            results={"error": f"Task failed: {str(e)}"},
        )
    finally:
        # Try to start the next queued job
        try_start_queued_job(EVAL_JOB_TYPES)


@router.post("/evaluate", response_model=TaskCreateResponse)
async def evaluate_tts(
    request: TTSEvaluationRequest, user_id: str = Depends(get_current_user_id)
):
    """
    Start a background task to evaluate multiple TTS providers with text inputs.

    Returns a task ID that can be used to poll for status and results.
    """
    if not request.providers:
        raise HTTPException(
            status_code=400,
            detail="At least one provider must be specified",
        )

    resolved = resolve_dataset_inputs(
        dataset_id=request.dataset_id,
        user_id=user_id,
        expected_type="tts",
        texts=request.texts,
        dataset_name=request.dataset_name,
    )
    texts = resolved.texts
    resolved_dataset_id = resolved.dataset_id
    resolved_dataset_name = resolved.dataset_name
    dataset_item_ids = resolved.item_ids

    request.texts = texts

    # Get S3 configuration from environment
    try:
        s3_bucket = get_s3_output_config()
    except ValueError as e:
        raise HTTPException(status_code=500, detail=str(e))

    resolved_evaluators = _resolve_evaluators_for_tts_job(
        uuids=request.evaluator_uuids,
        user_id=user_id,
        default_slug="default-tts-audio-quality",
        expected_evaluator_type="tts",
    )

    # Check if we can start immediately or need to queue
    can_start = can_start_job(EVAL_JOB_TYPES, user_id)
    initial_status = (
        TaskStatus.IN_PROGRESS.value if can_start else TaskStatus.QUEUED.value
    )

    # Create job in database with details for recovery
    job_id = create_job(
        job_type="tts-eval",
        user_id=user_id,
        status=initial_status,
        details={
            "texts": texts,
            "providers": request.providers,
            "language": request.language,
            "s3_bucket": s3_bucket,
            "dataset_id": resolved_dataset_id,
            "dataset_name": resolved_dataset_name,
            "dataset_item_ids": dataset_item_ids,
            "evaluators": resolved_evaluators,
        },
        results=None,
    )

    if can_start:
        # Start background task in a separate thread
        thread = threading.Thread(
            target=run_tts_evaluation_task,
            args=(job_id, request, s3_bucket),
            daemon=True,
        )
        thread.start()
        logger.info(f"Started TTS evaluation job {job_id} immediately")
    else:
        logger.info(f"Queued TTS evaluation job {job_id}")

    return TaskCreateResponse(
        task_id=job_id,
        status=initial_status,
        dataset_id=resolved_dataset_id,
        dataset_name=resolved_dataset_name,
    )


class VisibilityRequest(BaseModel):
    is_public: bool


class VisibilityResponse(BaseModel):
    is_public: bool
    share_token: str | None = None


@router.patch("/evaluate/{task_id}/visibility", response_model=VisibilityResponse)
async def update_tts_visibility(
    task_id: str,
    body: VisibilityRequest,
    user_id: str = Depends(get_current_user_id),
):
    """Toggle public sharing for a TTS evaluation job."""
    job = get_job(task_id, user_id=user_id)
    if not job or job.get("type") != "tts-eval":
        raise HTTPException(status_code=404, detail="Task not found")

    token_to_persist, token_to_return = compute_share_token_toggle(
        job, body.is_public
    )
    update_job_visibility(task_id, body.is_public, token_to_persist)
    return VisibilityResponse(
        is_public=body.is_public, share_token=token_to_return
    )


@router.get("/evaluate/{task_id}", response_model=TaskStatusResponse)
async def get_tts_evaluation_status(
    task_id: str, user_id: str = Depends(get_current_user_id)
):
    """
    Get the status of a TTS evaluation task.

    Returns the current status and, if done, the provider results and leaderboard path.
    """
    job = get_job(task_id, user_id=user_id)
    if not job:
        raise HTTPException(status_code=404, detail="Task not found")

    status = job["status"]
    results = job.get("results") or {}
    details = job.get("details") or {}

    # Check for timeout on in-progress jobs
    if status == TaskStatus.IN_PROGRESS.value:
        updated_at = job.get("updated_at")
        if updated_at and is_job_timed_out(updated_at):
            logger.warning(f"Job {task_id} timed out, marking as failed")

            # Kill running process
            pid = details.get("pid") or details.get("pgid")
            if pid:
                kill_process_group(pid, task_id)

            # Preserve intermediate results from disk before marking as failed
            # IMPORTANT: Merge with existing results, don't overwrite successful ones
            requested_providers = details.get("providers", [])
            output_dir_str = details.get("output_dir")
            s3_bucket = details.get("s3_bucket", "")
            existing_provider_results = results.get("provider_results", [])

            # Build a map of existing successful results (don't overwrite these)
            existing_success_map = {}
            for pr in existing_provider_results:
                if pr.get("success") is True:
                    existing_success_map[pr.get("provider")] = pr

            if output_dir_str:
                try:
                    output_dir = Path(output_dir_str)
                    if output_dir.exists():
                        intermediate = _collect_tts_intermediate_results(
                            output_dir,
                            requested_providers,
                            task_id,
                            s3_bucket,
                            len(details.get("texts") or []),
                        )
                        # Merge: keep existing successful results, add new ones from disk
                        merged_results = []
                        intermediate_map = {
                            r.provider: r.model_dump() for r in intermediate
                        }
                        for provider in requested_providers:
                            if provider in existing_success_map:
                                # Keep the existing successful result
                                merged_results.append(existing_success_map[provider])
                            elif provider in intermediate_map:
                                # Use intermediate result from disk
                                merged_results.append(intermediate_map[provider])
                            else:
                                # Provider not found anywhere, mark as failed
                                merged_results.append(
                                    {
                                        "provider": provider,
                                        "success": False,
                                        "metrics": None,
                                        "results": None,
                                    }
                                )
                        results["provider_results"] = merged_results
                except Exception as exc:
                    logger.warning(
                        f"Failed to collect intermediate results on timeout: {exc}"
                    )
                    # Even on exception, preserve any existing successful results
                    if existing_provider_results:
                        results["provider_results"] = existing_provider_results

            # Mark job as failed
            results["error"] = "Job timed out after 5 minutes of inactivity"
            update_job(
                task_id,
                status=TaskStatus.FAILED.value,
                results=results,
            )
            status = TaskStatus.FAILED.value

            # Try to start the next queued job
            try_start_queued_job(EVAL_JOB_TYPES)

    # Get list of all requested providers from job details
    requested_providers = details.get("providers", [])

    # Build provider results
    provider_results = results.get("provider_results")
    # `evaluator_id_by_metric_key` lets the post-processor build
    # `evaluator_runs[]` mid-flight from a freshly-landed metrics.json.
    evaluator_id_by_metric_key = load_evaluator_metric_key_map(details)
    output_dir_str = details.get("output_dir")
    if provider_results is None and status == TaskStatus.IN_PROGRESS.value:
        # Job is in progress - try to read intermediate results from disk
        expected_total = len(details.get("texts", []))
        if output_dir_str:
            output_dir = Path(output_dir_str)
            s3 = get_s3_client()
            s3_bucket = details.get("s3_bucket", "")
            provider_results = []
            for provider in requested_providers:
                provider_output_dir = _find_tts_provider_output_dir(
                    output_dir, provider
                )
                results_data = _read_tts_results_csv(provider_output_dir)
                metrics_data = _read_tts_metrics_json(provider_output_dir)
                if results_data:
                    # Upload audio files to S3 and generate presigned URLs
                    results_prefix = f"tts/evals/{task_id}/outputs/{provider}"
                    for row in results_data:
                        audio_path = row.get("audio_path")
                        if audio_path and not audio_path.startswith("http"):
                            local_path = Path(audio_path)
                            if local_path.exists() and provider_output_dir:
                                try:
                                    relative_path = local_path.relative_to(
                                        provider_output_dir
                                    )
                                    s3_key = f"{results_prefix}/{relative_path}"
                                    upload_file_to_s3(s3, local_path, s3_bucket, s3_key)
                                    presigned_url = generate_presigned_download_url(
                                        s3_key
                                    )
                                    row["audio_path"] = presigned_url
                                except Exception:
                                    row["audio_path"] = None
                            else:
                                row["audio_path"] = None
                    # If all texts are processed and metrics are ready, mark as done
                    provider_done = (
                        len(results_data) >= expected_total and metrics_data is not None
                    )
                    provider_results.append(
                        {
                            "provider": provider,
                            "success": True if provider_done else None,
                            "message": (
                                f"Done ({len(results_data)} texts processed)"
                                if provider_done
                                else f"Running... ({len(results_data)} texts processed)"
                            ),
                            "metrics": metrics_data,
                            "results": results_data,
                        }
                    )
                else:
                    provider_results.append(
                        {
                            "provider": provider,
                            "success": None,
                            "message": "Queued...",
                            "metrics": None,
                            "results": None,
                        }
                    )

    if provider_results is None:
        # Job hasn't completed yet or no output dir available, show all as queued
        provider_results = [
            {
                "provider": provider,
                "success": None,
                "message": "Queued...",
                "metrics": None,
                "results": None,
            }
            for provider in requested_providers
        ]

    # Normalize metrics format for backward compatibility (list -> dict)
    for provider_result in provider_results:
        if provider_result.get("metrics"):
            provider_result["metrics"] = normalize_metrics(provider_result["metrics"])

    enrich_evaluator_runs_with_current_names(
        provider_results, details.get("evaluators") or []
    )

    # Canonical post-processing: lift per-row outputs into evaluator_outputs[uuid],
    # type-coerce values, build evaluator_runs from in-progress metrics, surface
    # per-row judge errors. Idempotent — safe across in-progress / done / failed.
    post_process_provider_results(
        provider_results,
        evaluator_snapshots=details.get("evaluators") or [],
        evaluator_id_by_metric_key=evaluator_id_by_metric_key,
    )

    # Generate presigned URLs on the fly for completed or failed jobs
    if status in (TaskStatus.DONE.value, TaskStatus.FAILED.value):
        for provider_result in provider_results:
            if provider_result.get("results"):
                for result_row in provider_result["results"]:
                    if "audio_path" in result_row and result_row["audio_path"]:
                        audio_s3_key = result_row["audio_path"]
                        # Skip if already a URL (backwards compatibility)
                        if audio_s3_key.startswith("http") or audio_s3_key.startswith(
                            "s3://"
                        ):
                            continue
                        presigned_url = generate_presigned_download_url(audio_s3_key)
                        if presigned_url:
                            result_row["audio_path"] = presigned_url

    return TaskStatusResponse(
        task_id=task_id,
        status=status,
        language=details.get("language"),
        dataset_id=details.get("dataset_id"),
        dataset_name=details.get("dataset_name"),
        provider_results=provider_results,
        leaderboard_summary=results.get("leaderboard_summary"),
        error=results.get("error"),
        is_public=bool(job.get("is_public")),
        share_token=job.get("share_token"),
    )
