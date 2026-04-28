import mimetypes
import os
import signal
import logging
import threading
import time
import json
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import List, Optional, Dict, Any, Union

import boto3
import openpyxl
import sentry_sdk
from pydantic import BaseModel

logger = logging.getLogger(__name__)


def capture_exception_to_sentry(exception: Exception) -> None:
    """
    Capture an exception to Sentry and mark it as unhandled.

    This ensures job failures appear as unresolved issues in Sentry
    rather than handled/resolved exceptions.
    """
    sentry_sdk.capture_exception(
        exception,
        hint={"mechanism": {"type": "generic", "handled": False}},
    )
    # Flush to ensure the event is sent immediately (important for background tasks)
    sentry_sdk.flush(timeout=2)


def build_tool_configs(agent_tools: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Build tool configurations for calibrate CLI from agent tools.

    Handles both structured output tools and webhook tools:
    - Structured output tools: Include type, name, description, and parameters
    - Webhook tools: Include type, name, description, and full webhook configuration

    Args:
        agent_tools: List of tool dicts from get_tools_for_agent()

    Returns:
        List of tool config dicts ready for calibrate config
    """
    tool_configs = []
    for tool in agent_tools:
        tool_config = tool.get("config", {})
        tool_type = tool_config.get("type", "structured_output")

        tool_entry = {
            "name": tool["name"],
            "description": tool["description"],
        }

        if tool_type == "webhook":
            # For webhook tools, include the full webhook configuration
            tool_entry["type"] = "webhook"
            tool_entry["parameters"] = tool_config.get("parameters", [])
            tool_entry["webhook"] = tool_config.get("webhook", {})
        else:
            # For structured output tools (default)
            tool_entry["type"] = "structured_output"
            tool_entry["parameters"] = tool_config.get("parameters", [])

        tool_configs.append(tool_entry)

    return tool_configs


# Timeout threshold for marking jobs as failed (5 minutes)
JOB_TIMEOUT_SECONDS = 3600
# Presigned URL caching constants
PRESIGNED_URL_EXPIRY_SECONDS = 3600  # 1 hour
PRESIGNED_URL_REFRESH_BUFFER_SECONDS = 300  # Refresh 5 minutes before expiry

# In-memory task storage (shared across routers)
tasks = {}
tasks_lock = threading.Lock()


class TaskStatus(str, Enum):
    QUEUED = "queued"
    IN_PROGRESS = "in_progress"
    CANCELLED = "cancelled"
    DONE = "done"
    FAILED = "failed"


class EvaluatorRunEntry(BaseModel):
    """One evaluator's aggregate metrics for an STT/TTS provider, paired with evaluator UUID."""

    evaluator_uuid: str
    metric_key: str  # key as emitted in metrics.json (derived from CLI/config at run time)
    aggregate: Dict[str, Any]
    name: Optional[str] = None  # filled on API read from DB + job snapshot


class ProviderResult(BaseModel):
    provider: str
    success: Optional[bool] = None  # None while in progress, True/False when done
    metrics: Optional[Union[Dict[str, Any], List[Dict[str, Any]]]] = (
        None  # dict (new format) or list (backward compat)
    )
    results: Optional[List[Dict[str, Any]]] = None
    evaluator_runs: Optional[List[EvaluatorRunEntry]] = None


class TaskCreateResponse(BaseModel):
    task_id: str
    status: str
    dataset_id: Optional[str] = None
    dataset_name: Optional[str] = None


class TaskStatusResponse(BaseModel):
    task_id: str
    status: str
    language: Optional[str] = None
    dataset_id: Optional[str] = None
    dataset_name: Optional[str] = None
    provider_results: Optional[List[ProviderResult]] = None
    leaderboard_summary: Optional[List[Dict[str, Any]]] = None
    error: Optional[str] = None
    is_public: bool = False
    share_token: Optional[str] = None


def kill_process_group(pid: int, job_id: str) -> bool:
    """Kill a process group by PID.

    Args:
        pid: Process ID (also used as PGID when start_new_session=True)
        job_id: Job ID for logging

    Returns:
        True if process was killed or didn't exist, False on error
    """
    if not pid:
        return True

    try:
        os.killpg(pid, signal.SIGTERM)
        logger.info(f"Job {job_id}: Sent SIGTERM to process group {pid}")

        time.sleep(0.5)

        try:
            os.killpg(pid, signal.SIGKILL)
            logger.info(f"Job {job_id}: Sent SIGKILL to process group {pid}")
        except (ProcessLookupError, PermissionError):
            # Process already terminated after SIGTERM
            logger.info(
                f"Job {job_id}: Process group {pid} already terminated after SIGTERM"
            )

        return True
    except ProcessLookupError:
        logger.info(f"Job {job_id}: Process group {pid} not found (already dead)")
        return True
    except PermissionError:
        logger.warning(f"Job {job_id}: No permission to kill process group {pid}")
        return False
    except Exception as e:
        logger.error(f"Job {job_id}: Error killing process group {pid}: {e}")
        return False


def kill_processes_from_dict(pids_dict: dict, job_id: str) -> None:
    """Kill multiple processes from a dict mapping (e.g., provider -> PID).

    Args:
        pids_dict: Dict mapping names to PIDs (e.g., {"deepgram": 12345, "openai": 12346})
        job_id: Job ID for logging
    """
    if not pids_dict:
        logger.info(f"Job {job_id}: No running PIDs to kill")
        return

    for name, pid in pids_dict.items():
        if not pid:
            continue
        try:
            os.killpg(pid, signal.SIGTERM)
            logger.info(f"Job {job_id}: Sent SIGTERM to process group {pid} ({name})")

            time.sleep(0.5)

            try:
                os.killpg(pid, signal.SIGKILL)
                logger.info(
                    f"Job {job_id}: Sent SIGKILL to process group {pid} ({name})"
                )
            except ProcessLookupError:
                logger.info(
                    f"Job {job_id}: Process group {pid} ({name}) already terminated"
                )
        except ProcessLookupError:
            logger.info(f"Job {job_id}: Process group {pid} ({name}) not found")
        except PermissionError:
            logger.warning(
                f"Job {job_id}: No permission to kill process group {pid} ({name})"
            )
        except Exception as e:
            logger.error(
                f"Job {job_id}: Error killing process group {pid} ({name}): {e}"
            )


def is_job_timed_out(
    updated_at: str, timeout_seconds: int = JOB_TIMEOUT_SECONDS
) -> bool:
    """Check if a job has timed out based on its updated_at timestamp.

    Args:
        updated_at: ISO format timestamp string (from SQLite, stored in UTC)

    Returns:
        True if the job hasn't been updated in more than JOB_TIMEOUT_SECONDS
    """
    try:
        # Parse the timestamp (SQLite format: "YYYY-MM-DD HH:MM:SS", stored in UTC)
        last_update = datetime.fromisoformat(updated_at.replace(" ", "T"))
        # Use UTC for comparison since SQLite CURRENT_TIMESTAMP is in UTC
        timeout_threshold = datetime.utcnow() - timedelta(seconds=timeout_seconds)
        return last_update < timeout_threshold
    except Exception as e:
        logger.warning(f"Error parsing timestamp {updated_at}: {e}")
        return False


def get_s3_client():
    """Get S3 client from environment variables."""
    aws_access_key_id = os.getenv("AWS_ACCESS_KEY_ID")
    aws_secret_access_key = os.getenv("AWS_SECRET_ACCESS_KEY")
    aws_region = os.getenv("AWS_REGION", "ap-south-1")

    if aws_access_key_id and aws_secret_access_key:
        return boto3.client(
            "s3",
            aws_access_key_id=aws_access_key_id,
            aws_secret_access_key=aws_secret_access_key,
            region_name=aws_region,
        )

    return boto3.client("s3", region_name=aws_region)


def get_s3_output_config():
    """Get S3 output configuration from environment variables."""
    bucket = os.getenv("S3_OUTPUT_BUCKET")

    if not bucket:
        raise ValueError("S3_OUTPUT_BUCKET environment variable is required")

    return bucket


def upload_file_to_s3(
    s3_client,
    local_path: Union[str, Path],
    bucket: str,
    s3_key: str,
) -> None:
    """Upload a local file to S3 with a guessed Content-Type.

    Setting Content-Type ensures browsers render supported files (JSON, audio,
    text) inline instead of forcing a download (which happens when S3 defaults
    to binary/octet-stream).
    """
    content_type, _ = mimetypes.guess_type(str(local_path))
    extra_args = {"ContentType": content_type} if content_type else None
    s3_client.upload_file(str(local_path), bucket, s3_key, ExtraArgs=extra_args)


def upload_top_level_files_to_s3(
    s3_client,
    local_dir: Path,
    bucket: str,
    key_prefix: str,
) -> None:
    """Upload only regular files directly under ``local_dir`` (e.g. run-level ``logs``, ``leaderboard.csv``).

    Subdirectories are ignored; use :func:`upload_directory_tree_to_s3` for a full tree.
    """
    if not local_dir or not local_dir.is_dir():
        return
    prefix = key_prefix.rstrip("/")
    for p in local_dir.iterdir():
        if p.is_file():
            s3_key = f"{prefix}/{p.name}"
            upload_file_to_s3(s3_client, p, bucket, s3_key)


def upload_directory_tree_to_s3(
    s3_client,
    local_root: Path,
    bucket: str,
    key_prefix: str,
) -> None:
    """Recursively upload every file under ``local_root`` to ``s3://bucket/{key_prefix}/<relative>``."""
    if not local_root or not local_root.exists():
        return
    local_root = local_root.resolve()
    prefix = key_prefix.rstrip("/")
    for root, dirs, files in os.walk(local_root):
        for file in files:
            local_file_path = Path(root) / file
            relative_path = local_file_path.relative_to(local_root)
            s3_key = f"{prefix}/{relative_path.as_posix()}"
            upload_file_to_s3(s3_client, local_file_path, bucket, s3_key)


def generate_presigned_download_url(
    s3_key: str,
    bucket: Optional[str] = None,
    expiration: int = PRESIGNED_URL_EXPIRY_SECONDS,
) -> Optional[str]:
    """Generate a presigned URL for downloading (get_object) from S3.

    Args:
        s3_key: The S3 object key
        bucket: S3 bucket name (defaults to S3_OUTPUT_BUCKET env var)
        expiration: URL expiration time in seconds (default: 1 hour)

    Returns:
        Presigned URL string, or None if generation fails
    """
    try:
        s3 = get_s3_client()
        s3_bucket = bucket or get_s3_output_config()

        return s3.generate_presigned_url(
            "get_object",
            Params={
                "Bucket": s3_bucket,
                "Key": s3_key,
            },
            ExpiresIn=expiration,
        )
    except Exception as e:
        logger.warning(f"Failed to generate presigned download URL for {s3_key}: {e}")
        return None


def presign_audio_path(audio_path: Optional[str]) -> Optional[str]:
    """Convert an s3://bucket/key path (or plain key) to a presigned download URL."""
    if not audio_path:
        return audio_path
    if audio_path.startswith("s3://"):
        parts = audio_path[5:].split("/", 1)
        bucket = parts[0]
        key = parts[1] if len(parts) > 1 else ""
        return generate_presigned_download_url(key, bucket=bucket) or audio_path
    if audio_path.startswith("http"):
        return audio_path
    return generate_presigned_download_url(audio_path) or audio_path


def generate_presigned_upload_url(
    s3_key: str,
    content_type: str,
    bucket: Optional[str] = None,
    expiration: int = PRESIGNED_URL_EXPIRY_SECONDS,
) -> Optional[str]:
    """Generate a presigned URL for uploading (put_object) to S3.

    Args:
        s3_key: The S3 object key
        content_type: The content type of the file to upload
        bucket: S3 bucket name (defaults to S3_OUTPUT_BUCKET env var)
        expiration: URL expiration time in seconds (default: 1 hour)

    Returns:
        Presigned URL string, or None if generation fails
    """
    try:
        s3 = get_s3_client()
        s3_bucket = bucket or get_s3_output_config()

        return s3.generate_presigned_url(
            "put_object",
            Params={
                "Bucket": s3_bucket,
                "Key": s3_key,
                "ContentType": content_type,
            },
            ExpiresIn=expiration,
        )
    except Exception as e:
        logger.warning(f"Failed to generate presigned upload URL for {s3_key}: {e}")
        return None


def get_max_concurrent_jobs() -> int:
    """Get the maximum number of concurrent jobs from environment variable.

    Defaults to 2 if not set.
    """
    return int(os.getenv("MAX_CONCURRENT_JOBS"))


def get_max_concurrent_jobs_per_user() -> int:
    """Get the maximum number of concurrent jobs per user from environment variable.

    Defaults to 1 if not set. Set to 0 to disable user-level limit.
    """
    return int(os.getenv("MAX_CONCURRENT_JOBS_PER_USER", "1"))


# Job queue lock to ensure thread-safe queue operations
_job_queue_lock = threading.Lock()

# Registry of job starter callbacks by job type
_job_starters: Dict[str, callable] = {}


def register_job_starter(job_type: str, starter_callback: callable) -> None:
    """Register a callback function for starting jobs of a specific type.

    Args:
        job_type: The job type (e.g., "stt-eval", "tts-eval")
        starter_callback: Function that takes a job dict and starts the job.
    """
    _job_starters[job_type] = starter_callback
    logger.info(f"Registered job starter for type: {job_type}")


def try_start_queued_job(job_types: List[str]) -> bool:
    """Try to start the next queued job if there's capacity.

    Checks both global limit and per-user limit for each queued job.

    Args:
        job_types: List of job types to consider (e.g., ["stt-eval", "tts-eval"])

    Returns:
        True if a job was started, False otherwise.
    """
    # Import here to avoid circular imports
    from db import (
        count_running_jobs,
        count_running_jobs_for_user,
        get_queued_jobs,
        update_job,
    )

    with _job_queue_lock:
        max_jobs = get_max_concurrent_jobs()
        running_count = count_running_jobs(job_types)

        logger.info(f"Job queue check: {running_count}/{max_jobs} jobs running")

        if running_count >= max_jobs:
            logger.info("Max concurrent jobs reached, not starting new job")
            return False

        # Get all queued jobs (FIFO order)
        queued_jobs = get_queued_jobs(job_types)
        if not queued_jobs:
            logger.info("No queued jobs to start")
            return False

        # Find the first job that can be started (respects per-user limit)
        max_jobs_per_user = get_max_concurrent_jobs_per_user()
        job_to_start = None

        for job in queued_jobs:
            user_id = job.get("user_id")
            if max_jobs_per_user > 0 and user_id:  # 0 means disabled
                user_running_count = count_running_jobs_for_user(user_id, job_types)
                if user_running_count >= max_jobs_per_user:
                    logger.info(
                        f"User {user_id} has {user_running_count}/{max_jobs_per_user} jobs running, skipping job {job['uuid']}"
                    )
                    continue
            job_to_start = job
            break

        if not job_to_start:
            logger.info("No queued jobs can be started (all users at their limit)")
            return False

        job_id = job_to_start["uuid"]
        job_type = job_to_start.get("type")

        # Find the appropriate starter callback
        starter_callback = _job_starters.get(job_type)
        if not starter_callback:
            logger.error(f"No job starter registered for type: {job_type}")
            return False

        # Update status to in_progress before starting
        update_job(job_id, status=TaskStatus.IN_PROGRESS.value)
        logger.info(f"Starting queued job {job_id} of type {job_type}")

        try:
            # Start the job (this should spawn a thread)
            starter_callback(job_to_start)
            return True
        except Exception as e:
            # If starting fails, mark as done with error
            logger.error(f"Failed to start job {job_id}: {e}")
            update_job(
                job_id,
                status=TaskStatus.DONE.value,
                results={"error": f"Failed to start job: {str(e)}"},
            )
            return False


def can_start_job(job_types: List[str], user_id: str) -> bool:
    """Check if there's capacity to start a new job immediately.

    Checks both global limit and per-user limit.

    Args:
        job_types: List of job types to consider for counting running jobs.
        user_id: UUID of the user requesting the job.

    Returns:
        True if a new job can be started, False otherwise.
    """
    from db import count_running_jobs, count_running_jobs_for_user

    with _job_queue_lock:
        # Check global limit
        max_jobs = get_max_concurrent_jobs()
        running_count = count_running_jobs(job_types)
        if running_count >= max_jobs:
            return False

        # Check per-user limit
        max_jobs_per_user = get_max_concurrent_jobs_per_user()
        if max_jobs_per_user > 0:  # 0 means disabled
            user_running_count = count_running_jobs_for_user(user_id, job_types)
            if user_running_count >= max_jobs_per_user:
                return False

        return True


# ============ Agent Test Job Queue Functions ============


def try_start_queued_agent_test_job(job_types: List[str]) -> bool:
    """Try to start the next queued agent test job if there's capacity.

    Checks both global limit and per-user limit for each queued job.

    Args:
        job_types: List of job types to consider (e.g., ["llm-unit-test", "llm-benchmark"])

    Returns:
        True if a job was started, False otherwise.
    """
    from db import (
        count_running_agent_test_jobs,
        count_running_agent_test_jobs_for_user,
        get_queued_agent_test_jobs,
        update_agent_test_job,
    )

    with _job_queue_lock:
        max_jobs = get_max_concurrent_jobs()
        running_count = count_running_agent_test_jobs(job_types)

        logger.info(
            f"Agent test job queue check: {running_count}/{max_jobs} jobs running"
        )

        if running_count >= max_jobs:
            logger.info("Max concurrent jobs reached, not starting new agent test job")
            return False

        # Get all queued jobs (FIFO order)
        queued_jobs = get_queued_agent_test_jobs(job_types)
        if not queued_jobs:
            logger.info("No queued agent test jobs to start")
            return False

        # Find the first job that can be started (respects per-user limit)
        max_jobs_per_user = get_max_concurrent_jobs_per_user()
        job_to_start = None

        for job in queued_jobs:
            user_id = job.get("user_id")
            if max_jobs_per_user > 0 and user_id:  # 0 means disabled
                user_running_count = count_running_agent_test_jobs_for_user(
                    user_id, job_types
                )
                if user_running_count >= max_jobs_per_user:
                    logger.info(
                        f"User {user_id} has {user_running_count}/{max_jobs_per_user} agent test jobs running, skipping job {job['uuid']}"
                    )
                    continue
            job_to_start = job
            break

        if not job_to_start:
            logger.info(
                "No queued agent test jobs can be started (all users at their limit)"
            )
            return False

        job_id = job_to_start["uuid"]
        job_type = job_to_start.get("type")

        # Find the appropriate starter callback
        starter_callback = _job_starters.get(job_type)
        if not starter_callback:
            logger.error(f"No job starter registered for type: {job_type}")
            return False

        # Update status to in_progress before starting
        update_agent_test_job(job_id, status=TaskStatus.IN_PROGRESS.value)
        logger.info(f"Starting queued agent test job {job_id} of type {job_type}")

        try:
            # Start the job (this should spawn a thread)
            starter_callback(job_to_start)
            return True
        except Exception as e:
            # If starting fails, mark as done with error
            logger.error(f"Failed to start agent test job {job_id}: {e}")
            update_agent_test_job(
                job_id,
                status=TaskStatus.DONE.value,
                results={"error": f"Failed to start job: {str(e)}"},
            )
            return False


def can_start_agent_test_job(job_types: List[str], user_id: str) -> bool:
    """Check if there's capacity to start a new agent test job immediately.

    Checks both global limit and per-user limit.

    Args:
        job_types: List of job types to consider for counting running jobs.
        user_id: UUID of the user requesting the job.

    Returns:
        True if a new job can be started, False otherwise.
    """
    from db import count_running_agent_test_jobs, count_running_agent_test_jobs_for_user

    with _job_queue_lock:
        # Check global limit
        max_jobs = get_max_concurrent_jobs()
        running_count = count_running_agent_test_jobs(job_types)
        if running_count >= max_jobs:
            return False

        # Check per-user limit
        max_jobs_per_user = get_max_concurrent_jobs_per_user()
        if max_jobs_per_user > 0:  # 0 means disabled
            user_running_count = count_running_agent_test_jobs_for_user(
                user_id, job_types
            )
            if user_running_count >= max_jobs_per_user:
                return False

        return True


# ============ Simulation Job Queue Functions ============


def try_start_queued_simulation_job(job_types: List[str]) -> bool:
    """Try to start the next queued simulation job if there's capacity.

    Checks both global limit and per-user limit for each queued job.

    Args:
        job_types: List of job types to consider (e.g., ["text", "voice"])

    Returns:
        True if a job was started, False otherwise.
    """
    from db import (
        count_running_simulation_jobs,
        count_running_simulation_jobs_for_user,
        get_queued_simulation_jobs,
        update_simulation_job,
    )

    with _job_queue_lock:
        max_jobs = get_max_concurrent_jobs()
        running_count = count_running_simulation_jobs(job_types)

        logger.info(
            f"Simulation job queue check: {running_count}/{max_jobs} jobs running"
        )

        if running_count >= max_jobs:
            logger.info("Max concurrent jobs reached, not starting new simulation job")
            return False

        # Get all queued jobs (FIFO order)
        queued_jobs = get_queued_simulation_jobs(job_types)
        if not queued_jobs:
            logger.info("No queued simulation jobs to start")
            return False

        # Find the first job that can be started (respects per-user limit)
        max_jobs_per_user = get_max_concurrent_jobs_per_user()
        job_to_start = None

        for job in queued_jobs:
            user_id = job.get("user_id")
            if max_jobs_per_user > 0 and user_id:  # 0 means disabled
                user_running_count = count_running_simulation_jobs_for_user(
                    user_id, job_types
                )
                if user_running_count >= max_jobs_per_user:
                    logger.info(
                        f"User {user_id} has {user_running_count}/{max_jobs_per_user} simulation jobs running, skipping job {job['uuid']}"
                    )
                    continue
            job_to_start = job
            break

        if not job_to_start:
            logger.info(
                "No queued simulation jobs can be started (all users at their limit)"
            )
            return False

        job_id = job_to_start["uuid"]
        job_type = job_to_start.get("type")

        # Find the appropriate starter callback
        starter_callback = _job_starters.get(job_type)
        if not starter_callback:
            logger.error(f"No job starter registered for type: {job_type}")
            return False

        # Update status to in_progress before starting
        update_simulation_job(job_id, status=TaskStatus.IN_PROGRESS.value)
        logger.info(f"Starting queued simulation job {job_id} of type {job_type}")

        try:
            # Start the job (this should spawn a thread)
            starter_callback(job_to_start)
            return True
        except Exception as e:
            # If starting fails, mark as done with error
            logger.error(f"Failed to start simulation job {job_id}: {e}")
            update_simulation_job(
                job_id,
                status=TaskStatus.DONE.value,
                results={"error": f"Failed to start job: {str(e)}"},
            )
            return False


def can_start_simulation_job(job_types: List[str], user_id: str) -> bool:
    """Check if there's capacity to start a new simulation job immediately.

    Checks both global limit and per-user limit.

    Args:
        job_types: List of job types to consider for counting running jobs.
        user_id: UUID of the user requesting the job.

    Returns:
        True if a new job can be started, False otherwise.
    """
    from db import count_running_simulation_jobs, count_running_simulation_jobs_for_user

    with _job_queue_lock:
        # Check global limit
        max_jobs = get_max_concurrent_jobs()
        running_count = count_running_simulation_jobs(job_types)
        if running_count >= max_jobs:
            return False

        # Check per-user limit
        max_jobs_per_user = get_max_concurrent_jobs_per_user()
        if max_jobs_per_user > 0:  # 0 means disabled
            user_running_count = count_running_simulation_jobs_for_user(
                user_id, job_types
            )
            if user_running_count >= max_jobs_per_user:
                return False

        return True


def normalize_metrics(metrics):
    """Convert old list-of-dicts metrics format to new dict format.

    Old format: [{"wer": 2.4}, {"string_similarity": 0.15}, {"metric_name": "ttfb", "mean": 0.1, ...}, ...]
    New format: {"wer": 2.4, "string_similarity": 0.15, "ttfb": {"mean": 0.1, ...}, ...}
    """
    if metrics is None:
        return None
    if isinstance(metrics, dict):
        return metrics
    if isinstance(metrics, list):
        result = {}
        for item in metrics:
            if isinstance(item, dict):
                if "metric_name" in item:
                    metric_name = item["metric_name"]
                    value = {k: v for k, v in item.items() if k != "metric_name"}
                    result[metric_name] = value
                else:
                    result.update(item)
        return result if result else metrics
    return metrics


def is_evaluator_metric_aggregate(value: Any) -> bool:
    """True for nested evaluator outputs in metrics.json (excludes wer scalars, ttfb, etc.)."""
    return isinstance(value, dict) and "type" in value


def ordered_evaluator_metric_keys(metrics: Optional[Dict[str, Any]]) -> List[str]:
    if not metrics or not isinstance(metrics, dict):
        return []
    return [k for k, v in metrics.items() if is_evaluator_metric_aggregate(v)]


def read_evaluators_map_from_config(output_dir: Optional[Path]) -> Dict[str, str]:
    """Read root config.json evaluators_map as {metric_key: evaluator_uuid}."""
    if not output_dir:
        return {}
    config_path = output_dir / "config.json"
    if not config_path.exists():
        return {}
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)
    except Exception as e:
        logger.warning(f"Failed to read evaluator map from {config_path}: {e}")
        return {}

    raw = config.get("evaluators_map")
    if not isinstance(raw, dict):
        return {}
    return {
        str(name): str(evaluator_id)
        for evaluator_id, name in raw.items()
        if evaluator_id and name
    }


def build_evaluator_runs_for_eval_job(
    metrics: Any,
    evaluator_id_by_metric_key: Optional[Dict[str, str]] = None,
) -> List[EvaluatorRunEntry]:
    """Pair evaluator aggregates with UUIDs from calibrate's config map."""
    metrics_dict = normalize_metrics(metrics)
    if not isinstance(metrics_dict, dict):
        return []
    keys = ordered_evaluator_metric_keys(metrics_dict)
    if not keys:
        return []
    evaluator_id_by_metric_key = evaluator_id_by_metric_key or {}
    runs: List[EvaluatorRunEntry] = []
    for mk in keys:
        eu = evaluator_id_by_metric_key.get(mk)
        if not eu:
            continue
        agg = metrics_dict.get(mk)
        if agg is None:
            continue
        runs.append(
            EvaluatorRunEntry(
                evaluator_uuid=eu,
                metric_key=mk,
                aggregate=dict(agg) if isinstance(agg, dict) else {"value": agg},
            )
        )
    return runs


def enrich_evaluator_runs_with_current_names(
    provider_results: Optional[List[Any]],
) -> None:
    """Mutates each provider result dict in place and refreshes evaluator run names."""
    if not provider_results:
        return
    from db import get_evaluator

    for pr in provider_results:
        runs_raw = pr.get("evaluator_runs")
        if not runs_raw:
            continue
        for run in runs_raw:
            if not isinstance(run, dict):
                continue
            uid = run.get("evaluator_uuid")
            row = get_evaluator(uid) if uid else None
            run["name"] = (
                (row.get("name") if row else None)
                or run.get("metric_key")
                or ""
            )


def read_leaderboard_xlsx(leaderboard_dir: Path) -> Optional[List[dict]]:
    """Read the leaderboard summary from the xlsx file in leaderboard directory.

    Looks for any .xlsx file in the directory (commonly stt_leaderboard.xlsx or tts_leaderboard.xlsx).
    """
    if not leaderboard_dir.exists():
        logger.warning(f"Leaderboard directory does not exist: {leaderboard_dir}")
        return None

    xlsx_files = list(leaderboard_dir.glob("*.xlsx"))
    if not xlsx_files:
        logger.warning(
            f"No xlsx files found in leaderboard directory: {leaderboard_dir}"
        )
        all_files = list(leaderboard_dir.iterdir())
        logger.info(f"Files in leaderboard directory: {[f.name for f in all_files]}")
        return None

    xlsx_file = xlsx_files[0]
    logger.info(f"Reading leaderboard from: {xlsx_file}")

    try:
        wb = openpyxl.load_workbook(str(xlsx_file), data_only=True)
        logger.info(f"Workbook sheets: {wb.sheetnames}")

        if "summary" not in wb.sheetnames:
            logger.warning(
                f"'summary' sheet not found in {xlsx_file.name}, sheets: {wb.sheetnames}"
            )
            return None

        ws = wb["summary"]
        headers = [cell.value for cell in ws[1] if cell.value is not None]
        logger.info(f"Leaderboard headers: {headers}")

        leaderboard_summary = []
        for row in ws.iter_rows(min_row=2, values_only=False):
            if any(cell.value is not None for cell in row):
                row_dict = {}
                for idx, cell in enumerate(row):
                    if idx < len(headers):
                        row_dict[headers[idx]] = cell.value
                if any(v is not None for v in row_dict.values()):
                    leaderboard_summary.append(row_dict)

        logger.info(f"Read {len(leaderboard_summary)} rows from leaderboard")
        return leaderboard_summary
    except Exception as e:
        logger.warning(f"Failed to read leaderboard xlsx: {e}")
        return None
