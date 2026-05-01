from typing import List, Optional, Any, Dict
from enum import Enum

from fastapi import APIRouter, Query, Depends, HTTPException
from pydantic import BaseModel

from db import get_all_jobs, get_job, delete_job, get_active_dataset_ids
from auth_utils import get_current_user_id
from utils import (
    TaskStatus,
    try_start_queued_job,
    kill_processes_from_dict,
)


router = APIRouter(prefix="/jobs", tags=["jobs"])

# Job types that share the same queue (used for triggering next job on delete)
EVAL_JOB_TYPES = ["stt-eval", "tts-eval", "annotation-eval"]


class JobType(str, Enum):
    STT = "stt"
    TTS = "tts"


class JobListItem(BaseModel):
    uuid: str
    type: str
    status: str
    dataset_id: Optional[str] = None
    dataset_name: Optional[str] = None
    details: Optional[Dict[str, Any]] = None
    results: Optional[Dict[str, Any]] = None
    created_at: str
    updated_at: str


class JobsListResponse(BaseModel):
    jobs: List[JobListItem]


# Map user-friendly job type to actual job type in database
JOB_TYPE_MAP = {
    JobType.STT: "stt-eval",
    JobType.TTS: "tts-eval",
}


@router.get("", response_model=JobsListResponse)
async def list_jobs(
    job_type: Optional[JobType] = Query(
        None, description="Filter jobs by type: 'stt' or 'tts'"
    ),
    user_id: str = Depends(get_current_user_id),
):
    """
    Get all jobs for the authenticated user, optionally filtered by job type.

    Returns a list of all jobs with their UUID, type, status, details, results, and timestamps.
    Jobs are sorted by created_at descending (most recent first).
    """
    # Map the user-friendly job type to the actual database job type
    db_job_type = JOB_TYPE_MAP.get(job_type) if job_type else None

    jobs = get_all_jobs(user_id=user_id, job_type=db_job_type)

    all_dataset_ids = [
        (job.get("details") or {}).get("dataset_id")
        for job in jobs
    ]
    active_dataset_ids = get_active_dataset_ids(
        [did for did in all_dataset_ids if did]
    )

    job_items = []
    for job in jobs:
        details = job.get("details") or {}
        dataset_id = details.get("dataset_id")
        dataset_active = dataset_id in active_dataset_ids if dataset_id else False
        job_items.append(
            JobListItem(
                uuid=job["uuid"],
                type=job["type"],
                status=job["status"],
                dataset_id=dataset_id if dataset_active else None,
                dataset_name=details.get("dataset_name") if dataset_active else None,
                details=job.get("details"),
                results=job.get("results"),
                created_at=job["created_at"],
                updated_at=job["updated_at"],
            )
        )

    return JobsListResponse(jobs=job_items)


@router.delete("/{job_uuid}")
async def delete_job_endpoint(
    job_uuid: str, user_id: str = Depends(get_current_user_id)
):
    """Delete a job."""
    # Check if job exists and user owns it
    job = get_job(job_uuid, user_id=user_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    # Check if this was a running job (to trigger next queued job after delete)
    was_running = job.get("status") == TaskStatus.IN_PROGRESS.value
    details = job.get("details") or {}

    # Kill running processes if job is in progress
    if was_running:
        running_pids = details.get("running_pids")
        if running_pids:
            kill_processes_from_dict(running_pids, job_uuid)

    deleted = delete_job(job_uuid)
    if not deleted:
        raise HTTPException(status_code=404, detail="Job not found")

    # If the deleted job was running, try to start the next queued job
    if was_running:
        try_start_queued_job(EVAL_JOB_TYPES)

    return {"message": "Job deleted successfully"}
