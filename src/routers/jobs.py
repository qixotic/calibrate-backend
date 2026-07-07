from typing import List, Optional, Any, Dict
from enum import Enum

from fastapi import APIRouter, Query, Depends, HTTPException, Path
from pydantic import BaseModel, Field

from db import get_all_jobs, get_job, delete_job, get_active_dataset_ids
from auth_utils import get_current_org, OrgContext
from utils import (
    TaskStatus,
    EvalJobType,
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
    uuid: str = Field(
        min_length=36,
        max_length=36,
        description="Job ID",
    )
    type: EvalJobType = Field(description="Underlying job type, e.g. `stt-eval`, `tts-eval`")
    status: TaskStatus = Field(description="Lifecycle state, e.g. `queued`, `in_progress`, `done`, `failed`")
    dataset_id: Optional[str] = Field(
        None,
        min_length=36,
        max_length=36,
        description="Source dataset ID; `null` when the dataset has since been deleted",
    )
    dataset_name: Optional[str] = Field(
        None,
        description="Source dataset name; `null` when the dataset has since been deleted",
    )
    details: Optional[Dict[str, Any]] = Field(
        None, description="Job configuration and runtime metadata; `null` if unset"
    )
    results: Optional[Dict[str, Any]] = Field(
        None, description="Job output payload; `null` until the job produces results"
    )
    created_at: str = Field(description="Creation timestamp (ISO 8601 UTC)")
    updated_at: str = Field(description="Last-update timestamp (ISO 8601 UTC)")


class JobsListResponse(BaseModel):
    jobs: List[JobListItem] = Field(description="Jobs, newest first")


# Map user-friendly job type to actual job type in database
JOB_TYPE_MAP = {
    JobType.STT: "stt-eval",
    JobType.TTS: "tts-eval",
}


@router.get("", response_model=JobsListResponse, summary="List jobs")
async def list_jobs(
    job_type: Optional[JobType] = Query(
        None, description="Filter jobs by type: `stt` or `tts`; omit for all types"
    ),
    ctx: OrgContext = Depends(get_current_org),
):
    """List jobs for your workspace, newest first."""
    db_job_type = JOB_TYPE_MAP.get(job_type) if job_type else None

    jobs = get_all_jobs(org_uuid=ctx.org_uuid, job_type=db_job_type)

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


@router.delete("/{job_uuid}", summary="Delete job")
async def delete_job_endpoint(
    job_uuid: str = Path(
        description="Job to delete",
        examples=["a3b2c1d0-e5f4-3210-abcd-ef1234567890"],
    ),
    ctx: OrgContext = Depends(get_current_org),
):
    """Delete a job, stopping it first if it is still running."""
    job = get_job(job_uuid, org_uuid=ctx.org_uuid)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    was_running = job.get("status") == TaskStatus.IN_PROGRESS.value
    details = job.get("details") or {}

    if was_running:
        running_pids = details.get("running_pids")
        if running_pids:
            kill_processes_from_dict(running_pids, job_uuid)

    deleted = delete_job(job_uuid)
    if not deleted:
        raise HTTPException(status_code=404, detail="Job not found")

    if was_running:
        try_start_queued_job(EVAL_JOB_TYPES)

    return {"message": "Job deleted successfully"}
