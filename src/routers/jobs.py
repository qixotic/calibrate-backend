from typing import List, Optional
from enum import Enum

from fastapi import APIRouter, Query, Depends, HTTPException, Path
from pydantic import BaseModel, Field

from db import (
    get_all_jobs_summary,
    get_job,
    delete_job,
    bulk_delete_finished_jobs,
    get_active_dataset_ids,
)
from auth_utils import get_current_org, OrgContext
from pagination import (
    OptionalPaginationParams,
    PaginatedResponse,
    count_and_page,
    page_envelope,
    make_sort_params,
    make_search_params,
)
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
    type: EvalJobType = Field(description="Underlying job type")
    status: TaskStatus = Field(description="Lifecycle state")
    dataset_id: Optional[str] = Field(
        None,
        min_length=36,
        max_length=36,
        description="Source dataset ID",
    )
    dataset_name: Optional[str] = Field(
        None,
        description="Source dataset name",
    )
    providers: List[str] = Field(
        description="Speech providers compared in this run"
    )
    language: Optional[str] = Field(
        None, description="Spoken language of the evaluated audio"
    )
    sample_count: int = Field(description="Number of samples evaluated in this run")
    created_at: str = Field(description="When the job was created (ISO 8601 UTC)")
    updated_at: str = Field(description="When the job was last updated (ISO 8601 UTC)")


# Map user-friendly job type to actual job type in database
JOB_TYPE_MAP = {
    JobType.STT: "stt-eval",
    JobType.TTS: "tts-eval",
}

_JobsSort = make_sort_params(
    sortable=["created_at", "updated_at"], default="created_at"
)
_JobsSearch = make_search_params(searchable=["dataset_name"])


@router.get(
    "", response_model=PaginatedResponse[JobListItem], summary="List jobs"
)
async def list_jobs(
    job_type: Optional[JobType] = Query(
        None, description="Filter jobs by type. Omit for all types"
    ),
    search: _JobsSearch = Depends(),
    sort: _JobsSort = Depends(),
    pagination: OptionalPaginationParams = Depends(),
    ctx: OrgContext = Depends(get_current_org),
):
    """List jobs, newest first"""
    db_job_type = JOB_TYPE_MAP.get(job_type) if job_type else None

    jobs = get_all_jobs_summary(org_uuid=ctx.org_uuid, job_type=db_job_type)

    active_dataset_ids = get_active_dataset_ids(
        [job["dataset_id"] for job in jobs if job.get("dataset_id")]
    )
    for job in jobs:
        if job.get("dataset_id") not in active_dataset_ids:
            job["dataset_id"] = None
            job["dataset_name"] = None

    jobs = search.apply(jobs)
    jobs = sort.apply(jobs)
    page, total = count_and_page(jobs, pagination)

    job_items = [JobListItem(**job) for job in page]
    return page_envelope(job_items, total, pagination)


class BulkDeleteJobsRequest(BaseModel):
    job_uuids: List[str] = Field(
        min_length=1,
        description="Jobs to delete",
    )


class BulkDeleteJobsResponse(BaseModel):
    deleted_count: int = Field(description="Number of jobs deleted")


@router.delete("", response_model=BulkDeleteJobsResponse, summary="Bulk delete jobs")
async def bulk_delete_jobs_endpoint(
    payload: BulkDeleteJobsRequest = ...,
    ctx: OrgContext = Depends(get_current_org),
):
    """Delete several finished jobs at once, rejecting the batch if any is unfinished or unknown"""
    unique_uuids = list(dict.fromkeys(payload.job_uuids))
    result = bulk_delete_finished_jobs(unique_uuids, ctx.org_uuid)
    if result["active"] or result["not_found"]:
        raise HTTPException(
            status_code=400,
            detail={
                "message": (
                    "No jobs were deleted. Every job must be finished "
                    "(done or failed) and belong to this workspace."
                ),
                "active": result["active"],
                "not_found": result["not_found"],
            },
        )
    return BulkDeleteJobsResponse(deleted_count=len(result["deleted"]))


@router.delete("/{job_uuid}", summary="Delete job")
async def delete_job_endpoint(
    job_uuid: str = Path(
        description="Job to delete",
        examples=["a3b2c1d0-e5f4-3210-abcd-ef1234567890"],
    ),
    ctx: OrgContext = Depends(get_current_org),
):
    """Delete a job, stopping it first if it is still running"""
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
