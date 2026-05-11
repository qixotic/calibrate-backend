from typing import Any, Dict, List, Optional
from fastapi import APIRouter, HTTPException, Depends, Query
from pydantic import BaseModel

from db import (
    create_annotator,
    get_annotator,
    get_all_annotators,
    update_annotator,
    delete_annotator,
    ensure_name_unique,
    get_jobs_for_annotator_detailed,
    get_job_counts_for_user_annotators,
    get_annotations_for_user,
    get_annotations_for_annotator_overlap_slots,
)
from auth_utils import get_current_user_id
from annotation_metrics import (
    aggregate_agreement_for_annotator,
    trend_series_for_annotator,
)


router = APIRouter(prefix="/annotators", tags=["annotators"])


class AnnotatorCreate(BaseModel):
    name: str


class AnnotatorUpdate(BaseModel):
    name: Optional[str] = None


class AnnotatorResponse(BaseModel):
    uuid: str
    name: str
    created_at: str
    updated_at: str
    jobs_count: Optional[int] = None
    current_agreement: Optional[float] = None
    pair_count: Optional[int] = None


class AnnotatorCreateResponse(BaseModel):
    uuid: str
    message: str


def _ensure_owned_annotator(annotator_uuid: str, user_id: str):
    annotator = get_annotator(annotator_uuid)
    if not annotator or annotator.get("user_id") != user_id:
        # 404 (not 403) — avoid leaking existence
        raise HTTPException(status_code=404, detail="Annotator not found")
    return annotator


@router.post("", response_model=AnnotatorCreateResponse)
async def create_annotator_endpoint(
    payload: AnnotatorCreate,
    user_id: str = Depends(get_current_user_id),
):
    """Create a new annotator. Name must be unique per account."""
    try:
        with ensure_name_unique(
            "annotators", payload.name, user_id, entity="Annotator"
        ):
            annotator_uuid = create_annotator(name=payload.name, user_id=user_id)
    except ValueError as e:
        # `create_annotator` raises ValueError for non-uniqueness validation
        # too (empty name, missing user_id). The uniqueness collision case
        # is now caught by `ensure_name_unique` directly from the DB
        # IntegrityError, before it gets wrapped to ValueError. Anything
        # reaching here is a genuine input-validation 400.
        raise HTTPException(status_code=400, detail=str(e))
    return AnnotatorCreateResponse(
        uuid=annotator_uuid, message="Annotator created successfully"
    )


@router.get("", response_model=List[AnnotatorResponse])
async def list_annotators(user_id: str = Depends(get_current_user_id)):
    """List all annotators on this account with their per-annotator stats:
    `jobs_count` and `current_agreement` (pairwise mean vs other annotators).
    Both are `null` when there's nothing to compute (no jobs / no overlap).

    Bulk-fetches once: annotators, job counts, and the user's full annotation
    set. Per-annotator agreement is then a Python-side filter over the
    shared annotation list — `aggregate_agreement_for_annotator` already
    selects only slots where the target annotator participated, so feeding
    it the account-wide list is equivalent to the per-annotator query.
    """
    annotators = get_all_annotators(user_id=user_id)
    if not annotators:
        return []
    jobs_count_by_annotator = get_job_counts_for_user_annotators(user_id)
    all_annotations = get_annotations_for_user(user_id)
    out: List[Dict[str, Any]] = []
    for a in annotators:
        agreement, pairs = aggregate_agreement_for_annotator(
            all_annotations, a["uuid"]
        )
        out.append(
            {
                **a,
                "jobs_count": jobs_count_by_annotator.get(a["uuid"], 0),
                "current_agreement": agreement,
                "pair_count": pairs if pairs else None,
            }
        )
    return out


@router.get("/{annotator_uuid}")
async def get_annotator_endpoint(
    annotator_uuid: str,
    bucket: str = Query("month", pattern="^(week|month|year)$"),
    days: int = Query(365, ge=1, le=3650),
    user_id: str = Depends(get_current_user_id),
):
    """Annotator detail: basic info, jobs assigned to this annotator (with
    task name + item/annotation counts), latest agreement vs other annotators,
    and agreement trend series."""
    annotator = _ensure_owned_annotator(annotator_uuid, user_id)

    jobs = get_jobs_for_annotator_detailed(annotator_uuid)

    annotations = get_annotations_for_annotator_overlap_slots(
        user_id=user_id, annotator_id=annotator_uuid
    )
    current, pair_count = aggregate_agreement_for_annotator(
        annotations, annotator_uuid
    )
    series = trend_series_for_annotator(
        annotations, annotator_uuid, bucket=bucket, days=days
    )

    return {
        "annotator": {
            "uuid": annotator["uuid"],
            "name": annotator["name"],
            "created_at": annotator["created_at"],
            "updated_at": annotator["updated_at"],
        },
        "stats": {
            "current_agreement": current,
            "pair_count": pair_count,
            "jobs_count": len(jobs),
        },
        "trend": {
            "bucket": bucket,
            "days": days,
            "series": series,
        },
        "jobs": jobs,
    }


@router.put("/{annotator_uuid}", response_model=AnnotatorResponse)
async def update_annotator_endpoint(
    annotator_uuid: str,
    payload: AnnotatorUpdate,
    user_id: str = Depends(get_current_user_id),
):
    _ensure_owned_annotator(annotator_uuid, user_id)
    try:
        with ensure_name_unique(
            "annotators",
            payload.name,
            user_id,
            entity="Annotator",
            exclude_uuid=annotator_uuid,
        ):
            updated = update_annotator(annotator_uuid=annotator_uuid, name=payload.name)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    if not updated:
        raise HTTPException(status_code=400, detail="No fields to update")
    return get_annotator(annotator_uuid)


@router.delete("/{annotator_uuid}")
async def delete_annotator_endpoint(
    annotator_uuid: str, user_id: str = Depends(get_current_user_id)
):
    _ensure_owned_annotator(annotator_uuid, user_id)
    deleted = delete_annotator(annotator_uuid)
    if not deleted:
        raise HTTPException(status_code=404, detail="Annotator not found")
    return {"message": "Annotator deleted successfully"}
