from typing import Any, Dict, List, Optional
from fastapi import APIRouter, HTTPException, Depends, Path, Query
from pydantic import BaseModel, Field

from db import (
    create_annotator,
    get_annotator,
    get_all_annotators,
    update_annotator,
    delete_annotator,
    ensure_name_unique,
    get_jobs_for_annotator_detailed,
    get_job_counts_for_org_annotators,
    get_annotations_for_org,
    get_annotations_for_annotator_overlap_slots,
)
from auth_utils import get_current_org, OrgContext
from annotation_metrics import (
    aggregate_agreement_for_annotator,
    trend_series_for_annotator,
)


router = APIRouter(prefix="/annotators", tags=["annotators"])

_EXAMPLE_ID = "f47ac10b-58cc-4372-a567-0e02b2c3d479"


class AnnotatorCreate(BaseModel):
    name: str = Field(description="Annotator name, unique within your workspace")


class AnnotatorUpdate(BaseModel):
    name: Optional[str] = Field(
        None, description="New annotator name, unique within your workspace. Omit to leave unchanged"
    )


class AnnotatorResponse(BaseModel):
    uuid: str = Field(
        min_length=36,
        max_length=36,
        description="Annotator ID",
        examples=[_EXAMPLE_ID],
    )
    name: str = Field(description="Annotator name")
    created_at: str = Field(description="When the annotator was created (ISO 8601 UTC)")
    updated_at: str = Field(description="When the annotator was last updated (ISO 8601 UTC)")
    jobs_count: Optional[int] = Field(
        None, description="Number of labelling jobs assigned to this annotator"
    )
    current_agreement: Optional[float] = Field(
        None,
        description="Latest pairwise mean agreement with other annotators, from 0 to 1",
    )
    pair_count: Optional[int] = Field(
        None, description="Number of comparable annotation pairs behind the current agreement score"
    )


class AnnotatorCreateResponse(BaseModel):
    uuid: str = Field(
        min_length=36,
        max_length=36,
        description="ID of the newly created annotator",
        examples=[_EXAMPLE_ID],
    )
    message: str = Field(description="Success message")


def _ensure_owned_annotator(annotator_uuid: str, org_uuid: str):
    annotator = get_annotator(annotator_uuid)
    if not annotator or annotator.get("org_uuid") != org_uuid:
        raise HTTPException(status_code=404, detail="Annotator not found")
    return annotator


@router.post("", response_model=AnnotatorCreateResponse, summary="Create annotator")
def create_annotator_endpoint(
    payload: AnnotatorCreate,
    ctx: OrgContext = Depends(get_current_org),
):
    """Create an annotator, a human labeller who can be assigned annotation tasks"""
    try:
        with ensure_name_unique(
            "annotators", payload.name, ctx.org_uuid, entity="Annotator"
        ):
            annotator_uuid = create_annotator(
                name=payload.name, org_uuid=ctx.org_uuid, user_id=ctx.user_id
            )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return AnnotatorCreateResponse(
        uuid=annotator_uuid, message="Annotator created successfully"
    )


@router.get("", response_model=List[AnnotatorResponse], summary="List annotators")
def list_annotators(ctx: OrgContext = Depends(get_current_org)):
    """List annotators with job counts and agreement stats"""
    annotators = get_all_annotators(org_uuid=ctx.org_uuid)
    if not annotators:
        return []
    jobs_count_by_annotator = get_job_counts_for_org_annotators(ctx.org_uuid)
    all_annotations = get_annotations_for_org(ctx.org_uuid)
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


@router.get("/{annotator_uuid}", summary="Get annotator")
def get_annotator_endpoint(
    annotator_uuid: str = Path(
        description="Annotator to retrieve",
        examples=["f47ac10b-58cc-4372-a567-0e02b2c3d479"],
    ),
    bucket: str = Query(
        "month",
        pattern="^(week|month|year)$",
        description="Time bucket for the agreement trend series (`week`, `month`, or `year`)",
    ),
    days: int = Query(
        365, ge=1, le=3650, description="Trailing window in days for the trend series"
    ),
    ctx: OrgContext = Depends(get_current_org),
):
    """Get one annotator with assigned jobs and an agreement trend series"""
    annotator = _ensure_owned_annotator(annotator_uuid, ctx.org_uuid)

    jobs = get_jobs_for_annotator_detailed(annotator_uuid)

    annotations = get_annotations_for_annotator_overlap_slots(
        org_uuid=ctx.org_uuid, annotator_id=annotator_uuid
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


@router.put("/{annotator_uuid}", response_model=AnnotatorResponse, summary="Update annotator")
def update_annotator_endpoint(
    annotator_uuid: str = Path(
        description="Annotator to update",
        examples=["f47ac10b-58cc-4372-a567-0e02b2c3d479"],
    ),
    payload: AnnotatorUpdate = ...,
    ctx: OrgContext = Depends(get_current_org),
):
    """Update an annotator's name"""
    _ensure_owned_annotator(annotator_uuid, ctx.org_uuid)
    try:
        with ensure_name_unique(
            "annotators",
            payload.name,
            ctx.org_uuid,
            entity="Annotator",
            exclude_uuid=annotator_uuid,
        ):
            updated = update_annotator(annotator_uuid=annotator_uuid, name=payload.name)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    if not updated:
        raise HTTPException(status_code=400, detail="No fields to update")
    return get_annotator(annotator_uuid)


@router.delete("/{annotator_uuid}", summary="Delete annotator")
def delete_annotator_endpoint(
    annotator_uuid: str = Path(
        description="Annotator to delete",
        examples=["f47ac10b-58cc-4372-a567-0e02b2c3d479"],
    ),
    ctx: OrgContext = Depends(get_current_org),
):
    """Soft-delete an annotator"""
    _ensure_owned_annotator(annotator_uuid, ctx.org_uuid)
    deleted = delete_annotator(annotator_uuid)
    if not deleted:
        raise HTTPException(status_code=404, detail="Annotator not found")
    return {"message": "Annotator deleted successfully"}
