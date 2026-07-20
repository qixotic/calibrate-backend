"""Workspace limits (superadmin configuration).

Set caps for each workspace on dataset rows per eval run and on stored traces.
Members can read their workspace's effective limits via `/me/max-rows-per-eval`
and `/me/max-traces`.
"""

import os
import sqlite3

from typing import Optional

from fastapi import APIRouter, HTTPException, Depends, Path
from pydantic import BaseModel, Field

from db import (
    create_org_limits,
    get_member_role,
    get_organization,
    get_org_limits,
    update_org_limits,
    delete_org_limits,
)
from auth_utils import get_current_org, OrgContext, require_superadmin, is_superadmin_user

router = APIRouter(prefix="/org-limits", tags=["org-limits"])

DEFAULT_MAX_ROWS_PER_EVAL = int(os.getenv("DEFAULT_MAX_ROWS_PER_EVAL", "20"))
DEFAULT_MAX_TRACES = int(os.getenv("DEFAULT_MAX_TRACES", "50000"))


class OrgLimits(BaseModel):
    max_rows_per_eval: int = Field(
        gt=0,
        le=10000,
        description="Maximum dataset rows a single eval run may process",
    )
    # Traces are machine-ingested, so the ceiling is orders of magnitude above
    # max_rows_per_eval's; don't reuse that field's le=10000 bound.
    max_traces: Optional[int] = Field(
        None,
        gt=0,
        le=1_000_000,
        description="Maximum traces the workspace can store. Omit to keep the server default",
    )


def get_max_traces_for_org(org_uuid: str) -> int:
    """Effective trace cap for a workspace: its org_limits row, else the
    DEFAULT_MAX_TRACES env fallback. Enforced by POST /traces."""
    limits = get_org_limits(org_uuid)
    if limits and limits.get("limits", {}).get("max_traces"):
        return limits["limits"]["max_traces"]
    return DEFAULT_MAX_TRACES


class OrgLimitsCreate(BaseModel):
    org_uuid: str = Field(
        min_length=36,
        max_length=36,
        description="Workspace to create limits for",
        examples=["f47ac10b-58cc-4372-a567-0e02b2c3d479"],
    )
    limits: OrgLimits = Field(description="Limit values to set")


class OrgLimitsUpdate(BaseModel):
    limits: OrgLimits = Field(description="New limit values")


class OrgLimitsResponse(BaseModel):
    uuid: str = Field(
        min_length=36,
        max_length=36,
        description="Limits record ID",
    )
    org_uuid: str = Field(
        min_length=36,
        max_length=36,
        description="Workspace these limits apply to",
        examples=["f47ac10b-58cc-4372-a567-0e02b2c3d479"],
    )
    limits: OrgLimits = Field(description="Current limit values")
    created_at: str = Field(description="When the limits record was created (ISO 8601 UTC)")
    updated_at: str = Field(description="When the limits record was last updated (ISO 8601 UTC)")


class OrgLimitsCreateResponse(BaseModel):
    uuid: str = Field(
        min_length=36,
        max_length=36,
        description="ID of the newly created limits record",
    )
    message: str = Field(description="Status message")


@router.get("/me/max-rows-per-eval", summary="Get own max rows per eval")
async def get_max_rows_per_eval(ctx: OrgContext = Depends(get_current_org)):
    """Get the max rows per eval"""
    # Falls back to DEFAULT_MAX_ROWS_PER_EVAL when no workspace-specific limit is set.
    limits = get_org_limits(ctx.org_uuid)
    if limits and "max_rows_per_eval" in limits.get("limits", {}):
        return {"max_rows_per_eval": limits["limits"]["max_rows_per_eval"]}
    return {"max_rows_per_eval": DEFAULT_MAX_ROWS_PER_EVAL}


@router.get("/me/max-traces", summary="Get own max traces")
async def get_max_traces(ctx: OrgContext = Depends(get_current_org)):
    """Get the maximum number of traces your workspace can store"""
    return {"max_traces": get_max_traces_for_org(ctx.org_uuid)}


@router.post("", response_model=OrgLimitsCreateResponse, summary="Create workspace limits")
async def create_org_limits_endpoint(
    data: OrgLimitsCreate, user_id: str = Depends(require_superadmin)
):
    """Create limits for a workspace. Superadmin only"""
    # 404 if workspace missing; 409 if limits already exist (use PUT to update).
    if not get_organization(data.org_uuid):
        raise HTTPException(status_code=404, detail="Organization not found")
    existing = get_org_limits(data.org_uuid)
    if existing:
        raise HTTPException(
            status_code=409,
            detail="Limits already exist for this organization. Use PUT to update.",
        )
    try:
        row_uuid = create_org_limits(org_uuid=data.org_uuid, limits=data.limits)
    except sqlite3.IntegrityError:
        raise HTTPException(
            status_code=409,
            detail="Limits already exist for this organization. Use PUT to update.",
        )
    return OrgLimitsCreateResponse(
        uuid=row_uuid, message="Organization limits created successfully"
    )


@router.get("/{target_org_uuid}", response_model=OrgLimitsResponse, summary="Get workspace limits")
async def get_org_limits_endpoint(
    target_org_uuid: str = Path(
        description="The workspace whose limits to read. You must be a member",
        examples=["f47ac10b-58cc-4372-a567-0e02b2c3d479"],
    ),
    ctx: OrgContext = Depends(get_current_org),
):
    """Get limits for a workspace you belong to"""
    if get_member_role(target_org_uuid, ctx.user_id) is None and not is_superadmin_user(ctx.user_id):
        raise HTTPException(status_code=404, detail="Organization limits not found")
    limits = get_org_limits(target_org_uuid)
    if not limits:
        raise HTTPException(status_code=404, detail="Organization limits not found")
    return limits


@router.put("/{target_org_uuid}", response_model=OrgLimitsResponse, summary="Update workspace limits")
async def update_org_limits_endpoint(
    target_org_uuid: str = Path(
        description="The workspace whose limits to update",
        examples=["f47ac10b-58cc-4372-a567-0e02b2c3d479"],
    ),
    data: OrgLimitsUpdate = ...,
    user_id: str = Depends(require_superadmin),
):
    """Update limits for a workspace. Superadmin only"""
    updated = update_org_limits(org_uuid=target_org_uuid, limits=data.limits)
    if not updated:
        raise HTTPException(status_code=404, detail="Organization limits not found")
    return updated


@router.delete("/{target_org_uuid}", summary="Delete workspace limits")
async def delete_org_limits_endpoint(
    target_org_uuid: str = Path(
        description="The workspace whose limits to delete",
        examples=["f47ac10b-58cc-4372-a567-0e02b2c3d479"],
    ),
    user_id: str = Depends(require_superadmin),
):
    """Delete limits for a workspace, reverting it to the server default. Superadmin only"""
    deleted = delete_org_limits(target_org_uuid)
    if not deleted:
        raise HTTPException(status_code=404, detail="Organization limits not found")
    return {"message": "Organization limits deleted successfully"}
