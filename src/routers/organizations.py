"""Manage workspaces and membership.

Create workspaces, rename them, and add or remove members.
"""

from fastapi import APIRouter, Depends, HTTPException, Path
from pydantic import BaseModel, Field
from typing import List, Optional

from auth_utils import get_current_user_id, is_superadmin_user
from utils import MemberRoleLiteral
from db import (
    add_organization_member,
    create_organization,
    get_member_role,
    get_organization,
    list_organization_members,
    list_organizations_for_user,
    remove_organization_member,
    update_organization_name,
)

router = APIRouter(prefix="/organizations", tags=["organizations"])


class OrganizationResponse(BaseModel):
    uuid: str = Field(
        min_length=36,
        max_length=36,
        description="Workspace ID",
    )
    name: str = Field(description="Workspace display name")
    is_personal: bool = Field(
        description="`true` for your auto-created personal workspace, `false` for shared workspaces"
    )
    created_by_user_id: str = Field(
        min_length=36,
        max_length=36,
        description="ID of the user who created the workspace",
    )
    member_role: Optional[MemberRoleLiteral] = Field(
        None,
        description="Your role in this workspace (`owner` | `admin`); `null` when not resolved",
    )
    created_at: str = Field(description="When the workspace was created (ISO 8601 UTC)")
    updated_at: str = Field(description="When the workspace was last updated (ISO 8601 UTC)")


class CreateOrganizationRequest(BaseModel):
    name: str = Field(..., min_length=1, description="Display name for the new workspace")


class UpdateOrganizationRequest(BaseModel):
    name: str = Field(..., min_length=1, description="New display name for the workspace")


class AddMemberRequest(BaseModel):
    email: str = Field(
        ...,
        min_length=3,
        description="Email of the person to add; a stub account is created if they have not signed up yet",
    )


class MemberResponse(BaseModel):
    user_id: str = Field(
        min_length=36,
        max_length=36,
        description="Member's user ID",
    )
    email: str = Field(description="Member's email address")
    first_name: str = Field(description="Member's given name")
    last_name: str = Field(description="Member's family name")
    role: MemberRoleLiteral = Field(
        description="Member's role in the workspace (`owner` | `admin`)"
    )
    created_at: str = Field(description="When the member was added (ISO 8601 UTC)")


def _require_membership(org_uuid: str, user_id: str) -> str:
    """Resolve the caller's role in `org_uuid`, 404ing if not a member.

    Superadmin bypass: any existing workspace grants owner-level access.
    """
    role = get_member_role(org_uuid, user_id)
    if role is None:
        if is_superadmin_user(user_id) and get_organization(org_uuid) is not None:
            return "owner"
        raise HTTPException(status_code=404, detail="Organization not found")
    return role


@router.get("", response_model=List[OrganizationResponse], summary="List workspaces")
async def list_orgs(user_id: str = Depends(get_current_user_id)):
    """List every workspace you are an active member of."""
    return [OrganizationResponse(**o) for o in list_organizations_for_user(user_id)]


@router.post("", response_model=OrganizationResponse, status_code=201, summary="Create workspace")
async def create_org(
    request: CreateOrganizationRequest,
    user_id: str = Depends(get_current_user_id),
):
    """Create a new (non-personal) workspace with you as owner."""
    org_uuid = create_organization(name=request.name, owner_user_id=user_id)
    org = get_organization(org_uuid)
    return OrganizationResponse(**org, member_role="owner")


@router.patch("/{org_uuid}", response_model=OrganizationResponse, summary="Update workspace")
async def rename_org(
    org_uuid: str = Path(
        description="The workspace to rename. You must be a member.",
        examples=["f47ac10b-58cc-4372-a567-0e02b2c3d479"],
    ),
    request: UpdateOrganizationRequest = ...,
    user_id: str = Depends(get_current_user_id),
):
    """Rename a workspace you belong to."""
    role = _require_membership(org_uuid, user_id)
    update_organization_name(org_uuid, request.name)
    org = get_organization(org_uuid)
    return OrganizationResponse(**org, member_role=role)


@router.get("/{org_uuid}/members", response_model=List[MemberResponse], summary="List members")
async def list_members(
    org_uuid: str = Path(
        description="The workspace whose members to list. You must be a member.",
        examples=["f47ac10b-58cc-4372-a567-0e02b2c3d479"],
    ),
    user_id: str = Depends(get_current_user_id),
):
    """List members of a workspace you belong to."""
    _require_membership(org_uuid, user_id)
    return [MemberResponse(**m) for m in list_organization_members(org_uuid)]


@router.post(
    "/{org_uuid}/members",
    response_model=MemberResponse,
    status_code=201,
    summary="Add member",
)
async def add_member(
    org_uuid: str = Path(
        description="The workspace to add a member to. You must be a member.",
        examples=["f47ac10b-58cc-4372-a567-0e02b2c3d479"],
    ),
    request: AddMemberRequest = ...,
    user_id: str = Depends(get_current_user_id),
):
    """Add a member to a workspace as admin."""
    # Stub accounts are hydrated when the invitee signs up; they then see this workspace immediately.
    _require_membership(org_uuid, user_id)
    try:
        member = add_organization_member(
            org_uuid=org_uuid, email=request.email, role="admin"
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    # Re-read the full member row so the response has the joined user fields.
    for m in list_organization_members(org_uuid):
        if m["user_id"] == member["user_id"]:
            return MemberResponse(**m)
    raise HTTPException(status_code=500, detail="Member not found after insert")


@router.delete("/{org_uuid}/members/{target_user_id}", status_code=204, summary="Remove member")
async def remove_member(
    org_uuid: str = Path(
        description="The workspace to remove the member from. You must be a member.",
        examples=["f47ac10b-58cc-4372-a567-0e02b2c3d479"],
    ),
    target_user_id: str = Path(
        description="The member to remove",
        examples=["f47ac10b-58cc-4372-a567-0e02b2c3d479"],
    ),
    user_id: str = Depends(get_current_user_id),
):
    """Remove a member from a workspace. Owners cannot be removed."""
    _require_membership(org_uuid, user_id)
    try:
        removed = remove_organization_member(org_uuid, target_user_id)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    if not removed:
        raise HTTPException(status_code=404, detail="Member not found")
    return None
