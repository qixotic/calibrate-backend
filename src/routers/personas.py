from typing import Optional, List, Dict, Any
from fastapi import APIRouter, HTTPException, Depends, Path
from pydantic import BaseModel, Field

from db import (
    create_persona,
    get_persona,
    get_all_personas,
    update_persona,
    delete_persona,
    ensure_name_unique,
)
from auth_utils import get_current_org, OrgContext


router = APIRouter(prefix="/personas", tags=["personas"])

_EXAMPLE_ID = "f47ac10b-58cc-4372-a567-0e02b2c3d479"


class PersonaCreate(BaseModel):
    name: str = Field(description="Persona name, unique within the workspace")
    description: Optional[str] = Field(
        None, description="A description of the persona. Omit to leave unset"
    )
    config: Optional[Dict[str, Any]] = Field(
        None, description="Behavioral config. Omit to leave unset"
    )


class PersonaUpdate(BaseModel):
    name: Optional[str] = Field(
        None, description="New persona name, unique within the workspace. Omit to leave unchanged"
    )
    description: Optional[str] = Field(
        None, description="New description. Omit to leave unchanged"
    )
    config: Optional[Dict[str, Any]] = Field(
        None, description="New behavioral config. Omit to leave unchanged"
    )


class PersonaResponse(BaseModel):
    uuid: str = Field(
        min_length=36,
        max_length=36,
        description="ID of the persona",
        examples=[_EXAMPLE_ID],
    )
    name: str = Field(description="Persona name")
    description: Optional[str] = Field(
        None, description="Description of the persona"
    )
    config: Optional[Dict[str, Any]] = Field(
        None, description="Behavioral config"
    )
    created_at: str = Field(description="When the persona was created (ISO 8601 UTC)")
    updated_at: str = Field(description="When the persona was last updated (ISO 8601 UTC)")


class PersonaCreateResponse(BaseModel):
    uuid: str = Field(
        min_length=36,
        max_length=36,
        description="ID of the newly created persona",
        examples=[_EXAMPLE_ID],
    )
    message: str = Field(description="Success message")


@router.post("", response_model=PersonaCreateResponse, summary="Create persona")
def create_persona_endpoint(
    persona: PersonaCreate, ctx: OrgContext = Depends(get_current_org)
):
    """Create a new persona"""
    with ensure_name_unique("personas", persona.name, ctx.org_uuid, entity="Persona"):
        persona_uuid = create_persona(
            name=persona.name,
            description=persona.description,
            config=persona.config,
            org_uuid=ctx.org_uuid,
            user_id=ctx.user_id,
        )
    return PersonaCreateResponse(
        uuid=persona_uuid, message="Persona created successfully"
    )


@router.get("", response_model=List[PersonaResponse], summary="List personas")
def list_personas(ctx: OrgContext = Depends(get_current_org)):
    """List your personas"""
    personas = get_all_personas(org_uuid=ctx.org_uuid)
    return personas


@router.get("/{persona_uuid}", response_model=PersonaResponse, summary="Get persona")
def get_persona_endpoint(
    persona_uuid: str = Path(
        description="The persona to retrieve",
        examples=[_EXAMPLE_ID],
    ),
    ctx: OrgContext = Depends(get_current_org),
):
    """Get one persona by ID"""
    persona = get_persona(persona_uuid)
    if not persona or persona.get("org_uuid") != ctx.org_uuid:
        raise HTTPException(status_code=404, detail="Persona not found")
    return persona


@router.put("/{persona_uuid}", response_model=PersonaResponse, summary="Update persona")
def update_persona_endpoint(
    persona: PersonaUpdate,
    persona_uuid: str = Path(
        description="The persona to update",
        examples=[_EXAMPLE_ID],
    ),
    ctx: OrgContext = Depends(get_current_org),
):
    """Update a persona, a simulated user profile used in simulations"""
    existing_persona = get_persona(persona_uuid)
    if not existing_persona or existing_persona.get("org_uuid") != ctx.org_uuid:
        raise HTTPException(status_code=404, detail="Persona not found")

    with ensure_name_unique(
        "personas",
        persona.name,
        ctx.org_uuid,
        entity="Persona",
        exclude_uuid=persona_uuid,
    ):
        updated = update_persona(
            persona_uuid=persona_uuid,
            name=persona.name,
            description=persona.description,
            config=persona.config,
        )

    if not updated:
        raise HTTPException(status_code=400, detail="No fields to update")

    updated_persona = get_persona(persona_uuid)
    return updated_persona


@router.delete("/{persona_uuid}", summary="Delete persona")
def delete_persona_endpoint(
    persona_uuid: str = Path(
        description="The persona to delete",
        examples=[_EXAMPLE_ID],
    ),
    ctx: OrgContext = Depends(get_current_org),
):
    """Soft-delete a persona"""
    existing_persona = get_persona(persona_uuid)
    if not existing_persona or existing_persona.get("org_uuid") != ctx.org_uuid:
        raise HTTPException(status_code=404, detail="Persona not found")

    deleted = delete_persona(persona_uuid)
    if not deleted:
        raise HTTPException(status_code=404, detail="Persona not found")
    return {"message": "Persona deleted successfully"}
