from typing import Optional, List
from fastapi import APIRouter, HTTPException, Depends, Path
from pydantic import BaseModel, Field

from db import (
    create_scenario,
    get_scenario,
    get_all_scenarios,
    update_scenario,
    delete_scenario,
    ensure_name_unique,
)
from auth_utils import get_current_org, OrgContext


router = APIRouter(prefix="/scenarios", tags=["scenarios"])

_EXAMPLE_ID = "f47ac10b-58cc-4372-a567-0e02b2c3d479"


class ScenarioCreate(BaseModel):
    name: str = Field(description="Scenario name, unique within the workspace")
    description: Optional[str] = Field(
        None, description="A description of the scenario. Omit to leave unset"
    )


class ScenarioUpdate(BaseModel):
    name: Optional[str] = Field(
        None, description="New scenario name, unique within the workspace. Omit to leave unchanged"
    )
    description: Optional[str] = Field(
        None, description="New description. Omit to leave unchanged"
    )


class ScenarioResponse(BaseModel):
    uuid: str = Field(
        min_length=36,
        max_length=36,
        description="ID of the scenario",
        examples=[_EXAMPLE_ID],
    )
    name: str = Field(description="Scenario name")
    description: Optional[str] = Field(
        None, description="Description of the scenario"
    )
    created_at: str = Field(description="When the scenario was created (ISO 8601 UTC)")
    updated_at: str = Field(description="When the scenario was last updated (ISO 8601 UTC)")


class ScenarioCreateResponse(BaseModel):
    uuid: str = Field(
        min_length=36,
        max_length=36,
        description="ID of the newly created scenario",
        examples=[_EXAMPLE_ID],
    )
    message: str = Field(description="Success message")


@router.post("", response_model=ScenarioCreateResponse, summary="Create scenario")
def create_scenario_endpoint(
    scenario: ScenarioCreate, ctx: OrgContext = Depends(get_current_org)
):
    """Create a new scenario"""
    with ensure_name_unique("scenarios", scenario.name, ctx.org_uuid, entity="Scenario"):
        scenario_uuid = create_scenario(
            name=scenario.name,
            description=scenario.description,
            org_uuid=ctx.org_uuid,
            user_id=ctx.user_id,
        )
    return ScenarioCreateResponse(
        uuid=scenario_uuid, message="Scenario created successfully"
    )


@router.get("", response_model=List[ScenarioResponse], summary="List scenarios")
def list_scenarios(ctx: OrgContext = Depends(get_current_org)):
    """List your scenarios"""
    scenarios = get_all_scenarios(org_uuid=ctx.org_uuid)
    return scenarios


@router.get("/{scenario_uuid}", response_model=ScenarioResponse, summary="Get scenario")
def get_scenario_endpoint(
    scenario_uuid: str = Path(
        description="The scenario to retrieve",
        examples=[_EXAMPLE_ID],
    ),
    ctx: OrgContext = Depends(get_current_org),
):
    """Get one scenario by ID"""
    scenario = get_scenario(scenario_uuid)
    if not scenario or scenario.get("org_uuid") != ctx.org_uuid:
        raise HTTPException(status_code=404, detail="Scenario not found")
    return scenario


@router.put("/{scenario_uuid}", response_model=ScenarioResponse, summary="Update scenario")
def update_scenario_endpoint(
    scenario: ScenarioUpdate,
    scenario_uuid: str = Path(
        description="The scenario to update",
        examples=[_EXAMPLE_ID],
    ),
    ctx: OrgContext = Depends(get_current_org),
):
    """Update a scenario, the situation and goal a simulation plays out"""
    existing_scenario = get_scenario(scenario_uuid)
    if not existing_scenario or existing_scenario.get("org_uuid") != ctx.org_uuid:
        raise HTTPException(status_code=404, detail="Scenario not found")

    with ensure_name_unique(
        "scenarios",
        scenario.name,
        ctx.org_uuid,
        entity="Scenario",
        exclude_uuid=scenario_uuid,
    ):
        updated = update_scenario(
            scenario_uuid=scenario_uuid,
            name=scenario.name,
            description=scenario.description,
        )

    if not updated:
        raise HTTPException(status_code=400, detail="No fields to update")

    updated_scenario = get_scenario(scenario_uuid)
    return updated_scenario


@router.delete("/{scenario_uuid}", summary="Delete scenario")
def delete_scenario_endpoint(
    scenario_uuid: str = Path(
        description="The scenario to delete",
        examples=[_EXAMPLE_ID],
    ),
    ctx: OrgContext = Depends(get_current_org),
):
    """Soft-delete a scenario"""
    existing_scenario = get_scenario(scenario_uuid)
    if not existing_scenario or existing_scenario.get("org_uuid") != ctx.org_uuid:
        raise HTTPException(status_code=404, detail="Scenario not found")

    deleted = delete_scenario(scenario_uuid)
    if not deleted:
        raise HTTPException(status_code=404, detail="Scenario not found")
    return {"message": "Scenario deleted successfully"}
