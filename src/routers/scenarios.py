from typing import Optional, List
from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel

from db import (
    create_scenario,
    get_scenario,
    get_all_scenarios,
    update_scenario,
    delete_scenario,
    ensure_name_unique,
)
from auth_utils import get_current_user_id


router = APIRouter(prefix="/scenarios", tags=["scenarios"])


class ScenarioCreate(BaseModel):
    name: str
    description: Optional[str] = None


class ScenarioUpdate(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None


class ScenarioResponse(BaseModel):
    uuid: str
    name: str
    description: Optional[str] = None
    created_at: str
    updated_at: str


class ScenarioCreateResponse(BaseModel):
    uuid: str
    message: str


@router.post("", response_model=ScenarioCreateResponse)
async def create_scenario_endpoint(
    scenario: ScenarioCreate, user_id: str = Depends(get_current_user_id)
):
    """Create a new scenario."""
    with ensure_name_unique("scenarios", scenario.name, user_id, entity="Scenario"):
        scenario_uuid = create_scenario(
            name=scenario.name,
            description=scenario.description,
            user_id=user_id,
        )
    return ScenarioCreateResponse(
        uuid=scenario_uuid, message="Scenario created successfully"
    )


@router.get("", response_model=List[ScenarioResponse])
async def list_scenarios(user_id: str = Depends(get_current_user_id)):
    """List all scenarios for the authenticated user."""
    scenarios = get_all_scenarios(user_id=user_id)
    return scenarios


@router.get("/{scenario_uuid}", response_model=ScenarioResponse)
async def get_scenario_endpoint(
    scenario_uuid: str, user_id: str = Depends(get_current_user_id)
):
    """Get a scenario by UUID."""
    scenario = get_scenario(scenario_uuid)
    if not scenario:
        raise HTTPException(status_code=404, detail="Scenario not found")
    # Verify user owns this scenario
    if scenario.get("user_id") != user_id:
        raise HTTPException(status_code=403, detail="Access denied")
    return scenario


@router.put("/{scenario_uuid}", response_model=ScenarioResponse)
async def update_scenario_endpoint(
    scenario_uuid: str,
    scenario: ScenarioUpdate,
    user_id: str = Depends(get_current_user_id),
):
    """Update a scenario."""
    existing_scenario = get_scenario(scenario_uuid)
    if not existing_scenario:
        raise HTTPException(status_code=404, detail="Scenario not found")

    # Verify user owns this scenario
    if existing_scenario.get("user_id") != user_id:
        raise HTTPException(status_code=403, detail="Access denied")

    with ensure_name_unique(
        "scenarios", scenario.name, user_id, entity="Scenario", exclude_uuid=scenario_uuid
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


@router.delete("/{scenario_uuid}")
async def delete_scenario_endpoint(
    scenario_uuid: str, user_id: str = Depends(get_current_user_id)
):
    """Delete a scenario."""
    # Check if scenario exists and user owns it
    existing_scenario = get_scenario(scenario_uuid)
    if not existing_scenario:
        raise HTTPException(status_code=404, detail="Scenario not found")
    if existing_scenario.get("user_id") != user_id:
        raise HTTPException(status_code=403, detail="Access denied")

    deleted = delete_scenario(scenario_uuid)
    if not deleted:
        raise HTTPException(status_code=404, detail="Scenario not found")
    return {"message": "Scenario deleted successfully"}
