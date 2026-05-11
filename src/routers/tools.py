from typing import Optional, List, Dict, Any
from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel

from db import create_tool, get_tool, get_all_tools, update_tool, delete_tool, ensure_name_unique
from auth_utils import get_current_user_id


router = APIRouter(prefix="/tools", tags=["tools"])


class ToolCreate(BaseModel):
    name: str
    description: str
    config: Optional[Dict[str, Any]] = None


class ToolUpdate(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None
    config: Optional[Dict[str, Any]] = None


class ToolResponse(BaseModel):
    uuid: str
    name: str
    description: str
    config: Optional[Dict[str, Any]] = None
    created_at: str
    updated_at: str


class ToolCreateResponse(BaseModel):
    uuid: str
    message: str


@router.post("", response_model=ToolCreateResponse)
async def create_tool_endpoint(
    tool: ToolCreate, user_id: str = Depends(get_current_user_id)
):
    """Create a new tool."""
    with ensure_name_unique("tools", tool.name, user_id, entity="Tool"):
        tool_uuid = create_tool(
            name=tool.name,
            description=tool.description,
            config=tool.config,
            user_id=user_id,
        )
    return ToolCreateResponse(uuid=tool_uuid, message="Tool created successfully")


@router.get("", response_model=List[ToolResponse])
async def list_tools(user_id: str = Depends(get_current_user_id)):
    """List all tools for the authenticated user."""
    tools = get_all_tools(user_id=user_id)
    return tools


@router.get("/{tool_uuid}", response_model=ToolResponse)
async def get_tool_endpoint(
    tool_uuid: str, user_id: str = Depends(get_current_user_id)
):
    """Get a tool by UUID."""
    tool = get_tool(tool_uuid)
    if not tool:
        raise HTTPException(status_code=404, detail="Tool not found")
    # Verify user owns this tool
    if tool.get("user_id") != user_id:
        raise HTTPException(status_code=403, detail="Access denied")
    return tool


@router.put("/{tool_uuid}", response_model=ToolResponse)
async def update_tool_endpoint(
    tool_uuid: str, tool: ToolUpdate, user_id: str = Depends(get_current_user_id)
):
    """Update a tool."""
    # Check if tool exists
    existing_tool = get_tool(tool_uuid)
    if not existing_tool:
        raise HTTPException(status_code=404, detail="Tool not found")

    # Verify user owns this tool
    if existing_tool.get("user_id") != user_id:
        raise HTTPException(status_code=403, detail="Access denied")

    # Update only provided fields
    with ensure_name_unique(
        "tools", tool.name, user_id, entity="Tool", exclude_uuid=tool_uuid
    ):
        updated = update_tool(
            tool_uuid=tool_uuid,
            name=tool.name,
            description=tool.description,
            config=tool.config,
        )

    if not updated:
        raise HTTPException(status_code=400, detail="No fields to update")

    # Return updated tool
    updated_tool = get_tool(tool_uuid)
    return updated_tool


@router.delete("/{tool_uuid}")
async def delete_tool_endpoint(
    tool_uuid: str, user_id: str = Depends(get_current_user_id)
):
    """Delete a tool."""
    # Check if tool exists and user owns it
    existing_tool = get_tool(tool_uuid)
    if not existing_tool:
        raise HTTPException(status_code=404, detail="Tool not found")
    if existing_tool.get("user_id") != user_id:
        raise HTTPException(status_code=403, detail="Access denied")

    deleted = delete_tool(tool_uuid)
    if not deleted:
        raise HTTPException(status_code=404, detail="Tool not found")
    return {"message": "Tool deleted successfully"}
