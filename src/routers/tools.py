from typing import Optional, List, Dict, Any
from fastapi import APIRouter, HTTPException, Depends, Path
from pydantic import BaseModel, Field, model_validator

from db import create_tool, get_tool, get_all_tools, update_tool, delete_tool, ensure_name_unique
from auth_utils import get_current_org, OrgContext


router = APIRouter(prefix="/tools", tags=["tools"])

_EXAMPLE_ID = "f47ac10b-58cc-4372-a567-0e02b2c3d479"

# A tool parameter's identifier lives in its `id` field, and a duplicate is only
# representable in the array-valued param lists: object children are stored as a
# JSON object keyed by the child's name (`properties: {city: {...}}`), where a
# duplicate key can't survive parsing (last-wins). So uniqueness is enforced on
# the three arrays that carry entries — the structured-output `parameters` list
# and the two webhook lists — descending each entry's nested schema
# (`properties` children, `items` element) to reach any deeper array.


def _reject_duplicate_param_ids(entries: Any) -> None:
    """Reject an array of parameter entries where two share an `id` (compared
    case-insensitively after trimming), descending each entry's nested schema.
    Raises `ValueError` so it surfaces as a 422 from a model validator."""
    if not isinstance(entries, list):
        return
    seen: set = set()
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        entry_id = entry.get("id")
        if isinstance(entry_id, str) and entry_id.strip():
            key = entry_id.strip().lower()
            if key in seen:
                raise ValueError(
                    f'A parameter with id "{entry_id}" already exists in this list'
                )
            seen.add(key)
        _descend_param_schema(entry)


def _descend_param_schema(node: Dict[str, Any]) -> None:
    """Follow a parameter's nested schema — `properties` (object children, keyed
    by name) and `items` (array element) — to reach any deeper param array. Keyed
    object children can't hold a duplicate; only a nested array is checkable."""
    props = node.get("properties")
    if isinstance(props, dict):
        for child in props.values():
            if isinstance(child, dict):
                _descend_param_schema(child)
    items = node.get("items")
    if isinstance(items, dict):
        _descend_param_schema(items)
    elif isinstance(items, list):
        _reject_duplicate_param_ids(items)


def _validate_tool_config_params(config: Optional[Dict[str, Any]]) -> None:
    if not isinstance(config, dict):
        return
    _reject_duplicate_param_ids(config.get("parameters"))
    webhook = config.get("webhook")
    if isinstance(webhook, dict):
        _reject_duplicate_param_ids(webhook.get("queryParameters"))
        body = webhook.get("body")
        if isinstance(body, dict):
            _reject_duplicate_param_ids(body.get("parameters"))


class ToolCreate(BaseModel):
    name: str = Field(description="Tool name, unique within the workspace")
    description: str = Field(description="What the tool does. Surfaced to agents and the UI")
    config: Optional[Dict[str, Any]] = Field(
        None, description="Tool config (e.g. JSON schema, parameters). Omit to leave unset"
    )

    @model_validator(mode="after")
    def _validate_config(self):
        _validate_tool_config_params(self.config)
        return self


class ToolUpdate(BaseModel):
    name: Optional[str] = Field(
        None, description="New tool name, unique within the workspace. Omit to leave unchanged"
    )
    description: Optional[str] = Field(
        None, description="New description. Omit to leave unchanged"
    )
    config: Optional[Dict[str, Any]] = Field(
        None, description="New tool config. Omit to leave unchanged"
    )

    @model_validator(mode="after")
    def _validate_config(self):
        _validate_tool_config_params(self.config)
        return self


class ToolResponse(BaseModel):
    uuid: str = Field(
        min_length=36,
        max_length=36,
        description="ID of the tool",
        examples=[_EXAMPLE_ID],
    )
    name: str = Field(description="Tool name")
    description: str = Field(description="What the tool does")
    config: Optional[Dict[str, Any]] = Field(
        None, description="Tool config"
    )
    created_at: str = Field(description="When the tool was created (ISO 8601 UTC)")
    updated_at: str = Field(description="When the tool was last updated (ISO 8601 UTC)")


class ToolCreateResponse(BaseModel):
    uuid: str = Field(
        min_length=36,
        max_length=36,
        description="ID of the newly created tool",
        examples=[_EXAMPLE_ID],
    )
    message: str = Field(description="Success message")


@router.post("", response_model=ToolCreateResponse, summary="Create tool")
def create_tool_endpoint(
    tool: ToolCreate, ctx: OrgContext = Depends(get_current_org)
):
    """Create a new tool"""
    with ensure_name_unique("tools", tool.name, ctx.org_uuid, entity="Tool"):
        tool_uuid = create_tool(
            name=tool.name,
            description=tool.description,
            config=tool.config,
            org_uuid=ctx.org_uuid,
            user_id=ctx.user_id,
        )
    return ToolCreateResponse(uuid=tool_uuid, message="Tool created successfully")


@router.get("", response_model=List[ToolResponse], summary="List tools")
def list_tools(ctx: OrgContext = Depends(get_current_org)):
    """List your tools"""
    tools = get_all_tools(org_uuid=ctx.org_uuid)
    return tools


@router.get("/{tool_uuid}", response_model=ToolResponse, summary="Get tool")
def get_tool_endpoint(
    tool_uuid: str = Path(
        description="The tool to retrieve",
        examples=[_EXAMPLE_ID],
    ),
    ctx: OrgContext = Depends(get_current_org),
):
    """Get one tool by ID"""
    tool = get_tool(tool_uuid)
    if not tool or tool.get("org_uuid") != ctx.org_uuid:
        raise HTTPException(status_code=404, detail="Tool not found")
    return tool


@router.put("/{tool_uuid}", response_model=ToolResponse, summary="Update tool")
def update_tool_endpoint(
    tool: ToolUpdate,
    tool_uuid: str = Path(
        description="The tool to update",
        examples=[_EXAMPLE_ID],
    ),
    ctx: OrgContext = Depends(get_current_org),
):
    """Update a tool, a function your agent can call"""
    existing_tool = get_tool(tool_uuid)
    if not existing_tool or existing_tool.get("org_uuid") != ctx.org_uuid:
        raise HTTPException(status_code=404, detail="Tool not found")

    with ensure_name_unique(
        "tools", tool.name, ctx.org_uuid, entity="Tool", exclude_uuid=tool_uuid
    ):
        updated = update_tool(
            tool_uuid=tool_uuid,
            name=tool.name,
            description=tool.description,
            config=tool.config,
        )

    if not updated:
        raise HTTPException(status_code=400, detail="No fields to update")

    updated_tool = get_tool(tool_uuid)
    return updated_tool


@router.delete("/{tool_uuid}", summary="Delete tool")
def delete_tool_endpoint(
    tool_uuid: str = Path(
        description="The tool to delete",
        examples=[_EXAMPLE_ID],
    ),
    ctx: OrgContext = Depends(get_current_org),
):
    """Soft-delete a tool"""
    existing_tool = get_tool(tool_uuid)
    if not existing_tool or existing_tool.get("org_uuid") != ctx.org_uuid:
        raise HTTPException(status_code=404, detail="Tool not found")

    deleted = delete_tool(tool_uuid)
    if not deleted:
        raise HTTPException(status_code=404, detail="Tool not found")
    return {"message": "Tool deleted successfully"}
