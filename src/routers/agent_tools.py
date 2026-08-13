from typing import List, Dict, Any, Literal
from fastapi import APIRouter, HTTPException, Depends, Path
from pydantic import BaseModel, Field
from sqlite3 import IntegrityError

from auth_utils import get_current_org, OrgContext
from utils import AGENT_TYPE_DESCRIPTION
from db import (
    add_tool_to_agent,
    remove_tool_from_agent,
    get_tools_for_agent,
    get_agents_for_tool,
    get_agent_tool_link,
    get_all_agent_tools,
    get_agent,
    get_tool,
)


router = APIRouter(prefix="/agent-tools", tags=["agent-tools"])

_EXAMPLE_ID = "f47ac10b-58cc-4372-a567-0e02b2c3d479"


class AgentToolsCreate(BaseModel):
    agent_uuid: str = Field(
        min_length=36,
        max_length=36,
        description="The agent to link tools to",
        examples=[_EXAMPLE_ID],
    )
    tool_uuids: List[str] = Field(
        description="Tools to link. Any that are already linked are skipped",
        examples=[[_EXAMPLE_ID, "6ba7b810-9dad-11d1-80b4-00c04fd430c8"]],
    )


class AgentToolDelete(BaseModel):
    agent_uuid: str = Field(
        min_length=36,
        max_length=36,
        description="The agent to unlink a tool from",
        examples=[_EXAMPLE_ID],
    )
    tool_uuid: str = Field(
        min_length=36,
        max_length=36,
        description="The tool to unlink from the agent",
        examples=[_EXAMPLE_ID],
    )


class AgentToolResponse(BaseModel):
    id: int = Field(description="ID of the link")
    agent_id: str = Field(
        min_length=36,
        max_length=36,
        description="Linked agent ID",
        examples=[_EXAMPLE_ID],
    )
    tool_id: str = Field(
        min_length=36,
        max_length=36,
        description="Linked tool ID",
        examples=[_EXAMPLE_ID],
    )
    created_at: str = Field(description="When the link was created (ISO 8601 UTC)")


class AgentToolsCreateResponse(BaseModel):
    ids: List[int] = Field(
        description="IDs of the links created by this call. Tools that were already linked are excluded"
    )
    message: str = Field(description="Confirmation message")


class ToolResponse(BaseModel):
    uuid: str = Field(
        min_length=36,
        max_length=36,
        description="ID of the tool",
        examples=[_EXAMPLE_ID],
    )
    name: str = Field(description="Tool name")
    description: str = Field(description="What the tool does")
    config: Dict[str, Any] | None = Field(
        None, description="Tool configuration"
    )
    created_at: str = Field(description="When the tool was created (ISO 8601 UTC)")
    updated_at: str = Field(description="When the tool was last updated (ISO 8601 UTC)")


class AgentResponse(BaseModel):
    uuid: str = Field(
        min_length=36,
        max_length=36,
        description="ID of the agent",
        examples=[_EXAMPLE_ID],
    )
    name: str = Field(description="Agent name")
    type: Literal["agent", "connection"] = Field(
        description=AGENT_TYPE_DESCRIPTION
    )
    config: Dict[str, Any] | None = Field(
        None, description="Behavioral config"
    )
    created_at: str = Field(description="When the agent was created (ISO 8601 UTC)")
    updated_at: str = Field(description="When the agent was last updated (ISO 8601 UTC)")


def _require_owned_agent(agent_uuid: str, org_uuid: str) -> Dict[str, Any]:
    agent = get_agent(agent_uuid)
    if not agent or agent.get("org_uuid") != org_uuid:
        raise HTTPException(status_code=404, detail="Agent not found")
    return agent


def _require_owned_tool(tool_uuid: str, org_uuid: str) -> Dict[str, Any]:
    tool = get_tool(tool_uuid)
    if not tool or tool.get("org_uuid") != org_uuid:
        raise HTTPException(status_code=404, detail=f"Tool {tool_uuid} not found")
    return tool


@router.post(
    "", response_model=AgentToolsCreateResponse, summary="Link tools to agent"
)
def create_agent_tool_links(
    agent_tools: AgentToolsCreate,
    ctx: OrgContext = Depends(get_current_org),
):
    """Link one or more tools to an agent. Tools that are already linked are skipped"""
    _require_owned_agent(agent_tools.agent_uuid, ctx.org_uuid)
    for tool_uuid in agent_tools.tool_uuids:
        _require_owned_tool(tool_uuid, ctx.org_uuid)

    link_ids = []
    for tool_uuid in agent_tools.tool_uuids:
        existing = get_agent_tool_link(agent_tools.agent_uuid, tool_uuid)
        if existing:
            continue
        try:
            link_id = add_tool_to_agent(
                agent_id=agent_tools.agent_uuid,
                tool_id=tool_uuid,
            )
            link_ids.append(link_id)
        except IntegrityError:
            continue

    return AgentToolsCreateResponse(
        ids=link_ids, message="Tools added to agent successfully"
    )


@router.get(
    "", response_model=List[AgentToolResponse], summary="List agent-tool links"
)
def list_agent_tools(ctx: OrgContext = Depends(get_current_org)):
    """List which tools are linked to which agents"""
    return get_all_agent_tools(org_uuid=ctx.org_uuid)


@router.get(
    "/agent/{agent_uuid}/tools",
    response_model=List[ToolResponse],
    summary="List tools for agent",
)
def get_agent_tools(
    agent_uuid: str = Path(
        description="The agent whose linked tools to list",
        examples=[_EXAMPLE_ID],
    ),
    ctx: OrgContext = Depends(get_current_org),
):
    """List the tools linked to an agent"""
    _require_owned_agent(agent_uuid, ctx.org_uuid)
    return get_tools_for_agent(agent_uuid)


@router.get(
    "/tool/{tool_uuid}/agents",
    response_model=List[AgentResponse],
    summary="List agents for tool",
)
def get_tool_agents(
    tool_uuid: str = Path(
        description="The tool whose linked agents to list",
        examples=[_EXAMPLE_ID],
    ),
    ctx: OrgContext = Depends(get_current_org),
):
    """List the agents a tool is linked to"""
    _require_owned_tool(tool_uuid, ctx.org_uuid)
    return get_agents_for_tool(tool_uuid)


@router.delete("", summary="Unlink tool from agent")
def delete_agent_tool_link(
    agent_tool: AgentToolDelete, ctx: OrgContext = Depends(get_current_org)
):
    """Unlink a tool from an agent so the agent can no longer call it"""
    _require_owned_agent(agent_tool.agent_uuid, ctx.org_uuid)
    deleted = remove_tool_from_agent(agent_tool.agent_uuid, agent_tool.tool_uuid)
    if not deleted:
        raise HTTPException(status_code=404, detail="Agent-tool link not found")
    return {"message": "Tool removed from agent successfully"}
