import copy
import ipaddress
import logging
import socket
from datetime import datetime, timezone
from typing import Optional, List, Dict, Any, Literal
from urllib.parse import urlparse
from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
from calibrate.connections import TextAgentConnection

from utils import env_bool, env_int, env_str

from db import (
    create_agent,
    ensure_name_unique,
    get_agent,
    get_all_agents,
    update_agent,
    delete_agent,
    get_tools_for_agent,
    get_tests_for_agent,
    add_tool_to_agent,
    add_test_to_agent,
)
from auth_utils import get_current_user_id

logger = logging.getLogger(__name__)

BLOCKED_HEADERS = frozenset(
    {
        "host",
        "transfer-encoding",
        "content-length",
        "connection",
        "upgrade",
        "te",
        "trailer",
        "keep-alive",
        "proxy-authorization",
        "proxy-authenticate",
        "proxy-connection",
    }
)


def _is_private_ip(addr: str) -> bool:
    """Return True if addr is loopback, private, link-local, or otherwise non-public."""
    try:
        ip = ipaddress.ip_address(addr)
    except ValueError:
        return False
    return (
        ip.is_loopback
        or ip.is_private
        or ip.is_reserved
        or ip.is_link_local
        or ip.is_multicast
        or ip.is_unspecified
    )


def _validate_agent_url(url: str) -> None:
    """Raise HTTPException if url is not a valid public HTTP(S) endpoint.

    Checks both the hostname string and the resolved IP addresses to
    prevent SSRF via DNS rebinding or numeric IP encoding tricks.
    """
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https"):
        raise HTTPException(status_code=400, detail="agent_url must use http or https")
    if not parsed.hostname:
        raise HTTPException(status_code=400, detail="agent_url must include a hostname")
    hostname = parsed.hostname.lower()

    if hostname in ("localhost", "127.0.0.1", "0.0.0.0", "::1", "[::1]"):
        raise HTTPException(
            status_code=400, detail="agent_url must not point to localhost"
        )
    if hostname.endswith(".local"):
        raise HTTPException(
            status_code=400,
            detail="agent_url must not point to a private network address",
        )

    try:
        addr_infos = socket.getaddrinfo(hostname, None, proto=socket.IPPROTO_TCP)
    except socket.gaierror:
        raise HTTPException(
            status_code=400, detail="agent_url hostname could not be resolved"
        )

    if not addr_infos:
        raise HTTPException(
            status_code=400, detail="agent_url hostname could not be resolved"
        )

    for family, _type, _proto, _canonname, sockaddr in addr_infos:
        ip_str = sockaddr[0]
        if _is_private_ip(ip_str):
            raise HTTPException(
                status_code=400,
                detail="agent_url must not resolve to a private or reserved network address",
            )


def _sanitize_headers(headers: Optional[Dict[str, str]]) -> Optional[Dict[str, str]]:
    """Remove hop-by-hop and security-sensitive headers."""
    if not headers:
        return headers
    return {k: v for k, v in headers.items() if k.lower() not in BLOCKED_HEADERS}


async def _verify_agent_connection(
    agent_url: str,
    agent_headers: Optional[Dict[str, str]] = None,
    model: Optional[str] = None,
    messages: Optional[List[Dict[str, str]]] = None,
) -> Dict[str, Any]:
    """Verify agent connection using calibrate's TextAgentConnection."""
    _validate_agent_url(agent_url)
    safe_headers = _sanitize_headers(agent_headers)
    agent = TextAgentConnection(url=agent_url, headers=safe_headers)

    try:
        kwargs = {}
        if model:
            kwargs["model"] = model
        if messages:
            kwargs["messages"] = messages
        result = await agent.verify(**kwargs)
    except Exception as e:
        logger.exception(
            "Agent connection verification failed unexpectedly: %s", str(e)
        )
        return {
            "success": False,
            "error": f"Unexpected error: {str(e)}",
            "sample_response": None,
        }

    sample_output = result.get("sample_output")

    if result["ok"]:
        return {
            "success": True,
            "error": None,
            "sample_response": sample_output,
        }

    return {
        "success": False,
        "error": result.get("error", "Verification failed"),
        "sample_response": sample_output,
    }


router = APIRouter(prefix="/agents", tags=["agents"])


def _default_agent_config() -> Dict[str, Any]:
    """Build the default config block for a freshly-created `type=agent`.

    Each field is overridable via env var so tenants can pin different
    defaults without a code change. The env helpers in utils.py treat
    empty strings as unset, so compose's `${VAR:-}` passthrough and the
    `VAR=` placeholders in .env.example both fall through to the
    hardcoded values below.
    """
    return {
        "system_prompt": env_str(
            "DEFAULT_AGENT_SYSTEM_PROMPT", "You are a helpful assistant."
        ),
        "llm": {
            "model": env_str("DEFAULT_AGENT_LLM_MODEL", "google/gemini-2.5-flash"),
        },
        "stt": {
            "provider": env_str("DEFAULT_AGENT_STT_PROVIDER", "google"),
        },
        "tts": {
            "provider": env_str("DEFAULT_AGENT_TTS_PROVIDER", "google"),
        },
        "settings": {
            # Match the simulation runtime fallback (True) so a freshly-created
            # agent and a legacy agent missing this key behave identically when
            # the env var is unset. Override via DEFAULT_AGENT_SPEAKS_FIRST.
            "agent_speaks_first": env_bool("DEFAULT_AGENT_SPEAKS_FIRST", True),
            "max_assistant_turns": env_int("DEFAULT_AGENT_MAX_TURNS", 50),
        },
    }


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively merge `override` into a copy of `base`. Caller wins per key.

    Dict-vs-dict at the same key recurses; anything else (scalar, list, None)
    in `override` replaces the corresponding `base` value entirely.
    """
    result = copy.deepcopy(base)
    for key, ov_value in override.items():
        base_value = result.get(key)
        if isinstance(base_value, dict) and isinstance(ov_value, dict):
            result[key] = _deep_merge(base_value, ov_value)
        else:
            result[key] = copy.deepcopy(ov_value)
    return result


class AgentCreate(BaseModel):
    name: str
    type: Literal["agent", "connection"] = "agent"
    config: Optional[Dict[str, Any]] = None


class AgentUpdate(BaseModel):
    name: Optional[str] = None
    config: Optional[Dict[str, Any]] = None
    connection_verified: Optional[bool] = None
    benchmark_models_verified: Optional[Dict[str, Any]] = None


class AgentResponse(BaseModel):
    uuid: str
    name: str
    type: Literal["agent", "connection"]
    config: Optional[Dict[str, Any]] = None
    created_at: str
    updated_at: str


class AgentCreateResponse(BaseModel):
    uuid: str
    message: str


class AgentDuplicateRequest(BaseModel):
    name: str


class AgentDuplicateResponse(BaseModel):
    uuid: str
    message: str


class VerifyConnectionRequest(BaseModel):
    agent_url: Optional[str] = None
    agent_headers: Optional[Dict[str, str]] = None
    model: Optional[str] = None
    messages: Optional[List[Dict[str, str]]] = None


class VerifyConnectionResponse(BaseModel):
    success: bool
    error: Optional[str] = None
    sample_response: Optional[Dict[str, Any]] = None


@router.post("/verify-connection", response_model=VerifyConnectionResponse)
async def verify_agent_connection_presave(
    request: VerifyConnectionRequest,
    user_id: str = Depends(get_current_user_id),
):
    """
    Verify an agent connection before saving (no agent UUID needed).
    Requires agent_url in the request body.
    """
    if not request.agent_url:
        raise HTTPException(status_code=400, detail="agent_url is required")

    result = await _verify_agent_connection(
        agent_url=request.agent_url,
        agent_headers=request.agent_headers,
        model=request.model,
        messages=request.messages,
    )
    return VerifyConnectionResponse(**result)


@router.post("/{agent_uuid}/verify-connection", response_model=VerifyConnectionResponse)
async def verify_agent_connection(
    agent_uuid: str,
    request: VerifyConnectionRequest,
    user_id: str = Depends(get_current_user_id),
):
    """
    Verify a saved agent's connection and persist the result in the agent config.

    - No model: basic check (required before running LLM unit tests or text simulations).
    - With model: model-specific check (required for each model before running a benchmark).
    """
    agent = get_agent(agent_uuid)
    if not agent:
        raise HTTPException(status_code=404, detail="Agent not found")
    if agent.get("user_id") != user_id:
        raise HTTPException(status_code=403, detail="Access denied")

    agent_config = agent.get("config") or {}
    agent_url = agent_config.get("agent_url")
    if not agent_url:
        raise HTTPException(
            status_code=400,
            detail="This agent does not have an agent_url configured. Add agent_url to use connection mode.",
        )

    agent_headers = agent_config.get("agent_headers")
    model = request.model

    # Strip provider/ prefix for non-openrouter providers so the agent
    # receives just the model name (e.g. "gpt-4.1" not "openai/gpt-4.1").
    verify_model = model
    if model and "/" in model:
        benchmark_provider = agent_config.get("benchmark_provider", "openrouter")
        if benchmark_provider != "openrouter":
            verify_model = model.split("/", 1)[-1]

    result = await _verify_agent_connection(
        agent_url=agent_url,
        agent_headers=agent_headers,
        model=verify_model,
        messages=request.messages,
    )

    # Only persist successful verification results into agent config.
    # Re-read the agent to get the latest config, avoiding a race condition
    # where two concurrent verify calls (different models) would each snapshot
    # the config before the await, then the second write would overwrite the first.
    if result["success"]:
        now = datetime.now(timezone.utc).isoformat()
        fresh_agent = get_agent(agent_uuid)
        new_config = copy.deepcopy(fresh_agent.get("config") or {})

        if model:
            benchmark_verified = new_config.get("benchmark_models_verified") or {}
            benchmark_verified[model] = {
                "verified": True,
                "verified_at": now,
                "error": None,
            }
            new_config["benchmark_models_verified"] = benchmark_verified
        else:
            new_config["connection_verified"] = True
            new_config["connection_verified_at"] = now
            new_config["connection_verified_error"] = None

        update_agent(agent_uuid=agent_uuid, config=new_config)

    return VerifyConnectionResponse(**result)


@router.post("", response_model=AgentCreateResponse)
async def create_agent_endpoint(
    agent: AgentCreate, user_id: str = Depends(get_current_user_id)
):
    """Create a new agent.

    For `type=agent`, the backend applies sensible defaults (system_prompt,
    llm.model, stt, tts, settings) overridable via DEFAULT_AGENT_* env vars.
    Any caller-supplied `config` is deep-merged on top, so partial overrides
    only replace the specific fields the caller cares about.

    For `type=connection`, no defaults are injected — the caller-supplied
    config (which must eventually contain `agent_url`) is stored as-is.
    """
    if agent.type == "agent":
        merged_config = _deep_merge(_default_agent_config(), agent.config or {})
    else:
        merged_config = agent.config

    with ensure_name_unique("agents", agent.name, user_id, entity="Agent"):
        agent_uuid = create_agent(
            name=agent.name,
            agent_type=agent.type,
            config=merged_config,
            user_id=user_id,
        )
    return AgentCreateResponse(uuid=agent_uuid, message="Agent created successfully")


@router.get("", response_model=List[AgentResponse])
async def list_agents(user_id: str = Depends(get_current_user_id)):
    """List all agents for the authenticated user."""
    agents = get_all_agents(user_id=user_id)
    return agents


@router.get("/{agent_uuid}", response_model=AgentResponse)
async def get_agent_endpoint(
    agent_uuid: str, user_id: str = Depends(get_current_user_id)
):
    """Get an agent by UUID."""
    agent = get_agent(agent_uuid)
    if not agent:
        raise HTTPException(status_code=404, detail="Agent not found")
    # Verify user owns this agent
    if agent.get("user_id") != user_id:
        raise HTTPException(status_code=403, detail="Access denied")
    return agent


@router.put("/{agent_uuid}", response_model=AgentResponse)
async def update_agent_endpoint(
    agent_uuid: str, agent: AgentUpdate, user_id: str = Depends(get_current_user_id)
):
    """Update an agent."""
    # Check if agent exists
    existing_agent = get_agent(agent_uuid)
    if not existing_agent:
        raise HTTPException(status_code=404, detail="Agent not found")

    # Verify user owns this agent
    if existing_agent.get("user_id") != user_id:
        raise HTTPException(status_code=403, detail="Access denied")

    # If agent_url or agent_headers changed, reset all verification flags
    if agent.config is not None:
        existing_config = existing_agent.get("config") or {}
        if agent.config.get("agent_url") != existing_config.get(
            "agent_url"
        ) or agent.config.get("agent_headers") != existing_config.get("agent_headers"):
            agent.config["connection_verified"] = False
            agent.config["connection_verified_at"] = None
            agent.config["connection_verified_error"] = None
            agent.config["benchmark_models_verified"] = {}

    # Merge top-level verification fields into config
    if (
        agent.connection_verified is not None
        or agent.benchmark_models_verified is not None
    ):
        if agent.config is None:
            agent.config = copy.deepcopy(existing_agent.get("config") or {})
        if agent.connection_verified is not None:
            agent.config["connection_verified"] = agent.connection_verified
        if agent.benchmark_models_verified is not None:
            agent.config["benchmark_models_verified"] = agent.benchmark_models_verified

    # Update only provided fields
    with ensure_name_unique(
        "agents", agent.name, user_id, entity="Agent", exclude_uuid=agent_uuid
    ):
        updated = update_agent(
            agent_uuid=agent_uuid,
            name=agent.name,
            config=agent.config,
        )

    if not updated:
        raise HTTPException(status_code=400, detail="No fields to update")

    # Return updated agent
    updated_agent = get_agent(agent_uuid)
    return updated_agent


@router.delete("/{agent_uuid}")
async def delete_agent_endpoint(
    agent_uuid: str, user_id: str = Depends(get_current_user_id)
):
    """Delete an agent."""
    # Check if agent exists and user owns it
    existing_agent = get_agent(agent_uuid)
    if not existing_agent:
        raise HTTPException(status_code=404, detail="Agent not found")
    if existing_agent.get("user_id") != user_id:
        raise HTTPException(status_code=403, detail="Access denied")

    deleted = delete_agent(agent_uuid)
    if not deleted:
        raise HTTPException(status_code=404, detail="Agent not found")
    return {"message": "Agent deleted successfully"}


@router.post("/{agent_uuid}/duplicate", response_model=AgentDuplicateResponse)
async def duplicate_agent_endpoint(
    agent_uuid: str,
    request: AgentDuplicateRequest,
    user_id: str = Depends(get_current_user_id),
):
    """
    Duplicate an agent with all its linked data.

    This will:
    - Copy the agent (with the provided name, config including speaks_first, data extraction fields, etc.)
    - Copy all linked tools
    - Copy all linked tests
    - Return the new agent UUID
    """
    # Get the original agent
    original_agent = get_agent(agent_uuid)
    if not original_agent:
        raise HTTPException(status_code=404, detail="Agent not found")

    # Verify user owns this agent
    if original_agent.get("user_id") != user_id:
        raise HTTPException(status_code=403, detail="Access denied")

    # Use the provided name
    new_name = request.name

    # Copy the entire config (includes speaks_first, data extraction fields, llm config, etc.)
    new_config = original_agent.get("config")
    if new_config:
        # Deep copy the config to avoid reference issues
        new_config = copy.deepcopy(new_config)
        # Strip verification flags — the duplicated agent's connection is unverified
        new_config.pop("connection_verified", None)
        new_config.pop("connection_verified_at", None)
        new_config.pop("connection_verified_error", None)
        new_config.pop("benchmark_models_verified", None)

    # Create the new agent
    with ensure_name_unique("agents", new_name, user_id, entity="Agent"):
        new_agent_uuid = create_agent(
            name=new_name,
            agent_type=original_agent.get("type", "agent"),
            config=new_config,
            user_id=user_id,
        )

    # Copy all linked tools
    linked_tools = get_tools_for_agent(agent_uuid)
    for tool in linked_tools:
        try:
            add_tool_to_agent(new_agent_uuid, tool["uuid"])
        except Exception as e:
            # Log but continue - don't fail the entire duplication
            logger.warning(
                f"Failed to link tool {tool['uuid']} to duplicated agent: {e}"
            )

    # Copy all linked tests
    linked_tests = get_tests_for_agent(agent_uuid)
    for test in linked_tests:
        try:
            add_test_to_agent(new_agent_uuid, test["uuid"])
        except Exception as e:
            # Log but continue - don't fail the entire duplication
            logger.warning(
                f"Failed to link test {test['uuid']} to duplicated agent: {e}"
            )

    return AgentDuplicateResponse(
        uuid=new_agent_uuid,
        message="Agent duplicated successfully with all linked tools and tests",
    )
