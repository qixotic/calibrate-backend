import copy
import ipaddress
import json
import logging
import socket
from datetime import datetime, timezone
from typing import Optional, List, Dict, Any, Literal
from urllib.parse import urlparse
from fastapi import APIRouter, HTTPException, Depends, Path, Body
from pagination import (
    OptionalPaginationParams,
    PaginatedResponse,
    count_and_page,
    make_search_params,
    page_envelope,
)

_AgentSearch = make_search_params(searchable=["name"])
_AgentEvaluatorSearch = make_search_params(searchable=["name"])
from pydantic import BaseModel, Field
from calibrate_agent.connections import TextAgentConnection

from utils import env_bool, env_int, env_str, AGENT_TYPE_DESCRIPTION, EvaluatorUuid

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
    get_evaluators_for_agent,
    add_evaluator_to_agent,
    remove_evaluator_from_agent,
)
from auth_utils import get_current_org, get_org_jwt_or_api_key, OrgContext
from org_scope import ensure_owned_agent, ensure_owned_evaluator

# Evaluators router imports no routers, so this edge (agents -> evaluators) is
# acyclic and safe at module load — needed because the list endpoint's
# `response_model` references `EvaluatorResponse` and it shares that router's
# page-shaping helper.
from routers.evaluators import EvaluatorResponse, build_evaluator_page

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

    Dict-vs-dict at the same key recurses. Anything else (scalar, list, None)
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


_VERIFICATION_CONFIG_KEYS = (
    "connection_verified",
    "connection_verified_at",
    "connection_verified_error",
    "benchmark_models_verified",
)


def _strip_verification_fields(
    config: Optional[Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    """Remove server-owned verification flags from a caller-supplied config dict.

    Only `POST /agents/{uuid}/verify-connection` may set these — it's the sole
    place that runs `_validate_agent_url`'s SSRF guard before contacting
    `agent_url`. An API-key client self-attesting `connection_verified=true`
    would skip that check entirely and let the job runner hit an arbitrary
    URL, so API-key writes always have these keys stripped regardless of
    whether they arrive via the dedicated fields or smuggled inside `config`.
    """
    if not config:
        return config
    for key in _VERIFICATION_CONFIG_KEYS:
        config.pop(key, None)
    return config


# Full agent-config schema, shared by create + update so it renders identically.
# Free-form on purpose (see the type-decision note): the shape is a discriminated
# union by `type` plus open extension keys, so it's documented, not enforced.
_AGENT_CONFIG_DESCRIPTION = """Agent behavioral config. The keys depend on `type`.

**`type=agent`**, built inside Calibrate:
- `system_prompt`: the agent's instructions
- `llm.model`: `provider/model`, e.g. `openai/gpt-4.1` or `google/gemini-2.5-flash`
- `stt.provider`: `deepgram`, `openai`, `cartesia`, `elevenlabs`, `google`, `sarvam`, or `smallest`
- `tts.provider`: `cartesia`, `openai`, `google`, `elevenlabs`, `sarvam`, or `smallest`
- `settings.agent_speaks_first`, `settings.max_assistant_turns`
- `system_tools.end_call`: let the agent end the call
- `data_extraction_fields`: `[{name, type, description, required}]`

```json
{
  "system_prompt": "You are a helpful support agent.",
  "llm": {"model": "openai/gpt-4.1"},
  "stt": {"provider": "deepgram"},
  "tts": {"provider": "elevenlabs"},
  "settings": {"agent_speaks_first": true, "max_assistant_turns": 50}
}
```

**`type=connection`**, your own HTTP endpoint:
- `agent_url`: public HTTP(S) endpoint your agent is called at
- `agent_headers`: headers sent on each request, e.g. auth
- `benchmark_provider`: `openrouter` by default. Other values: `openai`, `google`, `anthropic`, `meta-llama`, `mistralai`, `deepseek`, `x-ai`, `cohere`, `qwen`, or `ai21`

```json
{
  "agent_url": "https://api.example.com/agent",
  "agent_headers": {"Authorization": "Bearer <token>"},
  "benchmark_provider": "openrouter"
}
```"""


class AgentCreate(BaseModel):
    name: str = Field(description="Agent name, unique within the workspace")
    type: Literal["agent", "connection"] = Field(
        "agent",
        description=AGENT_TYPE_DESCRIPTION,
    )
    config: Optional[Dict[str, Any]] = Field(
        None,
        description=_AGENT_CONFIG_DESCRIPTION
        + "\n\nFor `type=agent`, omitted keys inherit managed defaults. Omit `config` entirely to use all defaults. For `type=connection`, `config` is stored as-is and must contain `agent_url`",
    )


# Named request-body examples for `POST /agents`. Rendered as a switchable
# dropdown in the API reference (and as per-variant snippets in the generated
# SDK/CLI docs) so a reader can toggle between building an agent inside Calibrate
# and connecting their own HTTP endpoint. The Calibrate example's config is built
# from `_default_agent_config()` itself, so it always shows the exact managed
# defaults a caller gets — it can't drift out of sync.
_CREATE_AGENT_EXAMPLES = {
    "agent_within_calibrate": {
        "summary": "Agent within Calibrate",
        "description": (
            "Build a voice/chat agent inside Calibrate. This config is the managed "
            "defaults spelled out. Override only the keys you want to change; "
            "omitted keys still inherit the defaults."
        ),
        "value": {
            "name": "Support Agent",
            "type": "agent",
            "config": _default_agent_config(),
        },
    },
    "openai_compatible_connection": {
        "summary": "Connect OpenAI-compatible agent",
        "description": (
            "Connect your own agent over an OpenAI-compatible HTTP endpoint. "
            "`config.agent_url` is required; `agent_headers` carries the auth "
            "token the endpoint expects."
        ),
        "value": {
            "name": "My Hosted Agent",
            "type": "connection",
            "config": {
                "agent_url": "https://api.example.com/v1/chat/completions",
                "agent_headers": {"Authorization": "Bearer <token>"},
                "benchmark_provider": "openrouter",
            },
        },
    },
}


def _curl_code_samples(examples: Dict[str, Any], path: str) -> List[Dict[str, str]]:
    """Render each named request example as a copy-ready cURL snippet.

    Mintlify's code-sample panel is otherwise schema-generated and collapses to
    the required fields only (just `name` here, since `type` defaults and
    `config` is optional), so the copyable body never shows the real `config`
    shape. `x-codeSamples` overrides that panel with these verbatim snippets.
    Built FROM `examples` so the sample bodies can't drift from the request-body
    dropdown — one source of truth.
    """
    base_url = env_str("PUBLIC_API_BASE_URL", "http://localhost:8000").rstrip("/")
    return [
        {
            "lang": "curl",
            "label": ex["summary"],
            "source": (
                "curl --request POST \\\n"
                f"  --url {base_url}{path} \\\n"
                "  --header 'Content-Type: application/json' \\\n"
                "  --header 'X-API-Key: <api-key>' \\\n"
                f"  --data '{json.dumps(ex['value'], indent=2)}'"
            ),
        }
        for ex in examples.values()
    ]


class AgentUpdate(BaseModel):
    name: Optional[str] = Field(
        None, description="New agent name. Omit to leave the name unchanged"
    )
    config: Optional[Dict[str, Any]] = Field(
        None,
        description=_AGENT_CONFIG_DESCRIPTION
        + "\n\nReplaces the stored config. Omit to leave unchanged"
        + "\n\nFor `type=connection`, changing `agent_url` or `agent_headers` resets the connection and benchmark verification flags",
    )
    connection_verified: Optional[bool] = Field(
        None,
        description="Set the connection verification flag for a `type=connection` agent. Omit to leave it untouched",
    )
    benchmark_models_verified: Optional[Dict[str, Any]] = Field(
        None,
        description="Set the benchmark verification map, keyed by model, for a `type=connection` agent. Omit to leave it untouched",
        examples=[{"openai/gpt-4.1": {"verified": True, "verified_at": "2026-01-01T00:00:00Z", "error": None}}],
    )


class AgentResponse(BaseModel):
    uuid: str = Field(
        min_length=36,
        max_length=36,
        description="ID of the agent",
        examples=["f47ac10b-58cc-4372-a567-0e02b2c3d479"],
    )
    name: str = Field(description="Name of the agent")
    type: Literal["agent", "connection"] = Field(description=AGENT_TYPE_DESCRIPTION)
    config: Optional[Dict[str, Any]] = Field(None, description="Agent configuration")
    created_at: str = Field(description="When the agent was created (ISO 8601 UTC)")
    updated_at: str = Field(
        description="When the agent was last updated (ISO 8601 UTC)"
    )


class AgentSummary(BaseModel):
    uuid: str = Field(
        min_length=36,
        max_length=36,
        description="ID of the agent",
        examples=["f47ac10b-58cc-4372-a567-0e02b2c3d479"],
    )
    name: str = Field(description="Name of the agent")
    type: Literal["agent", "connection"] = Field(description=AGENT_TYPE_DESCRIPTION)
    updated_at: str = Field(
        description="When the agent was last updated (ISO 8601 UTC)"
    )
    connection_verified: Optional[bool] = Field(
        None,
        description="Whether the agent's connection has been verified, for a `type=connection` agent",
    )


def _to_agent_summary(agent: Dict[str, Any]) -> AgentSummary:
    """Project an agent row to the trimmed list shape, lifting
    `config.connection_verified` to a top-level flag (None when absent)."""
    verified = (agent.get("config") or {}).get("connection_verified")
    return AgentSummary(
        uuid=agent["uuid"],
        name=agent["name"],
        type=agent["type"],
        updated_at=agent["updated_at"],
        connection_verified=None if verified is None else bool(verified),
    )


class AgentCreateResponse(BaseModel):
    uuid: str = Field(
        min_length=36,
        max_length=36,
        description="ID of the newly created agent",
        examples=["f47ac10b-58cc-4372-a567-0e02b2c3d479"],
    )
    message: str = Field(description="Confirmation message")


class AgentDuplicateRequest(BaseModel):
    name: str = Field(
        description="Name for the duplicated agent, unique within the workspace"
    )


class AgentDuplicateResponse(BaseModel):
    uuid: str = Field(
        min_length=36,
        max_length=36,
        description="ID of the newly created duplicate agent",
        examples=["f47ac10b-58cc-4372-a567-0e02b2c3d479"],
    )
    message: str = Field(description="Confirmation message")


class EvaluatorLinkRequest(BaseModel):
    evaluator_ids: List[EvaluatorUuid] = Field(
        description="The evaluators to link to the agent. Ones that are already linked are skipped. Each must be one you created or a built-in default",
        examples=[["f47ac10b-58cc-4372-a567-0e02b2c3d479"]],
    )


class EvaluatorLinkResponse(BaseModel):
    message: str = Field(
        description="Confirmation that the evaluators were linked",
        examples=["Evaluators linked to agent"],
    )
    linked: List[str] = Field(
        description="Evaluator IDs newly linked by this request"
    )
    already_linked: List[str] = Field(
        description="Evaluator IDs skipped because they were already linked"
    )


class ResolveAgentNamesRequest(BaseModel):
    names: List[str] = Field(
        description="Agent names to resolve to IDs",
        examples=[["my-agent", "support-bot"]],
    )


class ResolveAgentNamesResponse(BaseModel):
    resolved: Dict[str, str] = Field(
        description="Map of name to agent ID for each name that matched"
    )
    not_found: List[str] = Field(
        description="Names with no matching agent in your workspace"
    )


class AgentVerifyRequest(BaseModel):
    """Body for verifying an existing agent by ID. The endpoint (`agent_url`,
    `agent_headers`) comes from the agent's stored config, so only the probe
    inputs are accepted here."""

    model: Optional[str] = Field(
        None,
        description="Model to verify. Omit for a basic connection check. Provide it for a model-specific check before benchmarking that model",
        examples=["openai/gpt-4.1"],
    )
    messages: Optional[List[Dict[str, str]]] = Field(
        None,
        description="Sample chat messages to send during verification. Omit to use the default probe",
        examples=[[{"role": "user", "content": "Hello"}]],
    )


class VerifyConnectionRequest(AgentVerifyRequest):
    agent_url: Optional[str] = Field(
        None,
        description="Public HTTP(S) agent endpoint to verify",
        examples=["https://api.example.com/agent"],
    )
    agent_headers: Optional[Dict[str, str]] = Field(
        None,
        description="Extra request headers to send to your agent, e.g. an auth token. Omit if none are needed",
        examples=[{"Authorization": "Bearer <token>"}],
    )


class VerifyConnectionResponse(BaseModel):
    success: bool = Field(
        description="True if the agent responded successfully to the verification probe"
    )
    error: Optional[str] = Field(None, description="Reason the verification failed")
    sample_response: Optional[Dict[str, Any]] = Field(
        None,
        description="Sample output returned by the agent during verification",
    )


@router.post(
    "/verify-connection",
    response_model=VerifyConnectionResponse,
    summary="Verify an agent connection",
)
async def verify_agent_connection_presave(
    request: VerifyConnectionRequest,
    ctx: OrgContext = Depends(get_current_org),
):
    """Verify an agent connection without creating an agent. Nothing is persisted"""
    if not request.agent_url:
        raise HTTPException(status_code=400, detail="agent_url is required")

    result = await _verify_agent_connection(
        agent_url=request.agent_url,
        agent_headers=request.agent_headers,
        model=request.model,
        messages=request.messages,
    )
    return VerifyConnectionResponse(**result)


@router.post(
    "/{agent_uuid}/verify-connection",
    response_model=VerifyConnectionResponse,
    summary="Verify agent connection",
    tags=["Public API"],
)
async def verify_agent_connection(
    agent_uuid: str = Path(
        description="The agent whose connection to verify",
        examples=["f47ac10b-58cc-4372-a567-0e02b2c3d479"],
    ),
    request: AgentVerifyRequest = ...,
    ctx: OrgContext = Depends(get_org_jwt_or_api_key),
):
    """Verify an agent's connection and persist the result when successful"""
    agent = get_agent(agent_uuid)
    if not agent or agent.get("org_uuid") != ctx.org_uuid:
        raise HTTPException(status_code=404, detail="Agent not found")

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


@router.post(
    "/resolve",
    response_model=ResolveAgentNamesResponse,
    tags=["Public API"],
    summary="Resolve agent names to IDs",
)
async def resolve_agent_names(
    request: ResolveAgentNamesRequest,
    ctx: OrgContext = Depends(get_org_jwt_or_api_key),
):
    """Get the IDs for your agents by their names"""
    # Public API. Auth via get_org_jwt_or_api_key (JWT for the web app, API key
    # for CI). Maps human-friendly names to the UUIDs the run/poll endpoints expect.
    agents = get_all_agents(org_uuid=ctx.org_uuid)
    name_to_uuid = {agent["name"]: agent["uuid"] for agent in agents}

    resolved: Dict[str, str] = {}
    not_found: List[str] = []
    for name in request.names:
        if name in name_to_uuid:
            resolved[name] = name_to_uuid[name]
        elif name not in not_found:
            not_found.append(name)

    return ResolveAgentNamesResponse(resolved=resolved, not_found=not_found)


@router.post(
    "",
    response_model=AgentCreateResponse,
    tags=["Public API"],
    summary="Create agent",
    openapi_extra={"x-codeSamples": _curl_code_samples(_CREATE_AGENT_EXAMPLES, "/agents")},
)
async def create_agent_endpoint(
    agent: AgentCreate = Body(openapi_examples=_CREATE_AGENT_EXAMPLES),
    ctx: OrgContext = Depends(get_org_jwt_or_api_key),
):
    """Create an agent to test inside Calibrate or connect your existing agent to Calibrate"""
    if agent.type == "agent":
        merged_config = _deep_merge(_default_agent_config(), agent.config or {})
    else:
        merged_config = agent.config

    if ctx.auth_method == "api_key":
        merged_config = _strip_verification_fields(merged_config)

    with ensure_name_unique("agents", agent.name, ctx.org_uuid, entity="Agent"):
        agent_uuid = create_agent(
            name=agent.name,
            agent_type=agent.type,
            config=merged_config,
            org_uuid=ctx.org_uuid,
            user_id=ctx.user_id,
        )
    return AgentCreateResponse(uuid=agent_uuid, message="Agent created successfully")


@router.get(
    "",
    response_model=PaginatedResponse[AgentSummary],
    tags=["Public API"],
    summary="List agents",
)
async def list_agents(
    ctx: OrgContext = Depends(get_org_jwt_or_api_key),
    search: _AgentSearch = Depends(),
    pagination: OptionalPaginationParams = Depends(),
):
    """Get the list of all your agents"""
    # Public API. Auth via get_org_jwt_or_api_key (JWT for the web app, API key
    # for CI); the run/poll and /resolve endpoints accept the same key, so CI can
    # enumerate agent UUIDs without knowing names up front.
    # Optional `?q=` name search + `?limit=&offset=` paging. Returns the
    # `{items, total, limit, offset}` envelope. Each item is a trimmed summary
    # (no full `config`) so the bulk list never ships agent auth credentials
    # (`config.agent_headers`) — the detail endpoint (`GET /agents/{uuid}`)
    # refetches the full config; the summary transform runs only on the page.
    agents = get_all_agents(org_uuid=ctx.org_uuid)
    agents = search.apply(agents)
    page, total = count_and_page(agents, pagination)
    return page_envelope([_to_agent_summary(a) for a in page], total, pagination)


@router.get(
    "/{agent_uuid}",
    response_model=AgentResponse,
    tags=["Public API"],
    summary="Get agent",
)
async def get_agent_endpoint(
    agent_uuid: str = Path(
        description="The agent to retrieve",
        examples=["f47ac10b-58cc-4372-a567-0e02b2c3d479"],
    ),
    ctx: OrgContext = Depends(get_org_jwt_or_api_key),
):
    """Get one agent by its ID"""
    agent = get_agent(agent_uuid)
    if not agent or agent.get("org_uuid") != ctx.org_uuid:
        raise HTTPException(status_code=404, detail="Agent not found")
    return agent


@router.put(
    "/{agent_uuid}",
    response_model=AgentResponse,
    tags=["Public API"],
    summary="Update agent",
)
async def update_agent_endpoint(
    agent_uuid: str = Path(
        description="The agent to update",
        examples=["f47ac10b-58cc-4372-a567-0e02b2c3d479"],
    ),
    agent: AgentUpdate = ...,
    ctx: OrgContext = Depends(get_org_jwt_or_api_key),
):
    """Update an agent's configuration"""
    existing_agent = get_agent(agent_uuid)
    if not existing_agent or existing_agent.get("org_uuid") != ctx.org_uuid:
        raise HTTPException(status_code=404, detail="Agent not found")

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

    if ctx.auth_method == "api_key":
        # API-key clients can't self-attest verification — see
        # `_strip_verification_fields`. They must call the JWT-only
        # `/agents/{uuid}/verify-connection` to actually flip these flags.
        agent.config = _strip_verification_fields(agent.config)
    elif (
        agent.connection_verified is not None
        or agent.benchmark_models_verified is not None
    ):
        if agent.config is None:
            agent.config = copy.deepcopy(existing_agent.get("config") or {})
        if agent.connection_verified is not None:
            agent.config["connection_verified"] = agent.connection_verified
        if agent.benchmark_models_verified is not None:
            agent.config["benchmark_models_verified"] = agent.benchmark_models_verified

    with ensure_name_unique(
        "agents", agent.name, ctx.org_uuid, entity="Agent", exclude_uuid=agent_uuid
    ):
        updated = update_agent(
            agent_uuid=agent_uuid,
            name=agent.name,
            config=agent.config,
        )

    if not updated:
        raise HTTPException(status_code=400, detail="No fields to update")

    updated_agent = get_agent(agent_uuid)
    return updated_agent


@router.delete("/{agent_uuid}", summary="Delete agent")
async def delete_agent_endpoint(
    agent_uuid: str = Path(
        description="The agent to delete",
        examples=["f47ac10b-58cc-4372-a567-0e02b2c3d479"],
    ),
    ctx: OrgContext = Depends(get_current_org),
):
    """Soft-delete an agent"""
    existing_agent = get_agent(agent_uuid)
    if not existing_agent or existing_agent.get("org_uuid") != ctx.org_uuid:
        raise HTTPException(status_code=404, detail="Agent not found")

    deleted = delete_agent(agent_uuid)
    if not deleted:
        raise HTTPException(status_code=404, detail="Agent not found")
    return {"message": "Agent deleted successfully"}


@router.post(
    "/{agent_uuid}/duplicate",
    response_model=AgentDuplicateResponse,
    summary="Duplicate agent",
)
async def duplicate_agent_endpoint(
    agent_uuid: str = Path(
        description="The agent to duplicate",
        examples=["f47ac10b-58cc-4372-a567-0e02b2c3d479"],
    ),
    request: AgentDuplicateRequest = ...,
    ctx: OrgContext = Depends(get_current_org),
):
    """Duplicate an agent along with its linked tools and tests. Verification flags are not copied"""
    original_agent = get_agent(agent_uuid)
    if not original_agent or original_agent.get("org_uuid") != ctx.org_uuid:
        raise HTTPException(status_code=404, detail="Agent not found")

    new_name = request.name

    new_config = original_agent.get("config")
    if new_config:
        new_config = copy.deepcopy(new_config)
        # Strip verification flags — the duplicated agent's connection is unverified
        new_config = _strip_verification_fields(new_config)

    with ensure_name_unique("agents", new_name, ctx.org_uuid, entity="Agent"):
        new_agent_uuid = create_agent(
            name=new_name,
            agent_type=original_agent.get("type", "agent"),
            config=new_config,
            org_uuid=ctx.org_uuid,
            user_id=ctx.user_id,
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

    # Copy all linked evaluators
    linked_evaluators = get_evaluators_for_agent(agent_uuid)
    for evaluator in linked_evaluators:
        try:
            add_evaluator_to_agent(new_agent_uuid, evaluator["uuid"])
        except Exception as e:
            # Log but continue - don't fail the entire duplication
            logger.warning(
                f"Failed to link evaluator {evaluator['uuid']} to duplicated agent: {e}"
            )

    return AgentDuplicateResponse(
        uuid=new_agent_uuid,
        message="Agent duplicated successfully with all linked tools, tests, and evaluators",
    )


# ============ Evaluator linking ============


@router.get(
    "/{agent_uuid}/evaluators",
    response_model=PaginatedResponse[EvaluatorResponse],
    summary="List agent evaluators",
    tags=["Public API"],
)
async def list_agent_evaluators(
    agent_uuid: str = Path(
        description="The agent whose evaluators to list",
        examples=["f47ac10b-58cc-4372-a567-0e02b2c3d479"],
    ),
    ctx: OrgContext = Depends(get_org_jwt_or_api_key),
    search: _AgentEvaluatorSearch = Depends(),
    pagination: OptionalPaginationParams = Depends(),
):
    """List evaluators linked to an agent"""
    ensure_owned_agent(agent_uuid, ctx.org_uuid)
    evaluators = get_evaluators_for_agent(agent_uuid)
    evaluators = search.apply(evaluators)
    page, total = count_and_page(evaluators, pagination)
    return build_evaluator_page(page, total, pagination)


@router.post(
    "/{agent_uuid}/evaluators",
    response_model=EvaluatorLinkResponse,
    summary="Link evaluators to agent",
    tags=["Public API"],
)
async def link_evaluators_to_agent(
    agent_uuid: str = Path(
        description="The agent to link the evaluators to",
        examples=["f47ac10b-58cc-4372-a567-0e02b2c3d479"],
    ),
    payload: EvaluatorLinkRequest = ...,
    ctx: OrgContext = Depends(get_org_jwt_or_api_key),
):
    """Link one or more existing evaluators to an agent, skipping any already linked"""
    ensure_owned_agent(agent_uuid, ctx.org_uuid)
    # Validate every id up front so a bad one links nothing.
    for evaluator_id in payload.evaluator_ids:
        ensure_owned_evaluator(evaluator_id, ctx.org_uuid)
    current = {e["uuid"] for e in get_evaluators_for_agent(agent_uuid)}
    linked: List[str] = []
    already_linked: List[str] = []
    seen: set = set()
    for evaluator_id in payload.evaluator_ids:
        if evaluator_id in seen:
            continue
        seen.add(evaluator_id)
        if evaluator_id in current:
            already_linked.append(evaluator_id)
        else:
            add_evaluator_to_agent(agent_uuid, evaluator_id)
            linked.append(evaluator_id)
    return EvaluatorLinkResponse(
        message="Evaluators linked to agent",
        linked=linked,
        already_linked=already_linked,
    )


@router.delete(
    "/{agent_uuid}/evaluators/{evaluator_uuid}",
    summary="Unlink evaluator from agent",
)
async def unlink_evaluator_from_agent(
    agent_uuid: str = Path(
        description="The agent to unlink the evaluator from",
        examples=["f47ac10b-58cc-4372-a567-0e02b2c3d479"],
    ),
    evaluator_uuid: str = Path(
        description="The evaluator to unlink from the agent",
        examples=["f47ac10b-58cc-4372-a567-0e02b2c3d479"],
    ),
    ctx: OrgContext = Depends(get_current_org),
):
    """Unlink an evaluator from an agent"""
    ensure_owned_agent(agent_uuid, ctx.org_uuid)
    removed = remove_evaluator_from_agent(agent_uuid, evaluator_uuid)
    if not removed:
        raise HTTPException(
            status_code=404, detail="Evaluator is not linked to this agent"
        )
    return {"message": "Evaluator unlinked from agent"}
