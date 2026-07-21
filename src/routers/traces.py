"""Production trace ingestion and curation.

Customer backends POST one trace per agent turn: the conversation history as
`input` plus the produced `output`. Rows persist in the dedicated traces store
(src/traces/), not pense.db. The contract deliberately mirrors test creation:
`input` is `tests.config.history` verbatim, and `output.tool_calls` matches
the expected-tool-call shape, so curated traces convert to tests without
transformation. New contract needs go into `metadata` keys, not new top-level
fields: customers integrate against this shape, and every field deepens the
eventual OTel-gateway migration.
"""

import logging
from typing import Any, Dict, List, Literal, Optional

from fastapi import APIRouter, Depends, HTTPException, Path, Query
from pydantic import BaseModel, ConfigDict, Field, model_validator

from auth_utils import OrgContext, get_current_org, get_org_jwt_or_api_key
from db import (
    add_test_to_agent,
    bulk_create_tests,
    get_agent,
    get_all_tests_summary,
    set_test_evaluators,
)
from pagination import PaginatedResponse, PaginationParams, page_envelope
from routers.org_limits import get_max_traces_for_org

# Reuse the test router's evaluator ref + validation so converted tests accept
# exactly what POST /tests does (workspace-visible, evaluator_type matches).
from routers.tests import EvaluatorRef, _validate_evaluators
from traces import store as traces_store
from utils import EXAMPLE_TEST_UUID

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/traces", tags=["traces"])

MAX_INPUT_TURNS = 500
MAX_TURN_CONTENT_CHARS = 50_000
MAX_TOOL_CALLS = 50
MAX_METADATA_ENTRIES = 100
# One conversion request creates at most this many tests; a wider select_all
# match 400s with a "narrow the filter" hint rather than mass-creating.
MAX_CONVERT_BATCH = 500

_EXAMPLE_TRACE_UUID = "f47ac10b-58cc-4372-a567-0e02b2c3d479"

_TRACE_UUID_DESCRIPTION = "Unique ID for the trace"

_Q_DESCRIPTION = (
    "Case-insensitive substring search on `message_id`, `conversation_id`, "
    "and message content"
)


class TraceTurn(BaseModel):
    # Extra keys (OpenAI `tool_calls`, `tool_call_id`, `name`, ...) are stored
    # verbatim so the history stays lossless for test conversion.
    model_config = ConfigDict(extra="allow")

    role: str = Field(
        min_length=1,
        max_length=64,
        description="Message author role in the conversation history",
    )
    content: Optional[str] = Field(
        None,
        max_length=MAX_TURN_CONTENT_CHARS,
        description="Message text. Omit for turns that only carry tool calls",
    )


class TraceToolCall(BaseModel):
    model_config = ConfigDict(extra="forbid")

    tool: str = Field(
        min_length=1,
        max_length=255,
        description="Name of the tool the agent called",
    )
    arguments: Optional[Dict[str, Any]] = Field(
        None,
        description="Argument values the agent passed to the tool. Omit when the call had none",
    )


class TraceOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    response: Optional[str] = Field(
        None,
        max_length=MAX_TURN_CONTENT_CHARS,
        description="The assistant reply text for this turn. Omit for turns that only issued tool calls",
    )
    tool_calls: Optional[List[TraceToolCall]] = Field(
        None,
        max_length=MAX_TOOL_CALLS,
        description="Tool calls the agent issued for this turn. Omit for plain text replies",
    )

    @model_validator(mode="after")
    def _require_response_or_tool_calls(self):
        if not (self.response and self.response.strip()) and not self.tool_calls:
            raise ValueError("output must include a response or at least one tool call")
        return self


class TraceMetadataEntry(BaseModel):
    model_config = ConfigDict(extra="forbid")

    key: str = Field(
        min_length=1,
        max_length=256,
        description="Name of the metadata entry",
    )
    value: str = Field(
        max_length=8192,
        description="Value of the metadata entry",
    )


class TraceIngest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    message_id: str = Field(
        min_length=1,
        max_length=255,
        description="Your ID for the last user message in `input`, unique within your workspace. Sending the same ID again returns the stored trace instead of creating a duplicate",
    )
    conversation_id: str = Field(
        min_length=1,
        max_length=255,
        description="Your ID for the conversation this turn belongs to. Reuse `message_id` when there is no conversation to group by",
    )
    input: List[TraceTurn] = Field(
        min_length=1,
        max_length=MAX_INPUT_TURNS,
        description="Conversation history up to the reported output, oldest turn first, in OpenAI chat format",
    )
    output: TraceOutput = Field(description="What the agent produced for this turn")
    metadata: Optional[List[TraceMetadataEntry]] = Field(
        None,
        max_length=MAX_METADATA_ENTRIES,
        description="Key-value pairs stored with the trace. Prefer OTel `gen_ai.*` key names where they fit. Omit if you have none",
    )


class TraceIngestResponse(BaseModel):
    uuid: str = Field(
        min_length=36,
        max_length=36,
        description=_TRACE_UUID_DESCRIPTION,
        examples=[_EXAMPLE_TRACE_UUID],
    )
    message_id: str = Field(description="Your ID for the trace's last user message")
    conversation_id: str = Field(
        description="Your ID for the conversation the trace belongs to"
    )
    created: bool = Field(
        description="Whether this call stored a new trace. False when a trace with this `message_id` already existed"
    )
    created_at: str = Field(description="When the trace was created (ISO 8601 UTC)")


class TraceSummary(BaseModel):
    uuid: str = Field(
        min_length=36,
        max_length=36,
        description=_TRACE_UUID_DESCRIPTION,
        examples=[_EXAMPLE_TRACE_UUID],
    )
    message_id: str = Field(description="Your ID for the trace's last user message")
    conversation_id: str = Field(
        description="Your ID for the conversation the trace belongs to"
    )
    input_preview: Optional[str] = Field(
        None, description="The last user message, truncated for display"
    )
    response_preview: Optional[str] = Field(
        None, description="The agent reply, truncated for display"
    )
    turn_count: int = Field(
        description="Number of turns in the stored conversation history"
    )
    tool_call_count: int = Field(
        description="Number of tool calls the agent issued for this turn"
    )
    metadata_count: int = Field(
        description="Number of metadata entries stored with the trace"
    )
    created_at: str = Field(description="When the trace was created (ISO 8601 UTC)")


class TraceResponse(BaseModel):
    uuid: str = Field(
        min_length=36,
        max_length=36,
        description=_TRACE_UUID_DESCRIPTION,
        examples=[_EXAMPLE_TRACE_UUID],
    )
    message_id: str = Field(description="Your ID for the trace's last user message")
    conversation_id: str = Field(
        description="Your ID for the conversation the trace belongs to"
    )
    input: List[TraceTurn] = Field(
        description="Conversation history stored for this trace, oldest turn first"
    )
    output: TraceOutput = Field(description="What the agent produced for this turn")
    metadata: Optional[List[TraceMetadataEntry]] = Field(
        None, description="Key-value pairs stored with the trace"
    )
    created_at: str = Field(description="When the trace was created (ISO 8601 UTC)")
    updated_at: str = Field(
        description="When the trace was last updated (ISO 8601 UTC)"
    )


class BulkDeleteTracesRequest(BaseModel):
    trace_ids: Optional[List[str]] = Field(
        None,
        description="IDs of the traces to delete. **Required when `select_all` is false.** Ignored otherwise",
    )
    select_all: bool = Field(
        False,
        description="Delete every trace matching `q` and `conversation_id` instead of an explicit ID list",
    )
    q: Optional[str] = Field(
        None,
        description=_Q_DESCRIPTION + ". Applied when `select_all` is true",
    )
    conversation_id: Optional[str] = Field(
        None,
        description="Limit `select_all` to traces from this conversation",
    )


class BulkDeleteTracesResponse(BaseModel):
    deleted: int = Field(description="Number of traces deleted")


_PREVIEW_CHARS = 160


def _preview(text: Optional[str]) -> Optional[str]:
    if not text:
        return None
    text = text.strip()
    if len(text) <= _PREVIEW_CHARS:
        return text
    return text[: _PREVIEW_CHARS - 1] + "…"


def _last_user_content(input_turns: List[Dict[str, Any]]) -> Optional[str]:
    for turn in reversed(input_turns or []):
        if turn.get("role") == "user" and isinstance(turn.get("content"), str):
            return turn["content"]
    return None


def _to_summary(row: Dict[str, Any]) -> Dict[str, Any]:
    output = row.get("output") or {}
    return {
        "uuid": row["uuid"],
        "message_id": row["message_id"],
        "conversation_id": row["conversation_id"],
        "input_preview": _preview(_last_user_content(row.get("input") or [])),
        "response_preview": _preview(output.get("response")),
        "turn_count": len(row.get("input") or []),
        "tool_call_count": len(output.get("tool_calls") or []),
        "metadata_count": len(row.get("metadata") or []),
        "created_at": row["created_at"],
    }


def _ingest_response(row: Dict[str, Any], created: bool) -> Dict[str, Any]:
    return {
        "uuid": row["uuid"],
        "message_id": row["message_id"],
        "conversation_id": row["conversation_id"],
        "created": created,
        "created_at": row["created_at"],
    }


@router.post("", response_model=TraceIngestResponse, summary="Create trace")
async def ingest_trace(
    payload: TraceIngest, ctx: OrgContext = Depends(get_org_jwt_or_api_key)
):
    """Store a production agent turn and its conversation history for later curation"""
    # Idempotency outranks the cap: a retry of an already-stored message_id
    # must succeed even when the workspace is at its limit.
    existing = traces_store.get_trace_by_message_id(ctx.org_uuid, payload.message_id)
    if existing:
        return _ingest_response(existing, created=False)

    cap = get_max_traces_for_org(ctx.org_uuid)
    current = traces_store.count_live_traces(ctx.org_uuid)
    if current >= cap:
        raise HTTPException(
            status_code=429,
            detail={
                "error": "Trace limit reached for this workspace",
                "current": current,
                "max_traces": cap,
                "hint": "Delete traces to free capacity or ask an administrator to raise the workspace limit",
            },
        )

    row, created = traces_store.create_trace(
        org_uuid=ctx.org_uuid,
        message_id=payload.message_id,
        conversation_id=payload.conversation_id,
        input=[turn.model_dump(exclude_none=True) for turn in payload.input],
        output=payload.output.model_dump(exclude_none=True),
        metadata=(
            [entry.model_dump() for entry in payload.metadata]
            if payload.metadata
            else None
        ),
    )
    return _ingest_response(row, created=created)


@router.get("", response_model=PaginatedResponse[TraceSummary], summary="List traces")
async def list_traces_endpoint(
    ctx: OrgContext = Depends(get_current_org),
    pagination: PaginationParams = Depends(),
    q: Optional[str] = Query(None, description=_Q_DESCRIPTION + ". Blank is a no-op"),
    conversation_id: Optional[str] = Query(
        None, description="Return only traces from this conversation"
    ),
):
    """List ingested traces, newest first"""
    # Search/filter/count run in SQL (traces.store), not the post-fetch
    # pagination helpers, and paging uses the bounded PaginationParams rather
    # than the unbounded OptionalPaginationParams: traces are machine-written
    # and outgrow in-memory filtering fast.
    rows, total = traces_store.list_traces(
        ctx.org_uuid,
        limit=pagination.limit,
        offset=pagination.offset,
        q=q,
        conversation_id=conversation_id,
    )
    return page_envelope([_to_summary(row) for row in rows], total, pagination)


@router.post(
    "/bulk-delete",
    response_model=BulkDeleteTracesResponse,
    summary="Bulk delete traces",
)
async def bulk_delete_traces(
    payload: BulkDeleteTracesRequest, ctx: OrgContext = Depends(get_current_org)
):
    """Soft-delete traces, freeing their capacity and message IDs for re-ingestion"""
    if not payload.select_all and not payload.trace_ids:
        raise HTTPException(
            status_code=400,
            detail="trace_ids must be non-empty when select_all is false",
        )
    deleted = traces_store.soft_delete_traces(
        ctx.org_uuid,
        trace_ids=payload.trace_ids,
        select_all=payload.select_all,
        q=payload.q,
        conversation_id=payload.conversation_id,
    )
    return {"deleted": deleted}


_CONVERT_TYPE_DESCRIPTION = (
    "What the created tests judge:\n\n"
    "- `response`: judge the agent's regenerated reply against the linked evaluators\n"
    "- `tool_call`: diff the agent's regenerated tool calls against the ones the trace recorded"
)


class ConvertTracesToTestsRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    trace_ids: Optional[List[str]] = Field(
        None,
        description="IDs of the traces to convert. **Required when `select_all` is false.** Ignored otherwise",
    )
    select_all: bool = Field(
        False,
        description="Convert every trace matching `q` and `conversation_id` instead of an explicit ID list",
    )
    q: Optional[str] = Field(
        None,
        description=_Q_DESCRIPTION + ". Applied when `select_all` is true",
    )
    conversation_id: Optional[str] = Field(
        None,
        description="Limit `select_all` to traces from this conversation",
    )
    type: Literal["response", "tool_call"] = Field(description=_CONVERT_TYPE_DESCRIPTION)
    evaluators: Optional[List[EvaluatorRef]] = Field(
        None,
        description="Evaluators to link to each created test. **Required for `response`**, unused for `tool_call`",
    )
    agent_uuids: Optional[List[str]] = Field(
        None,
        description="IDs of agents to link every created test to. Omit to link none",
    )
    accept_any_arguments: bool = Field(
        False,
        description="For `tool_call`, match only the tool name and ignore the arguments the trace recorded",
    )


class ConvertTracesToTestsResponse(BaseModel):
    created: int = Field(description="Number of tests created")
    test_uuids: List[str] = Field(
        description="IDs of the created tests, in creation order",
        examples=[[EXAMPLE_TEST_UUID]],
    )


def _dedupe_test_names(candidates: List[str], taken: set) -> List[str]:
    """Make each candidate test name unique against `taken` (existing workspace
    test names, mutated as names are claimed) by appending ` (2)`, ` (3)`, …
    Converting the same traces twice yields new names rather than a 400."""
    out: List[str] = []
    for base in candidates:
        name = base
        n = 2
        while name in taken:
            name = f"{base} ({n})"
            n += 1
        taken.add(name)
        out.append(name)
    return out


@router.post(
    "/convert-to-tests",
    response_model=ConvertTracesToTestsResponse,
    summary="Convert traces to tests",
)
async def convert_traces_to_tests(
    payload: ConvertTracesToTestsRequest, ctx: OrgContext = Depends(get_current_org)
):
    """Turn production traces into regression tests you can run and benchmark"""
    if not payload.select_all and not payload.trace_ids:
        raise HTTPException(
            status_code=400,
            detail="trace_ids must be non-empty when select_all is false",
        )

    if payload.agent_uuids:
        for agent_uuid in payload.agent_uuids:
            agent = get_agent(agent_uuid)
            if not agent or agent.get("org_uuid") != ctx.org_uuid:
                raise HTTPException(
                    status_code=404, detail=f"Agent {agent_uuid} not found"
                )

    # A converted response test re-runs the agent and judges the fresh reply, so
    # it has no fallback judge — require at least one (llm) evaluator up front.
    resolved_refs: Optional[List[Dict[str, Any]]] = None
    if payload.type == "response":
        if not payload.evaluators:
            raise HTTPException(
                status_code=400,
                detail="response tests require at least one evaluator",
            )
        resolved_refs = _validate_evaluators(
            payload.evaluators, ctx.org_uuid, "response"
        )

    traces = traces_store.select_traces(
        ctx.org_uuid,
        trace_ids=payload.trace_ids,
        select_all=payload.select_all,
        q=payload.q,
        conversation_id=payload.conversation_id,
        limit=MAX_CONVERT_BATCH + 1 if payload.select_all else None,
    )
    if not traces:
        raise HTTPException(status_code=404, detail="No matching traces found")
    if len(traces) > MAX_CONVERT_BATCH:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Too many traces selected (limit {MAX_CONVERT_BATCH}). "
                "Narrow the search or conversation filter and try again."
            ),
        )

    # tool_call tests assert the recorded calls, so every trace must have some.
    if payload.type == "tool_call":
        missing = [
            t["message_id"]
            for t in traces
            if not (t.get("output") or {}).get("tool_calls")
        ]
        if missing:
            raise HTTPException(
                status_code=400,
                detail={
                    "error": "Some traces have no tool calls to convert to a tool_call test",
                    "message_ids": missing,
                },
            )

    existing_names = {t["name"] for t in get_all_tests_summary(org_uuid=ctx.org_uuid)}
    names = _dedupe_test_names([t["message_id"] for t in traces], existing_names)

    db_tests: List[Dict[str, Any]] = []
    for trace, name in zip(traces, names):
        evaluation: Dict[str, Any] = {"type": payload.type}
        if payload.type == "tool_call":
            evaluation["tool_calls"] = [
                {
                    "tool": tc["tool"],
                    "arguments": tc.get("arguments"),
                    "accept_any_arguments": payload.accept_any_arguments,
                }
                for tc in (trace["output"].get("tool_calls") or [])
            ]
        db_tests.append(
            {
                "name": name,
                "type": payload.type,
                # `input` is already OpenAI history; `output` is discarded for
                # response tests (the agent is re-run) and captured as the
                # expected assertion for tool_call tests above.
                "config": {"history": trace["input"], "evaluation": evaluation},
            }
        )

    try:
        uuids = bulk_create_tests(
            tests=db_tests, org_uuid=ctx.org_uuid, user_id=ctx.user_id
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    if resolved_refs:
        for test_uuid in uuids:
            set_test_evaluators(test_uuid, resolved_refs)

    if payload.agent_uuids:
        for agent_uuid in payload.agent_uuids:
            for test_uuid in uuids:
                try:
                    add_test_to_agent(agent_uuid, test_uuid)
                except Exception as e:
                    logger.warning(
                        f"Failed to link test {test_uuid} to agent {agent_uuid}: {e}"
                    )

    return {"created": len(uuids), "test_uuids": uuids}


@router.get("/{trace_uuid}", response_model=TraceResponse, summary="Get trace")
async def get_trace_endpoint(
    trace_uuid: str = Path(
        description="The trace to retrieve",
        examples=[_EXAMPLE_TRACE_UUID],
    ),
    ctx: OrgContext = Depends(get_current_org),
):
    """Get one trace by its ID"""
    row = traces_store.get_trace(ctx.org_uuid, trace_uuid)
    if not row:
        raise HTTPException(status_code=404, detail="Trace not found")
    return row
