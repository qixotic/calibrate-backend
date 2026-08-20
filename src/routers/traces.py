"""Production trace ingestion and curation into tests.

Customer backends POST one trace per agent turn: the conversation history as
`input` plus the produced `output`. Rows persist as a normal `traces` table in
pense.db.

The stored shape deliberately mirrors test creation. `input` is
`tests.config.history` verbatim, and `output.tool_calls` matches the
expected-tool-call shape, so `POST /traces/convert-to-tests` needs no
transformation.

New contract needs go into `metadata` keys, not new top-level fields.
Customers integrate against this shape, and every field deepens the eventual
OTel-gateway migration.
"""

import logging
import os
from typing import Annotated, Any, Dict, List, Literal, Optional

from fastapi import APIRouter, Depends, HTTPException, Path, Query
from pydantic import BaseModel, ConfigDict, Field, StringConstraints, model_validator

from auth_utils import OrgContext, get_current_org, get_org_jwt_or_api_key
from db import (
    add_test_to_agent,
    bulk_create_tests,
    count_live_traces,
    create_trace_with_eval_queue,
    get_agent,
    get_all_tests_summary,
    get_evaluator_versions_by_uuids,
    get_evaluators_by_uuids,
    get_trace,
    get_traces_by_uuids,
    list_traces,
    set_test_evaluators,
    soft_delete_traces,
)
from org_scope import ensure_owned_agent
from pagination import PaginatedResponse, PaginationParams, page_envelope

# Reuse the tests router's validation so a converted test accepts exactly what
# POST /tests does (evaluator visible to the workspace, evaluator_type matches).
from routers.tests import EvaluatorRef, _validate_evaluators
from utils import EXAMPLE_TEST_UUID, EvaluatorUuid

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/traces", tags=["traces"])

# A fixed ceiling, not a per-workspace setting: one number is enough until a
# customer actually needs a different one.
MAX_TRACES_PER_WORKSPACE = int(os.getenv("DEFAULT_MAX_TRACES", "50000"))

# How many traces one delete call accepts. Independent of the storage cap:
# lowering that must not shrink a user's ability to delete their way back
# under it.
MAX_DELETE_IDS = 50_000

# A list row parses each trace's whole stored conversation to build its
# previews, so pages stay small. The shared PaginationParams allows a million,
# which would load every trace in a full workspace into memory at once.
MAX_LIST_LIMIT = 200

# How many traces one conversion accepts. Each becomes a test row plus its
# evaluator and agent links, all in one request.
MAX_CONVERT_TRACES = 500

MAX_INPUT_TURNS = 500
MAX_TURN_CONTENT_CHARS = 50_000
MAX_TOOL_CALLS = 50
MAX_METADATA_ENTRIES = 100
_EXAMPLE_TRACE_UUID = "f47ac10b-58cc-4372-a567-0e02b2c3d479"

_TRACE_UUID_DESCRIPTION = "Unique ID for the trace"

_AGENT_ID_DESCRIPTION = "ID of the agent that produced the turn"

# Bounds each entry so a malformed list is rejected before it reaches the
# database rather than being bound into a query.
TraceUuid = Annotated[str, StringConstraints(min_length=36, max_length=36)]


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

    agent_id: str = Field(
        min_length=1,
        max_length=36,
        description=_AGENT_ID_DESCRIPTION + ". Must be an agent in your workspace",
    )
    message_id: Optional[str] = Field(
        None,
        min_length=1,
        max_length=255,
        description="Your own ID for the last user message in `input`, stored for reference only. Omit if you have none",
    )
    conversation_id: Optional[str] = Field(
        None,
        min_length=1,
        max_length=255,
        description="Your own ID for the conversation this turn belongs to, stored for reference only. Omit if you have none",
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
    message_id: Optional[str] = Field(
        None, description="The message ID you sent, if any"
    )
    conversation_id: Optional[str] = Field(
        None, description="The conversation ID you sent, if any"
    )
    created_at: str = Field(description="When the trace was created (ISO 8601 UTC)")


class TraceSummary(BaseModel):
    uuid: str = Field(
        min_length=36,
        max_length=36,
        description=_TRACE_UUID_DESCRIPTION,
        examples=[_EXAMPLE_TRACE_UUID],
    )
    agent_id: str = Field(description=_AGENT_ID_DESCRIPTION)
    message_id: Optional[str] = Field(
        None, description="The message ID you sent, if any"
    )
    conversation_id: Optional[str] = Field(
        None, description="The conversation ID you sent, if any"
    )
    input_preview: Optional[str] = Field(
        None, description="The last user message, truncated for display"
    )
    response_preview: Optional[str] = Field(
        None, description="The agent reply, truncated for display"
    )
    tool_names: List[str] = Field(
        description="Names of the tools the agent issued on this turn, in order"
    )
    tool_calls: List[TraceToolCall] = Field(
        description="Tools the agent issued on this turn, with the arguments it passed"
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
    agent_id: str = Field(description=_AGENT_ID_DESCRIPTION)
    message_id: Optional[str] = Field(
        None, description="The message ID you sent, if any"
    )
    conversation_id: Optional[str] = Field(
        None, description="The conversation ID you sent, if any"
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
    # Unknown keys must not be silently dropped: a misspelled field would
    # otherwise look like it filtered something.
    model_config = ConfigDict(extra="forbid")

    trace_ids: List[TraceUuid] = Field(
        min_length=1,
        max_length=MAX_DELETE_IDS,
        description="IDs of the traces to delete",
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
    calls = [
        call
        for call in (output.get("tool_calls") or [])
        if isinstance(call, dict) and call.get("tool")
    ]
    return {
        "uuid": row["uuid"],
        "agent_id": row["agent_id"],
        "message_id": row["message_id"],
        "conversation_id": row["conversation_id"],
        "input_preview": _preview(_last_user_content(row.get("input") or [])),
        "response_preview": _preview(output.get("response")),
        "tool_names": [call["tool"] for call in calls],
        "tool_calls": [
            {"tool": call["tool"], "arguments": call.get("arguments")}
            for call in calls
        ],
        "turn_count": len(row.get("input") or []),
        "tool_call_count": len(calls),
        "metadata_count": len(row.get("metadata") or []),
        "created_at": row["created_at"],
    }


@router.post(
    "",
    response_model=TraceIngestResponse,
    summary="Create trace",
    tags=["Public API"],
)
async def ingest_trace(
    payload: TraceIngest, ctx: OrgContext = Depends(get_org_jwt_or_api_key)
):
    """Store a production agent turn and its conversation history for later review"""
    agent = ensure_owned_agent(payload.agent_id, ctx.org_uuid)

    cap = MAX_TRACES_PER_WORKSPACE
    current = count_live_traces(ctx.org_uuid)
    if current >= cap:
        raise HTTPException(
            status_code=429,
            detail={
                "error": "Trace limit reached for this workspace",
                "current": current,
                "max_traces": cap,
                "hint": "Delete traces to free capacity",
            },
        )

    row = create_trace_with_eval_queue(
        org_uuid=ctx.org_uuid,
        agent=agent,
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
    return {
        "uuid": row["uuid"],
        "message_id": row["message_id"],
        "conversation_id": row["conversation_id"],
        "created_at": row["created_at"],
    }


@router.get("", response_model=PaginatedResponse[TraceSummary], summary="List traces")
async def list_traces_endpoint(
    ctx: OrgContext = Depends(get_current_org),
    pagination: PaginationParams = Depends(),
    agent_id: Optional[str] = Query(
        None, description="Return only traces from this agent"
    ),
    q: Optional[str] = Query(
        None,
        description="Return only traces containing this text in their message ID, conversation ID, conversation history, output, or metadata",
    ),
):
    """List ingested traces, newest first"""
    if pagination.limit > MAX_LIST_LIMIT:
        raise HTTPException(
            status_code=422, detail=f"limit must be {MAX_LIST_LIMIT} or less"
        )
    # Search/filter/count run in SQL (db.list_traces), not the post-fetch
    # pagination helpers, and paging uses the bounded PaginationParams rather
    # than the unbounded OptionalPaginationParams: traces are machine-written
    # and outgrow in-memory filtering fast.
    rows, total = list_traces(
        ctx.org_uuid,
        limit=pagination.limit,
        offset=pagination.offset,
        agent_id=agent_id,
        q=q,
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
    """Soft-delete traces, freeing their capacity"""
    deleted = soft_delete_traces(ctx.org_uuid, trace_ids=payload.trace_ids)
    return {"deleted": deleted}


_CONVERT_TYPE_DESCRIPTION = (
    "What the created tests judge:\n\n"
    "- `response`: re-run the agent on the trace's history and judge the fresh reply against the linked evaluators\n"
    "- `tool_call`: re-run the agent and diff its tool calls against the ones the trace recorded"
)


class ConvertTracesToTestsRequest(BaseModel):
    # Unknown keys must not be silently dropped: a misspelled `evaluators` would
    # look like it linked judges when it linked none.
    model_config = ConfigDict(extra="forbid")

    trace_ids: List[TraceUuid] = Field(
        min_length=1,
        max_length=MAX_CONVERT_TRACES,
        description="IDs of the traces to convert, one test per trace",
    )
    type: Literal["response", "tool_call"] = Field(
        description=_CONVERT_TYPE_DESCRIPTION
    )
    evaluators: Optional[List[EvaluatorUuid]] = Field(
        None,
        description="IDs of the evaluators to link to every created test. **Required for `response`**, rejected for `tool_call`, which compares the recorded calls instead of judging. Each evaluator must judge on its prompt alone, with no `{{placeholder}}` variables to fill in",
    )
    accept_any_arguments: bool = Field(
        False,
        description="For `tool_call`, match only the tool name and ignore the arguments the trace recorded",
    )


class ConvertTracesToTestsResponse(BaseModel):
    created: int = Field(description="Number of tests created")
    test_uuids: List[str] = Field(
        description="IDs of the created tests, in the order their traces were given",
        examples=[[EXAMPLE_TEST_UUID]],
    )
    warnings: Optional[List[str]] = Field(
        None,
        description="What was created but not fully wired up, such as a test whose agent no longer exists",
    )


def _resolve_evaluators(
    evaluator_uuids: List[str], org_uuid: str
) -> List[Dict[str, Any]]:
    """Validate as POST /tests does, then reject any evaluator whose prompt needs
    variables filled in. A conversion has no per-test place to supply them, and a
    half-rendered prompt would reach the judge with `{{placeholders}}` intact."""
    refs = _validate_evaluators(
        [EvaluatorRef(evaluator_uuid=u) for u in evaluator_uuids], org_uuid, "response"
    )
    evaluators = get_evaluators_by_uuids(evaluator_uuids)
    live_ids = {
        u: (evaluators.get(u) or {}).get("live_version_id") for u in evaluator_uuids
    }
    versions = get_evaluator_versions_by_uuids([v for v in live_ids.values() if v])

    problems: List[str] = []
    for evaluator_uuid in evaluator_uuids:
        # Deleted between the two reads: report it rather than raising KeyError.
        name = (evaluators.get(evaluator_uuid) or {}).get("name", evaluator_uuid)
        version = versions.get(live_ids[evaluator_uuid] or "")
        if not version:
            problems.append(f'Evaluator "{name}" has no live version to run.')
            continue
        variables = version.get("variables") or []
        if variables:
            declared = ", ".join(v["name"] for v in variables)
            problems.append(
                f'Evaluator "{name}" defines variables ({declared}). Converting '
                "traces cannot fill them in. Use an evaluator with no variables, "
                "or add a version with the criteria written into the prompt."
            )
    if problems:
        raise HTTPException(
            status_code=400,
            detail={
                "error": "Some evaluators cannot be used for converted tests",
                "evaluators": problems,
            },
        )
    return refs


def _dedupe_test_names(candidates: List[str], taken: set) -> List[str]:
    """Make each candidate unique against `taken` (the workspace's existing test
    names, mutated as names are claimed) by appending ` (2)`, ` (3)`, … so
    converting the same traces twice creates new tests instead of failing."""
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
def convert_traces_to_tests(
    payload: ConvertTracesToTestsRequest, ctx: OrgContext = Depends(get_current_org)
):
    """Turn production traces into regression tests you can run and benchmark"""
    # A converted response test re-runs the agent and judges the fresh reply, so
    # it has no fallback judge. A tool_call run only diffs the recorded calls
    # (see the row-type skip in agent_tests._build_calibrate_config), so an
    # evaluator attached here would never judge anything: refuse it rather than
    # store a judge the user never sees run.
    if payload.type == "response" and not payload.evaluators:
        raise HTTPException(
            status_code=400,
            detail="response tests require at least one evaluator",
        )
    if payload.type == "tool_call" and payload.evaluators:
        raise HTTPException(
            status_code=400,
            detail="tool_call tests compare the recorded tool calls and cannot take evaluators",
        )
    resolved_refs: List[Dict[str, Any]] = []
    if payload.evaluators:
        resolved_refs = _resolve_evaluators(
            list(dict.fromkeys(payload.evaluators)), ctx.org_uuid
        )

    requested = list(dict.fromkeys(payload.trace_ids))
    traces = get_traces_by_uuids(ctx.org_uuid, requested)
    found = {trace["uuid"] for trace in traces}
    missing = [trace_uuid for trace_uuid in requested if trace_uuid not in found]
    if missing:
        raise HTTPException(
            status_code=404,
            detail={"error": "Some traces were not found", "trace_ids": missing},
        )

    if payload.type == "tool_call":
        no_calls = [
            trace["uuid"]
            for trace in traces
            if not (trace["output"] or {}).get("tool_calls")
        ]
        if no_calls:
            raise HTTPException(
                status_code=400,
                detail={
                    "error": "Some traces recorded no tool calls to assert",
                    "trace_ids": no_calls,
                },
            )

    existing_names = {t["name"] for t in get_all_tests_summary(org_uuid=ctx.org_uuid)}
    names = _dedupe_test_names(
        [trace["message_id"] or trace["uuid"] for trace in traces], existing_names
    )

    db_tests: List[Dict[str, Any]] = []
    for trace, name in zip(traces, names):
        evaluation: Dict[str, Any] = {"type": payload.type}
        if payload.type == "tool_call":
            evaluation["tool_calls"] = [
                {
                    "tool": call["tool"],
                    "arguments": call.get("arguments"),
                    "accept_any_arguments": payload.accept_any_arguments,
                }
                for call in trace["output"]["tool_calls"]
            ]
        # `input` is already OpenAI history. The recorded reply is dropped for a
        # response test (the agent is re-run) and captured as the assertion for
        # a tool_call test above.
        db_tests.append(
            {
                "name": name,
                "type": payload.type,
                "config": {"history": trace["input"], "evaluation": evaluation},
            }
        )

    try:
        test_uuids = bulk_create_tests(
            tests=db_tests, org_uuid=ctx.org_uuid, user_id=ctx.user_id
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    # The tests are already committed, so a failure here must not 500 and lose
    # them: report what did not get wired up and let the user fix it in place.
    warnings: List[str] = []
    unjudged = 0
    if resolved_refs:
        for test_uuid in test_uuids:
            try:
                set_test_evaluators(test_uuid, resolved_refs)
            except Exception as e:
                unjudged += 1
                logger.warning(f"Failed to link evaluators to test {test_uuid}: {e}")
    if unjudged:
        warnings.append(
            f"{unjudged} of {len(test_uuids)} tests were created without evaluators "
            "and will not run until you attach one"
        )

    # Each test links to the agent that produced its trace. One agent deleted
    # since ingest must not fail a batch the user cannot retry.
    linkable: Dict[str, bool] = {}
    unlinked = 0
    for trace, test_uuid in zip(traces, test_uuids):
        agent_id = trace["agent_id"]
        if agent_id not in linkable:
            agent = get_agent(agent_id)
            linkable[agent_id] = bool(agent and agent["org_uuid"] == ctx.org_uuid)
        if not linkable[agent_id]:
            unlinked += 1
            continue
        try:
            add_test_to_agent(agent_id, test_uuid)
        except Exception as e:
            unlinked += 1
            logger.warning(f"Failed to link test {test_uuid} to agent {agent_id}: {e}")
    if unlinked:
        warnings.append(
            f"{unlinked} of {len(test_uuids)} tests were not linked to an agent, "
            "so they will not appear on any agent's test list"
        )

    return {
        "created": len(test_uuids),
        "test_uuids": test_uuids,
        "warnings": warnings or None,
    }


@router.get("/{trace_uuid}", response_model=TraceResponse, summary="Get trace")
async def get_trace_endpoint(
    trace_uuid: str = Path(
        description="The trace to retrieve",
        examples=[_EXAMPLE_TRACE_UUID],
    ),
    ctx: OrgContext = Depends(get_current_org),
):
    """Get one trace by its ID"""
    row = get_trace(ctx.org_uuid, trace_uuid)
    if not row:
        raise HTTPException(status_code=404, detail="Trace not found")
    return row
