"""Production trace ingestion and curation into tests.

Customer backends POST one trace per agent turn: the conversation history as
`input` plus the produced `output`. Rows persist as a normal `traces` table in
pense.db.

The stored shape deliberately mirrors test creation. `input` is
`tests.config.history` verbatim for a conversational agent and
`tests.config.input` verbatim for a `general` one, and `output.tool_calls`
matches the expected-tool-call shape, so `POST /traces/convert-to-tests` needs
no transformation.

New contract needs go into `metadata` keys, not new top-level fields.
Customers integrate against this shape, and every field deepens the eventual
OTel-gateway migration.
"""

import logging
import os
from typing import Annotated, Any, ClassVar, Dict, List, Literal, Optional, Union

from fastapi import APIRouter, Depends, HTTPException, Path, Query
from pydantic import BaseModel, ConfigDict, Field, StringConstraints, model_validator

from auth_utils import OrgContext, get_current_org, get_org_jwt_or_api_key
from db import (
    add_test_to_agent,
    bulk_create_tests,
    count_live_traces,
    create_trace_with_eval_run,
    get_agent,
    get_all_tests_summary,
    get_evaluator_versions_by_uuids,
    get_evaluators_by_uuids,
    get_trace,
    get_traces_by_uuids,
    list_traces,
    set_test_evaluators,
    soft_delete_traces,
    soft_delete_traces_matching,
)
from org_scope import ensure_owned_agent
from pagination import PaginatedResponse, PaginationParams, page_envelope

# Reuse the tests router's validation so a converted test accepts exactly what
# POST /tests does (evaluator visible to the workspace, evaluator_type matches).
from routers.tests import (
    DEFAULT_AGENT_INTERACTION_TYPE,
    EvaluatorRef,
    _validate_evaluators,
    required_agent_interaction_type,
)
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


# A trace's input takes the shape its agent is called with: one standalone
# prompt for a `general` agent, a conversation for the rest. Separate aliases
# rather than one Field so each side keeps its own bound (turn count vs
# characters) instead of sharing whichever one is declared.
TraceInputText = Annotated[
    str, StringConstraints(min_length=1, max_length=MAX_TURN_CONTENT_CHARS)
]
TraceInputHistory = Annotated[
    List[TraceTurn], Field(min_length=1, max_length=MAX_INPUT_TURNS)
]
TraceInput = Union[TraceInputText, TraceInputHistory]

_TRACE_INPUT_DESCRIPTION = (
    "What the agent was given for this turn. For a `general` agent, the "
    "standalone prompt as a string. For a `conversation` agent, the history up "
    "to the reported output, oldest turn first, in OpenAI chat format"
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
    # Stored and displayed only. Conversion asserts the call, not its result,
    # so `POST /traces/convert-to-tests` never reads this.
    output: Any = Field(
        None,
        description="What the tool returned for this call. Any JSON value. Omit when you do not record it",
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
    input: TraceInput = Field(description=_TRACE_INPUT_DESCRIPTION)
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
        None,
        description="The standalone prompt or the last user message, truncated for display",
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
        description="Number of turns in the stored input. A standalone prompt counts as 1"
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
    input: TraceInput = Field(description=_TRACE_INPUT_DESCRIPTION)
    output: TraceOutput = Field(description="What the agent produced for this turn")
    metadata: Optional[List[TraceMetadataEntry]] = Field(
        None, description="Key-value pairs stored with the trace"
    )
    created_at: str = Field(description="When the trace was created (ISO 8601 UTC)")
    updated_at: str = Field(
        description="When the trace was last updated (ISO 8601 UTC)"
    )


_SELECT_ALL_DESCRIPTION = (
    "Act on every trace matching the filters below instead of a list of IDs. "
    "`trace_ids` is ignored when this is on"
)
_FILTER_AGENT_DESCRIPTION = (
    "With `select_all` on, act only on traces from this agent"
)
_FILTER_Q_DESCRIPTION = (
    "With `select_all` on, act only on traces containing this text in their "
    "message ID, conversation ID, conversation history, output, or metadata"
)
_FILTER_OUTPUT_TYPE_DESCRIPTION = (
    "With `select_all` on, act only on traces whose output is of this kind. "
    "`response` covers every trace carrying a reply, including one that also "
    "issued tool calls. `tool_call` covers traces that only issued tool calls"
)


class _TraceSelection(BaseModel):
    """Either a list of IDs or `select_all` plus the same filters `GET /traces`
    takes, so a caller acting on a whole filtered set never has to name its rows."""

    # Unknown keys must not be silently dropped: a misspelled field would
    # otherwise look like it filtered something.
    model_config = ConfigDict(extra="forbid")

    # The cap is enforced in the validator, not as `max_length`, so a caller
    # that posts its whole selection alongside `select_all` is not rejected on
    # a list the handler never reads.
    max_trace_ids: ClassVar[int] = 0

    trace_ids: List[TraceUuid] = Field(
        default_factory=list, description="IDs of the traces to act on"
    )
    select_all: bool = Field(False, description=_SELECT_ALL_DESCRIPTION)
    agent_id: Optional[str] = Field(None, description=_FILTER_AGENT_DESCRIPTION)
    q: Optional[str] = Field(None, description=_FILTER_Q_DESCRIPTION)
    output_type: Optional[Literal["response", "tool_call"]] = Field(
        None, description=_FILTER_OUTPUT_TYPE_DESCRIPTION
    )

    @model_validator(mode="after")
    def _require_a_selection(self):
        if self.select_all:
            return self
        if not self.trace_ids:
            raise ValueError("provide trace_ids, or select_all=true")
        if len(self.trace_ids) > self.max_trace_ids:
            raise ValueError(
                f"trace_ids accepts at most {self.max_trace_ids} IDs"
            )
        return self


class BulkDeleteTracesRequest(_TraceSelection):
    max_trace_ids: ClassVar[int] = MAX_DELETE_IDS

    trace_ids: List[TraceUuid] = Field(
        default_factory=list,
        description="IDs of the traces to delete. Omit when `select_all` is on",
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


def _last_user_content(stored_input: Any) -> Optional[str]:
    if isinstance(stored_input, str):
        return stored_input
    for turn in reversed(stored_input or []):
        if turn.get("role") == "user" and isinstance(turn.get("content"), str):
            return turn["content"]
    return None


def _turn_count(stored_input: Any) -> int:
    """A standalone prompt is the single turn it stands for."""
    return 1 if isinstance(stored_input, str) else len(stored_input or [])


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
        "input_preview": _preview(_last_user_content(row.get("input"))),
        "response_preview": _preview(output.get("response")),
        "tool_names": [call["tool"] for call in calls],
        "tool_calls": [
            {
                "tool": call["tool"],
                "arguments": call.get("arguments"),
                "output": call.get("output"),
            }
            for call in calls
        ],
        "turn_count": _turn_count(row.get("input")),
        "tool_call_count": len(calls),
        "metadata_count": len(row.get("metadata") or []),
        "created_at": row["created_at"],
    }


def _ensure_input_matches_agent(stored_input: Any, agent: Dict[str, Any]) -> None:
    """Reject an input whose shape the trace's agent is never called with.

    Tying the two together at ingest is what lets `convert-to-tests` build each
    test from the trace alone: the stored shape already matches what the agent
    can be given, so a converted test can only ever fit it.
    """
    interaction_type = (
        agent.get("interaction_type") or DEFAULT_AGENT_INTERACTION_TYPE
    )
    is_text = isinstance(stored_input, str)
    if is_text and interaction_type != "general":
        raise HTTPException(
            status_code=400,
            detail=(
                f"Agent has interaction_type='{interaction_type}' and is called "
                "with a conversation, so input must be a list of turns, not a string."
            ),
        )
    if not is_text and interaction_type == "general":
        raise HTTPException(
            status_code=400,
            detail=(
                "Agent has interaction_type='general' and is called with a "
                "standalone prompt, so input must be a string, not a list of turns."
            ),
        )
    if is_text and not stored_input.strip():
        raise HTTPException(status_code=400, detail="input must not be blank")


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
    _ensure_input_matches_agent(payload.input, agent)

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

    row = create_trace_with_eval_run(
        org_uuid=ctx.org_uuid,
        agent=agent,
        message_id=payload.message_id,
        conversation_id=payload.conversation_id,
        input=(
            payload.input
            if isinstance(payload.input, str)
            else [turn.model_dump(exclude_none=True) for turn in payload.input]
        ),
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
    output_type: Optional[Literal["response", "tool_call"]] = Query(
        None,
        description="Return only traces whose output is of this kind. `response` covers every trace carrying a reply, including one that also issued tool calls. `tool_call` covers traces that only issued tool calls",
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
        output_type=output_type,
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
    if payload.select_all:
        deleted = soft_delete_traces_matching(
            ctx.org_uuid,
            agent_id=payload.agent_id,
            q=payload.q,
            output_type=payload.output_type,
        )
    else:
        deleted = soft_delete_traces(ctx.org_uuid, trace_ids=payload.trace_ids)
    return {"deleted": deleted}


_CONVERT_TYPE_DESCRIPTION = (
    "What the created tests judge:\n\n"
    "- `response`: re-run the agent on the trace's conversation and judge the fresh reply against the linked evaluators\n"
    "- `general`: re-run the agent on the trace's standalone prompt and judge the fresh reply against the linked evaluators\n"
    "- `tool_call`: re-run the agent and diff its tool calls against the ones the trace recorded\n\n"
    "`response` needs traces carrying a conversation and `general` needs traces "
    "carrying a standalone prompt, so each one fits the agent it came from. "
    "`tool_call` takes either and keeps the shape the trace has"
)


class ConvertTracesToTestsRequest(_TraceSelection):
    max_trace_ids: ClassVar[int] = MAX_CONVERT_TRACES

    trace_ids: List[TraceUuid] = Field(
        default_factory=list,
        description="IDs of the traces to convert, one test per trace. Omit when `select_all` is on",
    )
    type: Literal["response", "general", "tool_call"] = Field(
        description=_CONVERT_TYPE_DESCRIPTION
    )
    evaluators: Optional[List[EvaluatorUuid]] = Field(
        None,
        description="IDs of the evaluators to link to every created test. **Required for `response` and `general`**, rejected for `tool_call`, which compares the recorded calls instead of judging. Each evaluator must match the created tests: `llm` for `response`, `llm-general` for `general`. Each must also judge on its prompt alone, with no `{{placeholder}}` variables to fill in",
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
    evaluator_uuids: List[str], org_uuid: str, test_type: str
) -> List[Dict[str, Any]]:
    """Validate as POST /tests does, then reject any evaluator whose prompt needs
    variables filled in. A conversion has no per-test place to supply them, and a
    half-rendered prompt would reach the judge with `{{placeholders}}` intact."""
    refs = _validate_evaluators(
        [EvaluatorRef(evaluator_uuid=u) for u in evaluator_uuids], org_uuid, test_type
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


# How many offending IDs a conflict lists before it just reports the count. A
# `select_all` conversion can hit hundreds, and a wall of UUIDs buries the hint
# that actually resolves it.
_MAX_REPORTED_CONFLICTS = 20


def _shape_conflict_detail(
    error: str,
    offending: List[str],
    considered: int,
    *,
    select_all: bool,
    hint: str,
) -> Dict[str, Any]:
    """Build the 400 body for traces the requested test type cannot take."""
    detail: Dict[str, Any] = {
        "error": error,
        "trace_ids": offending[:_MAX_REPORTED_CONFLICTS],
    }
    if select_all:
        detail["error"] = (
            f"{error}. {len(offending)} of the {considered} matching traces "
            "cannot be converted, so nothing was created"
        )
        detail["hint"] = hint
        if len(offending) > _MAX_REPORTED_CONFLICTS:
            detail["trace_ids_truncated"] = True
    return detail


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
    # A converted response or general test re-runs the agent and judges the fresh
    # reply, so it has no fallback judge. A tool_call run only diffs the recorded
    # calls (see the row-type skip in agent_tests._build_calibrate_config), so an
    # evaluator attached here would never judge anything: refuse it rather than
    # store a judge the user never sees run.
    if payload.type in ("response", "general") and not payload.evaluators:
        raise HTTPException(
            status_code=400,
            detail=f"{payload.type} tests require at least one evaluator",
        )
    if payload.type == "tool_call" and payload.evaluators:
        raise HTTPException(
            status_code=400,
            detail="tool_call tests compare the recorded tool calls and cannot take evaluators",
        )
    resolved_refs: List[Dict[str, Any]] = []
    if payload.evaluators:
        resolved_refs = _resolve_evaluators(
            list(dict.fromkeys(payload.evaluators)), ctx.org_uuid, payload.type
        )

    if payload.select_all:
        traces, total = list_traces(
            ctx.org_uuid,
            limit=MAX_CONVERT_TRACES,
            offset=0,
            agent_id=payload.agent_id,
            q=payload.q,
            output_type=payload.output_type,
        )
        if total > MAX_CONVERT_TRACES:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"{total} traces match, more than the {MAX_CONVERT_TRACES} "
                    "one conversion accepts. Narrow the filters and convert in batches."
                ),
            )
        if not traces:
            raise HTTPException(
                status_code=400, detail="No traces matched the filters"
            )
    else:
        requested = list(dict.fromkeys(payload.trace_ids))
        traces = get_traces_by_uuids(ctx.org_uuid, requested)
        found = {trace["uuid"] for trace in traces}
        missing = [trace_uuid for trace_uuid in requested if trace_uuid not in found]
        if missing:
            raise HTTPException(
                status_code=404,
                detail={"error": "Some traces were not found", "trace_ids": missing},
            )

    # `general` tests carry a standalone `input` and `response` tests a
    # `history`; calibrate refuses a row holding the wrong one. Since ingest ties
    # each trace's shape to its agent, this also keeps every created test
    # linkable to the agent it came from.
    if payload.type in ("response", "general"):
        wants_text = payload.type == "general"
        wrong_shape = [
            trace["uuid"]
            for trace in traces
            if isinstance(trace["input"], str) is not wants_text
        ]
        if wrong_shape:
            raise HTTPException(
                status_code=400,
                detail=_shape_conflict_detail(
                    (
                        "general tests take a standalone prompt, but these traces "
                        "carry a conversation"
                        if wants_text
                        else "response tests take a conversation, but these traces "
                        "carry a standalone prompt"
                    ),
                    wrong_shape,
                    len(traces),
                    select_all=payload.select_all,
                    hint=(
                        "Pass agent_id to convert one agent's traces at a time. "
                        "A trace's shape follows the agent that produced it, so a "
                        f"single {'general' if wants_text else 'conversation'} "
                        "agent's traces are all the right shape."
                    ),
                ),
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
                detail=_shape_conflict_detail(
                    "Some traces recorded no tool calls to assert",
                    no_calls,
                    len(traces),
                    select_all=payload.select_all,
                    hint=(
                        "Pass output_type=tool_call to convert only the traces "
                        "that issued tool calls."
                    ),
                ),
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
        # A trace's `input` is already the test config field it maps to: a
        # standalone prompt becomes `input`, a conversation becomes `history`.
        # The recorded reply is dropped for a response or general test (the agent
        # is re-run) and captured as the assertion for a tool_call test above.
        stored_input = trace["input"]
        given = (
            {"input": stored_input}
            if isinstance(stored_input, str)
            else {"history": stored_input}
        )
        db_tests.append(
            {
                "name": name,
                "type": payload.type,
                "config": {**given, "evaluation": evaluation},
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
    # since ingest, or switched to an interaction_type the created test no longer
    # fits, must not fail a batch the user cannot retry.
    agents_seen: Dict[str, Optional[Dict[str, Any]]] = {}
    unlinked = 0
    for trace, test_uuid, db_test in zip(traces, test_uuids, db_tests):
        agent_id = trace["agent_id"]
        if agent_id not in agents_seen:
            found = get_agent(agent_id)
            agents_seen[agent_id] = (
                found if found and found["org_uuid"] == ctx.org_uuid else None
            )
        agent = agents_seen[agent_id]
        if not agent or required_agent_interaction_type(
            payload.type, db_test["config"]
        ) != (agent.get("interaction_type") or DEFAULT_AGENT_INTERACTION_TYPE):
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
