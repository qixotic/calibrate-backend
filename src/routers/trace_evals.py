"""Judging ingested traces with the evaluators linked to their agent.

A trace records what an agent actually did, so it carries no expected answer:
every run judges the stored output as-is and nothing re-runs the agent. The
selected traces are split by inferred type and each type becomes its own run,
judged by the evaluators of the matching type, so one request can produce
several runs.

Automatic judging is off until it is turned on for the agent. Ingestion is
machine paced, and every verdict spends judge tokens, so arriving traces must
never start that spend on their own.
"""

from typing import Any, Dict, List, Literal, Optional

from fastapi import APIRouter, Depends, HTTPException, Path
from pydantic import BaseModel, ConfigDict, Field

from auth_utils import OrgContext, get_current_org
from db import get_evaluators_for_agent, update_agent
from org_scope import ensure_owned_agent
from pagination import PaginatedResponse, PaginationParams, page_envelope
from traces import eval_store, inference
from traces import store as traces_store
from traces.eval_runner import launch_trace_eval
from utils import TaskStatus

router = APIRouter(prefix="/trace-evals", tags=["trace-evals"])

# Bound on a single `select_all` request. The backlog for a busy agent is
# unbounded, and one request must not turn into an unbounded judge spend;
# callers drain the rest by calling again.
_SELECT_ALL_TRACE_LIMIT = 500

_EXAMPLE_AGENT_UUID = "86186be6-d898-404a-b79c-4f6ff5336afb"

_EXAMPLE_RUN_UUID = "9f2a7c14-3b6e-4a5d-9c81-2d4e6f8a0b13"

_EXAMPLE_TRACE_UUID = "f47ac10b-58cc-4372-a567-0e02b2c3d479"

TraceInferredType = Literal["conversation", "response", "tool_call"]

TraceEvalTrigger = Literal["auto", "manual"]

_TASK_ID_DESCRIPTION = "Unique ID for the trace evaluation run"

_AGENT_ID_DESCRIPTION = "ID of the agent whose traces the run judges"

_INFERRED_TYPE_DESCRIPTION = (
    "How the traces in this run are judged:\n\n"
    "- `conversation`: judges the conversation history together with the reply\n"
    "- `response`: judges the reply on its own\n"
    "- `tool_call`: judges the tool calls the turn produced\n"
)

_TRACE_COUNT_DESCRIPTION = "Number of traces the run judges"

_SKIPPED_COUNT_DESCRIPTION = (
    "Number of selected traces left unjudged because the agent has no evaluator "
    "that fits them"
)

_STATUS_DESCRIPTION = "Current status of the trace evaluation run"

_ERROR_DESCRIPTION = "What went wrong, for a run that failed"

_AUTO_EVAL_DESCRIPTION = (
    "Whether traces for this agent are judged automatically as they arrive"
)


class TraceEvalRunRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    trace_ids: Optional[List[str]] = Field(
        None,
        description="IDs of the traces to judge, each belonging to this agent. **Required when `select_all` is false.** Ignored otherwise",
        examples=[[_EXAMPLE_TRACE_UUID]],
    )
    select_all: bool = Field(
        False,
        description="Judge every trace of this agent that has no verdict yet, up to 500 in one call, instead of an explicit ID list",
    )


class LaunchedTraceEvalRun(BaseModel):
    task_id: str = Field(
        min_length=36,
        max_length=36,
        description=_TASK_ID_DESCRIPTION,
        examples=[_EXAMPLE_RUN_UUID],
    )
    inferred_type: TraceInferredType = Field(description=_INFERRED_TYPE_DESCRIPTION)
    trace_count: int = Field(description=_TRACE_COUNT_DESCRIPTION)
    status: TaskStatus = Field(description=_STATUS_DESCRIPTION)


class TraceEvalRunLaunchResponse(BaseModel):
    runs: List[LaunchedTraceEvalRun] = Field(
        description="The runs that started, one for each inferred type among the selected traces"
    )
    skipped_count: int = Field(description=_SKIPPED_COUNT_DESCRIPTION)


class TraceEvalRunStatusResponse(BaseModel):
    task_id: str = Field(
        min_length=36,
        max_length=36,
        description=_TASK_ID_DESCRIPTION,
        examples=[_EXAMPLE_RUN_UUID],
    )
    agent_id: str = Field(
        description=_AGENT_ID_DESCRIPTION, examples=[_EXAMPLE_AGENT_UUID]
    )
    status: TaskStatus = Field(description=_STATUS_DESCRIPTION)
    inferred_type: TraceInferredType = Field(description=_INFERRED_TYPE_DESCRIPTION)
    trace_count: int = Field(description=_TRACE_COUNT_DESCRIPTION)
    skipped_count: int = Field(description=_SKIPPED_COUNT_DESCRIPTION)
    error: Optional[str] = Field(None, description=_ERROR_DESCRIPTION)
    created_at: str = Field(description="When the run was created (ISO 8601 UTC)")
    started_at: Optional[str] = Field(
        None, description="When judging began (ISO 8601 UTC)"
    )
    finished_at: Optional[str] = Field(
        None, description="When judging ended (ISO 8601 UTC)"
    )


class TraceEvalRunSummary(BaseModel):
    task_id: str = Field(
        min_length=36,
        max_length=36,
        description=_TASK_ID_DESCRIPTION,
        examples=[_EXAMPLE_RUN_UUID],
    )
    agent_id: str = Field(
        description=_AGENT_ID_DESCRIPTION, examples=[_EXAMPLE_AGENT_UUID]
    )
    trigger: TraceEvalTrigger = Field(
        description=(
            "What started the run:\n\n"
            "- `auto`: automatic judging of traces as they arrived\n"
            "- `manual`: someone asked for these traces to be judged\n"
        )
    )
    status: TaskStatus = Field(description=_STATUS_DESCRIPTION)
    inferred_type: TraceInferredType = Field(description=_INFERRED_TYPE_DESCRIPTION)
    trace_count: int = Field(description=_TRACE_COUNT_DESCRIPTION)
    skipped_count: int = Field(description=_SKIPPED_COUNT_DESCRIPTION)
    error: Optional[str] = Field(None, description=_ERROR_DESCRIPTION)
    created_at: str = Field(description="When the run was created (ISO 8601 UTC)")
    started_at: Optional[str] = Field(
        None, description="When judging began (ISO 8601 UTC)"
    )
    finished_at: Optional[str] = Field(
        None, description="When judging ended (ISO 8601 UTC)"
    )


class TraceEvalSettings(BaseModel):
    auto_eval_enabled: bool = Field(description=_AUTO_EVAL_DESCRIPTION)


class TraceEvalSettingsUpdate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    auto_eval_enabled: bool = Field(
        description="Turn automatic judging of arriving traces on or off"
    )


def _run_fields(run: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "task_id": run["uuid"],
        "agent_id": run["agent_id"],
        "status": run["status"],
        "inferred_type": run["inferred_type"],
        "trace_count": run["trace_count"],
        "skipped_count": run["skipped_count"],
        "error": run["error"],
        "created_at": run["created_at"],
        "started_at": run["started_at"],
        "finished_at": run["finished_at"],
    }


def _run_summary(run: Dict[str, Any]) -> Dict[str, Any]:
    return {**_run_fields(run), "trigger": run["trigger"]}


def _select_traces(
    org_uuid: str, agent_uuid: str, payload: TraceEvalRunRequest
) -> List[Dict[str, Any]]:
    """Resolve the request's trace selection, 404ing on anything out of scope."""
    if payload.select_all:
        return eval_store.list_pending_traces(
            org_uuid, agent_uuid, limit=_SELECT_ALL_TRACE_LIMIT
        )
    if not payload.trace_ids:
        raise HTTPException(
            status_code=400,
            detail="trace_ids must be non-empty when select_all is false",
        )
    selected: List[Dict[str, Any]] = []
    for trace_uuid in payload.trace_ids:
        # A trace from another workspace and one from a sibling agent land on
        # the same 404, so neither leaks that the trace exists.
        row = traces_store.get_trace(org_uuid, trace_uuid)
        if not row or row["agent_id"] != agent_uuid:
            raise HTTPException(
                status_code=404,
                detail=f"Trace not found for this agent: {trace_uuid}",
            )
        selected.append(row)
    return selected


def _auto_eval_enabled(agent: Dict[str, Any]) -> bool:
    return bool((agent.get("config") or {}).get("auto_eval_enabled", False))


@router.post(
    "/agent/{agent_uuid}/run",
    response_model=TraceEvalRunLaunchResponse,
    summary="Run trace evaluations",
)
async def run_trace_evals(
    payload: TraceEvalRunRequest,
    agent_uuid: str = Path(
        description="The agent whose traces to judge",
        examples=[_EXAMPLE_AGENT_UUID],
    ),
    ctx: OrgContext = Depends(get_current_org),
):
    """Judge an agent's traces with the evaluators linked to it, starting one run for each inferred trace type"""
    agent = ensure_owned_agent(agent_uuid, ctx.org_uuid)

    evaluators = get_evaluators_for_agent(agent_uuid)
    if not evaluators:
        raise HTTPException(
            status_code=400,
            detail="This agent has no evaluators linked to it, so its traces cannot be judged. Link at least one evaluator first",
        )

    traces = _select_traces(ctx.org_uuid, agent_uuid, payload)
    batches, skipped = inference.plan_batches(traces, evaluators)

    runs = []
    for inferred_type, batch in batches.items():
        task_id, status = launch_trace_eval(
            org_uuid=ctx.org_uuid,
            agent=agent,
            inferred_type=inferred_type,
            traces=batch,
            evaluators=inference.evaluators_for_type(inferred_type, evaluators),
            trigger=eval_store.TRIGGER_MANUAL,
        )
        runs.append(
            {
                "task_id": task_id,
                "inferred_type": inferred_type,
                "trace_count": len(batch),
                "status": status,
            }
        )
    return {"runs": runs, "skipped_count": len(skipped)}


@router.get(
    "/run/{task_id}",
    response_model=TraceEvalRunStatusResponse,
    summary="Get trace evaluation run status",
)
async def get_trace_eval_run(
    task_id: str = Path(
        description="The trace evaluation run to poll",
        examples=[_EXAMPLE_RUN_UUID],
    ),
    ctx: OrgContext = Depends(get_current_org),
):
    """Poll a trace evaluation run for its status and progress"""
    run = eval_store.get_eval_run(ctx.org_uuid, task_id)
    if not run:
        raise HTTPException(status_code=404, detail="Trace evaluation run not found")
    return _run_fields(run)


@router.get(
    "/agent/{agent_uuid}/runs",
    response_model=PaginatedResponse[TraceEvalRunSummary],
    summary="List trace evaluation runs",
)
async def list_trace_eval_runs(
    agent_uuid: str = Path(
        description="The agent whose runs to list",
        examples=[_EXAMPLE_AGENT_UUID],
    ),
    ctx: OrgContext = Depends(get_current_org),
    pagination: PaginationParams = Depends(),
):
    """List an agent's trace evaluation runs, newest first"""
    ensure_owned_agent(agent_uuid, ctx.org_uuid)
    rows, total = eval_store.list_eval_runs(
        ctx.org_uuid, agent_uuid, limit=pagination.limit, offset=pagination.offset
    )
    return page_envelope([_run_summary(row) for row in rows], total, pagination)


@router.get(
    "/agent/{agent_uuid}/settings",
    response_model=TraceEvalSettings,
    summary="Get trace evaluation settings",
)
async def get_trace_eval_settings(
    agent_uuid: str = Path(
        description="The agent to read the settings of",
        examples=[_EXAMPLE_AGENT_UUID],
    ),
    ctx: OrgContext = Depends(get_current_org),
):
    """Get whether an agent's traces are judged automatically as they arrive"""
    agent = ensure_owned_agent(agent_uuid, ctx.org_uuid)
    return {"auto_eval_enabled": _auto_eval_enabled(agent)}


@router.patch(
    "/agent/{agent_uuid}/settings",
    response_model=TraceEvalSettings,
    summary="Update trace evaluation settings",
)
async def update_trace_eval_settings(
    payload: TraceEvalSettingsUpdate,
    agent_uuid: str = Path(
        description="The agent to update the settings of",
        examples=[_EXAMPLE_AGENT_UUID],
    ),
    ctx: OrgContext = Depends(get_current_org),
):
    """Turn automatic judging of an agent's arriving traces on or off"""
    agent = ensure_owned_agent(agent_uuid, ctx.org_uuid)
    config = dict(agent.get("config") or {})
    config["auto_eval_enabled"] = payload.auto_eval_enabled
    update_agent(agent_uuid, config=config)
    return {"auto_eval_enabled": payload.auto_eval_enabled}
