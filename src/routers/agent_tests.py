import copy
import csv
import os
import json
import subprocess
import tempfile
import time
import traceback
import threading
import logging
from pathlib import Path
from typing import List, Dict, Any, Literal, Optional

from fastapi import APIRouter, HTTPException, Depends, Path as PathParam, Query
from pagination import (
    OptionalPaginationParams,
    PaginatedResponse,
    count_and_page,
    make_projection_params,
    make_search_params,
    page_envelope,
    paginate,
)

_AgentTestSearch = make_search_params(searchable=["name"])
from pydantic import BaseModel, Field
from sqlite3 import IntegrityError

from db import (
    add_test_to_agent,
    remove_test_from_agent,
    bulk_remove_tests_from_agent,
    bulk_delete_tests,
    get_tests_for_agent,
    get_tests_for_agent_summary,
    get_agents_for_test,
    get_agent_test_link,
    get_all_agent_tests,
    get_agent,
    get_all_agents,
    get_test,
    get_tools_for_agent,
    get_evaluators_for_test,
    get_evaluator_by_slug,
    get_evaluator_version,
    create_agent_test_job,
    get_agent_test_job,
    update_agent_test_job,
    update_agent_test_job_visibility,
    get_agent_test_jobs_for_agent_summary,
    get_agent_test_jobs_for_org_summary,
    delete_agent_test_job,
)
from llm_judge import build_test_evaluators_payload, evaluator_value_name
from auth_utils import get_current_org, get_org_jwt_or_api_key, OrgContext
from utils import (
    TaskStatus,
    InitialTaskStatus,
    TaskCreateResponse,
    EXAMPLE_TEST_UUID,
    TestListResponse,
    to_test_list_response,
    OutputTypeLiteral,
    AGENT_TYPE_DESCRIPTION,
    AgentTestJobType,
    get_s3_client,
    get_s3_output_config,
    can_start_agent_test_job,
    try_start_queued_agent_test_job,
    register_job_starter,
    is_job_timed_out,
    capture_exception_to_sentry,
    build_tool_configs,
    get_calibrate_agent_cli,
    upload_directory_tree_to_s3,
    upload_file_to_s3,
)

# Job types that share the same queue
AGENT_TEST_JOB_TYPES = ["llm-unit-test", "llm-benchmark"]


def _start_llm_unit_test_job_from_queue(job: dict) -> bool:
    """Start an LLM unit test job from the queue."""
    job_id = job["uuid"]
    details = job.get("details", {})

    agent_uuid = details.get("agent_uuid")
    test_uuids = details.get("test_uuids", [])
    s3_bucket = details.get("s3_bucket", "")

    # Get agent and tests
    agent = get_agent(agent_uuid)
    if not agent:
        return False

    tests = []
    for test_uuid in test_uuids:
        test = get_test(test_uuid)
        if test:
            tests.append(test)

    if not tests:
        return False

    # Start background task in a separate thread
    thread = threading.Thread(
        target=run_llm_test_task,
        args=(job_id, agent, tests, s3_bucket),
        daemon=True,
    )
    thread.start()

    return True


def _start_llm_benchmark_job_from_queue(job: dict) -> bool:
    """Start an LLM benchmark job from the queue."""
    job_id = job["uuid"]
    details = job.get("details", {})

    agent_uuid = details.get("agent_uuid")
    test_uuids = details.get("test_uuids", [])
    models = details.get("models", [])
    s3_bucket = details.get("s3_bucket", "")

    # Get agent and tests
    agent = get_agent(agent_uuid)
    if not agent:
        return False

    tests = []
    for test_uuid in test_uuids:
        test = get_test(test_uuid)
        if test:
            tests.append(test)

    if not tests or not models:
        return False

    # Start background task in a separate thread
    thread = threading.Thread(
        target=run_benchmark_task,
        args=(job_id, agent, tests, models, s3_bucket),
        daemon=True,
    )
    thread.start()

    return True


# Register the job starters for agent test jobs
register_job_starter("llm-unit-test", _start_llm_unit_test_job_from_queue)
register_job_starter("llm-benchmark", _start_llm_benchmark_job_from_queue)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/agent-tests", tags=["agent-tests"])

# Doc-only examples — IDs are `str(uuid.uuid4())` (36-char UUID v4).
_EXAMPLE_AGENT_UUID = "f47ac10b-58cc-4372-a567-0e02b2c3d479"
_EXAMPLE_TASK_UUID = "a3b2c1d0-e5f4-3210-abcd-ef1234567890"

_TASK_STATUS_DESCRIPTION = "Current status of the run"

# Shared so the benchmark leaderboard reads the same wherever it appears.
LEADERBOARD_SUMMARY_DESCRIPTION = (
    "Leaderboard comparing the models, one row per model. Columns vary by "
    "benchmark: a `model` column plus pass/fail counts, latency, cost, and one "
    "score column per evaluator, keyed by evaluator name"
)
LEADERBOARD_SUMMARY_EXAMPLE = [
    {
        "model": "openai/gpt-4.1",
        "passed": 8,
        "failed": 2,
        "helpfulness": 0.92,
        "safety": 1.0,
        "latency_p50": 480,
        "cost": 0.0021,
    },
    {
        "model": "anthropic/claude-sonnet-4",
        "passed": 9,
        "failed": 1,
        "helpfulness": 0.95,
        "safety": 1.0,
        "latency_p50": 520,
        "cost": 0.0034,
    },
]


class AgentTestsCreate(BaseModel):
    agent_uuid: str = Field(
        min_length=36,
        max_length=36,
        description="Agent to link tests to",
        examples=[_EXAMPLE_AGENT_UUID],
    )
    test_uuids: List[str] = Field(
        description="Tests to link. Any that are already linked are skipped",
        examples=[[EXAMPLE_TEST_UUID]],
    )


class AgentTestDelete(BaseModel):
    agent_uuid: str = Field(
        min_length=36,
        max_length=36,
        description="Agent to unlink the test from",
        examples=[_EXAMPLE_AGENT_UUID],
    )
    test_uuid: str = Field(
        min_length=36,
        max_length=36,
        description="Test to unlink from the agent",
        examples=[EXAMPLE_TEST_UUID],
    )


class AgentTestResponse(BaseModel):
    id: int = Field(description="Identifier for the agent-test link")
    agent_id: str = Field(
        min_length=36,
        max_length=36,
        description="The linked agent",
        examples=[_EXAMPLE_AGENT_UUID],
    )
    test_id: str = Field(
        min_length=36,
        max_length=36,
        description="The linked test",
        examples=[EXAMPLE_TEST_UUID],
    )
    created_at: str = Field(description="When the link was created (ISO 8601 UTC)")


class AgentTestsCreateResponse(BaseModel):
    ids: List[int] = Field(
        description="Identifiers for the links created by this call. Tests that were already linked are excluded"
    )
    message: str = Field(description="Confirmation message")


class AgentResponse(BaseModel):
    uuid: str = Field(
        min_length=36,
        max_length=36,
        description="Agent ID",
        examples=[_EXAMPLE_AGENT_UUID],
    )
    name: str = Field(description="Agent name")
    type: Literal["agent", "connection"] = Field(description=AGENT_TYPE_DESCRIPTION)
    config: Dict[str, Any] | None = Field(
        None, description="How the agent behaves"
    )
    created_at: str = Field(description="When the agent was created (ISO 8601 UTC)")
    updated_at: str = Field(description="When the agent was last updated (ISO 8601 UTC)")


class RunTestRequest(BaseModel):
    test_uuids: Optional[List[str]] = Field(
        None,
        description="Tests to run. Omit to run all tests linked to the agent",
        examples=[[EXAMPLE_TEST_UUID]],
    )


class AgentTestRunCreateResponse(BaseModel):
    task_id: str = Field(
        min_length=36,
        max_length=36,
        description="Test run job ID. Poll it for status and results",
        examples=[_EXAMPLE_TASK_UUID],
    )
    status: InitialTaskStatus = Field(description="Current status of the run")


class BatchRunRequest(BaseModel):
    agent_names: Optional[List[str]] = Field(
        None,
        description="Agents to run. Omit to run every agent in your workspace",
        examples=[["my-agent", "support-bot"]],
    )


class BatchTestRun(BaseModel):
    agent_name: str = Field(description="Name of the agent")
    agent_uuid: str = Field(
        min_length=36,
        max_length=36,
        description="ID of the agent that was run",
        examples=[_EXAMPLE_AGENT_UUID],
    )
    task_id: str = Field(
        min_length=36,
        max_length=36,
        description="Test run job ID. Poll it for status and results",
        examples=[_EXAMPLE_TASK_UUID],
    )
    status: InitialTaskStatus = Field(description="Current status of the run")


class BatchTestSkip(BaseModel):
    agent_name: str = Field(description="Name of the skipped agent")
    agent_uuid: str = Field(
        min_length=36,
        max_length=36,
        description="ID of the skipped agent",
        examples=[_EXAMPLE_AGENT_UUID],
    )
    reason: Literal["no_linked_tests", "connection_not_verified"] = Field(
        description=(
            "Why this agent was not run:\n"
            "- `no_linked_tests`: the agent has no tests linked\n"
            "- `connection_not_verified`: the agent's connection is not verified"
        )
    )


class BatchTestRunResponse(BaseModel):
    runs: List[BatchTestRun] = Field(description="Test runs that were launched")
    skipped: List[BatchTestSkip] = Field(
        default=[],
        description="Agents that were skipped instead of failing the batch",
    )


class ToolCallOutput(BaseModel):
    tool: str = Field(description="Name of the tool the agent called")
    arguments: Optional[Dict[str, Any]] = Field(
        None,
        description="Arguments the agent passed to the tool",
        examples=[{"city": "Paris"}],
    )
    output: Optional[Any] = Field(
        None,
        description="Tool execution result, when the agent ran the tool and returned its result",
    )


class TestOutput(BaseModel):
    response: Optional[str] = Field(
        None, description="The reply the agent generated"
    )
    tool_calls: Optional[List[ToolCallOutput]] = Field(
        None, description="Tool calls the agent generated"
    )


class JudgeResult(BaseModel):
    evaluator_uuid: Optional[str] = Field(
        None,
        min_length=36,
        max_length=36,
        description="ID of the evaluator that produced this verdict",
    )
    reasoning: Optional[str] = Field(
        None, description="The judge's rationale for this verdict"
    )
    match: Optional[bool] = Field(
        None,
        description="Pass/fail verdict, set for binary evaluators",
    )
    score: Optional[float] = Field(
        None,
        description="Numeric score, set for rating evaluators",
    )
    value_name: Optional[str] = Field(
        None,
        description="Readable label for the verdict, taken from the run's rubric",
    )
    variable_values: Optional[Dict[str, Any]] = Field(
        None,
        description="Values filled into the evaluator prompt's `{{variable}}` placeholders for this test case, keyed by variable name",
        examples=[{"criteria": "Is the reply correct and helpful?"}],
    )


class TestCaseResult(BaseModel):
    test_case_id: Optional[str] = Field(
        None, description="ID of the test case within the run"
    )
    name: Optional[str] = Field(None, description="Name of the test")
    passed: Optional[bool] = Field(
        None, description="Whether the case passed"
    )
    reasoning: Optional[str] = Field(
        None,
        description="The judge's reasoning, or the tool-call diff for a tool-call test",
    )
    output: Optional[TestOutput] = Field(
        None, description="The agent's output for this case"
    )
    test_case: Optional[Dict[str, Any]] = Field(
        None, description="The test case definition that was run"
    )
    judge_results: Optional[List[JudgeResult]] = Field(
        None,
        description="One verdict for each evaluator",
    )
    latency_ms: Optional[float] = Field(
        None,
        description="How long the agent took to respond, in milliseconds",
    )
    cost: Optional[float] = Field(
        None,
        description="Cost of this case (USD)",
    )


class TestRunEvaluator(BaseModel):
    uuid: Optional[str] = Field(None, description="ID of the evaluator")
    name: Optional[str] = Field(None, description="Name of the evaluator")
    description: Optional[str] = Field(None, description="What the evaluator checks")
    output_type: Optional[OutputTypeLiteral] = Field(
        None,
        description=(
            "The shape of the verdict:\n"
            "- `binary`: a pass/fail verdict\n"
            "- `rating`: a numeric score"
        ),
    )
    output_config: Optional[Dict[str, Any]] = Field(
        None,
        description="The rubric: the scale values, labels, and colors a verdict maps to",
        examples=[
            {
                "scale": [
                    {"value": False, "name": "Wrong", "color": "#e5484d"},
                    {"value": True, "name": "Correct", "color": "#30a46c"},
                ]
            }
        ],
    )
    scale_min: Optional[float] = Field(
        None, description="Lowest value on a rating scale"
    )
    scale_max: Optional[float] = Field(
        None, description="Highest value on a rating scale"
    )
    version_number: Optional[int] = Field(
        None, description="The evaluator version this run used"
    )


class TestRunStatusResponse(BaseModel):
    task_id: str = Field(
        min_length=36,
        max_length=36,
        description="Test run job ID",
        examples=[_EXAMPLE_TASK_UUID],
    )
    status: TaskStatus = Field(description=_TASK_STATUS_DESCRIPTION)
    test_uuids: Optional[List[str]] = Field(
        None,
        description="IDs of the tests this run executed, in run order",
    )
    total_tests: Optional[int] = Field(
        None, description="Total number of test cases"
    )
    passed: Optional[int] = Field(
        None, description="Number of test cases that passed"
    )
    failed: Optional[int] = Field(
        None, description="Number of test cases that failed"
    )
    latency_ms: Optional[Dict[str, Any]] = Field(
        None,
        description="Aggregated response latency in milliseconds, as `{p50, p95, p99, count}`",
    )
    cost: Optional[Dict[str, Any]] = Field(
        None,
        description="Aggregated cost as `{mean, min, max, count}` (USD)",
    )
    total_tokens: Optional[Dict[str, Any]] = Field(
        None,
        description="Aggregated token usage as `{mean, min, max, count}`",
    )
    evaluators: Optional[List[TestRunEvaluator]] = Field(
        None,
        description="The evaluators used in this run. Each verdict in `judge_results` links to one of these by `evaluator_uuid`",
    )
    results: Optional[List[TestCaseResult]] = Field(
        None, description="Results for each test case"
    )
    error: bool = Field(False, description="True if the run failed")
    is_public: bool = Field(False, description="Whether the run is shared publicly")
    share_token: Optional[str] = Field(
        None, description="Token for building the public share URL"
    )


class TestRunCaseSummary(BaseModel):
    """Flat summary for one test case in the run-LIST endpoints. Carries only
    enough to render a run's pass/fail breakdown and a case name. The full detail
    for each case (agent output, judge verdicts, reasoning, latency, cost, the
    test-case definition) lives on the run-DETAIL endpoint
    (`GET /agent-tests/run/{task_id}`)."""

    name: Optional[str] = Field(
        None, description="Name of the test case"
    )
    passed: Optional[bool] = Field(
        None, description="Whether the case passed (null if it errored or is still running)"
    )


class ModelRunSummary(BaseModel):
    """Flat summary for one model in a benchmark run-LIST item. The full results
    for each case of a model live on the benchmark detail endpoint
    (`GET /agent-tests/benchmark/{task_id}`), not here."""

    model: str = Field(
        description="Model name these results are for", examples=["openai/gpt-4.1"]
    )
    success: Optional[bool] = Field(
        None, description="Whether this model's run succeeded"
    )
    message: str = Field("", description="Status or result message for this model")
    total_tests: Optional[int] = Field(
        None, description="Total test cases for this model"
    )
    passed: Optional[int] = Field(
        None, description="Number of test cases that passed for this model"
    )
    failed: Optional[int] = Field(
        None, description="Number of test cases that failed for this model"
    )


class AgentTestRunListItem(BaseModel):
    uuid: str = Field(
        min_length=36,
        max_length=36,
        description="Test run job ID",
        examples=[_EXAMPLE_TASK_UUID],
    )
    name: str = Field(
        description="Display name, such as `Run 1` for a unit test or `Benchmark 1` for a benchmark"
    )
    status: TaskStatus = Field(description=_TASK_STATUS_DESCRIPTION)
    type: AgentTestJobType = Field(
        description=(
            "What kind of run this is:\n"
            "- `llm-unit-test`: a single run of the agent's tests\n"
            "- `llm-benchmark`: a multi-model comparison"
        )
    )
    updated_at: str = Field(description="When the run was last updated (ISO 8601 UTC)")
    total_tests: Optional[int] = Field(
        None,
        description="Total number of test cases",
    )
    passed: Optional[int] = Field(
        None, description="Number of test cases that passed"
    )
    failed: Optional[int] = Field(
        None, description="Number of test cases that failed"
    )
    results: Optional[List[TestRunCaseSummary]] = Field(
        None,
        description="Flat pass/fail summary for each test case (fetch the run detail for full results)",
    )
    latency_ms: Optional[Dict[str, Any]] = Field(
        None,
        description="Aggregated latency in milliseconds, as `{p50, p95, p99, count}`",
    )
    cost: Optional[Dict[str, Any]] = Field(
        None, description="Aggregated cost as `{mean, min, max, count}` (USD)"
    )
    total_tokens: Optional[Dict[str, Any]] = Field(
        None,
        description="Aggregated token usage as `{mean, min, max, count}`",
    )
    model_results: Optional[List[ModelRunSummary]] = Field(
        None,
        description="Flat summary for each model in a benchmark run (fetch the benchmark detail for full results)",
    )
    error: bool = Field(False, description="True if the run failed")
    is_public: bool = Field(False, description="Whether the run is shared publicly")
    share_token: Optional[str] = Field(
        None, description="Token for building the public share URL"
    )


class GlobalTestRunListItem(AgentTestRunListItem):
    agent_id: str = Field(
        min_length=36,
        max_length=36,
        description="Agent this run belongs to",
        examples=[_EXAMPLE_AGENT_UUID],
    )
    agent_name: str = Field(description="Name of the agent this run belongs to")


@router.post(
    "",
    response_model=AgentTestsCreateResponse,
    summary="Link tests to agent",
    tags=["Public API"],
)
async def create_agent_test_links(
    agent_tests: AgentTestsCreate,
    ctx: OrgContext = Depends(get_org_jwt_or_api_key),
):
    """Link one or more tests to an agent. Tests that are already linked are skipped."""
    # Public API (auth via get_org_jwt_or_api_key). Verify the agent exists and
    # belongs to the caller's workspace (404 otherwise).
    agent = get_agent(agent_tests.agent_uuid)
    if not agent or agent.get("org_uuid") != ctx.org_uuid:
        raise HTTPException(status_code=404, detail="Agent not found")

    # Every test must exist and belong to the caller's workspace (404 otherwise).
    for test_uuid in agent_tests.test_uuids:
        test = get_test(test_uuid)
        if not test or test.get("org_uuid") != ctx.org_uuid:
            raise HTTPException(status_code=404, detail=f"Test {test_uuid} not found")

    link_ids = []
    for test_uuid in agent_tests.test_uuids:
        # Check if link already exists
        existing = get_agent_test_link(agent_tests.agent_uuid, test_uuid)
        if existing:
            continue  # Skip already linked tests

        try:
            link_id = add_test_to_agent(
                agent_id=agent_tests.agent_uuid,
                test_id=test_uuid,
            )
            link_ids.append(link_id)
        except IntegrityError:
            continue  # Skip if already linked

    return AgentTestsCreateResponse(
        ids=link_ids, message="Tests added to agent successfully"
    )


@router.get("", response_model=List[AgentTestResponse], summary="List agent-test links")
async def list_agent_tests():
    """List which tests are linked to which agents."""
    links = get_all_agent_tests()
    return links


@router.get(
    "/agent/{agent_uuid}/tests",
    response_model=PaginatedResponse[TestListResponse],
    summary="List tests for agent",
    tags=["Public API"],
)
async def get_agent_tests_endpoint(
    agent_uuid: str = PathParam(
        description="Agent whose linked tests to list",
        examples=[_EXAMPLE_AGENT_UUID],
    ),
    ctx: OrgContext = Depends(get_org_jwt_or_api_key),
    search: _AgentTestSearch = Depends(),
    pagination: OptionalPaginationParams = Depends(),
):
    """List the tests linked to an agent."""
    # Public API (auth via get_org_jwt_or_api_key). Verify the agent exists and
    # belongs to the caller's workspace (404 otherwise).
    agent = get_agent(agent_uuid)
    if not agent or agent.get("org_uuid") != ctx.org_uuid:
        raise HTTPException(status_code=404, detail="Agent not found")

    # Optional `?q=` name search + `?limit=&offset=` paging. Returns the
    # `{items, total, limit, offset}` envelope. Each item is the trimmed list
    # shape (uuid/name/type + config.description, no evaluator hydration);
    # the transform runs only on the returned page.
    tests = get_tests_for_agent_summary(agent_uuid)
    tests = search.apply(tests)
    page, total = count_and_page(tests, pagination)
    return page_envelope([to_test_list_response(t) for t in page], total, pagination)


def _slim_test_results(test_results: Any) -> Optional[List[Dict[str, Any]]]:
    """Flatten stored per-case results into `{name, passed}` rows for the run-list.
    Lifts the nested `test_case.name` up onto `name` so the row carries no nested
    objects; the full per-case detail stays on the run-detail endpoint."""
    if not test_results:
        return None
    slim = []
    for r in test_results:
        if not isinstance(r, dict):
            continue
        test_case = r.get("test_case") if isinstance(r.get("test_case"), dict) else {}
        slim.append(
            {
                "name": r.get("name") or (test_case or {}).get("name"),
                "passed": r.get("passed"),
            }
        )
    return slim or None


def _slim_model_results(model_results: Any) -> Optional[List[Dict[str, Any]]]:
    """Flatten stored per-model benchmark results into scalar-only rows for the
    run-list, dropping each model's per-case `test_results`. Full per-case detail
    stays on the benchmark-detail endpoint."""
    if not model_results:
        return None
    slim = []
    for m in model_results:
        if not isinstance(m, dict):
            continue
        slim.append(
            {
                "model": m.get("model", ""),
                "success": m.get("success"),
                "message": m.get("message", ""),
                "total_tests": m.get("total_tests"),
                "passed": m.get("passed"),
                "failed": m.get("failed"),
            }
        )
    return slim or None


def _build_agent_test_run_item_fields(job: Dict[str, Any], name: str) -> Dict[str, Any]:
    """Shared field mapping for the run-list item models (``AgentTestRunListItem``
    and its ``GlobalTestRunListItem`` subclass).

    The list is a lightweight index: it carries flat per-case (`{name, passed}`)
    and per-model scalar summaries plus run-level aggregates — NOT the heavy
    per-case detail (agent output, judge verdicts, reasoning, the test-case
    definition, per-model `test_results`, leaderboard, evaluator rubrics). Those
    live on the run-detail endpoints, which every viewer re-fetches by task id.
    Callers spread the result and append any model-specific fields (e.g.
    ``agent_id``/``agent_name`` for the global view).
    """
    job_results = job.get("results") or {}

    return {
        "uuid": job["uuid"],
        "name": name,
        "status": job["status"],
        "type": job.get("type", ""),
        "updated_at": job.get("updated_at", job.get("created_at", "")),
        # Unit test results
        "total_tests": job_results.get("total_tests"),
        "passed": job_results.get("passed"),
        "failed": job_results.get("failed"),
        "latency_ms": job_results.get("latency_ms"),
        "cost": job_results.get("cost"),
        "total_tokens": job_results.get("total_tokens"),
        "results": _slim_test_results(job_results.get("test_results")),
        # Benchmark results
        "model_results": _slim_model_results(job_results.get("model_results")),
        # Common fields
        "error": bool(job_results.get("error")),
        "is_public": bool(job.get("is_public")),
        "share_token": job.get("share_token"),
    }


def _run_item_has_failures(item: AgentTestRunListItem) -> bool:
    """True if a run has any failing test case or model. Covers both shapes:
    a unit-test run's aggregate `failed`/`error`, and a benchmark run where any
    single model failed (`failed > 0`) or its run didn't succeed."""
    if item.error:
        return True
    if item.failed and item.failed > 0:
        return True
    for m in item.model_results or []:
        if (m.failed and m.failed > 0) or m.success is False:
            return True
    return False


@router.get(
    "/agent/{agent_uuid}/runs",
    response_model=PaginatedResponse[AgentTestRunListItem],
    summary="List test runs for agent",
    tags=["Public API"],
)
async def get_agent_test_runs(
    agent_uuid: str = PathParam(
        description="Agent whose test runs to list",
        examples=[_EXAMPLE_AGENT_UUID],
    ),
    ctx: OrgContext = Depends(get_org_jwt_or_api_key),
    type: Optional[AgentTestJobType] = Query(
        None,
        description=(
            "Filter by run type. Omit to return both:\n"
            "- `llm-unit-test`: single runs of an agent's tests\n"
            "- `llm-benchmark`: multi-model comparisons"
        ),
    ),
    status: Optional[TaskStatus] = Query(
        None,
        description="Filter by run status. Omit for all statuses",
    ),
    has_failures: Optional[bool] = Query(
        None,
        description=(
            "Filter by whether the run has any failing test case or model. "
            "`true` returns only runs with failures (or errors), `false` only "
            "clean runs. Omit for both"
        ),
    ),
    pagination: OptionalPaginationParams = Depends(),
):
    """List an agent's test runs with their results"""
    # Public API (auth via get_org_jwt_or_api_key). Verify the agent exists and
    # belongs to the caller's workspace (404 otherwise).
    agent = get_agent(agent_uuid)
    if not agent or agent.get("org_uuid") != ctx.org_uuid:
        raise HTTPException(status_code=404, detail="Agent not found")

    # Get all jobs for this agent
    jobs = get_agent_test_jobs_for_agent_summary(agent_uuid)

    # Name in chronological order (oldest = "Run 1") so a run's number never
    # shifts when a newer run is added, then build items newest-first. Names are
    # assigned before any filtering so they stay stable regardless of filters.
    jobs_asc = sorted(jobs, key=lambda j: (j.get("created_at", ""), j.get("id", 0)))
    unit_test_count = 0
    benchmark_count = 0
    name_map: Dict[str, str] = {}
    for job in jobs_asc:
        job_type = job.get("type", "")
        if job_type == "llm-unit-test":
            unit_test_count += 1
            name_map[job["uuid"]] = f"Run {unit_test_count}"
        elif job_type == "llm-benchmark":
            benchmark_count += 1
            name_map[job["uuid"]] = f"Benchmark {benchmark_count}"
        else:
            name_map[job["uuid"]] = "Job"

    runs = []
    for job in jobs:  # already newest-first
        name = name_map.get(job["uuid"], "Job")
        run_item = AgentTestRunListItem(**_build_agent_test_run_item_fields(job, name))
        runs.append(run_item)

    # Optional filters — each narrows the set an MCP/agent client pages through,
    # so it can ask "the failing benchmark runs" in one small call instead of
    # pulling every run and filtering client-side.
    if type is not None:
        runs = [r for r in runs if r.type == type]
    if status is not None:
        runs = [r for r in runs if r.status == status]
    if has_failures is not None:
        runs = [r for r in runs if _run_item_has_failures(r) == has_failures]

    # `total` = matches after filtering, before the page slice.
    return paginate(runs, pagination)


@router.get(
    "/runs",
    response_model=PaginatedResponse[GlobalTestRunListItem],
    summary="List test runs for workspace",
)
async def get_all_test_runs_for_user(
    ctx: OrgContext = Depends(get_current_org),
    type: Optional[AgentTestJobType] = Query(
        None,
        description=(
            "Filter by run type. Omit to return both:\n"
            "- `llm-unit-test`: single runs of an agent's tests\n"
            "- `llm-benchmark`: multi-model comparisons"
        ),
    ),
    status: Optional[TaskStatus] = Query(
        None,
        description="Filter by run status. Omit for all statuses",
    ),
    has_failures: Optional[bool] = Query(
        None,
        description=(
            "Filter by whether the run has any failing test case or model. "
            "`true` returns only runs with failures (or errors), `false` only "
            "clean runs. Omit for both"
        ),
    ),
    pagination: OptionalPaginationParams = Depends(),
):
    """List all test runs, most recent first."""
    jobs = get_agent_test_jobs_for_org_summary(ctx.org_uuid, job_type=type)

    # Per-agent counters for naming ("Run 1", "Benchmark 2", …).
    # We need ascending order to assign names correctly, then flip back.
    jobs_asc = sorted(jobs, key=lambda j: (j.get("created_at", ""), j.get("id", 0)))
    agent_unit_counts: Dict[str, int] = {}
    agent_benchmark_counts: Dict[str, int] = {}
    name_map: Dict[str, str] = {}  # job uuid → display name

    for job in jobs_asc:
        agent_id = job.get("agent_id", "")
        job_type = job.get("type", "")
        if job_type == "llm-unit-test":
            agent_unit_counts[agent_id] = agent_unit_counts.get(agent_id, 0) + 1
            name_map[job["uuid"]] = f"Run {agent_unit_counts[agent_id]}"
        elif job_type == "llm-benchmark":
            agent_benchmark_counts[agent_id] = (
                agent_benchmark_counts.get(agent_id, 0) + 1
            )
            name_map[job["uuid"]] = f"Benchmark {agent_benchmark_counts[agent_id]}"
        else:
            name_map[job["uuid"]] = "Job"

    # Build every run item FIRST (before status/has_failures filtering) so the
    # "Run N"/"Benchmark N" names stay stable regardless of which filters apply.
    runs = []
    for job in jobs:  # already newest-first
        run_item = GlobalTestRunListItem(
            **_build_agent_test_run_item_fields(job, name_map[job["uuid"]]),
            # Agent identity (global-only fields)
            agent_id=job.get("agent_id", ""),
            agent_name=job.get("agent_name", ""),
        )
        runs.append(run_item)

    if status is not None:
        runs = [r for r in runs if r.status == status]
    if has_failures is not None:
        runs = [r for r in runs if _run_item_has_failures(r) == has_failures]

    # `total` = matches after filtering, before the page slice.
    return paginate(runs, pagination)


@router.get(
    "/test/{test_uuid}/agents",
    response_model=List[AgentResponse],
    summary="List agents for test",
)
async def get_test_agents(
    test_uuid: str = PathParam(
        description="Test whose linked agents to list",
        examples=[EXAMPLE_TEST_UUID],
    ),
):
    """List the agents linked to a test."""
    # Verify test exists
    test = get_test(test_uuid)
    if not test:
        raise HTTPException(status_code=404, detail="Test not found")

    agents = get_agents_for_test(test_uuid)
    return agents


@router.delete("", summary="Unlink test from agent")
async def delete_agent_test_link(agent_test: AgentTestDelete):
    """Unlink a test from an agent so it no longer runs for that agent."""
    deleted = remove_test_from_agent(agent_test.agent_uuid, agent_test.test_uuid)
    if not deleted:
        raise HTTPException(status_code=404, detail="Agent-test link not found")
    return {"message": "Test removed from agent successfully"}


class AgentTestBulkDelete(BaseModel):
    agent_uuid: str = Field(
        min_length=36,
        max_length=36,
        description="Agent to unlink tests from",
        examples=[_EXAMPLE_AGENT_UUID],
    )
    test_uuids: List[str] = Field(
        description="Tests to unlink from the agent",
        examples=[[EXAMPLE_TEST_UUID]],
    )


@router.post("/bulk-unlink", summary="Bulk unlink tests from agent")
async def bulk_delete_agent_test_links(payload: AgentTestBulkDelete):
    """Unlink multiple tests from an agent."""
    if not payload.test_uuids:
        raise HTTPException(status_code=400, detail="test_uuids must not be empty")

    agent = get_agent(payload.agent_uuid)
    if not agent:
        raise HTTPException(status_code=404, detail="Agent not found")

    deleted_count = bulk_remove_tests_from_agent(
        agent_id=payload.agent_uuid,
        test_ids=payload.test_uuids,
    )

    return {
        "deleted_count": deleted_count,
        "message": f"Successfully unlinked {deleted_count} test(s) from agent",
    }


class AgentTestsBulkDeleteAll(BaseModel):
    agent_uuid: str = Field(
        min_length=36,
        max_length=36,
        description="Agent whose linked tests define the deletion scope",
        examples=[_EXAMPLE_AGENT_UUID],
    )
    test_uuids: List[str] = Field(
        description="Tests to delete. Only tests linked to this agent in your workspace are deleted. Others are skipped",
        examples=[[EXAMPLE_TEST_UUID]],
    )


@router.post("/bulk-delete-tests", summary="Bulk delete agent tests")
async def bulk_delete_agent_tests(
    payload: AgentTestsBulkDeleteAll,
    ctx: OrgContext = Depends(get_current_org),
):
    """Delete tests linked to an agent, removing their links across every agent."""
    # Unlike `/bulk-unlink`, this deletes the test rows (not just links). Only
    # tests in your workspace that are linked to `agent_uuid` are deleted;
    # foreign or unlinked IDs are silently skipped. Deleting a test cascades
    # to remove its links across every agent, not just this one.
    if not payload.test_uuids:
        raise HTTPException(status_code=400, detail="test_uuids must not be empty")

    agent = get_agent(payload.agent_uuid)
    if not agent or agent.get("org_uuid") != ctx.org_uuid:
        raise HTTPException(status_code=404, detail="Agent not found")

    requested = set(payload.test_uuids)
    linked_owned_uuids = [
        t["uuid"]
        for t in get_tests_for_agent(payload.agent_uuid)
        if t.get("org_uuid") == ctx.org_uuid and t["uuid"] in requested
    ]

    if not linked_owned_uuids:
        return {
            "deleted_count": 0,
            "deleted_test_uuids": [],
            "message": "No tests eligible for deletion",
        }

    # bulk_delete_tests cascades to soft-delete every agent_tests link row
    # for these test_ids — across every agent, not just this one. Intentional:
    # the test itself is gone, so no link should survive.
    deleted_count = bulk_delete_tests(
        test_uuids=linked_owned_uuids, org_uuid=ctx.org_uuid
    )
    return {
        "deleted_count": deleted_count,
        # Eligible set, not a blind first-N slice. With test_uuids required,
        # the caller knows exactly what they asked for and we report the
        # exact subset that passed the (linked + owned) gate.
        "deleted_test_uuids": linked_owned_uuids,
        "message": (
            f"Successfully deleted {deleted_count} test(s) associated with agent"
        ),
    }


# ============ Shared Helper Functions ============


def _build_calibrate_config(
    agent: Dict[str, Any],
    tests: List[Dict[str, Any]],
    model: Optional[str] = None,
) -> tuple[Dict[str, Any], Dict[str, List[Dict[str, Any]]]]:
    """
    Build the calibrate test config from agent and tests.

    Args:
        agent: Agent dict with config
        tests: List of test dicts with config
        model: Optional model override. If None, uses agent's llm.model or defaults to gpt-4.1

    Returns:
        Tuple of (config, evaluators_by_test_id).

        ``evaluators_by_test_id`` maps test UUID → ordered list of evaluator
        snapshot dicts for response-type tests. Each entry has:
          - ``uuid`` / ``name`` — evaluator UUID and the calibrate-rendered name
            (the key calibrate uses inside ``metrics.judge_results``), used to
            map results back to the evaluator at read time.
          - ``output_type`` — ``"binary"`` or ``"rating"``.
          - ``variable_values`` — the per-test ``{{var}}`` substitutions sent to
            calibrate (frozen at submission time, so the value reflects what the
            run actually used even if the link is later edited).
          - ``scale_min`` / ``scale_max`` — present only for ``rating``, derived
            from the pinned evaluator-version ``output_config.scale``.
          - ``version_number`` — the live-at-run-time evaluator version number,
            frozen so the finished run keeps displaying the version it judged
            against even after the evaluator is edited again.

        Tests without linked evaluators (e.g. tool_call tests) are absent from
        this dict.
    """
    agent_config = agent.get("config") or {}

    # First pass: collect linked evaluators per response-type test so we can build a
    # deduped top-level `evaluators` list. Calibrate keys results by evaluator
    # name across the whole run, so each unique evaluator appears once at top level and
    # each test case references it by name (with optional `arguments` for {{var}} subs).
    #
    # Legacy back-compat: a response-type test created with `evaluation.criteria` as a
    # plain string (no `test_evaluators` link) is auto-promoted to a synthesized link
    # against the seeded `default-llm-next-reply` evaluator with
    # `variable_values={"criteria": <string>}`. That way the calibrate handoff always
    # uses our canonical evaluator (with our exact prompt) rather than relying on
    # calibrate's built-in `criteria-passed` fallback.
    default_llm_link_template: Optional[Dict[str, Any]] = None

    def _synthetic_default_llm_link(criteria_text: str) -> Optional[Dict[str, Any]]:
        nonlocal default_llm_link_template
        if default_llm_link_template is None:
            default_evaluator = get_evaluator_by_slug("default-llm-next-reply")
            if not default_evaluator or not default_evaluator.get("live_version_id"):
                logger.warning(
                    "default-llm-next-reply evaluator missing or has no live version; "
                    "legacy string criteria will be skipped"
                )
                return None
            version = get_evaluator_version(default_evaluator["live_version_id"])
            if not version:
                return None
            default_llm_link_template = {
                **default_evaluator,
                "evaluator_version_id": version["uuid"],
                "judge_model": version["judge_model"],
                "system_prompt": version["system_prompt"],
                "output_config": version.get("output_config"),
                "variables": version.get("variables"),
            }
        return {
            **default_llm_link_template,
            "variable_values": {"criteria": criteria_text},
        }

    # Both `response` and `conversation` tests carry a list of evaluators that
    # calibrate references by name per test case (`evaluation.criteria`). They
    # differ only in how calibrate judges each row (response ⇒ judge a generated
    # reply; conversation ⇒ judge the supplied transcript), which calibrate
    # decides from `evaluation.type` — the evaluator payload is identical.
    #
    # The IMMUTABLE row `type` is authoritative here, not `config.evaluation.type`:
    # `PUT /tests` keeps the row `type` fixed but accepts an arbitrary `config`
    # dict, so a stored `evaluation.type` can drift from what the test actually
    # is (and what its evaluators were validated against). We dispatch on the row
    # type and normalize `evaluation.type` to it below, so a drifted config can
    # never make calibrate judge the wrong way.
    tests_with_evaluators: List[Dict[str, Any]] = []
    for test in tests:
        test_config = test.get("config")
        if not test_config:
            continue
        evaluation = test_config.get("evaluation", {})
        if test.get("type") not in ("response", "conversation"):
            continue

        linked_evaluators = get_evaluators_for_test(test["uuid"])

        # Legacy string-criteria fallback is RESPONSE-ONLY: it synthesizes the
        # `default-llm-next-reply` LLM evaluator, which must never be attached to
        # a conversation test (those are validated to use `simulation`
        # evaluators). A conversation test can reach here with no linked
        # evaluators (the create path only validates refs when provided), so
        # gating on the row type keeps the evaluator-type contract intact —
        # such a test simply contributes no evaluators.
        if not linked_evaluators and test.get("type") == "response":
            legacy_criteria = evaluation.get("criteria")
            if isinstance(legacy_criteria, str) and legacy_criteria.strip():
                synth = _synthetic_default_llm_link(legacy_criteria)
                if synth is not None:
                    linked_evaluators = [synth]

        tests_with_evaluators.append(
            {"test_uuid": test["uuid"], "evaluators": linked_evaluators}
        )

    top_level_evaluators, criteria_per_test = build_test_evaluators_payload(
        tests_with_evaluators
    )

    # Snapshot mapping test UUID → ordered list of evaluator snapshot dicts for
    # response-type evaluators. Each entry carries enough info to reconstruct
    # the API response at read time without re-reading the (possibly edited
    # since) evaluator-version row:
    #   - {uuid, name}: lookup keys (descriptions are read live from the DB)
    #   - output_type: distinguishes binary vs rating in JudgeResult
    #   - variable_values: {{var}} substitutions sent to calibrate (frozen)
    #   - scale_min / scale_max: rating bounds from the pinned version's
    #     output_config.scale; omitted for binary
    evaluators_by_test_id: Dict[str, List[Dict[str, Any]]] = {}
    for entry in tests_with_evaluators:
        test_uuid = entry["test_uuid"]
        refs = criteria_per_test.get(test_uuid) or []
        evals = entry.get("evaluators") or []
        pairs: List[Dict[str, Any]] = []
        for i, ev in enumerate(evals):
            ev_uuid = ev.get("uuid")
            if not ev_uuid or i >= len(refs):
                continue
            output_type = ev.get("output_type") or "binary"
            output_config = ev.get("output_config")
            snap_entry: Dict[str, Any] = {
                "uuid": ev_uuid,
                "name": refs[i].get("name", ""),
                "output_type": output_type,
                "variable_values": ev.get("variable_values") or {},
                # Pin the version that actually ran (LLM tests resolve LIVE, so
                # this is the live-at-run-time number — frozen here so the
                # finished run keeps showing the version it judged against even
                # after the evaluator is edited again).
                "version_number": ev.get("version_number"),
            }
            # Snapshot the live-at-run-time version's rubric so `value_name` can
            # be resolved at read time without re-reading the version row (which
            # may have drifted since — test runs resolve the live version, so a
            # later evaluator edit would otherwise change how this finished run
            # renders). Applies to binary and rating both.
            if isinstance(output_config, dict):
                snap_entry["output_config"] = output_config
            if output_type == "rating":
                if isinstance(output_config, dict):
                    scale = output_config.get("scale")
                    if isinstance(scale, list) and scale:
                        # Reject bool because bool is a subclass of int; rating
                        # scales must be numeric (int/float), not True/False.
                        numeric_values = [
                            e.get("value")
                            for e in scale
                            if isinstance(e, dict)
                            and isinstance(e.get("value"), (int, float))
                            and not isinstance(e.get("value"), bool)
                        ]
                        if numeric_values:
                            snap_entry["scale_min"] = min(numeric_values)
                            snap_entry["scale_max"] = max(numeric_values)
            pairs.append(snap_entry)
        if pairs:
            evaluators_by_test_id[test_uuid] = pairs

    # Second pass: shape each test case with its evaluation.criteria refs.
    all_test_cases = []
    for test in tests:
        test_name = test.get("name")
        test_config = test.get("config")
        if not test_config:
            continue

        test_config["name"] = test_name
        test_config["id"] = test["uuid"]
        evaluation = test_config.get("evaluation", {})

        # The immutable row `type` wins over a possibly-drifted
        # `config.evaluation.type` (see first-pass note). Normalize the value
        # calibrate dispatches on so it always matches what the test is and what
        # its evaluators were validated against.
        row_type = test.get("type")
        evaluation["type"] = row_type
        test_config["evaluation"] = evaluation

        if row_type == "tool_call":
            tool_calls = []
            for tool_call in evaluation.get("tool_calls", []):
                tool_calls.append(
                    {
                        "tool": tool_call["tool"],
                        "arguments": (
                            tool_call["arguments"]
                            if not tool_call.get("accept_any_arguments", False)
                            else None
                        ),
                    }
                )
            evaluation["tool_calls"] = tool_calls
        elif row_type in ("response", "conversation"):
            # Reference the top-level evaluators by name (with per-test {{var}}
            # arguments). For `response`, legacy string criteria were promoted to a
            # synthetic evaluator link in the first pass; either way we overwrite
            # with the structured refs from criteria_per_test. `conversation` rows
            # keep their `history` and carry no `output`/`tool_calls`.
            evaluation["criteria"] = criteria_per_test.get(test["uuid"], [])

        all_test_cases.append(test_config)

    if agent_config.get("agent_url"):
        # Agent connection mode — agent owns its LLM; no system_prompt/tools/model in config
        config: Dict[str, Any] = {
            "agent_url": agent_config["agent_url"],
            "test_cases": all_test_cases,
        }
        if top_level_evaluators:
            config["evaluators"] = top_level_evaluators
        if agent_config.get("agent_headers"):
            config["agent_headers"] = agent_config["agent_headers"]
        return config, evaluators_by_test_id

    # Calibrate agent mode
    if model is None:
        llm_config = agent_config.get("llm", {})
        model = llm_config.get("model", "gpt-4.1")

    agent_tools = get_tools_for_agent(agent["uuid"])
    tool_configs = build_tool_configs(agent_tools)

    config = {
        "params": {"model": model},
        "system_prompt": agent_config.get("system_prompt", ""),
        "tools": tool_configs,
        "test_cases": all_test_cases,
    }
    if top_level_evaluators:
        config["evaluators"] = top_level_evaluators
    return config, evaluators_by_test_id


def _calibrate_config_from_agent_test_job(
    task_id: str,
    agent: Dict[str, Any],
    tests: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Load the calibrate JSON config frozen at job creation, or rebuild for legacy jobs."""
    job = get_agent_test_job(task_id)
    details = (job or {}).get("details") or {}
    stored = details.get("calibrate_config")
    if isinstance(stored, dict) and stored:
        return copy.deepcopy(stored)
    calibrate_config, _ = _build_calibrate_config(agent, tests)
    return calibrate_config


def _read_agent_test_results_json(output_dir: Path) -> Optional[List[dict]]:
    """Read results.json from agent test output directory if it exists."""
    if not output_dir or not output_dir.exists():
        return None
    for root, dirs, files in os.walk(output_dir):
        for file in files:
            if file == "results.json":
                file_path = Path(root) / file
                try:
                    with open(file_path, "r", encoding="utf-8") as f:
                        return json.load(f)
                except Exception:
                    return None
    return None


def _read_agent_test_metrics_json(output_dir: Path) -> Optional[dict]:
    """Read metrics.json from agent test output directory if it exists."""
    if not output_dir or not output_dir.exists():
        return None
    for root, dirs, files in os.walk(output_dir):
        for file in files:
            if file == "metrics.json":
                file_path = Path(root) / file
                try:
                    with open(file_path, "r", encoding="utf-8") as f:
                        return json.load(f)
                except Exception:
                    return None
    return None


def _parse_agent_test_results(results_data: Optional[List[dict]]) -> List[dict]:
    """Parse results.json data into the format expected by the API.

    `judge_results` is preserved as the raw calibrate-emitted dict
    (``{<calibrate_name>: {reasoning, match|score}}``) for response-type tests. It
    is later converted to a structured list (with evaluator UUIDs and current DB
    names) by ``_enrich_test_results_with_evaluators`` at API read time.
    Tool-call tests have no ``judge_results`` from calibrate, so the field is None.
    """
    if not results_data or not isinstance(results_data, list):
        return []
    test_results = []
    for r in results_data:
        output_data = r.get("output", {})
        metrics = r.get("metrics", {})
        test_case = r.get("test_case", {})
        test_results.append(
            {
                "name": test_case.get("name"),
                "test_case_id": r.get("test_case_id") or test_case.get("id"),
                "passed": metrics.get("passed", False),
                "reasoning": metrics.get("reasoning"),
                "output": {
                    "response": output_data.get("response"),
                    "tool_calls": output_data.get("tool_calls"),
                },
                "test_case": test_case,
                "judge_results": metrics.get("judge_results"),
                # latency_ms is top-level on the calibrate result object (sibling of
                # output/metrics); cost is nested inside output. Different depths by
                # design — see CLAUDE.md. We lift cost up so the API surfaces both
                # symmetrically. Both present only for live runs (None otherwise).
                "latency_ms": r.get("latency_ms"),
                "cost": output_data.get("cost"),
            }
        )
    return test_results


def _pending_test_case_result_placeholder(name: str) -> Dict[str, Any]:
    """``TestCaseResult`` shape for rows not yet finished (explicit nulls for clients)."""
    return {
        "test_case_id": None,
        "name": name,
        "passed": None,
        "reasoning": None,
        "output": None,
        "test_case": None,
        "judge_results": None,
        "latency_ms": None,
        "cost": None,
    }


def _merge_test_results_by_test_names(
    test_names: List[str], parsed: List[dict]
) -> List[dict]:
    """Interleave calibrate rows with pending placeholders in suite order (same
    fields as :func:`_pending_test_case_result_placeholder`). Used for unit-test
    and per-model benchmark rows."""
    if not test_names:
        return []
    completed = {r.get("name"): r for r in parsed if r.get("name")}
    out: List[dict] = []
    for name in test_names:
        if name in completed:
            out.append(completed[name])
        else:
            out.append(_pending_test_case_result_placeholder(name))
    return out


def _benchmark_queued_model_results(
    models: List[str], test_names: List[str]
) -> List[Dict[str, Any]]:
    """Per-model result shell with placeholder ``test_results`` (queued / not started)."""
    placeholders = [_pending_test_case_result_placeholder(n) for n in test_names]
    return [
        {
            "model": model,
            "success": None,
            "message": "Queued...",
            "total_tests": None,
            "passed": None,
            "failed": None,
            "evaluator_summary": None,
            "test_results": placeholders,
        }
        for model in models
    ]


def _get_evaluator_cached_for_enrichment(
    uid: str, cache: Dict[str, Optional[Dict[str, Any]]]
) -> Optional[Dict[str, Any]]:
    """Single SQLite round-trip per UUID per enrichment pass (see simulations
    ``current_by_uuid`` in ``apply_simulation_job_evaluator_enrichment``)."""
    if uid not in cache:
        from db import get_evaluator

        cache[uid] = get_evaluator(uid)
    return cache[uid]


_REDUNDANT_JUDGE_RESULT_KEYS = ("name", "description", "scale_min", "scale_max")


def _build_evaluators_block_for_test_run(
    evaluators_by_test_id: Optional[Dict[str, List[Dict[str, Any]]]],
    test_results: Optional[List[Dict[str, Any]]] = None,
    model_results: Optional[List[Dict[str, Any]]] = None,
    evaluator_cache: Optional[Dict[str, Optional[Dict[str, Any]]]] = None,
) -> List[Dict[str, Any]]:
    """Build the top-level `evaluators[]` block for an agent-test / benchmark
    response: one entry per unique evaluator UUID, carrying the
    `name`/`description`/`output_type`/`output_config`/scale bounds/
    `version_number` that would otherwise be duplicated across every
    judge_results row.

    Sources, in order of preference:
      1. `evaluators_by_test_id` snapshot — fixed at submission time, so
         `output_config` reflects the rubric the run actually used.
      2. `test_results` / `model_results` — fallback for legacy runs whose
         snapshot is missing the evaluator (we still want SOMETHING in the
         block so the FE doesn't 'unknown evaluator' the row).
      3. `get_evaluator(uuid)` — for current name/description (the snapshot
         only carried the calibrate-aligned name, not the live one).

    Binary evaluators with no rubric snapshot get the Correct/Wrong default
    (via `default_output_config`) so the FE always has a scale to render.
    """
    from llm_judge import default_output_config

    cache: Dict[str, Optional[Dict[str, Any]]] = (
        evaluator_cache if evaluator_cache is not None else {}
    )

    # uuid → snapshot entry (first one wins via setdefault; snapshots for
    # the same evaluator across test cases carry the same rubric — only
    # `variable_values` varies and that's per-row, not block-level).
    by_uuid: Dict[str, Dict[str, Any]] = {}
    for entries in (evaluators_by_test_id or {}).values():
        if not isinstance(entries, list):
            continue
        for e in entries:
            if isinstance(e, dict) and e.get("uuid"):
                by_uuid.setdefault(e["uuid"], e)

    # Fallback: pick up evaluator UUIDs that appear on rows but not in the
    # snapshot (legacy runs).
    def _scan_rows(rows: Optional[List[Dict[str, Any]]]) -> None:
        if not rows:
            return
        for r in rows:
            if not isinstance(r, dict):
                continue
            jr = r.get("judge_results")
            if isinstance(jr, list):
                iterator = (e for e in jr if isinstance(e, dict))
                uid_key = "evaluator_uuid"
            elif isinstance(jr, dict):
                iterator = (e for e in jr.values() if isinstance(e, dict))
                uid_key = "evaluator_id"
            else:
                continue
            for e in iterator:
                uid = e.get(uid_key)
                if uid and uid not in by_uuid:
                    by_uuid[uid] = {"uuid": uid}

    _scan_rows(test_results)
    for mr in model_results or []:
        if isinstance(mr, dict):
            _scan_rows(mr.get("test_results"))

    block: List[Dict[str, Any]] = []
    for uid, snap in by_uuid.items():
        ev = _get_evaluator_cached_for_enrichment(uid, cache)
        output_type = snap.get("output_type") or (ev.get("output_type") if ev else None)
        output_config = snap.get("output_config")
        if output_config is None:
            output_config = default_output_config(output_type)
        block.append(
            {
                "uuid": uid,
                "name": (ev.get("name") if ev else None) or snap.get("name"),
                "description": ev.get("description") if ev else None,
                "output_type": output_type,
                "output_config": output_config,
                "scale_min": snap.get("scale_min"),
                "scale_max": snap.get("scale_max"),
                # Version the run executed against (snapshot only — None for
                # legacy runs whose snapshot predates this field).
                "version_number": snap.get("version_number"),
            }
        )
    return block


def _enrich_test_results_with_evaluators(
    test_results: Optional[List[Dict[str, Any]]],
    evaluators_by_test_id: Optional[Dict[str, List[Dict[str, Any]]]],
    evaluator_cache: Optional[Dict[str, Optional[Dict[str, Any]]]] = None,
) -> None:
    """Mutate ``test_results`` in place: convert each row's raw ``judge_results``
    dict (keyed by calibrate evaluator name) into a structured list of
    ``{evaluator_uuid, name, description, reasoning, match, score,
    variable_values, scale_min, scale_max}`` entries.

    ``name`` and ``description`` reflect the **current** evaluator row from the DB (latest).
    ``variable_values`` and the rating ``scale_min``/``scale_max`` come from
    the snapshot frozen at submission time, so they always match what the run
    actually used.

    Idempotent: if ``judge_results`` is already a list (e.g. re-enriched), the
    ``name`` and ``description`` fields are refreshed against the current DB row.

    Pass ``evaluator_cache`` to dedupe ``get_evaluator`` across rows and across
    nested benchmark models (same dict as ``_enrich_model_results_with_evaluators``).
    """
    if not test_results:
        return

    cache: Dict[str, Optional[Dict[str, Any]]] = (
        evaluator_cache if evaluator_cache is not None else {}
    )
    snapshot_map = evaluators_by_test_id or {}
    for r in test_results:
        if not isinstance(r, dict):
            continue
        raw = r.get("judge_results")
        if raw is None:
            continue

        test_id = r.get("test_case_id")
        snapshot = snapshot_map.get(test_id) if test_id else None
        uuid_to_meta: Dict[str, Dict[str, Any]] = {}
        if isinstance(snapshot, list):
            for e in snapshot:
                if isinstance(e, dict) and e.get("uuid"):
                    uuid_to_meta[e["uuid"]] = e

        if isinstance(raw, list):
            for entry in raw:
                if not isinstance(entry, dict):
                    continue
                uid = entry.get("evaluator_uuid")
                if not uid:
                    continue
                meta = uuid_to_meta.get(uid) or {}
                ev = _get_evaluator_cached_for_enrichment(uid, cache)
                snap_output_type = meta.get("output_type") or (
                    ev.get("output_type") if ev else None
                )
                value = (
                    entry.get("match")
                    if entry.get("match") is not None
                    else entry.get("score")
                )
                entry["value_name"] = evaluator_value_name(
                    value, snap_output_type, meta.get("output_config")
                )
                # Drop fields that have been promoted to the top-level
                # evaluators[] block so the row stays minimal.
                for k in _REDUNDANT_JUDGE_RESULT_KEYS:
                    entry.pop(k, None)
            continue

        if not isinstance(raw, dict):
            continue

        out: List[Dict[str, Any]] = []
        for cal_name, entry in raw.items():
            if not isinstance(entry, dict):
                continue
            echoed_uid = entry.get("evaluator_id")
            meta = (uuid_to_meta.get(echoed_uid) if echoed_uid else None) or {}
            uid = echoed_uid
            # Warm the cache so the block builder can reuse it. Also use
            # the live evaluator's output_type as a fallback when the
            # snapshot lacks it (legacy jobs) — matches the list-path
            # behavior so `value_name` resolves consistently across both
            # enrichment paths.
            ev: Optional[Dict[str, Any]] = None
            if uid:
                ev = _get_evaluator_cached_for_enrichment(uid, cache)
            snap_output_type = meta.get("output_type") or (
                ev.get("output_type") if ev else None
            )
            match_val = entry.get("match")
            score_val = entry.get("score")
            scalar = match_val if match_val is not None else score_val
            out.append(
                {
                    "evaluator_uuid": uid,
                    "reasoning": entry.get("reasoning"),
                    "match": match_val,
                    "score": score_val,
                    "value_name": evaluator_value_name(
                        scalar,
                        snap_output_type,
                        meta.get("output_config"),
                    ),
                    "variable_values": meta.get("variable_values") or None,
                }
            )
        r["judge_results"] = out


def _enrich_model_results_with_evaluators(
    model_results: Optional[List[Dict[str, Any]]],
    evaluators_by_test_id: Optional[Dict[str, List[Dict[str, Any]]]],
    evaluator_cache: Optional[Dict[str, Optional[Dict[str, Any]]]] = None,
) -> None:
    """Run ``_enrich_test_results_with_evaluators`` for each model's nested
    ``test_results`` list. The same snapshot applies to every model in a
    benchmark run because all models execute the same test suite."""
    if not model_results:
        return
    cache: Dict[str, Optional[Dict[str, Any]]] = (
        evaluator_cache if evaluator_cache is not None else {}
    )
    for mr in model_results:
        if isinstance(mr, dict):
            _enrich_test_results_with_evaluators(
                mr.get("test_results"), evaluators_by_test_id, cache
            )
            _enrich_evaluator_summary(mr.get("evaluator_summary"), cache)


def _build_evaluator_summary(
    metrics_data: Optional[dict],
) -> Optional[List[Dict[str, Any]]]:
    """Extract per-evaluator benchmark aggregates from calibrate metrics.json."""
    if not isinstance(metrics_data, dict):
        return None
    criteria = metrics_data.get("criteria")
    if not isinstance(criteria, dict):
        return None

    summary: List[Dict[str, Any]] = []
    for metric_key, aggregate in criteria.items():
        if not isinstance(aggregate, dict):
            continue
        evaluator_type = aggregate.get("type")
        if evaluator_type not in {"binary", "rating"}:
            continue

        entry: Dict[str, Any] = {
            "metric_key": metric_key,
            "name": metric_key,
            "type": evaluator_type,
            "evaluator_uuid": aggregate.get("evaluator_id"),
        }

        if evaluator_type == "binary":
            entry.update(
                {
                    "passed": aggregate.get("passed"),
                    "total": aggregate.get("total"),
                    "pass_rate": aggregate.get("pass_rate"),
                }
            )
        else:
            entry.update(
                {
                    "mean": aggregate.get("mean"),
                    "min": aggregate.get("min"),
                    "max": aggregate.get("max"),
                    "count": aggregate.get("count"),
                    "scale_min": aggregate.get("scale_min"),
                    "scale_max": aggregate.get("scale_max"),
                }
            )

        summary.append(entry)

    return summary or None


def _enrich_evaluator_summary(
    evaluator_summary: Optional[List[Dict[str, Any]]],
    evaluator_cache: Optional[Dict[str, Optional[Dict[str, Any]]]] = None,
) -> None:
    """Refresh evaluator names/descriptions in per-model aggregate summaries."""
    if not evaluator_summary:
        return

    cache: Dict[str, Optional[Dict[str, Any]]] = (
        evaluator_cache if evaluator_cache is not None else {}
    )
    for entry in evaluator_summary:
        if not isinstance(entry, dict):
            continue
        uid = entry.get("evaluator_uuid")
        if not uid:
            entry.setdefault("description", None)
            continue
        ev = _get_evaluator_cached_for_enrichment(uid, cache)
        if ev and ev.get("name"):
            entry["name"] = ev["name"]
        entry["description"] = ev.get("description") if ev else None


def _find_all_results_in_output(output_dir: Path) -> Dict[str, tuple]:
    """
    Walk output_dir and find all results.json and metrics.json files.
    Returns a dict mapping folder names to (results_data, metrics_data) tuples.
    """
    found = {}
    if not output_dir.exists():
        return found

    for root, dirs, files in os.walk(output_dir):
        root_path = Path(root)
        results_data = None
        metrics_data = None

        if "results.json" in files:
            try:
                with open(root_path / "results.json", "r", encoding="utf-8") as f:
                    results_data = json.load(f)
            except Exception:
                pass

        if "metrics.json" in files:
            try:
                with open(root_path / "metrics.json", "r", encoding="utf-8") as f:
                    metrics_data = json.load(f)
            except Exception:
                pass

        if results_data is not None or metrics_data is not None:
            # Use the folder name as key (this contains the model name)
            found[root_path.name] = (results_data, metrics_data)

    return found


def _match_model_to_folder(model: str, folder_names: List[str]) -> Optional[str]:
    """Find folder name that matches the model.

    Uses exact matching across known calibrate separator conventions (single
    underscore, double underscore, dash) rather than substring matching.
    Substring matching caused false matches when one model name was a prefix of
    another (e.g. `gpt-5.4` matching the `gpt-5.4-mini` folder).
    """
    candidates = {
        model.replace("/", "_").replace(":", "_").lower(),
        model.replace("/", "-").replace(":", "-").lower(),
        model.replace("/", "__").replace(":", "__").lower(),
        model.lower(),
    }

    for folder in folder_names:
        if folder.lower() in candidates:
            return folder
    return None


def _read_leaderboard_csv(
    leaderboard_dir: Path, models: Optional[List[str]] = None
) -> Optional[List[dict]]:
    """Read the leaderboard CSV from the leaderboard directory.

    The calibrate CLI writes the ``model`` column using the model name converted
    to a filesystem-safe form (``/`` and ``:`` replaced with ``__``/``_``/``-``).
    When ``models`` is supplied (the original slash-form names), each row's
    ``model`` field is normalized back to the matching original string so the
    API response shape matches ``model_results[].model``.
    """
    if not leaderboard_dir.exists():
        logger.warning(f"Leaderboard directory does not exist: {leaderboard_dir}")
        return None

    # Find CSV file in leaderboard directory
    csv_files = list(leaderboard_dir.glob("*.csv"))
    if not csv_files:
        logger.warning(
            f"No CSV files found in leaderboard directory: {leaderboard_dir}"
        )
        all_files = list(leaderboard_dir.iterdir())
        logger.info(f"Files in leaderboard directory: {[f.name for f in all_files]}")
        return None

    csv_file = csv_files[0]
    logger.info(f"Reading leaderboard from: {csv_file}")

    folder_to_model: Dict[str, str] = {}
    if models:
        for original in models:
            for variant in {
                original.replace("/", "_").replace(":", "_").lower(),
                original.replace("/", "-").replace(":", "-").lower(),
                original.replace("/", "__").replace(":", "__").lower(),
                original.lower(),
            }:
                folder_to_model.setdefault(variant, original)

    try:
        leaderboard_summary = []
        with open(csv_file, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                row_dict = dict(row)
                folder_name = row_dict.get("model")
                if folder_name and folder_to_model:
                    original = folder_to_model.get(folder_name.lower())
                    if original:
                        row_dict["model"] = original
                leaderboard_summary.append(row_dict)
        logger.info(f"Read {len(leaderboard_summary)} rows from leaderboard")
        return leaderboard_summary
    except Exception as e:
        logger.warning(f"Failed to read leaderboard CSV: {e}")
        return None


def _update_agent_test_intermediate_results(
    task_id: str,
    output_dir: Path,
    test_names: List[str],
) -> int:
    """
    Update intermediate results for an agent test job.
    Returns the number of completed tests.
    """
    results_data = _read_agent_test_results_json(output_dir)
    if results_data is None:
        return 0

    # Parse results
    test_results = _parse_agent_test_results(results_data)
    completed_count = len(test_results)

    # Build intermediate results: show completed tests with results, pending tests with just name
    intermediate_results = _merge_test_results_by_test_names(test_names, test_results)

    # Check if metrics.json exists (all tests complete)
    metrics_data = _read_agent_test_metrics_json(output_dir)

    update_agent_test_job(
        task_id,
        results={
            "total_tests": (
                metrics_data.get("total") if metrics_data else len(test_names)
            ),
            "passed": metrics_data.get("passed") if metrics_data else None,
            "failed": (
                (metrics_data.get("total", 0) - metrics_data.get("passed", 0))
                if metrics_data
                else None
            ),
            "latency_ms": metrics_data.get("latency_ms") if metrics_data else None,
            "cost": metrics_data.get("cost") if metrics_data else None,
            "total_tokens": (
                metrics_data.get("total_tokens") if metrics_data else None
            ),
            "test_results": intermediate_results,
        },
    )

    return completed_count


# ============ Conversation tests ============
#
# A conversation-type test runs in LIVE mode through the *same* `calibrate llm`
# command and output shape as response/tool_call tests: a top-level `evaluators`
# list plus per-test-case `evaluation = {type: "conversation", criteria:
# [{name, arguments?}]}`, with `history` ending at the user turn the agent
# should answer. Each row's `evaluation.type` tells calibrate how to handle it:
#   - response   ⇒ run the agent, judge only its generated reply
#   - conversation ⇒ run the agent, append its reply, judge the FULL conversation
#   - tool_call  ⇒ run the agent, diff the tool calls
# All three invoke the agent, so conversation tests flow through the normal
# `_build_calibrate_config` / `run_llm_test_task` path and are subject to the
# same agent-connection-verified guard — see `_build_calibrate_config`.


def run_llm_test_task(
    task_id: str,
    agent: Dict[str, Any],
    tests: List[Dict[str, Any]],
    s3_bucket: str,
):
    """Run the LLM tests in the background using a single CLI command with intermediate updates.

    Handles response, tool_call, and conversation test cases uniformly — the
    calibrate CLI dispatches per row on each test case's `evaluation.type`."""
    try:
        logger.info(
            f"Running LLM test task {task_id} for agent {agent['uuid']} with {len(tests)} test(s)"
        )

        # Extract test names for progress tracking
        test_names = [test.get("name") for test in tests if test.get("name")]

        update_agent_test_job(
            task_id,
            status=TaskStatus.IN_PROGRESS.value,
            results={
                "test_results": [
                    _pending_test_case_result_placeholder(name) for name in test_names
                ]
            },
        )

        s3 = get_s3_client()

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            try:
                calibrate_config = _calibrate_config_from_agent_test_job(
                    task_id, agent, tests
                )
                agent_config = agent.get("config") or {}

                # Create directories
                input_dir = temp_path / "input"
                output_dir = temp_path / "output"
                input_dir.mkdir(parents=True, exist_ok=True)
                output_dir.mkdir(parents=True, exist_ok=True)

                # Write config file
                config_file = input_dir / "test_config.json"
                with open(config_file, "w", encoding="utf-8") as f:
                    json.dump(calibrate_config, f, indent=2)

                if agent_config.get("agent_url"):
                    # Agent connection mode: agent owns its model — no -m or -p
                    model = "agent-connection"
                    run_cmd = [
                        get_calibrate_agent_cli(),
                        "llm",
                        "-c",
                        str(config_file),
                        "-o",
                        str(output_dir),
                        "--skip-verify",
                    ]
                else:
                    # Calibrate agent mode: use model + provider from agent config
                    model = calibrate_config["params"]["model"]
                    llm_config = agent_config.get("llm", {})
                    provider = llm_config.get("provider", "openrouter")
                    run_cmd = [
                        get_calibrate_agent_cli(),
                        "llm",
                        "-c",
                        str(config_file),
                        "-m",
                        model,
                        "-p",
                        provider,
                        "-o",
                        str(output_dir),
                    ]

                # Test-case parallelism is left to calibrate, which resolves it as
                # `-n flag > CALIBRATE_TEST_PARALLEL env > default(4)`. The subprocess
                # inherits this process's env, so set CALIBRATE_TEST_PARALLEL to tune it.

                logger.info(f"Running LLM test command: {' '.join(run_cmd)}")

                # Create temp files for stdout/stderr
                stdout_path = output_dir / "stdout.log"
                stderr_path = output_dir / "stderr.log"

                with (
                    open(stdout_path, "w") as stdout_f,
                    open(stderr_path, "w") as stderr_f,
                ):
                    process = subprocess.Popen(
                        run_cmd,
                        stdout=stdout_f,
                        stderr=stderr_f,
                        text=True,
                        start_new_session=True,
                        cwd=str(temp_path),
                    )

                    # Poll for process completion while updating intermediate results
                    prev_completed = 0
                    while process.poll() is None:
                        completed = _update_agent_test_intermediate_results(
                            task_id, output_dir, test_names
                        )
                        if completed != prev_completed:
                            logger.info(
                                f"LLM test {task_id}: {completed}/{len(test_names)} tests completed"
                            )
                            prev_completed = completed
                        time.sleep(2)  # Poll every 2 seconds

                    # Final update after process completes
                    _update_agent_test_intermediate_results(
                        task_id, output_dir, test_names
                    )

                # Read stdout/stderr
                with open(stdout_path, "r") as f:
                    stdout = f.read()
                with open(stderr_path, "r") as f:
                    stderr = f.read()

                if stdout:
                    logger.info(f"LLM test stdout: {stdout}")
                if stderr:
                    logger.info(f"LLM test stderr: {stderr}")

                if process.returncode != 0:
                    error_msg = (
                        f"LLM test failed with exit code {process.returncode}: {stderr}"
                    )
                    logger.error(error_msg)
                    capture_exception_to_sentry(RuntimeError(error_msg))
                    raise subprocess.CalledProcessError(
                        process.returncode, run_cmd, stdout, stderr
                    )

                logger.info("LLM test command completed successfully")

                # Log output directory contents for debugging
                logger.info(
                    f"Output directory contents: {[f.name for f in output_dir.iterdir()]}"
                )

                # Find results.json and metrics.json files
                results_data = None
                metrics_data = None

                for root, dirs, files in os.walk(output_dir):
                    for file in files:
                        file_path = Path(root) / file
                        if file == "results.json" and results_data is None:
                            with open(file_path, "r", encoding="utf-8") as f:
                                results_data = json.load(f)
                        elif file == "metrics.json" and metrics_data is None:
                            with open(file_path, "r", encoding="utf-8") as f:
                                metrics_data = json.load(f)

                if results_data is None and metrics_data is None:
                    error_msg = f"LLM test produced no output files (results.json/metrics.json not found in {output_dir})"
                    logger.error(error_msg)
                    capture_exception_to_sentry(RuntimeError(error_msg))
                    raise subprocess.CalledProcessError(0, run_cmd, stdout, stderr)

                # Parse results
                test_results = _parse_agent_test_results(results_data)

                # Add name field for consistency
                for i, r in enumerate(test_results):
                    if not r.get("name") and results_data and i < len(results_data):
                        test_case = results_data[i].get("test_case", {})
                        r["name"] = test_case.get("name")

                # Parse metrics
                total_tests = 0
                passed = 0
                failed = 0
                latency_ms = None
                cost = None
                total_tokens = None

                if metrics_data and isinstance(metrics_data, dict):
                    total_tests = metrics_data.get("total", 0)
                    passed = metrics_data.get("passed", 0)
                    failed = total_tests - passed
                    latency_ms = metrics_data.get("latency_ms")
                    cost = metrics_data.get("cost")
                    total_tokens = metrics_data.get("total_tokens")
                elif results_data:
                    # Compute from results if metrics.json not found
                    total_tests = len(results_data)
                    passed = sum(
                        1
                        for r in results_data
                        if r.get("metrics", {}).get("passed", False)
                    )
                    failed = total_tests - passed

                # Upload results to S3 (calibrate ``logs``/``results.log`` per model, run-level ``logs``, etc.)
                results_prefix = f"agent-tests/runs/{task_id}"
                upload_directory_tree_to_s3(s3, output_dir, s3_bucket, results_prefix)

                # Upload the config file to S3
                config_s3_key = f"{results_prefix}/test_config.json"
                upload_file_to_s3(s3, config_file, s3_bucket, config_s3_key)
                logger.info(f"Uploaded config file to S3: {config_s3_key}")

                # Update job with results
                update_agent_test_job(
                    task_id,
                    status=TaskStatus.DONE.value,
                    results={
                        "total_tests": total_tests,
                        "passed": passed,
                        "failed": failed,
                        "latency_ms": latency_ms,
                        "cost": cost,
                        "total_tokens": total_tokens,
                        "test_results": test_results,
                        "results_s3_prefix": results_prefix,
                        "error": None,
                    },
                )

                logger.info(
                    f"LLM test task {task_id} completed: {passed}/{total_tests} passed"
                )

            except subprocess.CalledProcessError as e:
                traceback.print_exc()
                capture_exception_to_sentry(e)
                # Preserve any existing results from the job
                existing_job = get_agent_test_job(task_id)
                existing_results = (
                    (existing_job.get("results") or {}) if existing_job else {}
                )
                existing_results["error"] = (
                    f"LLM test failed: {e.stderr if hasattr(e, 'stderr') else str(e)}"
                )
                try:
                    if output_dir.exists():
                        upload_directory_tree_to_s3(
                            s3,
                            output_dir,
                            s3_bucket,
                            f"agent-tests/runs/{task_id}",
                        )
                        existing_results["results_s3_prefix"] = (
                            f"agent-tests/runs/{task_id}"
                        )
                except Exception:
                    pass
                update_agent_test_job(
                    task_id,
                    status=TaskStatus.FAILED.value,
                    results=existing_results,
                )
            except Exception as e:
                traceback.print_exc()
                capture_exception_to_sentry(e)
                # Preserve any existing results from the job
                existing_job = get_agent_test_job(task_id)
                existing_results = (
                    (existing_job.get("results") or {}) if existing_job else {}
                )
                existing_results["error"] = (
                    f"Unexpected error during LLM test: {str(e)}"
                )
                try:
                    if output_dir.exists():
                        upload_directory_tree_to_s3(
                            s3,
                            output_dir,
                            s3_bucket,
                            f"agent-tests/runs/{task_id}",
                        )
                        existing_results["results_s3_prefix"] = (
                            f"agent-tests/runs/{task_id}"
                        )
                except Exception:
                    pass
                update_agent_test_job(
                    task_id,
                    status=TaskStatus.FAILED.value,
                    results=existing_results,
                )

    except Exception as e:
        traceback.print_exc()
        capture_exception_to_sentry(e)
        # Preserve any existing results from the job
        existing_job = get_agent_test_job(task_id)
        existing_results = (existing_job.get("results") or {}) if existing_job else {}
        existing_results["error"] = f"Task failed: {str(e)}"
        update_agent_test_job(
            task_id,
            status=TaskStatus.FAILED.value,
            results=existing_results,
        )
    finally:
        # Try to start the next queued job
        try_start_queued_agent_test_job(AGENT_TEST_JOB_TYPES)


def _agent_connection_unverified(agent: Dict[str, Any]) -> bool:
    """True when a connection-type agent hasn't passed connection verification.

    Every test type runs the agent live, so an unverified connection means a run
    would fail. The single-agent endpoint turns this into a 400. The batch
    endpoints use it to skip-and-report rather than failing the whole batch.
    """
    agent_config = agent.get("config") or {}
    return bool(
        agent_config.get("agent_url") and not agent_config.get("connection_verified")
    )


def _launch_agent_test_run(
    agent: Dict[str, Any],
    tests: List[Dict[str, Any]],
    s3_bucket: str,
) -> tuple[str, str]:
    """Create + start (or queue) one ``llm-unit-test`` job for ``agent`` over ``tests``.

    Shared by the single-agent run endpoint and the batch run endpoints. Returns
    ``(task_id, status)``. The caller owns agent-ownership, connection-verified,
    and non-empty ``tests`` checks. This just snapshots config and dispatches.
    """
    agent_uuid = agent["uuid"]
    org_uuid = agent.get("org_uuid")

    can_start = can_start_agent_test_job(AGENT_TEST_JOB_TYPES, org_uuid)
    initial_status = (
        TaskStatus.IN_PROGRESS.value if can_start else TaskStatus.QUEUED.value
    )

    # Extract test names for progress tracking
    test_names = [test.get("name") for test in tests if test.get("name")]

    # Snapshot calibrate config and per-test evaluator metadata at submission so the
    # worker (and enrichment) stay aligned even if links or live versions change later.
    calibrate_config, evaluators_by_test_id = _build_calibrate_config(agent, tests)

    # Create job in database with details for recovery
    test_uuids = [t["uuid"] for t in tests]
    job_id = create_agent_test_job(
        agent_id=agent_uuid,
        job_type="llm-unit-test",
        status=initial_status,
        details={
            "agent_uuid": agent_uuid,
            "test_uuids": test_uuids,
            "test_names": test_names,
            "s3_bucket": s3_bucket,
            "calibrate_config": calibrate_config,
            "evaluators_by_test_id": evaluators_by_test_id,
        },
        results={
            "test_results": [
                _pending_test_case_result_placeholder(name) for name in test_names
            ]
        },
    )

    if can_start:
        # Start background task in a separate thread
        thread = threading.Thread(
            target=run_llm_test_task,
            args=(job_id, agent, tests, s3_bucket),
            daemon=True,
        )
        thread.start()
        logger.info(f"Started LLM unit test job {job_id} immediately")
    else:
        logger.info(f"Queued LLM unit test job {job_id}")

    return job_id, initial_status


@router.post(
    "/agent/{agent_uuid}/run",
    response_model=AgentTestRunCreateResponse,
    tags=["Public API"],
    summary="Run agent tests",
)
async def run_agent_test(
    agent_uuid: str = PathParam(
        description="Agent to test",
        examples=[_EXAMPLE_AGENT_UUID],
    ),
    request: RunTestRequest = ...,
    ctx: OrgContext = Depends(get_org_jwt_or_api_key),
):
    """Run an agent's linked tests as a background job, returning a task ID to poll."""
    # Public API (auth via get_org_jwt_or_api_key). Verify the agent exists and
    # belongs to the caller's workspace (404 otherwise).
    agent = get_agent(agent_uuid)
    if not agent or agent.get("org_uuid") != ctx.org_uuid:
        raise HTTPException(status_code=404, detail="Agent not found")

    # Guard: agent connection must be verified before running tests. Every test
    # type runs the agent — response/tool_call generate the reply to judge, and
    # conversation tests run in live mode too (calibrate runs the agent on the
    # `history`, appends the generated reply, then the simulation judge scores
    # the full conversation). So the guard applies uniformly.
    if _agent_connection_unverified(agent):
        raise HTTPException(
            status_code=400,
            detail="Agent connection not verified. Call POST /agents/{agent_uuid}/verify-connection first.",
        )

    if request.test_uuids:
        # Verify all specified tests exist and belong to the caller's org — a
        # cross-org UUID must 404 identically to a missing one (existence-leak
        # parity), otherwise a leaked/guessed UUID from another org could be
        # run against this agent and its content read back via the result.
        tests = []
        for test_uuid in request.test_uuids:
            test = get_test(test_uuid)
            if not test or test.get("org_uuid") != ctx.org_uuid:
                raise HTTPException(
                    status_code=404, detail=f"Test {test_uuid} not found"
                )
            tests.append(test)
    else:
        # No test_uuids provided — run all tests linked to the agent
        tests = get_tests_for_agent(agent_uuid)
        if not tests:
            raise HTTPException(
                status_code=400,
                detail="No tests linked to this agent. Link tests first or provide test_uuids.",
            )

    # Get S3 configuration
    try:
        s3_bucket = get_s3_output_config()
    except ValueError as e:
        raise HTTPException(status_code=500, detail=str(e))

    job_id, initial_status = _launch_agent_test_run(agent, tests, s3_bucket)
    return AgentTestRunCreateResponse(task_id=job_id, status=initial_status)


def _run_tests_for_agents(
    agents: List[Dict[str, Any]], s3_bucket: str
) -> BatchTestRunResponse:
    """Launch all linked tests for each agent in ``agents``, one job per agent.

    Agents with no linked tests or an unverified connection are skipped and
    reported under ``skipped`` rather than aborting the whole batch. Each launched
    agent yields one ``llm-unit-test`` job (its task_id), subject to the normal
    per-workspace concurrency queue — over-limit jobs come back ``queued``.
    """
    runs: List[BatchTestRun] = []
    skipped: List[BatchTestSkip] = []

    for agent in agents:
        if _agent_connection_unverified(agent):
            skipped.append(
                BatchTestSkip(
                    agent_name=agent.get("name", ""),
                    agent_uuid=agent["uuid"],
                    reason="connection_not_verified",
                )
            )
            continue

        tests = get_tests_for_agent(agent["uuid"])
        if not tests:
            skipped.append(
                BatchTestSkip(
                    agent_name=agent.get("name", ""),
                    agent_uuid=agent["uuid"],
                    reason="no_linked_tests",
                )
            )
            continue

        job_id, status = _launch_agent_test_run(agent, tests, s3_bucket)
        runs.append(
            BatchTestRun(
                agent_name=agent.get("name", ""),
                agent_uuid=agent["uuid"],
                task_id=job_id,
                status=status,
            )
        )

    return BatchTestRunResponse(runs=runs, skipped=skipped)


@router.post(
    "/run",
    response_model=BatchTestRunResponse,
    tags=["Public API"],
    summary="Run agent tests in batch",
)
async def run_tests_batch(
    request: Optional[BatchRunRequest] = None,
    ctx: OrgContext = Depends(get_org_jwt_or_api_key),
):
    """Run agent tests for every agent, or for a selected set."""
    # Public API (auth via get_org_jwt_or_api_key).
    agent_names = request.agent_names if request else None

    org_agents = get_all_agents(org_uuid=ctx.org_uuid)
    if agent_names:
        # Validate ALL names before creating any tasks.
        name_to_agent = {a["name"]: a for a in org_agents}
        not_found: List[str] = []
        selected: List[Dict[str, Any]] = []
        seen: set[str] = set()
        for name in agent_names:
            if name in seen:
                continue
            seen.add(name)
            agent = name_to_agent.get(name)
            if agent is None:
                not_found.append(name)
            else:
                selected.append(agent)

        if not_found:
            raise HTTPException(
                status_code=404,
                detail={"message": "Unknown agent name(s)", "not_found": not_found},
            )
    else:
        selected = org_agents

    try:
        s3_bucket = get_s3_output_config()
    except ValueError as e:
        raise HTTPException(status_code=500, detail=str(e))

    return _run_tests_for_agents(selected, s3_bucket)


def _load_owned_agent_test_job(task_id: str, ctx: OrgContext) -> Dict[str, Any]:
    """Fetch an agent-test job and assert the caller's workspace owns it.

    Ownership is derived through the job's parent agent (``agent_test_jobs`` has
    no workspace column of its own). Returns the job dict on success. Raises 404 with
    the same generic ``"Task not found"`` detail for the missing / cross-workspace /
    orphaned cases so existence is never leaked. A soft-deleted agent makes its
    runs unreadable here (``get_agent`` filters ``deleted_at``), consistent with
    the workspace-wide runs list. Used by the run/benchmark status and visibility
    endpoints — keep the rule in this one place.
    """
    job = get_agent_test_job(task_id)
    if not job:
        raise HTTPException(status_code=404, detail="Task not found")

    agent_id = job.get("agent_id")
    agent = get_agent(agent_id) if agent_id else None
    if not agent or agent.get("org_uuid") != ctx.org_uuid:
        raise HTTPException(status_code=404, detail="Task not found")

    return job


class VisibilityRequest(BaseModel):
    is_public: bool = Field(
        description="True to make the run publicly shareable and generate a share token. False to make it private"
    )


class VisibilityResponse(BaseModel):
    is_public: bool = Field(description="Whether the run is now shared publicly")
    share_token: str | None = Field(
        None,
        description="Token for building the public share URL",
    )


@router.patch(
    "/run/{task_id}/visibility",
    response_model=VisibilityResponse,
    summary="Update test run visibility",
)
async def update_test_run_visibility(
    task_id: str = PathParam(
        description="Test run whose sharing settings to update",
        examples=[_EXAMPLE_TASK_UUID],
    ),
    body: VisibilityRequest = ...,
    ctx: OrgContext = Depends(get_current_org),
):
    """Toggle public sharing for a test run."""
    job = _load_owned_agent_test_job(task_id, ctx)

    if body.is_public:
        import uuid as _uuid

        share_token = job.get("share_token") or str(_uuid.uuid4())
    else:
        share_token = None

    update_agent_test_job_visibility(task_id, body.is_public, share_token)
    return VisibilityResponse(is_public=body.is_public, share_token=share_token)


_RunProjection = make_projection_params(
    heavy_fields=[
        "results[].output",
        "results[].test_case",
        "results[].judge_results",
        "results[].reasoning",
        "evaluators[].output_config",
    ]
)


@router.get(
    "/run/{task_id}",
    response_model=TestRunStatusResponse,
    tags=["Public API"],
    summary="Get test run status",
)
async def get_agent_test_run_status(
    task_id: str = PathParam(
        description="Test run to poll for status and results",
        examples=[_EXAMPLE_TASK_UUID],
    ),
    ctx: OrgContext = Depends(get_org_jwt_or_api_key),
    only_failed: bool = Query(
        False,
        description="Return only failing test cases. Omit to return every case",
    ),
    projection: _RunProjection = Depends(),
):
    """Poll a test run for its status and evaluation results."""
    # Public API (auth via get_org_jwt_or_api_key); ownership enforced below.
    job = _load_owned_agent_test_job(task_id, ctx)

    status = job["status"]
    results = job.get("results") or {}
    details = job.get("details") or {}

    # Check for timeout on in-progress jobs
    # if status == TaskStatus.IN_PROGRESS.value:
    #     updated_at = job.get("updated_at")
    #     if updated_at and is_job_timed_out(updated_at):
    #         logger.warning(f"Agent test job {task_id} timed out, marking as failed")

    #         # Mark job as failed (preserve existing results, add error)
    #         results["error"] = "Job timed out after 5 minutes of inactivity"
    #         update_agent_test_job(
    #             task_id,
    #             status=TaskStatus.FAILED.value,
    #             results=results,
    #         )
    #         status = TaskStatus.FAILED.value

    #         # Try to start the next queued job
    #         try_start_queued_agent_test_job(AGENT_TEST_JOB_TYPES)

    evaluators_snapshot = details.get("evaluators_by_test_id") or {}
    evaluator_cache: Dict[str, Optional[Dict[str, Any]]] = {}
    _enrich_test_results_with_evaluators(
        results.get("test_results"), evaluators_snapshot, evaluator_cache
    )
    evaluators_block = _build_evaluators_block_for_test_run(
        evaluators_snapshot,
        test_results=results.get("test_results"),
        evaluator_cache=evaluator_cache,
    )

    response = TestRunStatusResponse(
        task_id=task_id,
        status=status,
        test_uuids=details.get("test_uuids") or None,
        total_tests=results.get("total_tests"),
        passed=results.get("passed"),
        failed=results.get("failed"),
        latency_ms=results.get("latency_ms"),
        cost=results.get("cost"),
        total_tokens=results.get("total_tokens"),
        evaluators=evaluators_block or None,
        results=results.get("test_results"),
        error=bool(results.get("error")),
        is_public=bool(job.get("is_public")),
        share_token=job.get("share_token"),
    )
    data = response.model_dump()
    if only_failed and isinstance(data.get("results"), list):
        # `passed is None` is a pending case, not a failure — exclude it so a
        # mid-run poll doesn't report unfinished cases. Errored cases are False.
        data["results"] = [
            r for r in data["results"] if r.get("passed") is False
        ]
    return projection.apply(data)


# ============ Benchmark API ============


class BenchmarkRequest(BaseModel):
    models: List[str] = Field(
        description="Model names to benchmark",
        examples=[["openai/gpt-4.1", "anthropic/claude-sonnet-4"]],
    )
    test_uuids: Optional[List[str]] = Field(
        None,
        description="A subset of the agent's linked tests to benchmark. Each ID must be linked to the agent. Omit to run all linked tests",
        examples=[[EXAMPLE_TEST_UUID]],
    )


class ModelResult(BaseModel):
    model: str = Field(
        description="Model name these results are for",
        examples=["openai/gpt-4.1"],
    )
    success: Optional[bool] = Field(
        None,
        description="Whether this model's run succeeded",
    )
    message: str = Field(description="Status or result message for this model")
    total_tests: Optional[int] = Field(
        None, description="Total test cases for this model"
    )
    passed: Optional[int] = Field(
        None, description="Number of test cases that passed"
    )
    failed: Optional[int] = Field(
        None, description="Number of test cases that failed"
    )
    evaluator_summary: Optional[List[Dict[str, Any]]] = Field(
        None,
        description="Aggregate summary for each evaluator for this model",
    )
    test_results: Optional[List[TestCaseResult]] = Field(
        None,
        description="Results for each test case for this model",
    )
    latency_ms: Optional[Dict[str, Any]] = Field(
        None,
        description="Aggregated latency in milliseconds, as `{p50, p95, p99, count}`",
    )
    cost: Optional[Dict[str, Any]] = Field(
        None, description="Aggregated cost as `{mean, min, max, count}` (USD)"
    )
    total_tokens: Optional[Dict[str, Any]] = Field(
        None,
        description="Aggregated token usage as `{mean, min, max, count}`",
    )




class BenchmarkStatusResponse(BaseModel):
    task_id: str = Field(
        min_length=36,
        max_length=36,
        description="Benchmark run job ID",
        examples=[_EXAMPLE_TASK_UUID],
    )
    status: TaskStatus = Field(description=_TASK_STATUS_DESCRIPTION)
    test_uuids: Optional[List[str]] = Field(
        None,
        description="IDs of the tests this benchmark executed, in run order",
    )
    evaluators: Optional[List[TestRunEvaluator]] = Field(
        None,
        description="The evaluators used in this run. Each verdict in `judge_results` links to one of these by `evaluator_uuid`",
    )
    model_results: Optional[List[ModelResult]] = Field(
        None, description="Results for each model"
    )
    leaderboard_summary: Optional[List[Dict[str, Any]]] = Field(
        None,
        description=LEADERBOARD_SUMMARY_DESCRIPTION,
        examples=[LEADERBOARD_SUMMARY_EXAMPLE],
    )
    error: bool = Field(False, description="True if the run failed")
    is_public: bool = Field(False, description="Whether the run is shared publicly")
    share_token: Optional[str] = Field(
        None, description="Token for building the public share URL"
    )


def _update_benchmark_intermediate_results(
    task_id: str,
    output_dir: Path,
    models: List[str],
    test_names: List[str],
    cli_models: Optional[List[str]] = None,
) -> int:
    """
    Update intermediate results for a benchmark job.
    Returns the number of models with completed results.

    models: display names (original from frontend, e.g. "openai/gpt-4.1")
    test_names: ordered suite names. Pending rows are ``{name: ...}`` only, like unit tests.
    cli_models: names passed to CLI (may be stripped, e.g. "gpt-4.1"). Defaults to models
    """
    if cli_models is None:
        cli_models = models

    # Find all results in output directory
    all_results = _find_all_results_in_output(output_dir)
    folder_names = list(all_results.keys())

    model_results = []
    completed_count = 0

    for model, cli_model in zip(models, cli_models):
        matched_folder = _match_model_to_folder(cli_model, folder_names)

        if matched_folder and matched_folder in all_results:
            results_data, metrics_data = all_results[matched_folder]

            # Parse results
            test_results = _parse_agent_test_results(results_data)

            # Add name field for consistency
            for i, r in enumerate(test_results):
                if not r.get("name") and results_data and i < len(results_data):
                    test_case = results_data[i].get("test_case", {})
                    r["name"] = test_case.get("name")

            merged = (
                _merge_test_results_by_test_names(test_names, test_results)
                if test_names
                else test_results
            )

            if metrics_data:
                total = metrics_data.get("total", 0)
                passed = metrics_data.get("passed", 0)
                evaluator_summary = _build_evaluator_summary(metrics_data)
                model_results.append(
                    {
                        "model": model,
                        "success": True,
                        "message": f"Completed",
                        "total_tests": total,
                        "passed": passed,
                        "failed": total - passed,
                        "evaluator_summary": evaluator_summary,
                        "latency_ms": metrics_data.get("latency_ms"),
                        "cost": metrics_data.get("cost"),
                        "total_tokens": metrics_data.get("total_tokens"),
                        "test_results": merged,
                    }
                )
                completed_count += 1
            elif test_results:
                # Has partial results but no metrics yet
                total = len(merged) if test_names else len(test_results)
                passed = sum(1 for r in merged if r.get("passed", False))
                model_results.append(
                    {
                        "model": model,
                        "success": None,
                        "message": f"Running... ({len(test_results)} tests done)",
                        "total_tests": total,
                        "passed": passed,
                        "failed": total - passed,
                        "evaluator_summary": None,
                        "test_results": merged,
                    }
                )
            else:
                # No results yet for this model
                model_results.append(
                    {
                        "model": model,
                        "success": None,
                        "message": "Queued...",
                        "total_tests": None,
                        "passed": None,
                        "failed": None,
                        "evaluator_summary": None,
                        "test_results": (
                            _merge_test_results_by_test_names(test_names, [])
                            if test_names
                            else None
                        ),
                    }
                )
        else:
            # No folder found for this model yet
            model_results.append(
                {
                    "model": model,
                    "success": None,
                    "message": "Queued...",
                    "total_tests": None,
                    "passed": None,
                    "failed": None,
                    "evaluator_summary": None,
                    "test_results": (
                        _merge_test_results_by_test_names(test_names, [])
                        if test_names
                        else None
                    ),
                }
            )

    update_agent_test_job(
        task_id,
        results={"model_results": model_results},
    )

    return completed_count


def run_benchmark_task(
    task_id: str,
    agent: Dict[str, Any],
    tests: List[Dict[str, Any]],
    models: List[str],
    s3_bucket: str,
):
    """Run the benchmark for multiple models using a single CLI command with intermediate updates.

    The calibrate CLI handles parallelization internally and generates the leaderboard.
    """
    try:
        logger.info(
            f"Running benchmark task {task_id} for agent {agent['uuid']} "
            f"with {len(tests)} test(s) and {len(models)} model(s)"
        )

        test_names = [t.get("name") for t in tests if t.get("name")]

        # Initialize with pending model results (per-model test list like unit tests)
        update_agent_test_job(
            task_id,
            status=TaskStatus.IN_PROGRESS.value,
            results={
                "model_results": _benchmark_queued_model_results(models, test_names)
            },
        )

        s3 = get_s3_client()

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            try:
                calibrate_config = _calibrate_config_from_agent_test_job(
                    task_id, agent, tests
                )
                agent_config = agent.get("config") or {}

                # Clear any model from config — models are passed via CLI flags
                if "params" in calibrate_config:
                    calibrate_config["params"] = {}

                # Create directories
                input_dir = temp_path / "input"
                output_dir = temp_path / "output"
                input_dir.mkdir(parents=True, exist_ok=True)
                output_dir.mkdir(parents=True, exist_ok=True)

                # Write config file
                config_file = input_dir / "test_config.json"
                with open(config_file, "w", encoding="utf-8") as f:
                    json.dump(calibrate_config, f, indent=2)

                if agent_config.get("agent_url"):
                    # Agent connection mode: -m {models} but no -p
                    # Calibrate sends model in each request body; agent routes internally
                    # Frontend always sends models in openrouter format (provider/model).
                    # Strip the provider prefix for non-openrouter providers so the
                    # agent receives just the model name (e.g. "gpt-4.1" not "openai/gpt-4.1").
                    benchmark_provider = agent_config.get(
                        "benchmark_provider", "openrouter"
                    )
                    if benchmark_provider != "openrouter":
                        cli_models = [
                            m.split("/", 1)[-1] if "/" in m else m for m in models
                        ]
                    else:
                        cli_models = models
                    run_cmd = (
                        [get_calibrate_agent_cli(), "llm", "-c", str(config_file), "-m"]
                        + cli_models
                        + ["-o", str(output_dir), "--skip-verify"]
                    )
                else:
                    # Calibrate agent mode: -m {models} -p {provider}
                    llm_config = agent_config.get("llm", {})
                    provider = llm_config.get("provider", "openrouter")
                    cli_models = models
                    run_cmd = (
                        [get_calibrate_agent_cli(), "llm", "-c", str(config_file), "-m"]
                        + cli_models
                        + ["-p", provider, "-o", str(output_dir)]
                    )

                # Test-case parallelism is left to calibrate (CALIBRATE_TEST_PARALLEL
                # env / default 4); the subprocess inherits this process's env.

                logger.info(f"Running benchmark command: {' '.join(run_cmd)}")

                # Create temp files for stdout/stderr
                stdout_path = output_dir / "stdout.log"
                stderr_path = output_dir / "stderr.log"

                with (
                    open(stdout_path, "w") as stdout_f,
                    open(stderr_path, "w") as stderr_f,
                ):
                    process = subprocess.Popen(
                        run_cmd,
                        stdout=stdout_f,
                        stderr=stderr_f,
                        text=True,
                        start_new_session=True,
                        cwd=str(temp_path),
                    )

                    # Poll for process completion while updating intermediate results
                    prev_completed = 0
                    while process.poll() is None:
                        completed = _update_benchmark_intermediate_results(
                            task_id, output_dir, models, test_names, cli_models
                        )
                        if completed != prev_completed:
                            logger.info(
                                f"Benchmark {task_id}: {completed}/{len(models)} models completed"
                            )
                            prev_completed = completed
                        time.sleep(2)  # Poll every 2 seconds

                    # Final update after process completes
                    _update_benchmark_intermediate_results(
                        task_id, output_dir, models, test_names, cli_models
                    )

                # Read stdout/stderr
                with open(stdout_path, "r") as f:
                    stdout = f.read()
                with open(stderr_path, "r") as f:
                    stderr = f.read()

                if stdout:
                    logger.info(f"Benchmark stdout: {stdout}")
                if stderr:
                    logger.info(f"Benchmark stderr: {stderr}")

                if process.returncode != 0:
                    error_msg = f"Benchmark failed with exit code {process.returncode}: {stderr}"
                    logger.error(error_msg)
                    capture_exception_to_sentry(RuntimeError(error_msg))
                    raise subprocess.CalledProcessError(
                        process.returncode, run_cmd, stdout, stderr
                    )

                logger.info("Benchmark command completed successfully")

                # Log output directory contents for debugging
                logger.info(
                    f"Output directory contents: {[f.name for f in output_dir.iterdir()]}"
                )

                # Read results for each model from output directory
                all_results = _find_all_results_in_output(output_dir)

                if not all_results:
                    error_msg = f"Benchmark produced no output files (no results.json/metrics.json found in {output_dir})"
                    logger.error(error_msg)
                    capture_exception_to_sentry(RuntimeError(error_msg))
                    raise subprocess.CalledProcessError(0, run_cmd, stdout, stderr)
                folder_names = list(all_results.keys())
                logger.info(f"Found result folders: {folder_names}")

                model_results = []
                for model, cli_model in zip(models, cli_models):
                    matched_folder = _match_model_to_folder(cli_model, folder_names)

                    if matched_folder and matched_folder in all_results:
                        results_data, metrics_data = all_results[matched_folder]

                        # Parse results
                        test_results = _parse_agent_test_results(results_data)

                        # Add name field for consistency
                        for i, r in enumerate(test_results):
                            if (
                                not r.get("name")
                                and results_data
                                and i < len(results_data)
                            ):
                                test_case = results_data[i].get("test_case", {})
                                r["name"] = test_case.get("name")

                        if test_names:
                            test_results = _merge_test_results_by_test_names(
                                test_names, test_results
                            )

                        if metrics_data:
                            total = metrics_data.get("total", 0)
                            passed = metrics_data.get("passed", 0)
                            evaluator_summary = _build_evaluator_summary(metrics_data)
                            # Store raw parsed dicts (not a validated ``ModelResult``):
                            # calibrate emits ``judge_results`` as a dict keyed by
                            # evaluator name, but ``TestCaseResult.judge_results`` is
                            # typed ``List[JudgeResult]``. Validating at write time
                            # would raise; the dict→list conversion happens at read
                            # time in ``_enrich_model_results_with_evaluators``.
                            model_results.append(
                                {
                                    "model": model,
                                    "success": True,
                                    "message": f"Benchmark completed successfully for {model}",
                                    "total_tests": total,
                                    "passed": passed,
                                    "failed": total - passed,
                                    "evaluator_summary": evaluator_summary,
                                    "latency_ms": metrics_data.get("latency_ms"),
                                    "cost": metrics_data.get("cost"),
                                    "total_tokens": metrics_data.get("total_tokens"),
                                    "test_results": test_results,
                                }
                            )
                        else:
                            # No metrics but has results - compute from results
                            total = len(test_results) if test_results else 0
                            passed = sum(
                                1 for r in test_results if r.get("passed", False)
                            )
                            model_results.append(
                                {
                                    "model": model,
                                    "success": True,
                                    "message": f"Benchmark completed for {model}",
                                    "total_tests": total,
                                    "passed": passed,
                                    "failed": total - passed,
                                    "evaluator_summary": None,
                                    "test_results": test_results,
                                }
                            )
                    else:
                        logger.warning(f"No output found for model {model}")
                        model_results.append(
                            {
                                "model": model,
                                "success": False,
                                "message": f"No output found for model {model}",
                                "test_results": (
                                    _merge_test_results_by_test_names(test_names, [])
                                    if test_names
                                    else None
                                ),
                            }
                        )

                # Read leaderboard from output directory
                leaderboard_dir = output_dir / "leaderboard"
                leaderboard_summary = None
                if leaderboard_dir.exists():
                    logger.info(f"Leaderboard directory exists: {leaderboard_dir}")
                    leaderboard_summary = _read_leaderboard_csv(
                        leaderboard_dir, models=models
                    )

                    # Upload leaderboard to S3
                    results_prefix = f"agent-tests/benchmarks/{task_id}"
                    for root, dirs, files in os.walk(leaderboard_dir):
                        for file in files:
                            local_file_path = Path(root) / file
                            relative_path = local_file_path.relative_to(leaderboard_dir)
                            s3_key = f"{results_prefix}/leaderboard/{relative_path}"
                            upload_file_to_s3(s3, local_file_path, s3_bucket, s3_key)
                else:
                    logger.warning(
                        f"Leaderboard directory does not exist: {leaderboard_dir}"
                    )

                results_prefix = f"agent-tests/benchmarks/{task_id}"

                # Upload output directory to S3 (whole-run ``logs``, per-model logs/results.log, CSV/JSON, etc.)
                upload_directory_tree_to_s3(
                    s3,
                    output_dir,
                    s3_bucket,
                    f"{results_prefix}/outputs",
                )

                logger.info(
                    f"Uploaded benchmark outputs to s3://{s3_bucket}/{results_prefix}/outputs/"
                )

                # Create and upload benchmark config file to S3
                benchmark_config = {
                    **calibrate_config,
                    "models": models,
                }
                config_s3_key = f"{results_prefix}/benchmark_config.json"
                with open(config_file, "w", encoding="utf-8") as f:
                    json.dump(benchmark_config, f, indent=2)
                upload_file_to_s3(s3, config_file, s3_bucket, config_s3_key)
                logger.info(f"Uploaded benchmark config file to S3: {config_s3_key}")

                # Check if all models succeeded
                all_succeeded = all(r["success"] for r in model_results)
                final_status = (
                    TaskStatus.DONE.value if all_succeeded else TaskStatus.FAILED.value
                )

                error_msg = None
                if not all_succeeded:
                    failed = [r["model"] for r in model_results if not r["success"]]
                    error_msg = f"Some models failed: {', '.join(failed)}"

                # Update job with results
                update_agent_test_job(
                    task_id,
                    status=final_status,
                    results={
                        "model_results": model_results,
                        "leaderboard_summary": leaderboard_summary,
                        "results_s3_prefix": results_prefix,
                        "error": error_msg,
                    },
                )

                logger.info(
                    f"Benchmark task {task_id} completed, status={final_status}"
                )

            except subprocess.CalledProcessError as e:
                traceback.print_exc()
                capture_exception_to_sentry(e)
                failed_results: Dict[str, Any] = {
                    "error": f"Benchmark failed: {e.stderr if hasattr(e, 'stderr') else str(e)}",
                }
                try:
                    if output_dir.exists():
                        bp = f"agent-tests/benchmarks/{task_id}"
                        upload_directory_tree_to_s3(
                            s3, output_dir, s3_bucket, f"{bp}/outputs"
                        )
                        failed_results["results_s3_prefix"] = bp
                except Exception:
                    pass
                update_agent_test_job(
                    task_id,
                    status=TaskStatus.FAILED.value,
                    results=failed_results,
                )
            except Exception as e:
                traceback.print_exc()
                capture_exception_to_sentry(e)
                # Preserve any existing results from the job
                existing_job = get_agent_test_job(task_id)
                existing_results = (
                    (existing_job.get("results") or {}) if existing_job else {}
                )
                existing_results["error"] = (
                    f"Unexpected error during benchmark: {str(e)}"
                )
                try:
                    if output_dir.exists():
                        bp = f"agent-tests/benchmarks/{task_id}"
                        upload_directory_tree_to_s3(
                            s3, output_dir, s3_bucket, f"{bp}/outputs"
                        )
                        existing_results["results_s3_prefix"] = bp
                except Exception:
                    pass
                update_agent_test_job(
                    task_id,
                    status=TaskStatus.FAILED.value,
                    results=existing_results,
                )

    except Exception as e:
        traceback.print_exc()
        capture_exception_to_sentry(e)
        # Preserve any existing results from the job
        existing_job = get_agent_test_job(task_id)
        existing_results = (existing_job.get("results") or {}) if existing_job else {}
        existing_results["error"] = f"Task failed: {str(e)}"
        update_agent_test_job(
            task_id,
            status=TaskStatus.FAILED.value,
            results=existing_results,
        )
    finally:
        # Try to start the next queued job
        try_start_queued_agent_test_job(AGENT_TEST_JOB_TYPES)


@router.post(
    "/agent/{agent_uuid}/benchmark",
    response_model=AgentTestRunCreateResponse,
    summary="Run agent benchmark",
    tags=["Public API"],
)
async def run_agent_benchmark(
    agent_uuid: str = PathParam(
        description="Agent to benchmark",
        examples=[_EXAMPLE_AGENT_UUID],
    ),
    request: BenchmarkRequest = ...,
    ctx: OrgContext = Depends(get_org_jwt_or_api_key),
):
    """Run a multi-model benchmark on an agent's linked tests as a background job."""
    # Verify agent exists and belongs to the caller's org.
    agent = get_agent(agent_uuid)
    if not agent or agent.get("org_uuid") != ctx.org_uuid:
        raise HTTPException(status_code=404, detail="Agent not found")

    if not request.models:
        raise HTTPException(status_code=400, detail="At least one model is required")

    # Guard: for agent connection mode, verify each requested model is verified
    agent_config = agent.get("config") or {}
    if agent_config.get("agent_url"):
        benchmark_verified = agent_config.get("benchmark_models_verified") or {}
        unverified = [
            m
            for m in request.models
            if not benchmark_verified.get(m, {}).get("verified")
        ]
        if unverified:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"The following models are not verified for this agent connection: "
                    f"{', '.join(unverified)}. "
                    f"Call POST /agents/{agent_uuid}/verify-connection with each model first."
                ),
            )

    # Benchmarks run the agent's linked tests, optionally narrowed to a subset.
    linked_tests = get_tests_for_agent(agent_uuid)
    if not linked_tests:
        raise HTTPException(
            status_code=400,
            detail="No tests linked to this agent. Link tests first.",
        )

    if request.test_uuids:
        # Scope to the requested subset; every uuid must be linked to this agent.
        linked_by_uuid = {t["uuid"]: t for t in linked_tests}
        unknown = [u for u in request.test_uuids if u not in linked_by_uuid]
        if unknown:
            raise HTTPException(
                status_code=404,
                detail=(
                    "The following tests are not linked to this agent: "
                    f"{', '.join(unknown)}."
                ),
            )
        # Preserve request order while de-duplicating.
        seen: set = set()
        tests = []
        for u in request.test_uuids:
            if u not in seen:
                seen.add(u)
                tests.append(linked_by_uuid[u])
    else:
        tests = linked_tests

    # Get S3 configuration
    try:
        s3_bucket = get_s3_output_config()
    except ValueError as e:
        raise HTTPException(status_code=500, detail=str(e))

    # Per-org limit check uses the agent's org.
    org_uuid = agent.get("org_uuid")

    can_start = can_start_agent_test_job(AGENT_TEST_JOB_TYPES, org_uuid)
    initial_status = (
        TaskStatus.IN_PROGRESS.value if can_start else TaskStatus.QUEUED.value
    )

    # Extract test names for progress tracking
    test_names = [test.get("name") for test in tests if test.get("name")]

    calibrate_config, evaluators_by_test_id = _build_calibrate_config(agent, tests)

    # Create job in database with details for recovery
    test_uuids = [t["uuid"] for t in tests]
    job_id = create_agent_test_job(
        agent_id=agent_uuid,
        job_type="llm-benchmark",
        status=initial_status,
        details={
            "agent_uuid": agent_uuid,
            "test_uuids": test_uuids,
            "test_names": test_names,
            "models": request.models,
            "s3_bucket": s3_bucket,
            "calibrate_config": calibrate_config,
            "evaluators_by_test_id": evaluators_by_test_id,
        },
        results={
            "model_results": _benchmark_queued_model_results(request.models, test_names)
        },
    )

    if can_start:
        # Start background task
        thread = threading.Thread(
            target=run_benchmark_task,
            args=(job_id, agent, tests, request.models, s3_bucket),
            daemon=True,
        )
        thread.start()
        logger.info(f"Started LLM benchmark job {job_id} immediately")
    else:
        logger.info(f"Queued LLM benchmark job {job_id}")

    return AgentTestRunCreateResponse(task_id=job_id, status=initial_status)


@router.patch(
    "/benchmark/{task_id}/visibility",
    response_model=VisibilityResponse,
    summary="Update benchmark visibility",
)
async def update_benchmark_visibility(
    task_id: str = PathParam(
        description="Benchmark run whose sharing settings to update",
        examples=[_EXAMPLE_TASK_UUID],
    ),
    body: VisibilityRequest = ...,
    ctx: OrgContext = Depends(get_current_org),
):
    """Toggle public sharing for a benchmark run."""
    job = _load_owned_agent_test_job(task_id, ctx)

    if body.is_public:
        import uuid as _uuid

        share_token = job.get("share_token") or str(_uuid.uuid4())
    else:
        share_token = None

    update_agent_test_job_visibility(task_id, body.is_public, share_token)
    return VisibilityResponse(is_public=body.is_public, share_token=share_token)


_BenchmarkProjection = make_projection_params(
    heavy_fields=[
        "model_results[].test_results",
        "evaluators[].output_config",
    ]
)


@router.get(
    "/benchmark/{task_id}",
    response_model=BenchmarkStatusResponse,
    summary="Get benchmark status",
    tags=["Public API"],
)
async def get_benchmark_status(
    task_id: str = PathParam(
        description="Benchmark run to poll for status and results",
        examples=[_EXAMPLE_TASK_UUID],
    ),
    ctx: OrgContext = Depends(get_org_jwt_or_api_key),
    only_failed: bool = Query(
        False,
        description="Return only failing test cases for each model. Omit to return every case",
    ),
    projection: _BenchmarkProjection = Depends(),
):
    """Get the results of a benchmark run"""
    job = _load_owned_agent_test_job(task_id, ctx)

    status = job["status"]
    results = job.get("results") or {}
    details = job.get("details") or {}

    # Check for timeout on in-progress jobs
    # if status == TaskStatus.IN_PROGRESS.value:
    #     updated_at = job.get("updated_at")
    #     if updated_at and is_job_timed_out(updated_at):
    #         logger.warning(f"Benchmark job {task_id} timed out, marking as failed")

    #         # Mark job as failed (preserve existing results, add error)
    #         results["error"] = "Job timed out after 5 minutes of inactivity"
    #         update_agent_test_job(
    #             task_id,
    #             status=TaskStatus.FAILED.value,
    #             results=results,
    #         )
    #         status = TaskStatus.FAILED.value

    #         # Try to start the next queued job
    #         try_start_queued_agent_test_job(AGENT_TEST_JOB_TYPES)

    evaluators_snapshot = details.get("evaluators_by_test_id") or {}
    evaluator_cache: Dict[str, Optional[Dict[str, Any]]] = {}
    _enrich_model_results_with_evaluators(
        results.get("model_results"), evaluators_snapshot, evaluator_cache
    )
    evaluators_block = _build_evaluators_block_for_test_run(
        evaluators_snapshot,
        model_results=results.get("model_results"),
        evaluator_cache=evaluator_cache,
    )

    response = BenchmarkStatusResponse(
        task_id=task_id,
        status=status,
        test_uuids=details.get("test_uuids") or None,
        evaluators=evaluators_block or None,
        model_results=results.get("model_results"),
        leaderboard_summary=results.get("leaderboard_summary"),
        error=bool(results.get("error")),
        is_public=bool(job.get("is_public")),
        share_token=job.get("share_token"),
    )
    data = response.model_dump()
    if only_failed and isinstance(data.get("model_results"), list):
        for model in data["model_results"]:
            if isinstance(model, dict) and isinstance(
                model.get("test_results"), list
            ):
                # `passed is None` is pending, not a failure; errored is False.
                model["test_results"] = [
                    r for r in model["test_results"] if r.get("passed") is False
                ]
    return projection.apply(data)


@router.delete("/job/{job_uuid}", summary="Delete test job")
async def delete_agent_test_job_endpoint(
    job_uuid: str = PathParam(
        description="Test or benchmark job to delete",
        examples=[_EXAMPLE_TASK_UUID],
    ),
    ctx: OrgContext = Depends(get_current_org),
):
    """Delete an agent test or benchmark job."""
    job = get_agent_test_job(job_uuid)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    agent_id = job.get("agent_id")
    if agent_id:
        agent = get_agent(agent_id)
        if not agent or agent.get("org_uuid") != ctx.org_uuid:
            raise HTTPException(status_code=404, detail="Job not found")
    else:
        raise HTTPException(status_code=404, detail="Job not found")

    # Check if this was a running job (to trigger next queued job after delete)
    was_running = job.get("status") == TaskStatus.IN_PROGRESS.value

    # Delete the job
    deleted = delete_agent_test_job(job_uuid)
    if not deleted:
        raise HTTPException(status_code=404, detail="Job not found")

    # If the deleted job was running, try to start the next queued job
    if was_running:
        try_start_queued_agent_test_job(AGENT_TEST_JOB_TYPES)

    return {"message": "Agent test job deleted successfully"}
