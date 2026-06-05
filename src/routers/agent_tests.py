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

from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
from sqlite3 import IntegrityError

from db import (
    add_test_to_agent,
    remove_test_from_agent,
    bulk_remove_tests_from_agent,
    bulk_delete_tests,
    get_tests_for_agent,
    get_agents_for_test,
    get_agent_test_link,
    get_all_agent_tests,
    get_agent,
    get_test,
    get_tools_for_agent,
    get_evaluators_for_test,
    get_evaluator_by_slug,
    get_evaluator_version,
    create_agent_test_job,
    get_agent_test_job,
    update_agent_test_job,
    update_agent_test_job_visibility,
    get_agent_test_jobs_for_agent,
    get_agent_test_jobs_for_org,
    delete_agent_test_job,
)
from llm_judge import build_test_evaluators_payload, evaluator_value_name
from auth_utils import get_current_org, get_org_jwt_or_api_key, OrgContext
from utils import (
    TaskStatus,
    TaskCreateResponse,
    get_s3_client,
    get_s3_output_config,
    can_start_agent_test_job,
    try_start_queued_agent_test_job,
    register_job_starter,
    is_job_timed_out,
    capture_exception_to_sentry,
    build_tool_configs,
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


class AgentTestsCreate(BaseModel):
    agent_uuid: str
    test_uuids: List[str]


class AgentTestDelete(BaseModel):
    agent_uuid: str
    test_uuid: str


class AgentTestResponse(BaseModel):
    id: int
    agent_id: str
    test_id: str
    created_at: str


class AgentTestsCreateResponse(BaseModel):
    ids: List[int]
    message: str


class TestResponse(BaseModel):
    uuid: str
    name: str
    type: str
    config: Dict[str, Any] | None = None
    created_at: str
    updated_at: str


class AgentResponse(BaseModel):
    uuid: str
    name: str
    type: Literal["agent", "connection"]
    config: Dict[str, Any] | None = None
    created_at: str
    updated_at: str


class RunTestRequest(BaseModel):
    test_uuids: Optional[List[str]] = None


class ToolCallOutput(BaseModel):
    tool: str
    arguments: Optional[Dict[str, Any]] = None
    # Tool execution result, surfaced only for agent-connection tests where the
    # external agent actually runs the tool and echoes its return value. Any
    # JSON value (object/list/string/number/...). Absent for calibrate-agent
    # mode (tools are declared, never executed) and for agents that don't echo
    # it. calibrate passes this through verbatim in `output.tool_calls`; without
    # this field the response_model would drop it on serialization.
    output: Optional[Any] = None


class TestOutput(BaseModel):
    response: Optional[str] = None
    tool_calls: Optional[List[ToolCallOutput]] = None


class JudgeResult(BaseModel):
    """One evaluator's verdict for a response-type test case.

    Per-row data only — anything constant across rows for the same evaluator
    (name, description, output_type, output_config, scale_min/max) lives on
    the response-level `evaluators[]` block; look it up by `evaluator_uuid`.

    `evaluator_uuid` is None for legacy runs that pre-date the evaluator-
    snapshot capture or when the evaluator can't be resolved from the
    snapshot.

    Exactly one of `match` (binary) / `score` (rating) is set per entry;
    both are None for tool-call tests, but tool-call tests don't carry
    `judge_results`.

    `variable_values` are the {{var}} substitutions used for this evaluator
    on this test case, frozen from `test_evaluators.variable_values` at
    submission time — stays on the row because it varies per test case.

    `value_name` is the human-readable label for `match`/`score` resolved
    against the rubric the run actually used (snapshot's
    `output_config.scale.name`). Falls back to `Correct`/`Wrong` for binary
    or the stringified score for rating when the snapshot lacks named
    scale entries (e.g. legacy runs captured before the rubric was
    snapshotted).
    """

    evaluator_uuid: Optional[str] = None
    reasoning: Optional[str] = None
    match: Optional[bool] = None
    score: Optional[float] = None
    value_name: Optional[str] = None
    variable_values: Optional[Dict[str, Any]] = None


class TestCaseResult(BaseModel):
    """Result for a single test case matching calibrate results.json structure"""

    test_case_id: Optional[str] = None
    name: Optional[str] = None  # Test name, present during in-progress and done states
    passed: Optional[bool] = None  # Only present when done
    reasoning: Optional[str] = (
        None  # LLM judge reasoning or deterministic diff; null for passing tool call tests
    )
    output: Optional[TestOutput] = None  # Only present when done
    test_case: Optional[Dict[str, Any]] = None  # Only present when done
    # Per-evaluator verdicts for response-type tests; None for tool-call tests or
    # in-progress rows. Names reflect the current DB value (refreshed on each read).
    judge_results: Optional[List[JudgeResult]] = None


class TestRunStatusResponse(BaseModel):
    task_id: str
    status: str
    total_tests: Optional[int] = None
    passed: Optional[int] = None
    failed: Optional[int] = None
    # Top-level evaluator block — name/description/output_type/rubric
    # shared across every judge_results row. Per-row entries reference
    # back via `evaluator_uuid` so the rubric isn't duplicated per test.
    evaluators: Optional[List[Dict[str, Any]]] = None
    results: Optional[List[TestCaseResult]] = None
    results_s3_prefix: Optional[str] = None
    error: bool = False
    is_public: bool = False
    share_token: Optional[str] = None


class AgentTestRunListItem(BaseModel):
    uuid: str
    name: str  # Format: "Run {index}" or "Benchmark {index}"
    status: str
    type: str
    updated_at: str
    # Top-level evaluator block — see TestRunStatusResponse.evaluators.
    evaluators: Optional[List[Dict[str, Any]]] = None
    # Unit test results (for llm-unit-test type)
    total_tests: Optional[int] = None
    passed: Optional[int] = None
    failed: Optional[int] = None
    results: Optional[List[TestCaseResult]] = None
    # Benchmark results (for llm-benchmark type)
    model_results: Optional[List[Dict[str, Any]]] = None
    leaderboard_summary: Optional[List[Dict[str, Any]]] = None
    # Common fields
    results_s3_prefix: Optional[str] = None
    error: bool = False
    is_public: bool = False
    share_token: Optional[str] = None


class AgentTestRunsResponse(BaseModel):
    runs: List[AgentTestRunListItem]


class GlobalTestRunListItem(AgentTestRunListItem):
    """AgentTestRunListItem extended with agent identity for the global view."""

    agent_id: str
    agent_name: str


class GlobalTestRunsResponse(BaseModel):
    runs: List[GlobalTestRunListItem]


@router.post("", response_model=AgentTestsCreateResponse)
async def create_agent_test_links(agent_tests: AgentTestsCreate):
    """Add tests to an agent."""
    # Verify agent exists
    agent = get_agent(agent_tests.agent_uuid)
    if not agent:
        raise HTTPException(status_code=404, detail="Agent not found")

    # Verify all tests exist
    for test_uuid in agent_tests.test_uuids:
        test = get_test(test_uuid)
        if not test:
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


@router.get("", response_model=List[AgentTestResponse])
async def list_agent_tests():
    """List all agent-test links."""
    links = get_all_agent_tests()
    return links


@router.get("/agent/{agent_uuid}/tests", response_model=List[TestResponse])
async def get_agent_tests_endpoint(agent_uuid: str):
    """Get all tests for a specific agent."""
    # Verify agent exists
    agent = get_agent(agent_uuid)
    if not agent:
        raise HTTPException(status_code=404, detail="Agent not found")

    tests = get_tests_for_agent(agent_uuid)
    return tests


@router.get("/agent/{agent_uuid}/runs", response_model=AgentTestRunsResponse)
async def get_agent_test_runs(agent_uuid: str):
    """
    Get all test runs for an agent.

    Returns a list of all test runs (unit tests and benchmarks) with their UUID, status, type, name, and results.
    """
    # Verify agent exists
    agent = get_agent(agent_uuid)
    if not agent:
        raise HTTPException(status_code=404, detail="Agent not found")

    # Get all jobs for this agent
    jobs = get_agent_test_jobs_for_agent(agent_uuid)

    # Group jobs by type to generate run names
    unit_test_count = 0
    benchmark_count = 0

    runs = []
    for job in jobs:
        job_type = job.get("type", "")
        if job_type == "llm-unit-test":
            unit_test_count += 1
            name = f"Run {unit_test_count}"
        elif job_type == "llm-benchmark":
            benchmark_count += 1
            name = f"Benchmark {benchmark_count}"
        else:
            name = f"Job {len(runs) + 1}"

        # Extract results from job
        job_results = job.get("results") or {}
        job_details = job.get("details") or {}
        evaluators_snapshot = (
            job_details.get("evaluators_by_test_id")
            or {}
        )

        # Refresh evaluator names + uuids on per-row judge_results before serializing
        _evaluator_cache: Dict[str, Optional[Dict[str, Any]]] = {}
        _enrich_test_results_with_evaluators(
            job_results.get("test_results"),
            evaluators_snapshot,
            _evaluator_cache,
        )
        _enrich_model_results_with_evaluators(
            job_results.get("model_results"),
            evaluators_snapshot,
            _evaluator_cache,
        )
        evaluators_block = _build_evaluators_block_for_test_run(
            evaluators_snapshot,
            test_results=job_results.get("test_results"),
            model_results=job_results.get("model_results"),
            evaluator_cache=_evaluator_cache,
        )

        run_item = AgentTestRunListItem(
            uuid=job["uuid"],
            name=name,
            status=job["status"],
            type=job_type,
            updated_at=job.get("updated_at", job.get("created_at", "")),
            evaluators=evaluators_block or None,
            # Unit test results
            total_tests=job_results.get("total_tests"),
            passed=job_results.get("passed"),
            failed=job_results.get("failed"),
            results=job_results.get("test_results"),
            # Benchmark results
            model_results=job_results.get("model_results"),
            leaderboard_summary=job_results.get("leaderboard_summary"),
            # Common fields
            results_s3_prefix=job_results.get("results_s3_prefix"),
            error=bool(job_results.get("error")),
            is_public=bool(job.get("is_public")),
            share_token=job.get("share_token"),
        )
        runs.append(run_item)

    return AgentTestRunsResponse(runs=runs)


@router.get("/runs", response_model=GlobalTestRunsResponse)
async def get_all_test_runs_for_user(
    ctx: OrgContext = Depends(get_current_org),
    type: Optional[str] = None,
):
    """
    Get all test runs (unit tests and benchmarks) across every agent in the
    caller's current org.

    Optional query param:
      ?type=llm-unit-test   — return only unit-test runs
      ?type=llm-benchmark   — return only benchmark runs

    Results are ordered newest-updated-first. Each item includes ``agent_id``
    and ``agent_name`` so the frontend can group or label by agent.
    """
    jobs = get_agent_test_jobs_for_org(ctx.org_uuid, job_type=type)

    # Per-agent counters for naming ("Run 1", "Benchmark 2", …).
    # We need ascending order to assign names correctly, then flip back.
    jobs_asc = sorted(jobs, key=lambda j: j.get("created_at", ""))
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

    runs = []
    for job in jobs:  # already newest-first
        job_results = job.get("results") or {}
        job_details = job.get("details") or {}
        evaluators_snapshot = (
            job_details.get("evaluators_by_test_id")
            or {}
        )

        # Refresh evaluator names + uuids on per-row judge_results before serializing
        _evaluator_cache: Dict[str, Optional[Dict[str, Any]]] = {}
        _enrich_test_results_with_evaluators(
            job_results.get("test_results"),
            evaluators_snapshot,
            _evaluator_cache,
        )
        _enrich_model_results_with_evaluators(
            job_results.get("model_results"),
            evaluators_snapshot,
            _evaluator_cache,
        )
        evaluators_block = _build_evaluators_block_for_test_run(
            evaluators_snapshot,
            test_results=job_results.get("test_results"),
            model_results=job_results.get("model_results"),
            evaluator_cache=_evaluator_cache,
        )

        run_item = GlobalTestRunListItem(
            uuid=job["uuid"],
            name=name_map[job["uuid"]],
            status=job["status"],
            type=job.get("type", ""),
            updated_at=job.get("updated_at", job.get("created_at", "")),
            evaluators=evaluators_block or None,
            # Unit test fields
            total_tests=job_results.get("total_tests"),
            passed=job_results.get("passed"),
            failed=job_results.get("failed"),
            results=job_results.get("test_results"),
            # Benchmark fields
            model_results=job_results.get("model_results"),
            leaderboard_summary=job_results.get("leaderboard_summary"),
            # Common fields
            results_s3_prefix=job_results.get("results_s3_prefix"),
            error=bool(job_results.get("error")),
            is_public=bool(job.get("is_public")),
            share_token=job.get("share_token"),
            # Agent identity (global-only fields)
            agent_id=job.get("agent_id", ""),
            agent_name=job.get("agent_name", ""),
        )
        runs.append(run_item)

    return GlobalTestRunsResponse(runs=runs)


@router.get("/test/{test_uuid}/agents", response_model=List[AgentResponse])
async def get_test_agents(test_uuid: str):
    """Get all agents for a specific test."""
    # Verify test exists
    test = get_test(test_uuid)
    if not test:
        raise HTTPException(status_code=404, detail="Test not found")

    agents = get_agents_for_test(test_uuid)
    return agents


@router.delete("")
async def delete_agent_test_link(agent_test: AgentTestDelete):
    """Remove a test from an agent."""
    deleted = remove_test_from_agent(agent_test.agent_uuid, agent_test.test_uuid)
    if not deleted:
        raise HTTPException(status_code=404, detail="Agent-test link not found")
    return {"message": "Test removed from agent successfully"}


class AgentTestBulkDelete(BaseModel):
    agent_uuid: str
    test_uuids: List[str]


@router.post("/bulk-unlink")
async def bulk_delete_agent_test_links(payload: AgentTestBulkDelete):
    """Remove multiple tests from an agent at once."""
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
    agent_uuid: str
    # Required: the explicit set of test UUIDs to delete. Matches the
    # `/bulk-unlink` sibling's contract — the FE always names the targets
    # explicitly, so there's no implicit "delete every test on this agent"
    # mode. Removes a class of footgun (a buggy FE blanking out an agent)
    # and lets the response report exactly which UUIDs were processed
    # without slicing tricks.
    test_uuids: List[str]


@router.post("/bulk-delete-tests")
async def bulk_delete_agent_tests(
    payload: AgentTestsBulkDeleteAll,
    ctx: OrgContext = Depends(get_current_org),
):
    """Hard-cousin of `/bulk-unlink`: instead of just unlinking, soft-delete
    the named tests (and the link rows along with them). Only tests in the
    caller's CURRENT ORG are deleted — UUIDs from other orgs (or that aren't
    linked to this agent) are silently skipped, so this can't nuke a test
    in another org and can't be used as a reconnaissance probe for foreign
    test UUIDs.

    `agent_uuid` is a sanity scope: every requested UUID must be linked
    to this agent. Cross-agent deletes have to be sent as separate calls.
    """
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
            }
            # Snapshot the pinned version's rubric so `value_name` can be
            # resolved at read time without re-reading the version row
            # (which may have drifted). Applies to binary and rating both.
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
    (``{<calibrate_name>: {reasoning, match|score}}``) for response-type tests; it
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
    `name`/`description`/`output_type`/`output_config`/scale bounds that
    would otherwise be duplicated across every judge_results row.

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
        output_type = snap.get("output_type") or (
            ev.get("output_type") if ev else None
        )
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


def _build_evaluator_summary(metrics_data: Optional[dict]) -> Optional[List[Dict[str, Any]]]:
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
                        "calibrate",
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
                        "calibrate",
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

                if metrics_data and isinstance(metrics_data, dict):
                    total_tests = metrics_data.get("total", 0)
                    passed = metrics_data.get("passed", 0)
                    failed = total_tests - passed
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


@router.post("/agent/{agent_uuid}/run", response_model=TaskCreateResponse)
async def run_agent_test(
    agent_uuid: str,
    request: RunTestRequest,
    ctx: OrgContext = Depends(get_org_jwt_or_api_key),
):
    """
    Run one or more tests for an agent.

    This starts a background task that runs the calibrate LLM tests command
    with the agent's config and the combined test cases from all specified tests.

    Returns a task ID that can be used to poll for status and results.

    Auth: requires either a JWT (frontend) or an `sk_` API key. The agent
    must belong to the caller's org or this 404s.
    """
    # Verify agent exists and belongs to the caller's org.
    agent = get_agent(agent_uuid)
    if not agent or agent.get("org_uuid") != ctx.org_uuid:
        raise HTTPException(status_code=404, detail="Agent not found")

    # Guard: agent connection must be verified before running tests. Every test
    # type runs the agent — response/tool_call generate the reply to judge, and
    # conversation tests run in live mode too (calibrate runs the agent on the
    # `history`, appends the generated reply, then the simulation judge scores
    # the full conversation). So the guard applies uniformly.
    agent_config = agent.get("config") or {}
    if agent_config.get("agent_url") and not agent_config.get("connection_verified"):
        raise HTTPException(
            status_code=400,
            detail="Agent connection not verified. Call POST /agents/{agent_uuid}/verify-connection first.",
        )

    if request.test_uuids:
        # Verify all specified tests exist
        tests = []
        for test_uuid in request.test_uuids:
            test = get_test(test_uuid)
            if not test:
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

    # Per-org limit check uses the agent's org.
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

    return TaskCreateResponse(task_id=job_id, status=initial_status)


def _load_owned_agent_test_job(task_id: str, ctx: OrgContext) -> Dict[str, Any]:
    """Fetch an agent-test job and assert the caller's org owns it.

    Ownership is derived through the job's parent agent (``agent_test_jobs`` has
    no org column of its own). Returns the job dict on success; raises 404 with
    the same generic ``"Task not found"`` detail for the missing / cross-org /
    orphaned cases so existence is never leaked. A soft-deleted agent makes its
    runs unreadable here (``get_agent`` filters ``deleted_at``), consistent with
    the org-wide runs list. Used by the run/benchmark status and visibility
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
    is_public: bool


class VisibilityResponse(BaseModel):
    is_public: bool
    share_token: str | None = None


@router.patch("/run/{task_id}/visibility", response_model=VisibilityResponse)
async def update_test_run_visibility(
    task_id: str,
    body: VisibilityRequest,
    ctx: OrgContext = Depends(get_current_org),
):
    """Toggle public sharing for an agent test run."""
    job = _load_owned_agent_test_job(task_id, ctx)

    if body.is_public:
        import uuid as _uuid

        share_token = job.get("share_token") or str(_uuid.uuid4())
    else:
        share_token = None

    update_agent_test_job_visibility(task_id, body.is_public, share_token)
    return VisibilityResponse(is_public=body.is_public, share_token=share_token)


@router.get("/run/{task_id}", response_model=TestRunStatusResponse)
async def get_agent_test_run_status(
    task_id: str,
    ctx: OrgContext = Depends(get_org_jwt_or_api_key),
):
    """
    Get the status of an agent test run.

    Requires either a JWT (frontend) or an `sk_` API key, plus org
    ownership of the run. Unauthenticated access to a completed run is only
    possible once it is made public, via the share-token endpoint in the public
    router.

    Returns the current status and, if done, the test results.
    """
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

    return TestRunStatusResponse(
        task_id=task_id,
        status=status,
        total_tests=results.get("total_tests"),
        passed=results.get("passed"),
        failed=results.get("failed"),
        evaluators=evaluators_block or None,
        results=results.get("test_results"),
        results_s3_prefix=results.get("results_s3_prefix"),
        error=bool(results.get("error")),
        is_public=bool(job.get("is_public")),
        share_token=job.get("share_token"),
    )


# ============ Benchmark API ============


class BenchmarkRequest(BaseModel):
    models: List[str]  # List of model names to benchmark


class ModelResult(BaseModel):
    model: str
    success: Optional[bool] = None  # None while queued/processing, True/False when done
    message: str
    total_tests: Optional[int] = None
    passed: Optional[int] = None
    failed: Optional[int] = None
    evaluator_summary: Optional[List[Dict[str, Any]]] = None
    test_results: Optional[List[Dict[str, Any]]] = None


class BenchmarkStatusResponse(BaseModel):
    task_id: str
    status: str
    # Top-level evaluator block — see TestRunStatusResponse.evaluators.
    # Shared by every model's test_results since all models run the
    # same suite.
    evaluators: Optional[List[Dict[str, Any]]] = None
    model_results: Optional[List[ModelResult]] = None
    leaderboard_summary: Optional[List[Dict[str, Any]]] = None
    results_s3_prefix: Optional[str] = None
    error: bool = False
    is_public: bool = False
    share_token: Optional[str] = None


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
    test_names: ordered suite names; pending rows are ``{name: ...}`` only, like unit tests.
    cli_models: names passed to CLI (may be stripped, e.g. "gpt-4.1"); defaults to models
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
            results={"model_results": _benchmark_queued_model_results(models, test_names)},
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
                        ["calibrate", "llm", "-c", str(config_file), "-m"]
                        + cli_models
                        + ["-o", str(output_dir), "--skip-verify"]
                    )
                else:
                    # Calibrate agent mode: -m {models} -p {provider}
                    llm_config = agent_config.get("llm", {})
                    provider = llm_config.get("provider", "openrouter")
                    cli_models = models
                    run_cmd = (
                        ["calibrate", "llm", "-c", str(config_file), "-m"]
                        + cli_models
                        + ["-p", provider, "-o", str(output_dir)]
                    )

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
                            model_results.append(
                                ModelResult(
                                    model=model,
                                    success=True,
                                    message=f"Benchmark completed successfully for {model}",
                                    total_tests=total,
                                    passed=passed,
                                    failed=total - passed,
                                    evaluator_summary=evaluator_summary,
                                    test_results=test_results,
                                )
                            )
                        else:
                            # No metrics but has results - compute from results
                            total = len(test_results) if test_results else 0
                            passed = sum(
                                1 for r in test_results if r.get("passed", False)
                            )
                            model_results.append(
                                ModelResult(
                                    model=model,
                                    success=True,
                                    message=f"Benchmark completed for {model}",
                                    total_tests=total,
                                    passed=passed,
                                    failed=total - passed,
                                    evaluator_summary=None,
                                    test_results=test_results,
                                )
                            )
                    else:
                        logger.warning(f"No output found for model {model}")
                        model_results.append(
                            ModelResult(
                                model=model,
                                success=False,
                                message=f"No output found for model {model}",
                                test_results=_merge_test_results_by_test_names(
                                    test_names, []
                                )
                                if test_names
                                else None,
                            )
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
                all_succeeded = all(r.success for r in model_results)
                final_status = (
                    TaskStatus.DONE.value if all_succeeded else TaskStatus.FAILED.value
                )

                error_msg = None
                if not all_succeeded:
                    failed = [r.model for r in model_results if not r.success]
                    error_msg = f"Some models failed: {', '.join(failed)}"

                # Update job with results
                update_agent_test_job(
                    task_id,
                    status=final_status,
                    results={
                        "model_results": [r.model_dump() for r in model_results],
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


@router.post("/agent/{agent_uuid}/benchmark", response_model=TaskCreateResponse)
async def run_agent_benchmark(
    agent_uuid: str,
    request: BenchmarkRequest,
    ctx: OrgContext = Depends(get_current_org),
):
    """
    Run a benchmark comparing multiple models on the same tests.

    This starts a background task that runs the calibrate LLM tests command
    for each model in parallel, then generates a leaderboard comparing results.

    Returns a task ID that can be used to poll for status and results.
    """
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

    # Benchmarks always run all tests linked to the agent
    tests = get_tests_for_agent(agent_uuid)
    if not tests:
        raise HTTPException(
            status_code=400,
            detail="No tests linked to this agent. Link tests first.",
        )

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
            "model_results": _benchmark_queued_model_results(
                request.models, test_names
            )
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

    return TaskCreateResponse(task_id=job_id, status=initial_status)


@router.patch("/benchmark/{task_id}/visibility", response_model=VisibilityResponse)
async def update_benchmark_visibility(
    task_id: str,
    body: VisibilityRequest,
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


@router.get("/benchmark/{task_id}", response_model=BenchmarkStatusResponse)
async def get_benchmark_status(
    task_id: str,
    ctx: OrgContext = Depends(get_current_org),
):
    """
    Get the status of a benchmark run.

    Requires a valid JWT and org ownership of the run. Unauthenticated access
    to a completed run is only possible once it is made public, via the
    share-token endpoint in the public router.

    Returns the current status and, if done, results for each model and leaderboard.
    """
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

    return BenchmarkStatusResponse(
        task_id=task_id,
        status=status,
        evaluators=evaluators_block or None,
        model_results=results.get("model_results"),
        leaderboard_summary=results.get("leaderboard_summary"),
        results_s3_prefix=results.get("results_s3_prefix"),
        error=bool(results.get("error")),
        is_public=bool(job.get("is_public")),
        share_token=job.get("share_token"),
    )


@router.delete("/job/{job_uuid}")
async def delete_agent_test_job_endpoint(
    job_uuid: str, ctx: OrgContext = Depends(get_current_org)
):
    """Delete an agent test job. Only members of the parent agent's org can delete."""
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
