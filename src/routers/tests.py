from typing import ClassVar, Optional, List, Dict, Any, Literal
from fastapi import APIRouter, HTTPException, Depends, Path
from pydantic import BaseModel, ConfigDict, Field, model_validator

from db import (
    create_test,
    ensure_name_unique,
    get_test,
    get_all_tests,
    update_test,
    delete_test,
    bulk_create_tests,
    bulk_delete_tests,
    get_agent,
    add_test_to_agent,
    get_evaluator,
    get_evaluators_for_test,
    set_test_evaluators,
)
from auth_utils import get_current_org, get_org_jwt_or_api_key, OrgContext

import logging

logger = logging.getLogger(__name__)


router = APIRouter(prefix="/tests", tags=["tests"])

_EXAMPLE_TEST_UUID = "b1c2d3e4-f5a6-7890-bcde-f12345678901"
_EXAMPLE_EVALUATOR_UUID = "f47ac10b-58cc-4372-a567-0e02b2c3d479"
_EXAMPLE_AGENT_UUID = "a3b2c1d0-e5f4-3210-abcd-ef1234567890"


TestType = Literal["response", "tool_call", "conversation"]

# Each test type pins the evaluator_type it accepts. `conversation` tests judge whole
# simulated conversations, so only `conversation` evaluators apply; `response`/`tool_call`
# tests judge a single LLM reply, so only `llm` evaluators apply.
REQUIRED_EVALUATOR_TYPE_BY_TEST_TYPE: Dict[str, str] = {
    "response": "llm",
    "tool_call": "llm",
    "conversation": "conversation",
}


# Linked evaluators resolve to the live version at run time (not pinned per test).
class EvaluatorRef(BaseModel):
    model_config = ConfigDict(extra="forbid")

    evaluator_uuid: str = Field(
        min_length=36,
        max_length=36,
        description="Evaluator to attach to the test",
        examples=[_EXAMPLE_EVALUATOR_UUID],
    )
    variable_values: Optional[Dict[str, Any]] = Field(
        None,
        description="Values for the evaluator's `{{placeholder}}` variables, pinned per-test on the pivot. Omit to inherit the evaluator version's defaults",
    )


class TestCreate(BaseModel):
    name: str = Field(description="Human-readable test name, unique within the workspace")
    type: TestType = Field(
        description="Test kind (immutable after creation): `response` judges the generated reply, `tool_call` diffs generated tool calls, `conversation` judges the full conversation"
    )
    config: Optional[Dict[str, Any]] = Field(
        None,
        description="Calibrate test config (`history`, `evaluation`, optional `settings`). Omit to create an empty shell to fill in later",
    )
    evaluators: Optional[List[EvaluatorRef]] = Field(
        None,
        description="Evaluators to link. **Required (>=1) for `type=conversation`** (no fallback judge). Omit for `response`/`tool_call` to link none",
    )


class TestUpdate(BaseModel):
    name: Optional[str] = Field(None, description="New test name. Omit to leave unchanged")
    type: Optional[TestType] = Field(
        None,
        description="Test type. Immutable — may only echo the existing value; a different value is rejected (400). Omit to leave unchanged",
    )
    config: Optional[Dict[str, Any]] = Field(
        None, description="Replacement calibrate config. Omit to leave unchanged"
    )
    evaluators: Optional[List[EvaluatorRef]] = Field(
        None,
        description="Replacement evaluator links (replaces the existing set). Omit to leave links unchanged; an empty list clears them (**rejected for `conversation` tests**)",
    )


class TestResponse(BaseModel):
    uuid: str = Field(
        min_length=36,
        max_length=36,
        description="Test ID",
        examples=[_EXAMPLE_TEST_UUID],
    )
    name: str = Field(description="Human-readable test name")
    type: str = Field(description="Test kind: `response`, `tool_call`, or `conversation`")
    config: Optional[Dict[str, Any]] = Field(
        None, description="Calibrate test config (`history`, `evaluation`, optional `settings`)"
    )
    created_at: str = Field(description="Creation timestamp (ISO 8601 UTC)")
    updated_at: str = Field(description="Last-update timestamp (ISO 8601 UTC)")
    evaluators: List[Dict[str, Any]] = Field(
        default=[],
        description="Linked evaluators, resolved to their current live version at read time",
    )


class TestCreateResponse(BaseModel):
    uuid: str = Field(
        min_length=36,
        max_length=36,
        description="ID of the newly created test",
        examples=[_EXAMPLE_TEST_UUID],
    )
    message: str = Field(description="Human-readable confirmation message")


# --- Bulk upload models ---

class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant", "tool"] = Field(
        description="Message author role in the conversation history"
    )
    content: Optional[str] = Field(
        None, description="Message text. Omit for assistant messages that only carry `tool_calls`"
    )
    tool_calls: Optional[List[Dict[str, Any]]] = Field(
        None, description="Tool calls issued by the assistant. Omit for plain text turns"
    )
    tool_call_id: Optional[str] = Field(
        None, description="ID of the tool call this message responds to. **Required for `role=tool`**"
    )
    name: Optional[str] = Field(None, description="Tool name for `role=tool` messages. Omit otherwise")


class ExpectedToolCall(BaseModel):
    tool: str = Field(description="Name of the tool the agent is expected to call")
    arguments: Optional[Dict[str, Any]] = Field(
        None, description="Expected argument values, diffed against the generated call. Omit to expect no arguments"
    )
    accept_any_arguments: bool = Field(
        False, description="When `true`, only the tool name must match and `arguments` is ignored"
    )


class BulkTestItem(BaseModel):
    name: str = Field(description="Test name, unique within the batch")
    conversation_history: List[ChatMessage] = Field(
        description="Ordered messages ending at the user turn the agent should answer (must be non-empty)"
    )
    evaluators: Optional[List[EvaluatorRef]] = Field(
        None, description="Evaluators to link. **Required for `response`/`conversation` batches**"
    )
    tool_calls: Optional[List[ExpectedToolCall]] = Field(
        None, description="Expected tool calls. **Required for `tool_call` batches**"
    )


class BulkTestUpload(BaseModel):
    type: TestType = Field(description="Test kind applied to every item in the batch")
    tests: List[BulkTestItem] = Field(
        description=f"Test items to create (non-empty, max {500} per request, names unique within the batch)"
    )
    agent_uuids: Optional[List[str]] = Field(
        None,
        description="Agents (IDs) to link every created test to. Omit to link none",
        examples=[[_EXAMPLE_AGENT_UUID]],
    )
    language: Optional[str] = Field(
        None, description="Language written to each test's `config.settings.language`. Omit to leave unset"
    )

    MAX_BATCH_SIZE: ClassVar[int] = 500

    @model_validator(mode="after")
    def validate_tests(self):
        if not self.tests:
            raise ValueError("tests list must not be empty")

        if len(self.tests) > self.MAX_BATCH_SIZE:
            raise ValueError(f"Batch size {len(self.tests)} exceeds maximum of {self.MAX_BATCH_SIZE}")

        names = [t.name for t in self.tests]
        if len(names) != len(set(names)):
            seen = set()
            dupes = sorted({n for n in names if n in seen or seen.add(n)})
            raise ValueError(f"Duplicate test names in request: {', '.join(dupes)}")

        for t in self.tests:
            if not t.conversation_history:
                raise ValueError(f"Test '{t.name}' must have at least one message in conversation_history")
            if self.type == "response":
                if not t.evaluators:
                    raise ValueError(
                        f"Test '{t.name}' must have at least one evaluator for response type"
                    )
            elif self.type == "tool_call":
                if not t.tool_calls:
                    raise ValueError(f"Test '{t.name}' must have 'tool_calls' for tool_call type")
            elif self.type == "conversation":
                if not t.evaluators:
                    raise ValueError(
                        f"Test '{t.name}' must have at least one evaluator for conversation type"
                    )

        return self


class BulkTestUploadResponse(BaseModel):
    uuids: List[str] = Field(
        description="IDs of the created tests, in request order",
        examples=[[_EXAMPLE_TEST_UUID]],
    )
    count: int = Field(description="Number of tests created")
    message: str = Field(description="Human-readable confirmation message")
    warnings: Optional[List[str]] = Field(
        None, description="Non-fatal issues (e.g. agents some tests couldn't link to). Null when there were none"
    )


class BulkTestDelete(BaseModel):
    test_uuids: List[str] = Field(
        description="IDs of the tests to delete (non-empty)",
        examples=[[_EXAMPLE_TEST_UUID]],
    )


class BulkTestDeleteResponse(BaseModel):
    deleted_count: int = Field(description="Number of tests actually deleted (excludes IDs not in your workspace)")
    message: str = Field(description="Human-readable confirmation message")


def _validate_evaluators(
    refs: List[EvaluatorRef], org_uuid: str, test_type: str
) -> List[Dict[str, Any]]:
    """Validate that each referenced evaluator is visible to the workspace and that its
    `evaluator_type` matches the test's type (`response`/`tool_call` ⇒ `llm`,
    `conversation` ⇒ `simulation`). Returns db-ready refs."""
    required_evaluator_type = REQUIRED_EVALUATOR_TYPE_BY_TEST_TYPE.get(test_type)
    if required_evaluator_type is None:
        raise HTTPException(
            status_code=400, detail=f"Unknown test type '{test_type}'"
        )
    out: List[Dict[str, Any]] = []
    for ref in refs:
        evaluator = get_evaluator(ref.evaluator_uuid)
        if not evaluator:
            raise HTTPException(status_code=404, detail=f"Evaluator {ref.evaluator_uuid} not found")
        if evaluator.get("org_uuid") is not None and evaluator["org_uuid"] != org_uuid:
            raise HTTPException(status_code=404, detail=f"Evaluator {ref.evaluator_uuid} not found")
        if evaluator.get("evaluator_type") != required_evaluator_type:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"Evaluator {ref.evaluator_uuid} has evaluator_type="
                    f"'{evaluator.get('evaluator_type')}'. Tests of type "
                    f"'{test_type}' only accept '{required_evaluator_type}' evaluators."
                ),
            )
        out.append(
            {
                "evaluator_id": ref.evaluator_uuid,
                "evaluator_version_id": None,
                "variable_values": ref.variable_values,
            }
        )
    return out


def _with_evaluators(test_dict: Dict[str, Any]) -> Dict[str, Any]:
    """Attach linked evaluators to a test dict."""
    evaluators = get_evaluators_for_test(test_dict["uuid"])
    return {**test_dict, "evaluators": evaluators}


@router.post("/bulk-delete", response_model=BulkTestDeleteResponse, summary="Bulk delete tests")
async def bulk_delete_tests_endpoint(
    payload: BulkTestDelete, ctx: OrgContext = Depends(get_current_org)
):
    """Soft-delete multiple tests by ID."""
    if not payload.test_uuids:
        raise HTTPException(status_code=400, detail="test_uuids must not be empty")

    deleted_count = bulk_delete_tests(test_uuids=payload.test_uuids, org_uuid=ctx.org_uuid)

    return BulkTestDeleteResponse(
        deleted_count=deleted_count,
        message=f"Successfully deleted {deleted_count} test(s)",
    )


@router.post(
    "/bulk",
    response_model=BulkTestUploadResponse,
    tags=["Public API"],
    summary="Bulk create tests",
)
async def bulk_upload_tests(
    payload: BulkTestUpload, ctx: OrgContext = Depends(get_org_jwt_or_api_key)
):
    """Create many tests of one type in a single call, optionally linking them to agents."""
    if payload.agent_uuids:
        for agent_uuid in payload.agent_uuids:
            agent = get_agent(agent_uuid)
            if not agent or agent.get("org_uuid") != ctx.org_uuid:
                raise HTTPException(status_code=404, detail=f"Agent {agent_uuid} not found")

    resolved_evaluator_refs: List[Optional[List[Dict[str, Any]]]] = []
    for t in payload.tests:
        if t.evaluators:
            resolved_evaluator_refs.append(
                _validate_evaluators(t.evaluators, ctx.org_uuid, payload.type)
            )
        else:
            resolved_evaluator_refs.append(None)

    db_tests = []
    for t in payload.tests:
        evaluation: Dict[str, Any] = {"type": payload.type}
        if payload.type == "tool_call":
            evaluation["tool_calls"] = [tc.model_dump() for tc in t.tool_calls]

        config: Dict[str, Any] = {
            "history": [msg.model_dump(exclude_none=True) for msg in t.conversation_history],
            "evaluation": evaluation,
        }
        if payload.language:
            config["settings"] = {"language": payload.language}

        db_tests.append({
            "name": t.name,
            "type": payload.type,
            "config": config,
        })

    try:
        uuids = bulk_create_tests(
            tests=db_tests, org_uuid=ctx.org_uuid, user_id=ctx.user_id
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    for test_uuid, refs in zip(uuids, resolved_evaluator_refs):
        if refs:
            set_test_evaluators(test_uuid, refs)

    warnings: List[str] = []
    if payload.agent_uuids:
        linked_agents = set()
        for agent_uuid in payload.agent_uuids:
            agent_failed = False
            for test_uuid in uuids:
                try:
                    add_test_to_agent(agent_uuid, test_uuid)
                    linked_agents.add(agent_uuid)
                except Exception as e:
                    agent_failed = True
                    logger.warning(f"Failed to link test {test_uuid} to agent {agent_uuid}: {e}")
            if agent_failed:
                warnings.append(f"Some tests could not be linked to agent {agent_uuid}")

    message = f"Successfully created {len(uuids)} tests"
    if payload.agent_uuids:
        message += f" and linked to {len(linked_agents)} agent(s)"

    return BulkTestUploadResponse(
        uuids=uuids,
        count=len(uuids),
        message=message,
        warnings=warnings or None,
    )


@router.post(
    "",
    response_model=TestCreateResponse,
    tags=["Public API"],
    summary="Create test",
)
async def create_test_endpoint(
    test: TestCreate, ctx: OrgContext = Depends(get_org_jwt_or_api_key)
):
    """Create a test in your workspace."""
    # Conversation tests have no evaluator fallback (unlike `response`, which can
    # synthesize the default LLM judge from legacy string criteria) — without a
    # linked simulation evaluator a run produces an empty calibrate config with
    # nothing to judge. Require at least one up front. (The bulk endpoint already
    # enforces this; this closes the single-create gap.)
    if test.type == "conversation" and not test.evaluators:
        raise HTTPException(
            status_code=400,
            detail="Conversation tests require at least one evaluator.",
        )
    resolved = (
        _validate_evaluators(test.evaluators, ctx.org_uuid, test.type)
        if test.evaluators
        else None
    )
    with ensure_name_unique("tests", test.name, ctx.org_uuid, entity="Test"):
        test_uuid = create_test(
            name=test.name,
            type=test.type,
            config=test.config,
            org_uuid=ctx.org_uuid,
            user_id=ctx.user_id,
        )
    if resolved:
        set_test_evaluators(test_uuid, resolved)
    return TestCreateResponse(uuid=test_uuid, message="Test created successfully")


@router.get(
    "",
    response_model=List[TestResponse],
    tags=["Public API"],
    summary="List tests",
)
async def list_tests(ctx: OrgContext = Depends(get_org_jwt_or_api_key)):
    """List all tests for your workspace, each with its linked evaluators."""
    tests = get_all_tests(org_uuid=ctx.org_uuid)
    return [_with_evaluators(t) for t in tests]


@router.get(
    "/{test_uuid}",
    response_model=TestResponse,
    tags=["Public API"],
    summary="Get test",
)
async def get_test_endpoint(
    test_uuid: str = Path(
        description="Test to retrieve",
        examples=["b1c2d3e4-f5a6-7890-bcde-f12345678901"],
    ),
    ctx: OrgContext = Depends(get_org_jwt_or_api_key),
):
    """Get a test by ID, including its linked evaluators."""
    test = get_test(test_uuid)
    if not test or test.get("org_uuid") != ctx.org_uuid:
        raise HTTPException(status_code=404, detail="Test not found")
    return _with_evaluators(test)


@router.put(
    "/{test_uuid}",
    response_model=TestResponse,
    tags=["Public API"],
    summary="Update test",
)
async def update_test_endpoint(
    test: TestUpdate,
    test_uuid: str = Path(
        description="Test to update",
        examples=["b1c2d3e4-f5a6-7890-bcde-f12345678901"],
    ),
    ctx: OrgContext = Depends(get_org_jwt_or_api_key),
):
    """Update a test's name, config, and/or evaluator links."""
    existing_test = get_test(test_uuid)
    if not existing_test or existing_test.get("org_uuid") != ctx.org_uuid:
        raise HTTPException(status_code=404, detail="Test not found")

    # A test's `type` is immutable after creation. Allowing a change would
    # strand already-linked evaluators whose `evaluator_type` was validated
    # against the original type (e.g. a `response` test's `llm` evaluator
    # surviving a switch to `conversation`, which only accepts `simulation`).
    # Echoing back the same value is a no-op; a different value is rejected.
    existing_type = existing_test.get("type")
    if test.type is not None and test.type != existing_type:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Test type is immutable; cannot change from "
                f"'{existing_type}' to '{test.type}'. Create a new test instead."
            ),
        )

    # Conversation tests must keep at least one evaluator (see create-endpoint
    # note). Reject an update that would clear them all.
    if (
        existing_type == "conversation"
        and test.evaluators is not None
        and len(test.evaluators) == 0
    ):
        raise HTTPException(
            status_code=400,
            detail="Conversation tests require at least one evaluator; cannot remove all.",
        )

    resolved = (
        _validate_evaluators(test.evaluators, ctx.org_uuid, existing_type)
        if test.evaluators is not None
        else None
    )

    has_core_updates = any(
        v is not None for v in (test.name, test.type, test.config)
    )
    if has_core_updates:
        with ensure_name_unique(
            "tests", test.name, ctx.org_uuid, entity="Test", exclude_uuid=test_uuid
        ):
            updated = update_test(
                test_uuid=test_uuid,
                name=test.name,
                type=test.type,
                config=test.config,
            )
        if not updated and resolved is None:
            raise HTTPException(status_code=400, detail="No fields to update")

    if resolved is not None:
        set_test_evaluators(test_uuid, resolved)

    return _with_evaluators(get_test(test_uuid))


@router.delete("/{test_uuid}", summary="Delete test")
async def delete_test_endpoint(
    test_uuid: str = Path(
        description="Test to delete",
        examples=["b1c2d3e4-f5a6-7890-bcde-f12345678901"],
    ),
    ctx: OrgContext = Depends(get_current_org),
):
    """Soft-delete a test."""
    existing_test = get_test(test_uuid)
    if not existing_test or existing_test.get("org_uuid") != ctx.org_uuid:
        raise HTTPException(status_code=404, detail="Test not found")

    deleted = delete_test(test_uuid)
    if not deleted:
        raise HTTPException(status_code=404, detail="Test not found")
    return {"message": "Test deleted successfully"}
