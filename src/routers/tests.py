from typing import ClassVar, Optional, List, Dict, Any, Literal
from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel, ConfigDict, model_validator

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
from auth_utils import get_current_user_id

import logging

logger = logging.getLogger(__name__)


router = APIRouter(prefix="/tests", tags=["tests"])


class EvaluatorRef(BaseModel):
    """Reference to an evaluator attached to a test. The pinned version is always the
    evaluator's live version at write time (`set_test_evaluators` in `db.py`)."""

    model_config = ConfigDict(extra="forbid")

    evaluator_uuid: str
    variable_values: Optional[Dict[str, Any]] = None


class TestCreate(BaseModel):
    name: str
    type: str
    config: Optional[Dict[str, Any]] = None
    evaluators: Optional[List[EvaluatorRef]] = None


class TestUpdate(BaseModel):
    name: Optional[str] = None
    type: Optional[str] = None
    config: Optional[Dict[str, Any]] = None
    evaluators: Optional[List[EvaluatorRef]] = None


class TestResponse(BaseModel):
    uuid: str
    name: str
    type: str
    config: Optional[Dict[str, Any]] = None
    created_at: str
    updated_at: str
    evaluators: List[Dict[str, Any]] = []


class TestCreateResponse(BaseModel):
    uuid: str
    message: str


# --- Bulk upload models ---

class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant", "tool"]
    content: Optional[str] = None
    tool_calls: Optional[List[Dict[str, Any]]] = None
    tool_call_id: Optional[str] = None
    name: Optional[str] = None


class ExpectedToolCall(BaseModel):
    tool: str
    arguments: Optional[Dict[str, Any]] = None
    accept_any_arguments: bool = False


class BulkTestItem(BaseModel):
    name: str
    conversation_history: List[ChatMessage]
    evaluators: Optional[List[EvaluatorRef]] = None
    tool_calls: Optional[List[ExpectedToolCall]] = None


class BulkTestUpload(BaseModel):
    type: Literal["response", "tool_call"]
    tests: List[BulkTestItem]
    agent_uuids: Optional[List[str]] = None
    language: Optional[str] = None

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

        return self


class BulkTestUploadResponse(BaseModel):
    uuids: List[str]
    count: int
    message: str
    warnings: Optional[List[str]] = None


class BulkTestDelete(BaseModel):
    test_uuids: List[str]


class BulkTestDeleteResponse(BaseModel):
    deleted_count: int
    message: str


def _validate_evaluators(refs: List[EvaluatorRef], user_id: str) -> List[Dict[str, Any]]:
    """Validate that each referenced evaluator is visible to the user and that it has
    `evaluator_type == 'llm'` (response/next-reply tests only judge LLM output, so attaching
    a stt/tts/simulation evaluator is rejected at write time). Returns db-ready refs."""
    out: List[Dict[str, Any]] = []
    for ref in refs:
        evaluator = get_evaluator(ref.evaluator_uuid)
        if not evaluator:
            raise HTTPException(status_code=404, detail=f"Evaluator {ref.evaluator_uuid} not found")
        if evaluator.get("owner_user_id") is not None and evaluator["owner_user_id"] != user_id:
            raise HTTPException(status_code=404, detail=f"Evaluator {ref.evaluator_uuid} not found")
        if evaluator.get("evaluator_type") != "llm":
            raise HTTPException(
                status_code=400,
                detail=(
                    f"Evaluator {ref.evaluator_uuid} has evaluator_type="
                    f"'{evaluator.get('evaluator_type')}'. LLM tests only accept 'llm' evaluators."
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


@router.post("/bulk-delete", response_model=BulkTestDeleteResponse)
async def bulk_delete_tests_endpoint(
    payload: BulkTestDelete, user_id: str = Depends(get_current_user_id)
):
    """Bulk delete tests by UUIDs. Only deletes tests owned by the authenticated user."""
    if not payload.test_uuids:
        raise HTTPException(status_code=400, detail="test_uuids must not be empty")

    deleted_count = bulk_delete_tests(test_uuids=payload.test_uuids, user_id=user_id)

    return BulkTestDeleteResponse(
        deleted_count=deleted_count,
        message=f"Successfully deleted {deleted_count} test(s)",
    )


@router.post("/bulk", response_model=BulkTestUploadResponse)
async def bulk_upload_tests(
    payload: BulkTestUpload, user_id: str = Depends(get_current_user_id)
):
    """Bulk upload LLM tests. All tests must be the same type (response or tool_call)."""
    if payload.agent_uuids:
        for agent_uuid in payload.agent_uuids:
            agent = get_agent(agent_uuid)
            if not agent:
                raise HTTPException(status_code=404, detail=f"Agent {agent_uuid} not found")
            if agent.get("user_id") != user_id:
                raise HTTPException(status_code=403, detail=f"Access denied for agent {agent_uuid}")

    resolved_evaluator_refs: List[Optional[List[Dict[str, Any]]]] = []
    for t in payload.tests:
        if t.evaluators:
            resolved_evaluator_refs.append(_validate_evaluators(t.evaluators, user_id))
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
        uuids = bulk_create_tests(tests=db_tests, user_id=user_id)
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


@router.post("", response_model=TestCreateResponse)
async def create_test_endpoint(
    test: TestCreate, user_id: str = Depends(get_current_user_id)
):
    """Create a new test."""
    resolved = _validate_evaluators(test.evaluators, user_id) if test.evaluators else None
    with ensure_name_unique("tests", test.name, user_id, entity="Test"):
        test_uuid = create_test(
            name=test.name,
            type=test.type,
            config=test.config,
            user_id=user_id,
        )
    if resolved:
        set_test_evaluators(test_uuid, resolved)
    return TestCreateResponse(uuid=test_uuid, message="Test created successfully")


@router.get("", response_model=List[TestResponse])
async def list_tests(user_id: str = Depends(get_current_user_id)):
    """List all tests for the authenticated user."""
    tests = get_all_tests(user_id=user_id)
    return [_with_evaluators(t) for t in tests]


@router.get("/{test_uuid}", response_model=TestResponse)
async def get_test_endpoint(
    test_uuid: str, user_id: str = Depends(get_current_user_id)
):
    """Get a test by UUID."""
    test = get_test(test_uuid)
    if not test:
        raise HTTPException(status_code=404, detail="Test not found")
    if test.get("user_id") != user_id:
        raise HTTPException(status_code=403, detail="Access denied")
    return _with_evaluators(test)


@router.put("/{test_uuid}", response_model=TestResponse)
async def update_test_endpoint(
    test_uuid: str, test: TestUpdate, user_id: str = Depends(get_current_user_id)
):
    """Update a test."""
    existing_test = get_test(test_uuid)
    if not existing_test:
        raise HTTPException(status_code=404, detail="Test not found")
    if existing_test.get("user_id") != user_id:
        raise HTTPException(status_code=403, detail="Access denied")

    resolved = _validate_evaluators(test.evaluators, user_id) if test.evaluators is not None else None

    has_core_updates = any(
        v is not None for v in (test.name, test.type, test.config)
    )
    if has_core_updates:
        with ensure_name_unique(
            "tests", test.name, user_id, entity="Test", exclude_uuid=test_uuid
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


@router.delete("/{test_uuid}")
async def delete_test_endpoint(
    test_uuid: str, user_id: str = Depends(get_current_user_id)
):
    """Delete a test."""
    existing_test = get_test(test_uuid)
    if not existing_test:
        raise HTTPException(status_code=404, detail="Test not found")
    if existing_test.get("user_id") != user_id:
        raise HTTPException(status_code=403, detail="Access denied")

    deleted = delete_test(test_uuid)
    if not deleted:
        raise HTTPException(status_code=404, detail="Test not found")
    return {"message": "Test deleted successfully"}
