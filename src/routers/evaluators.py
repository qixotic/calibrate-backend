"""Evaluators router.

Replaces the legacy `metrics` router. Adds:
 - Built-in default vs user-created evaluators
 - Versioned prompts with judge_model, rating scale, variables
 - Duplicate a built-in default into an editable copy
 - Live-version selection
 - API-key-authenticated invocation endpoint
"""

from typing import Any, Dict, List, Literal, Optional

from fastapi import APIRouter, Depends, HTTPException, Path, Query
from pagination import (
    OptionalPaginationParams,
    PaginatedResponse,
    count_and_page,
    make_projection_params,
    make_search_params,
    page_envelope,
)

_EvaluatorSearch = make_search_params(searchable=["name"])
_EvaluatorDetailProjection = make_projection_params(
    heavy_fields=[
        "versions[].system_prompt",
        "versions[].output_config",
        "versions[].variables",
    ]
)
from pydantic import BaseModel, Field, model_validator

from auth_utils import get_current_org, get_current_user_id, get_org_jwt_or_api_key, OrgContext
from db import (
    DEFAULT_PROMPTS_BY_PURPOSE,
    create_evaluator,
    name_uniqueness_guard,
    create_evaluator_version,
    delete_evaluator,
    duplicate_evaluator,
    evaluator_name_exists,
    get_all_evaluators,
    get_evaluator,
    get_evaluator_version,
    get_evaluator_versions,
    get_evaluator_versions_by_uuids,
    set_evaluator_live_version,
    update_evaluator,
)
from llm_judge import render_template
from utils import (
    EvaluatorTypeLiteral,
    DataTypeLiteral,
    EVALUATOR_TYPE_DESCRIPTION,
    DATA_TYPE_DESCRIPTION,
    OUTPUT_TYPE_DESCRIPTION,
)

router = APIRouter(prefix="/evaluators", tags=["evaluators"])

_EXAMPLE_EVALUATOR_UUID = "f47ac10b-58cc-4372-a567-0e02b2c3d479"
_EXAMPLE_VERSION_UUID = "6ba7b810-9dad-11d1-80b4-00c04fd430c8"
_EXAMPLE_USER_UUID = "a3b2c1d0-e5f4-3210-abcd-ef1234567890"

_VERSION_NUMBER_DESCRIPTION = (
    "The version's number. The first version is 1, and it goes up by one for each "
    "new version of the evaluator"
)

_JUDGE_MODEL_DESCRIPTION = (
    "The model that runs the judge, named the way its provider does, for example "
    "`openai/gpt-4.1` or `anthropic/claude-sonnet-4`"
)

# The rubric is optional for binary (custom pass/fail labels) and required for
# rating, so the same wording is reused on request and response output_config.
_RUBRIC_DESCRIPTION = (
    "The scale points and their labels. Required for a `rating` evaluator. A "
    "`binary` evaluator uses the default Correct/Wrong labels unless you set "
    "your own"
)


# ============ Pydantic models ============


class OutputScaleEntry(BaseModel):
    """One entry in an evaluator's output_config.scale. `value` is bool|number|string
 depending on output_type."""

    value: Any = Field(
        description="The value for this scale point. Use a boolean for a `binary` evaluator, a number for a `rating` one"
    )
    name: str = Field(description="Short label for this scale point")
    description: Optional[str] = Field(
        None,
        description="Rubric text for this level, added to the judge prompt. Omit to leave this level undescribed",
    )
    color: Optional[str] = Field(None, description="Color to show for this level. Omit for the default")


class OutputConfig(BaseModel):
    scale: Optional[List[OutputScaleEntry]] = Field(
        None,
        description="The ordered scale points that make up the rubric, each with its label",
    )


class VariableSpec(BaseModel):
    name: str = Field(description="Name of a `{{placeholder}}` used in the system prompt")
    description: Optional[str] = Field(None, description="What the variable is for. Omit if self-evident")
    default: Optional[str] = Field(None, description="Default value used when you omit this variable. Omit for no default")


class EvaluatorVersionCreate(BaseModel):
    """One version of an evaluator: its judge prompt, model, variables, and rubric."""

    judge_model: str = Field(description=_JUDGE_MODEL_DESCRIPTION)
    system_prompt: str = Field(description="Judge system prompt. May contain `{{variable}}` placeholders")
    output_config: Optional[OutputConfig] = Field(
        None,
        description=_RUBRIC_DESCRIPTION,
    )
    variables: Optional[List[VariableSpec]] = Field(
        None, description="Declared prompt variables. Omit if the prompt has no `{{placeholders}}`"
    )


class EvaluatorVersionCreateRequest(EvaluatorVersionCreate):
    """Request body for adding a version to an existing evaluator."""

    variables: Optional[List[VariableSpec]] = Field(
        None,
        description=(
            "Declared prompt variables. Omit if the prompt has none. After the "
            "first version the variable names are fixed. You can change a "
            "variable's description or default, but not add, remove, or rename one"
        ),
    )
    make_live: bool = Field(
        False, description="When `true`, immediately point the evaluator's live version at this new version"
    )


class EvaluatorCreate(BaseModel):
    name: str = Field(..., min_length=1, description="Evaluator name, unique within your workspace")
    description: Optional[str] = Field(None, description="Description. Omit to leave blank")
    evaluator_type: EvaluatorTypeLiteral = Field(
        "llm",
        description=EVALUATOR_TYPE_DESCRIPTION,
    )
    data_type: DataTypeLiteral = Field("text", description=DATA_TYPE_DESCRIPTION)
    kind: Literal["single", "side_by_side"] = Field(
        "single",
        description=(
            "How the evaluator scores:\n\n"
            "- `single`: judges one output\n"
            "- `side_by_side`: compares two outputs and picks a winner\n"
        ),
    )
    output_type: Literal["binary", "rating"] = Field(
        "binary", description=OUTPUT_TYPE_DESCRIPTION
    )
    version: EvaluatorVersionCreate = Field(description="The evaluator's first version. Set as live when you create the evaluator")

    @model_validator(mode="after")
    def _validate_output(self):
        cfg = self.version.output_config if self.version else None
        if self.output_type == "rating":
            if not cfg or not cfg.scale or len(cfg.scale) < 2:
                raise ValueError(
                    "version.output_config.scale (>=2 entries) is required when output_type='rating'"
                )
        return self


class EvaluatorUpdate(BaseModel):
    name: Optional[str] = Field(None, min_length=1, description="New name. Omit to leave unchanged")
    description: Optional[str] = Field(None, description="New description. Omit to leave unchanged")
    evaluator_type: Optional[EvaluatorTypeLiteral] = Field(
        None, description="New value for what the evaluator judges. Omit to leave unchanged"
    )
    data_type: Optional[DataTypeLiteral] = Field(None, description="New modality. Omit to leave unchanged")
    kind: Optional[Literal["single", "side_by_side"]] = Field(
        None, description="New scoring mode. Omit to leave unchanged"
    )
    output_type: Optional[Literal["binary", "rating"]] = Field(
        None, description="New output shape. Omit to leave unchanged"
    )


class EvaluatorDuplicateRequest(BaseModel):
    name: str = Field(..., min_length=1, description="Name for the new copy, unique within your workspace")


class EvaluatorVersionResponse(BaseModel):
    uuid: str = Field(
        min_length=36,
        max_length=36,
        description="Version ID",
        examples=[_EXAMPLE_VERSION_UUID],
    )
    version_number: int = Field(description=_VERSION_NUMBER_DESCRIPTION)
    judge_model: str = Field(description=_JUDGE_MODEL_DESCRIPTION)
    system_prompt: str = Field(description="Judge system prompt, with `{{variable}}` placeholders unrendered")
    output_config: Optional[OutputConfig] = Field(
        None, description=_RUBRIC_DESCRIPTION
    )
    variables: Optional[List[VariableSpec]] = Field(None, description="Declared prompt variables")
    created_at: str = Field(description="When the version was created (ISO 8601 UTC)")


# Compact-mode shape for GET /evaluators/{uuid}: `system_prompt` is nullable
# only here so the always-full endpoints reusing the base keep it required.
# No docstring: Pydantic would publish it as the schema description.
class EvaluatorVersionCompact(EvaluatorVersionResponse):
    system_prompt: Optional[str] = Field(
        None, description="Judge system prompt, with `{{variable}}` placeholders unrendered"
    )


class EvaluatorLiveVersionSummary(BaseModel):
    """Slim view of the live version for list results, carrying only what the list needs."""

    uuid: str = Field(
        min_length=36,
        max_length=36,
        description="Version ID",
        examples=[_EXAMPLE_VERSION_UUID],
    )
    version_number: int = Field(description=_VERSION_NUMBER_DESCRIPTION)
    judge_model: str = Field(description=_JUDGE_MODEL_DESCRIPTION)
    variables: Optional[List[VariableSpec]] = Field(None, description="Declared prompt variables")


class EvaluatorResponseBase(BaseModel):
    """Identity and classification fields shared by every evaluator response."""

    uuid: str = Field(
        min_length=36,
        max_length=36,
        description="Evaluator ID",
        examples=[_EXAMPLE_EVALUATOR_UUID],
    )
    name: str = Field(description="Evaluator name")
    description: Optional[str] = Field(None, description="What the evaluator checks")
    evaluator_type: EvaluatorTypeLiteral = Field(
        description=EVALUATOR_TYPE_DESCRIPTION
    )
    data_type: DataTypeLiteral = Field(description=DATA_TYPE_DESCRIPTION)
    kind: Literal["single", "side_by_side"] = Field(
        description=(
            "How the evaluator scores:\n\n"
            "- `single`: judges one output\n"
            "- `side_by_side`: compares two outputs and picks a winner\n"
        )
    )
    output_type: Literal["binary", "rating"] = Field(description=OUTPUT_TYPE_DESCRIPTION)
    owner_user_id: Optional[str] = Field(
        None,
        min_length=36,
        max_length=36,
        description="Creator user ID",
        examples=[_EXAMPLE_USER_UUID],
    )
    is_default: bool = Field(
        description="True for a built-in default evaluator, which you can't edit. False for an evaluator you created, which you can edit and add versions to"
    )
    slug: Optional[str] = Field(None, description="Stable slug for a built-in default evaluator")
    live_version_id: Optional[str] = Field(
        None,
        min_length=36,
        max_length=36,
        description="ID of the version that is currently live",
        examples=[_EXAMPLE_VERSION_UUID],
    )
    created_at: str = Field(description="When the evaluator was created (ISO 8601 UTC)")
    updated_at: str = Field(description="When the evaluator was last updated (ISO 8601 UTC)")


class EvaluatorResponse(EvaluatorResponseBase):
    live_version: Optional[EvaluatorLiveVersionSummary] = Field(
        None, description="The version that is currently live"
    )


class EvaluatorDetailResponse(EvaluatorResponseBase):
    # Detail shape: `versions[]` is the full history; `live_version_index`
    # is the direct array position of the live version (None when the
    # evaluator has no live version, or when the live id doesn't resolve
    # to anything in `versions[]`). Clients should prefer the index over
    # scanning `versions[]` by uuid.
    versions: List[EvaluatorVersionResponse] = Field(description="Full version history, oldest first")
    live_version_index: Optional[int] = Field(
        None, description="Array position of the live version within `versions[]`"
    )


# Response for GET /evaluators/{uuid}: uses the compact version model so
# `?compact` can null version fields. The base stays tight for the
# annotation-task endpoints that reuse it. No docstring: it would be published.
class EvaluatorDetailResponseCompact(EvaluatorDetailResponse):
    versions: List[EvaluatorVersionCompact] = Field(description="Full version history, oldest first")


class EvaluatorCreateResponse(BaseModel):
    uuid: str = Field(
        min_length=36,
        max_length=36,
        description="ID of the created evaluator",
        examples=[_EXAMPLE_EVALUATOR_UUID],
    )
    version_uuid: str = Field(
        min_length=36,
        max_length=36,
        description="ID of its initial version",
        examples=[_EXAMPLE_VERSION_UUID],
    )


class VersionCreateResponse(BaseModel):
    version_uuid: str = Field(
        min_length=36,
        max_length=36,
        description="ID of the newly created version",
        examples=[_EXAMPLE_VERSION_UUID],
    )
    version_number: int = Field(description=_VERSION_NUMBER_DESCRIPTION)


class SetLiveVersionRequest(BaseModel):
    version_uuid: str = Field(
        min_length=36,
        max_length=36,
        description="ID of the version to mark as live. It must belong to this evaluator",
        examples=[_EXAMPLE_VERSION_UUID],
    )


# ============ Helpers ============


def _owner_check(evaluator: Dict[str, Any], org_uuid: str) -> None:
    """Seeded defaults (org_uuid IS NULL) are visible to every workspace but mutable by no one."""
    if evaluator.get("org_uuid") is None:
        raise HTTPException(status_code=403, detail="Default evaluators cannot be modified")
    if evaluator.get("org_uuid") != org_uuid:
        raise HTTPException(status_code=404, detail="Evaluator not found")


def _visible_or_404(
    evaluator: Optional[Dict[str, Any]], org_uuid: str
) -> Dict[str, Any]:
    if not evaluator:
        raise HTTPException(status_code=404, detail="Evaluator not found")
    if evaluator.get("org_uuid") is not None and evaluator["org_uuid"] != org_uuid:
        raise HTTPException(status_code=404, detail="Evaluator not found")
    return evaluator


def _ensure_unique_evaluator_name(
    name: str,
    org_uuid: str,
    exclude_uuid: Optional[str] = None,
) -> None:
    if evaluator_name_exists(name, org_uuid=org_uuid, exclude_uuid=exclude_uuid):
        raise HTTPException(status_code=409, detail="Evaluator name already exists")


def _version_dict(
    version: Dict[str, Any],
    output_type: Optional[str] = None,
) -> Dict[str, Any]:
    """Shape an evaluator_versions row for the API response. When `output_config`
    is missing on the stored row, fill in the evaluator's default rubric for
    the given `output_type` so consumers don't have to handle a null scale —
    binary rows get Correct/Wrong, rating rows stay null (the FE falls back to
    `str(value)`)."""
    from llm_judge import default_output_config

    output_config = version.get("output_config")
    if output_config is None:
        output_config = default_output_config(output_type)
    return {
        "uuid": version["uuid"],
        "version_number": version["version_number"],
        "judge_model": version["judge_model"],
        "system_prompt": version["system_prompt"],
        "output_config": output_config,
        "variables": version.get("variables"),
        "created_at": version["created_at"],
    }


def _live_version_index(
    versions: List[EvaluatorVersionResponse],
    live_version_id: Optional[str],
) -> Optional[int]:
    """Position of the live version in `versions[]`, or None if no live
    version is set or the id doesn't match any entry."""
    if not live_version_id:
        return None
    for i, v in enumerate(versions):
        if v.uuid == live_version_id:
            return i
    return None


def _evaluator_response(
    evaluator: Dict[str, Any],
    version_by_id: Optional[Dict[str, Dict[str, Any]]] = None,
) -> EvaluatorResponse:
    """Shape one evaluator row into the list/summary response.

    `version_by_id` is an optional preloaded `{version_uuid: version_row}` map so
    a list caller can resolve every evaluator's live version from ONE batched
    query instead of a per-evaluator `get_evaluator_version` (N+1). When omitted,
    the single live version is fetched inline (the single-detail path)."""
    live_version = None
    live_version_id = evaluator.get("live_version_id")
    if live_version_id:
        v = (
            version_by_id.get(live_version_id)
            if version_by_id is not None
            else get_evaluator_version(live_version_id)
        )
        if v:
            live_version = EvaluatorLiveVersionSummary(
                uuid=v["uuid"],
                version_number=v["version_number"],
                judge_model=v["judge_model"],
                variables=v.get("variables"),
            )
    return EvaluatorResponse(
        uuid=evaluator["uuid"],
        name=evaluator["name"],
        description=evaluator.get("description"),
        evaluator_type=evaluator.get("evaluator_type", "llm"),
        data_type=evaluator.get("data_type", "text"),
        kind=evaluator.get("kind", "single"),
        output_type=evaluator.get("output_type", "binary"),
        owner_user_id=evaluator.get("owner_user_id"),
        is_default=evaluator.get("owner_user_id") is None,
        slug=evaluator.get("slug"),
        live_version_id=evaluator.get("live_version_id"),
        created_at=evaluator["created_at"],
        updated_at=evaluator["updated_at"],
        live_version=live_version,
    )


# ============ CRUD ============


@router.post("", response_model=EvaluatorCreateResponse, summary="Create evaluator", tags=["Public API"])
async def create_evaluator_endpoint(
    payload: EvaluatorCreate, ctx: OrgContext = Depends(get_org_jwt_or_api_key)
):
    """Create an evaluator along with its first version, which is set live"""
    _ensure_unique_evaluator_name(payload.name, ctx.org_uuid)
    with name_uniqueness_guard("Evaluator"):
        evaluator_uuid = create_evaluator(
            name=payload.name,
            description=payload.description,
            evaluator_type=payload.evaluator_type,
            data_type=payload.data_type,
            kind=payload.kind,
            output_type=payload.output_type,
            owner_user_id=ctx.user_id,
            org_uuid=ctx.org_uuid,
        )
    version_cfg = (
        payload.version.output_config.model_dump(exclude_none=True)
        if payload.version.output_config
        else None
    )
    try:
        version = create_evaluator_version(
            evaluator_uuid=evaluator_uuid,
            judge_model=payload.version.judge_model,
            system_prompt=payload.version.system_prompt,
            output_config=version_cfg,
            variables=[v.model_dump() for v in payload.version.variables]
            if payload.version.variables
            else None,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    set_evaluator_live_version(evaluator_uuid, version["uuid"])
    return EvaluatorCreateResponse(uuid=evaluator_uuid, version_uuid=version["uuid"])


class DefaultPromptResponse(BaseModel):
    """Canonical default prompt for a given purpose. Used by the frontend to prefill the
 create-evaluator form. The `name` field is null for `purpose=conversation` because there's
 no built-in default conversation evaluator. The prompt is just a template."""

    purpose: Literal["llm", "llm-general", "stt", "tts", "conversation"] = Field(
        description="Evaluation purpose this default prompt targets"
    )
    name: Optional[str] = Field(
        None, description="Name of the built-in default evaluator"
    )
    system_prompt: str = Field(description="Suggested judge system prompt for prefilling the create form")
    judge_model: str = Field(description="Suggested judge model")
    evaluator_type: EvaluatorTypeLiteral = Field(description="Suggested value for what the evaluator judges")
    data_type: DataTypeLiteral = Field(description="Suggested modality")
    kind: Literal["single", "side_by_side"] = Field(description="Suggested scoring mode")
    output_type: Literal["binary", "rating"] = Field(description="Suggested output shape")
    output_config: Optional[OutputConfig] = Field(None, description="Suggested rubric")
    variables: List[VariableSpec] = Field(default=[], description="Suggested prompt variables")


@router.get("/default-prompt", response_model=DefaultPromptResponse, summary="Get default prompt")
async def get_default_prompt(
    purpose: Literal["llm", "llm-general", "stt", "tts", "conversation"] = Query(
        description="Evaluation purpose whose canonical default prompt + config to return"
    ),
    _user_id: str = Depends(get_current_user_id),
):
    """Get the canonical default prompt and suggested config for prefilling the create-evaluator form"""
    if purpose not in DEFAULT_PROMPTS_BY_PURPOSE:
        raise HTTPException(status_code=404, detail=f"Unknown purpose: {purpose}")
    p = DEFAULT_PROMPTS_BY_PURPOSE[purpose]
    return DefaultPromptResponse(
        purpose=purpose,
        name=p.get("name"),
        system_prompt=p["system_prompt"],
        judge_model=p["judge_model"],
        evaluator_type=p["evaluator_type"],
        data_type=p["data_type"],
        kind=p["kind"],
        output_type=p["output_type"],
        output_config=p.get("output_config"),
        variables=p.get("variables") or [],
    )


@router.get("", response_model=PaginatedResponse[EvaluatorResponse], summary="List evaluators", tags=["Public API"])
async def list_evaluators(
    evaluator_type: Optional[EvaluatorTypeLiteral] = Query(
        None, description="Filter by what the evaluator judges. Omit for all types"
    ),
    data_type: Optional[DataTypeLiteral] = Query(
        None, description="Filter by modality. Omit for all"
    ),
    include_defaults: bool = Query(
        True, description="When `true`, include the built-in default evaluators alongside the ones you created"
    ),
    ctx: OrgContext = Depends(get_org_jwt_or_api_key),
    search: _EvaluatorSearch = Depends(),
    pagination: OptionalPaginationParams = Depends(),
):
    """List your evaluators"""
    # `evaluator_type`/`data_type`/`include_defaults` filter server-side; then
    # optional `?q=` name search + `?limit=&offset=` paging. Returns the
    # `{items, total, limit, offset}` envelope.
    evaluators = get_all_evaluators(
        org_uuid=ctx.org_uuid,
        include_defaults=include_defaults,
        evaluator_type=evaluator_type,
        data_type=data_type,
    )
    evaluators = search.apply(evaluators)
    page, total = count_and_page(evaluators, pagination)
    # Resolve the PAGE's evaluators' live versions in ONE batched query, then
    # shape against that map — avoids a per-evaluator `get_evaluator_version`
    # (N+1), and only over the rows actually returned.
    version_by_id = get_evaluator_versions_by_uuids(
        [e["live_version_id"] for e in page if e.get("live_version_id")]
    )
    return page_envelope(
        [_evaluator_response(e, version_by_id=version_by_id) for e in page],
        total,
        pagination,
    )


@router.get("/{evaluator_uuid}", response_model=EvaluatorDetailResponseCompact, summary="Get evaluator", tags=["Public API"])
async def get_evaluator_endpoint(
    evaluator_uuid: str = Path(
        description="Evaluator to retrieve",
        examples=["f47ac10b-58cc-4372-a567-0e02b2c3d479"],
    ),
    ctx: OrgContext = Depends(get_org_jwt_or_api_key),
    projection: _EvaluatorDetailProjection = Depends(),
):
    """Get one evaluator with its full version history"""
    evaluator = _visible_or_404(get_evaluator(evaluator_uuid), ctx.org_uuid)
    base = _evaluator_response(evaluator)
    output_type = evaluator.get("output_type", "binary")
    versions = [
        EvaluatorVersionCompact(**_version_dict(v, output_type))
        for v in get_evaluator_versions(evaluator_uuid)
    ]
    # base carries `live_version` (list shape); drop it here — detail uses
    # `versions[]` + `live_version_id`/`live_version_index` so we don't
    # duplicate the version.
    response = EvaluatorDetailResponseCompact(
        **base.model_dump(exclude={"live_version"}),
        versions=versions,
        live_version_index=_live_version_index(versions, base.live_version_id),
    )
    # `?compact=true` nulls heavy per-version fields in place; a no-op otherwise.
    return projection.apply(response.model_dump())


@router.put("/{evaluator_uuid}", response_model=EvaluatorResponse, summary="Update evaluator")
async def update_evaluator_endpoint(
    payload: EvaluatorUpdate,
    evaluator_uuid: str = Path(
        description="Evaluator to update",
        examples=["f47ac10b-58cc-4372-a567-0e02b2c3d479"],
    ),
    ctx: OrgContext = Depends(get_current_org),
):
    """Update an evaluator, the judge used to grade outputs"""
    existing = _visible_or_404(get_evaluator(evaluator_uuid), ctx.org_uuid)
    _owner_check(existing, ctx.org_uuid)
    if payload.name is not None:
        _ensure_unique_evaluator_name(
            payload.name, ctx.org_uuid, exclude_uuid=evaluator_uuid
        )
    try:
        with name_uniqueness_guard("Evaluator"):
            updated = update_evaluator(
                evaluator_uuid=evaluator_uuid,
                name=payload.name,
                description=payload.description,
                evaluator_type=payload.evaluator_type,
                data_type=payload.data_type,
                kind=payload.kind,
                output_type=payload.output_type,
            )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    if not updated:
        raise HTTPException(status_code=400, detail="No fields to update")
    return _evaluator_response(get_evaluator(evaluator_uuid))


@router.delete("/{evaluator_uuid}", summary="Delete evaluator")
async def delete_evaluator_endpoint(
    evaluator_uuid: str = Path(
        description="Evaluator to delete",
        examples=["f47ac10b-58cc-4372-a567-0e02b2c3d479"],
    ),
    ctx: OrgContext = Depends(get_current_org),
):
    """Delete an evaluator you created"""
    existing = _visible_or_404(get_evaluator(evaluator_uuid), ctx.org_uuid)
    _owner_check(existing, ctx.org_uuid)
    if not delete_evaluator(evaluator_uuid):
        raise HTTPException(status_code=404, detail="Evaluator not found")
    return {"message": "Evaluator deleted"}


@router.post("/{evaluator_uuid}/duplicate", response_model=EvaluatorCreateResponse, summary="Duplicate evaluator")
async def duplicate_evaluator_endpoint(
    payload: EvaluatorDuplicateRequest,
    evaluator_uuid: str = Path(
        description="Evaluator to copy. Must be visible in your workspace",
        examples=["f47ac10b-58cc-4372-a567-0e02b2c3d479"],
    ),
    ctx: OrgContext = Depends(get_current_org),
):
    """Copy any evaluator you can see (including a built-in default) into a new one you can edit"""
    _visible_or_404(get_evaluator(evaluator_uuid), ctx.org_uuid)
    _ensure_unique_evaluator_name(payload.name, ctx.org_uuid)
    with name_uniqueness_guard("Evaluator"):
        new_uuid = duplicate_evaluator(
            evaluator_uuid,
            new_name=payload.name,
            org_uuid=ctx.org_uuid,
            owner_user_id=ctx.user_id,
        )
    if not new_uuid:
        raise HTTPException(status_code=404, detail="Evaluator not found")
    new_evaluator = get_evaluator(new_uuid)
    return EvaluatorCreateResponse(
        uuid=new_uuid, version_uuid=new_evaluator.get("live_version_id") or ""
    )


# ============ Versions ============


@router.get("/{evaluator_uuid}/versions", response_model=List[EvaluatorVersionResponse], summary="List evaluator versions")
async def list_versions(
    evaluator_uuid: str = Path(
        description="Evaluator whose versions to list",
        examples=["f47ac10b-58cc-4372-a567-0e02b2c3d479"],
    ),
    ctx: OrgContext = Depends(get_current_org),
):
    """List an evaluator's full version history, oldest first"""
    evaluator = _visible_or_404(get_evaluator(evaluator_uuid), ctx.org_uuid)
    # Pass `output_type` so binary versions stored with a null
    # output_config surface the Correct/Wrong default — consistent with
    # the detail / list / annotation-tasks evaluator endpoints.
    output_type = evaluator.get("output_type", "binary")
    return [
        EvaluatorVersionResponse(**_version_dict(v, output_type))
        for v in get_evaluator_versions(evaluator_uuid)
    ]


@router.post("/{evaluator_uuid}/versions", response_model=VersionCreateResponse, summary="Create evaluator version", tags=["Public API"])
async def create_version(
    payload: EvaluatorVersionCreateRequest,
    evaluator_uuid: str = Path(
        description="Evaluator to add a version to",
        examples=["f47ac10b-58cc-4372-a567-0e02b2c3d479"],
    ),
    ctx: OrgContext = Depends(get_org_jwt_or_api_key),
):
    """Add a new version to an evaluator you created"""
    existing = _visible_or_404(get_evaluator(evaluator_uuid), ctx.org_uuid)
    _owner_check(existing, ctx.org_uuid)
    cfg = payload.output_config.model_dump(exclude_none=True) if payload.output_config else None
    try:
        version = create_evaluator_version(
            evaluator_uuid=evaluator_uuid,
            judge_model=payload.judge_model,
            system_prompt=payload.system_prompt,
            output_config=cfg,
            variables=[v.model_dump() for v in payload.variables]
            if payload.variables
            else None,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    if payload.make_live:
        set_evaluator_live_version(evaluator_uuid, version["uuid"])
    return VersionCreateResponse(
        version_uuid=version["uuid"], version_number=version["version_number"]
    )


@router.post("/{evaluator_uuid}/versions/live", summary="Set live version")
async def mark_live(
    payload: SetLiveVersionRequest,
    evaluator_uuid: str = Path(
        description="Evaluator whose live version to set",
        examples=["f47ac10b-58cc-4372-a567-0e02b2c3d479"],
    ),
    ctx: OrgContext = Depends(get_current_org),
):
    """Set which version is the evaluator's live version"""
    existing = _visible_or_404(get_evaluator(evaluator_uuid), ctx.org_uuid)
    _owner_check(existing, ctx.org_uuid)
    ok = set_evaluator_live_version(evaluator_uuid, payload.version_uuid)
    if not ok:
        raise HTTPException(status_code=404, detail="Version not found")
    return {"message": "Live version updated"}


# ============ Prompt preview (authenticated) ============


class PromptPreviewRequest(BaseModel):
    version_uuid: Optional[str] = Field(
        None,
        min_length=36,
        max_length=36,
        description="Version to render. Omit to use the evaluator's live version",
        examples=[_EXAMPLE_VERSION_UUID],
    )
    variables: Optional[Dict[str, Any]] = Field(
        None, description="Values substituted into `{{placeholders}}`. Omit to render with none"
    )


@router.post("/{evaluator_uuid}/preview-prompt", summary="Preview evaluator prompt")
async def preview_prompt(
    payload: PromptPreviewRequest,
    evaluator_uuid: str = Path(
        description="Evaluator whose prompt to preview",
        examples=["f47ac10b-58cc-4372-a567-0e02b2c3d479"],
    ),
    ctx: OrgContext = Depends(get_current_org),
):
    """Render a version's system prompt with the supplied variables and return the resolved text"""
    evaluator = _visible_or_404(get_evaluator(evaluator_uuid), ctx.org_uuid)
    version_uuid = payload.version_uuid or evaluator.get("live_version_id")
    if not version_uuid:
        raise HTTPException(status_code=400, detail="Evaluator has no live version")
    version = get_evaluator_version(version_uuid)
    if not version or version["evaluator_id"] != evaluator_uuid:
        raise HTTPException(status_code=404, detail="Version not found")
    rendered = render_template(version["system_prompt"], payload.variables or {})
    return {"rendered_system_prompt": rendered}
