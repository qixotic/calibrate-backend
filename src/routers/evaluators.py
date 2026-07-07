"""Evaluators router.

Replaces the legacy `metrics` router. Adds:
 - Default (seeded) vs custom (per-user) evaluators
 - Versioned prompts with judge_model, rating scale, variables
 - Duplicate-to-custom
 - Live-version selection
 - API-key-authenticated invocation endpoint
"""

from typing import Any, Dict, List, Literal, Optional

from fastapi import APIRouter, Depends, HTTPException, Path, Query
from pydantic import BaseModel, Field, model_validator

from auth_utils import get_current_org, get_current_user_id, OrgContext
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
    set_evaluator_live_version,
    update_evaluator,
)
from llm_judge import render_template
from utils import EvaluatorTypeLiteral, DataTypeLiteral

router = APIRouter(prefix="/evaluators", tags=["evaluators"])

_EXAMPLE_EVALUATOR_UUID = "f47ac10b-58cc-4372-a567-0e02b2c3d479"
_EXAMPLE_VERSION_UUID = "6ba7b810-9dad-11d1-80b4-00c04fd430c8"
_EXAMPLE_USER_UUID = "a3b2c1d0-e5f4-3210-abcd-ef1234567890"


# ============ Pydantic models ============


class OutputScaleEntry(BaseModel):
    """One entry in an evaluator's output_config.scale. `value` is bool|number|string
 depending on output_type."""

    value: Any = Field(
        description="Scale point value: `bool` for `binary` (2 entries), numeric for `rating` (N>=2 entries)"
    )
    name: str = Field(description="Short human-readable label for this scale point")
    description: Optional[str] = Field(
        None,
        description="Rubric text for this level, injected into the judge prompt. Omit to leave this level undescribed",
    )
    color: Optional[str] = Field(None, description="UI color hint for this level. Omit for the default")


class OutputConfig(BaseModel):
    scale: Optional[List[OutputScaleEntry]] = Field(
        None,
        description="Ordered scale points defining the rubric. **Required (>=2 entries) when `output_type=rating`**",
    )


class VariableSpec(BaseModel):
    name: str = Field(description="`{{placeholder}}` name used in the system prompt (immutable across versions)")
    description: Optional[str] = Field(None, description="Human-readable description of the variable. Omit if self-evident")
    default: Optional[str] = Field(None, description="Default value used when you omit this variable. Omit for no default")


class EvaluatorVersionCreate(BaseModel):
    """Version-level config — prompt, model, variables, and the rubric (output_config).

 The rubric is version-owned so links that pin a version (tests, simulations) get
 reproducible judge prompts even if the evaluator's live version is later changed.
 """

    judge_model: str = Field(description="Model that runs the judge (e.g. an OpenRouter model slug)")
    system_prompt: str = Field(description="Judge system prompt; may contain `{{variable}}` placeholders")
    output_config: Optional[OutputConfig] = Field(
        None,
        description="Rubric definition. **Required for `output_type=rating`**; omit for `binary` to use the default Correct/Wrong scale",
    )
    variables: Optional[List[VariableSpec]] = Field(
        None, description="Declared prompt variables. Omit if the prompt has no `{{placeholders}}`"
    )


class EvaluatorVersionCreateRequest(EvaluatorVersionCreate):
    """Request body for adding a version to an existing evaluator."""

    make_live: bool = Field(
        False, description="When `true`, immediately point the evaluator's live version at this new version"
    )


class EvaluatorCreate(BaseModel):
    name: str = Field(..., min_length=1, description="Evaluator name, unique within your workspace")
    description: Optional[str] = Field(None, description="Human-readable description. Omit to leave blank")
    evaluator_type: EvaluatorTypeLiteral = Field(
        "llm",
        description="Semantic category: `tts` judges TTS audio, `stt` one transcript, `llm` a reply with history, `llm-general` a standalone input->output pair, `conversation` a full conversation",
    )
    data_type: DataTypeLiteral = Field(
        "text", description="Medium the judge consumes: `text` or `audio` (the only field gating audio routing)"
    )
    kind: Literal["single", "side_by_side"] = Field(
        "single", description="`single` judges one output; `side_by_side` compares two and picks a winner"
    )
    output_type: Literal["binary", "rating"] = Field(
        "binary", description="`binary` = pass/fail; `rating` = numeric scale (requires `version.output_config.scale`)"
    )
    version: EvaluatorVersionCreate = Field(description="Initial version (prompt, model, rubric, variables); set as live on creation")

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
        None, description="New semantic category. Omit to leave unchanged"
    )
    data_type: Optional[DataTypeLiteral] = Field(None, description="New medium (`text`/`audio`). Omit to leave unchanged")
    kind: Optional[Literal["single", "side_by_side"]] = Field(
        None, description="New kind (`single`/`side_by_side`). Omit to leave unchanged"
    )
    output_type: Optional[Literal["binary", "rating"]] = Field(
        None, description="New output type (`binary`/`rating`). Omit to leave unchanged"
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
    version_number: int = Field(description="1-based version number, incrementing per evaluator")
    judge_model: str = Field(description="Model that runs the judge for this version")
    system_prompt: str = Field(description="Judge system prompt, with `{{variable}}` placeholders unrendered")
    output_config: Optional[Dict[str, Any]] = Field(
        None, description="Rubric for this version; binary versions fall back to the default Correct/Wrong scale"
    )
    variables: Optional[List[Dict[str, Any]]] = Field(None, description="Declared prompt variables, or null if none")
    created_at: str = Field(description="Version creation timestamp (ISO 8601 UTC)")


class EvaluatorResponseBase(BaseModel):
    """Identity + classification fields shared by every evaluator response.

 `live_version_id` is the FK pointer to the current live version. List
 views inline the full version on `live_version` because there's no
 `versions[]` to look it up in. Detail views skip the inlined block and
 expose `versions[]` instead — clients resolve the live version by
 matching `live_version_id` to a version entry's `uuid` (avoids
 duplicating the same version payload twice in the same response).
 """

    uuid: str = Field(
        min_length=36,
        max_length=36,
        description="Evaluator ID",
        examples=[_EXAMPLE_EVALUATOR_UUID],
    )
    name: str = Field(description="Evaluator name")
    description: Optional[str] = Field(None, description="Human-readable description, or null")
    evaluator_type: EvaluatorTypeLiteral = Field(
        description="Semantic category (`tts`/`stt`/`llm`/`llm-general`/`conversation`)"
    )
    data_type: DataTypeLiteral = Field(description="Medium the judge consumes (`text`/`audio`)")
    kind: Literal["single", "side_by_side"] = Field(description="`single` or `side_by_side`")
    output_type: Literal["binary", "rating"] = Field(description="`binary` or `rating`")
    owner_user_id: Optional[str] = Field(
        None,
        min_length=36,
        max_length=36,
        description="Creator user ID; null for seeded defaults visible in your workspace but not editable by you",
        examples=[_EXAMPLE_USER_UUID],
    )
    slug: Optional[str] = Field(None, description="Stable slug for seeded defaults; null for custom evaluators")
    live_version_id: Optional[str] = Field(
        None,
        min_length=36,
        max_length=36,
        description="ID of the current live version; null if none is set",
        examples=[_EXAMPLE_VERSION_UUID],
    )
    created_at: str = Field(description="Creation timestamp (ISO 8601 UTC)")
    updated_at: str = Field(description="Last-update timestamp (ISO 8601 UTC)")


class EvaluatorResponse(EvaluatorResponseBase):
    # List shape: no `versions[]` here, so we inline the live version for
    # the FE.
    live_version: Optional[EvaluatorVersionResponse] = Field(
        None, description="Full live version inlined for list views; null if the evaluator has no live version"
    )


class EvaluatorDetailResponse(EvaluatorResponseBase):
    # Detail shape: `versions[]` is the full history; `live_version_index`
    # is the direct array position of the live version (None when the
    # evaluator has no live version, or when the live id doesn't resolve
    # to anything in `versions[]`). Clients should prefer the index over
    # scanning `versions[]` by uuid.
    versions: List[EvaluatorVersionResponse] = Field(description="Full version history, oldest first")
    live_version_index: Optional[int] = Field(
        None, description="Array position of the live version within `versions[]`; null if unset or unresolved"
    )


class EvaluatorCreateResponse(BaseModel):
    uuid: str = Field(
        min_length=36,
        max_length=36,
        description="ID of the created (or duplicated) evaluator",
        examples=[_EXAMPLE_EVALUATOR_UUID],
    )
    version_uuid: str = Field(
        min_length=36,
        max_length=36,
        description="ID of its initial/live version",
        examples=[_EXAMPLE_VERSION_UUID],
    )


class VersionCreateResponse(BaseModel):
    version_uuid: str = Field(
        min_length=36,
        max_length=36,
        description="ID of the newly created version",
        examples=[_EXAMPLE_VERSION_UUID],
    )
    version_number: int = Field(description="1-based number assigned to the new version")


class SetLiveVersionRequest(BaseModel):
    version_uuid: str = Field(
        min_length=36,
        max_length=36,
        description="ID of the version to mark as live (must belong to this evaluator)",
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


def _evaluator_response(evaluator: Dict[str, Any]) -> EvaluatorResponse:
    live_version = None
    output_type = evaluator.get("output_type", "binary")
    if evaluator.get("live_version_id"):
        v = get_evaluator_version(evaluator["live_version_id"])
        if v:
            live_version = EvaluatorVersionResponse(
                **_version_dict(v, output_type)
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
        slug=evaluator.get("slug"),
        live_version_id=evaluator.get("live_version_id"),
        created_at=evaluator["created_at"],
        updated_at=evaluator["updated_at"],
        live_version=live_version,
    )


# ============ CRUD ============


@router.post("", response_model=EvaluatorCreateResponse, summary="Create evaluator")
async def create_evaluator_endpoint(
    payload: EvaluatorCreate, ctx: OrgContext = Depends(get_current_org)
):
    """Create a custom evaluator in your workspace along with its first version, which is set live."""
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
 no seeded conversation evaluator — the prompt is just a template."""

    purpose: Literal["llm", "llm-general", "stt", "tts", "conversation"] = Field(
        description="Evaluation purpose this default prompt targets"
    )
    name: Optional[str] = Field(
        None, description="Seeded evaluator name; null for `conversation` (template only, no seeded evaluator)"
    )
    system_prompt: str = Field(description="Suggested judge system prompt for prefilling the create form")
    judge_model: str = Field(description="Suggested judge model")
    evaluator_type: EvaluatorTypeLiteral = Field(description="Suggested semantic category")
    data_type: DataTypeLiteral = Field(description="Suggested medium (`text`/`audio`)")
    kind: Literal["single", "side_by_side"] = Field(description="Suggested kind (`single`/`side_by_side`)")
    output_type: Literal["binary", "rating"] = Field(description="Suggested output type (`binary`/`rating`)")
    output_config: Optional[Dict[str, Any]] = Field(None, description="Suggested rubric, or null")
    variables: List[Dict[str, Any]] = Field(default=[], description="Suggested prompt variables (empty if none)")


@router.get("/default-prompt", response_model=DefaultPromptResponse, summary="Get default prompt")
async def get_default_prompt(
    purpose: Literal["llm", "llm-general", "stt", "tts", "conversation"] = Query(
        description="Evaluation purpose whose canonical default prompt + config to return"
    ),
    _user_id: str = Depends(get_current_user_id),
):
    """Get the canonical default prompt and suggested config for prefilling the create-evaluator form."""
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


@router.get("", response_model=List[EvaluatorResponse], summary="List evaluators")
async def list_evaluators(
    evaluator_type: Optional[EvaluatorTypeLiteral] = Query(
        None, description="Filter by semantic category. Omit for all types"
    ),
    data_type: Optional[DataTypeLiteral] = Query(
        None, description="Filter by medium (`text`/`audio`). Omit for all"
    ),
    include_defaults: bool = Query(
        True, description="When `true`, include seeded default evaluators alongside your workspace's custom ones"
    ),
    ctx: OrgContext = Depends(get_current_org),
):
    """List evaluators visible in your workspace, each with its inlined live version."""
    evaluators = get_all_evaluators(
        org_uuid=ctx.org_uuid,
        include_defaults=include_defaults,
        evaluator_type=evaluator_type,
        data_type=data_type,
    )
    return [_evaluator_response(e) for e in evaluators]


@router.get("/{evaluator_uuid}", response_model=EvaluatorDetailResponse, summary="Get evaluator")
async def get_evaluator_endpoint(
    evaluator_uuid: str = Path(
        description="Evaluator to retrieve. Must be in your workspace.",
        examples=["f47ac10b-58cc-4372-a567-0e02b2c3d479"],
    ),
    ctx: OrgContext = Depends(get_current_org),
):
    """Get one evaluator with its full version history."""
    evaluator = _visible_or_404(get_evaluator(evaluator_uuid), ctx.org_uuid)
    base = _evaluator_response(evaluator)
    output_type = evaluator.get("output_type", "binary")
    versions = [
        EvaluatorVersionResponse(**_version_dict(v, output_type))
        for v in get_evaluator_versions(evaluator_uuid)
    ]
    # base carries `live_version` (list shape); drop it here — detail uses
    # `versions[]` + `live_version_id`/`live_version_index` so we don't
    # duplicate the version.
    return EvaluatorDetailResponse(
        **base.model_dump(exclude={"live_version"}),
        versions=versions,
        live_version_index=_live_version_index(versions, base.live_version_id),
    )


@router.put("/{evaluator_uuid}", response_model=EvaluatorResponse, summary="Update evaluator")
async def update_evaluator_endpoint(
    payload: EvaluatorUpdate,
    evaluator_uuid: str = Path(
        description="Evaluator to update. Must be in your workspace.",
        examples=["f47ac10b-58cc-4372-a567-0e02b2c3d479"],
    ),
    ctx: OrgContext = Depends(get_current_org),
):
    """Update an evaluator's name, description, and classification fields."""
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
        description="Evaluator to delete. Must be in your workspace.",
        examples=["f47ac10b-58cc-4372-a567-0e02b2c3d479"],
    ),
    ctx: OrgContext = Depends(get_current_org),
):
    """Soft-delete a custom evaluator in your workspace."""
    existing = _visible_or_404(get_evaluator(evaluator_uuid), ctx.org_uuid)
    _owner_check(existing, ctx.org_uuid)
    if not delete_evaluator(evaluator_uuid):
        raise HTTPException(status_code=404, detail="Evaluator not found")
    return {"message": "Evaluator deleted"}


@router.post("/{evaluator_uuid}/duplicate", response_model=EvaluatorCreateResponse, summary="Duplicate evaluator")
async def duplicate_evaluator_endpoint(
    payload: EvaluatorDuplicateRequest,
    evaluator_uuid: str = Path(
        description="Evaluator to copy. Must be visible in your workspace.",
        examples=["f47ac10b-58cc-4372-a567-0e02b2c3d479"],
    ),
    ctx: OrgContext = Depends(get_current_org),
):
    """Copy any visible evaluator (including a seeded default) into a new editable custom evaluator that you own."""
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
        description="Evaluator whose versions to list. Must be in your workspace.",
        examples=["f47ac10b-58cc-4372-a567-0e02b2c3d479"],
    ),
    ctx: OrgContext = Depends(get_current_org),
):
    """List an evaluator's full version history, oldest first."""
    evaluator = _visible_or_404(get_evaluator(evaluator_uuid), ctx.org_uuid)
    # Pass `output_type` so binary versions stored with a null
    # output_config surface the Correct/Wrong default — consistent with
    # the detail / list / annotation-tasks evaluator endpoints.
    output_type = evaluator.get("output_type", "binary")
    return [
        EvaluatorVersionResponse(**_version_dict(v, output_type))
        for v in get_evaluator_versions(evaluator_uuid)
    ]


@router.post("/{evaluator_uuid}/versions", response_model=VersionCreateResponse, summary="Create evaluator version")
async def create_version(
    payload: EvaluatorVersionCreateRequest,
    evaluator_uuid: str = Path(
        description="Evaluator to add a version to. Must be in your workspace.",
        examples=["f47ac10b-58cc-4372-a567-0e02b2c3d479"],
    ),
    ctx: OrgContext = Depends(get_current_org),
):
    """Add a new version to a custom evaluator in your workspace."""
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
        description="Evaluator whose live version to set. Must be in your workspace.",
        examples=["f47ac10b-58cc-4372-a567-0e02b2c3d479"],
    ),
    ctx: OrgContext = Depends(get_current_org),
):
    """Set which version is the evaluator's live version."""
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
        description="Evaluator whose prompt to preview. Must be in your workspace.",
        examples=["f47ac10b-58cc-4372-a567-0e02b2c3d479"],
    ),
    ctx: OrgContext = Depends(get_current_org),
):
    """Render a version's system prompt with the supplied variables and return the resolved text."""
    evaluator = _visible_or_404(get_evaluator(evaluator_uuid), ctx.org_uuid)
    version_uuid = payload.version_uuid or evaluator.get("live_version_id")
    if not version_uuid:
        raise HTTPException(status_code=400, detail="Evaluator has no live version")
    version = get_evaluator_version(version_uuid)
    if not version or version["evaluator_id"] != evaluator_uuid:
        raise HTTPException(status_code=404, detail="Version not found")
    rendered = render_template(version["system_prompt"], payload.variables or {})
    return {"rendered_system_prompt": rendered}
