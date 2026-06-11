"""Evaluators router.

Replaces the legacy `metrics` router. Adds:
  - Default (seeded) vs custom (per-user) evaluators
  - Versioned prompts with judge_model, rating scale, variables
  - Duplicate-to-custom
  - Live-version selection
  - API-key-authenticated invocation endpoint
"""

from typing import Any, Dict, List, Literal, Optional

from fastapi import APIRouter, Depends, HTTPException
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

router = APIRouter(prefix="/evaluators", tags=["evaluators"])


# ============ Pydantic models ============


class OutputScaleEntry(BaseModel):
    """One entry in an evaluator's output_config.scale. `value` is bool|number|string
    depending on output_type."""

    value: Any
    name: str
    description: Optional[str] = None
    color: Optional[str] = None


class OutputConfig(BaseModel):
    scale: Optional[List[OutputScaleEntry]] = None


class VariableSpec(BaseModel):
    name: str
    description: Optional[str] = None
    default: Optional[str] = None


class EvaluatorVersionCreate(BaseModel):
    """Version-level config — prompt, model, variables, and the rubric (output_config).

    The rubric is version-owned so links that pin a version (tests, simulations) get
    reproducible judge prompts even if the evaluator's live version is later changed.
    """

    judge_model: str
    system_prompt: str
    output_config: Optional[OutputConfig] = None
    variables: Optional[List[VariableSpec]] = None


class EvaluatorVersionCreateRequest(EvaluatorVersionCreate):
    """Request body for adding a version to an existing evaluator."""

    make_live: bool = False


EvaluatorTypeLiteral = Literal["tts", "stt", "llm", "llm-general", "conversation"]
DataTypeLiteral = Literal["text", "audio"]


class EvaluatorCreate(BaseModel):
    name: str = Field(..., min_length=1)
    description: Optional[str] = None
    evaluator_type: EvaluatorTypeLiteral = "llm"
    data_type: DataTypeLiteral = "text"
    kind: Literal["single", "side_by_side"] = "single"
    output_type: Literal["binary", "rating"] = "binary"
    version: EvaluatorVersionCreate

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
    name: Optional[str] = Field(None, min_length=1)
    description: Optional[str] = None
    evaluator_type: Optional[EvaluatorTypeLiteral] = None
    data_type: Optional[DataTypeLiteral] = None
    kind: Optional[Literal["single", "side_by_side"]] = None
    output_type: Optional[Literal["binary", "rating"]] = None


class EvaluatorDuplicateRequest(BaseModel):
    name: str = Field(..., min_length=1)


class EvaluatorVersionResponse(BaseModel):
    uuid: str
    version_number: int
    judge_model: str
    system_prompt: str
    output_config: Optional[Dict[str, Any]] = None
    variables: Optional[List[Dict[str, Any]]] = None
    created_at: str


class EvaluatorResponseBase(BaseModel):
    """Identity + classification fields shared by every evaluator response.

    `live_version_id` is the FK pointer to the current live version. List
    views inline the full version on `live_version` because there's no
    `versions[]` to look it up in. Detail views skip the inlined block and
    expose `versions[]` instead — clients resolve the live version by
    matching `live_version_id` to a version entry's `uuid` (avoids
    duplicating the same version payload twice in the same response).
    """

    uuid: str
    name: str
    description: Optional[str] = None
    evaluator_type: str
    data_type: str
    kind: str
    output_type: str
    owner_user_id: Optional[str] = None
    slug: Optional[str] = None
    live_version_id: Optional[str] = None
    created_at: str
    updated_at: str


class EvaluatorResponse(EvaluatorResponseBase):
    # List shape: no `versions[]` here, so we inline the live version for
    # the FE.
    live_version: Optional[EvaluatorVersionResponse] = None


class EvaluatorDetailResponse(EvaluatorResponseBase):
    # Detail shape: `versions[]` is the full history; `live_version_index`
    # is the direct array position of the live version (None when the
    # evaluator has no live version, or when the live id doesn't resolve
    # to anything in `versions[]`). Clients should prefer the index over
    # scanning `versions[]` by uuid.
    versions: List[EvaluatorVersionResponse]
    live_version_index: Optional[int] = None


class EvaluatorCreateResponse(BaseModel):
    uuid: str
    version_uuid: str


class VersionCreateResponse(BaseModel):
    version_uuid: str
    version_number: int


class SetLiveVersionRequest(BaseModel):
    version_uuid: str


# ============ Helpers ============


def _owner_check(evaluator: Dict[str, Any], org_uuid: str) -> None:
    """Seeded defaults (org_uuid IS NULL) are visible to every org but mutable by no one."""
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


@router.post("", response_model=EvaluatorCreateResponse)
async def create_evaluator_endpoint(
    payload: EvaluatorCreate, ctx: OrgContext = Depends(get_current_org)
):
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

    purpose: Literal["llm", "llm-general", "stt", "tts", "conversation"]
    name: Optional[str] = None
    system_prompt: str
    judge_model: str
    evaluator_type: str
    data_type: str
    kind: str
    output_type: str
    output_config: Optional[Dict[str, Any]] = None
    variables: List[Dict[str, Any]] = []


@router.get("/default-prompt", response_model=DefaultPromptResponse)
async def get_default_prompt(
    purpose: Literal["llm", "llm-general", "stt", "tts", "conversation"],
    _user_id: str = Depends(get_current_user_id),
):
    """Return the canonical default prompt + suggested config for a given purpose.

    For `llm`, `stt`, `tts` this matches the seeded default evaluator. For `conversation`
    there is no seeded evaluator — the response gives a template the frontend can use
    to prefill a "create conversation evaluator" form (the user replaces the literal
    `<ENTER EVALUATION CRITERIA HERE>` placeholder in the prompt with their criteria).
    """
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


@router.get("", response_model=List[EvaluatorResponse])
async def list_evaluators(
    evaluator_type: Optional[EvaluatorTypeLiteral] = None,
    data_type: Optional[DataTypeLiteral] = None,
    include_defaults: bool = True,
    ctx: OrgContext = Depends(get_current_org),
):
    evaluators = get_all_evaluators(
        org_uuid=ctx.org_uuid,
        include_defaults=include_defaults,
        evaluator_type=evaluator_type,
        data_type=data_type,
    )
    return [_evaluator_response(e) for e in evaluators]


@router.get("/{evaluator_uuid}", response_model=EvaluatorDetailResponse)
async def get_evaluator_endpoint(
    evaluator_uuid: str, ctx: OrgContext = Depends(get_current_org)
):
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


@router.put("/{evaluator_uuid}", response_model=EvaluatorResponse)
async def update_evaluator_endpoint(
    evaluator_uuid: str,
    payload: EvaluatorUpdate,
    ctx: OrgContext = Depends(get_current_org),
):
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


@router.delete("/{evaluator_uuid}")
async def delete_evaluator_endpoint(
    evaluator_uuid: str, ctx: OrgContext = Depends(get_current_org)
):
    existing = _visible_or_404(get_evaluator(evaluator_uuid), ctx.org_uuid)
    _owner_check(existing, ctx.org_uuid)
    if not delete_evaluator(evaluator_uuid):
        raise HTTPException(status_code=404, detail="Evaluator not found")
    return {"message": "Evaluator deleted"}


@router.post("/{evaluator_uuid}/duplicate", response_model=EvaluatorCreateResponse)
async def duplicate_evaluator_endpoint(
    evaluator_uuid: str,
    payload: EvaluatorDuplicateRequest,
    ctx: OrgContext = Depends(get_current_org),
):
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


@router.get("/{evaluator_uuid}/versions", response_model=List[EvaluatorVersionResponse])
async def list_versions(
    evaluator_uuid: str, ctx: OrgContext = Depends(get_current_org)
):
    evaluator = _visible_or_404(get_evaluator(evaluator_uuid), ctx.org_uuid)
    # Pass `output_type` so binary versions stored with a null
    # output_config surface the Correct/Wrong default — consistent with
    # the detail / list / annotation-tasks evaluator endpoints.
    output_type = evaluator.get("output_type", "binary")
    return [
        EvaluatorVersionResponse(**_version_dict(v, output_type))
        for v in get_evaluator_versions(evaluator_uuid)
    ]


@router.post("/{evaluator_uuid}/versions", response_model=VersionCreateResponse)
async def create_version(
    evaluator_uuid: str,
    payload: EvaluatorVersionCreateRequest,
    ctx: OrgContext = Depends(get_current_org),
):
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


@router.post("/{evaluator_uuid}/versions/live")
async def mark_live(
    evaluator_uuid: str,
    payload: SetLiveVersionRequest,
    ctx: OrgContext = Depends(get_current_org),
):
    existing = _visible_or_404(get_evaluator(evaluator_uuid), ctx.org_uuid)
    _owner_check(existing, ctx.org_uuid)
    ok = set_evaluator_live_version(evaluator_uuid, payload.version_uuid)
    if not ok:
        raise HTTPException(status_code=404, detail="Version not found")
    return {"message": "Live version updated"}


# ============ Prompt preview (authenticated) ============


class PromptPreviewRequest(BaseModel):
    version_uuid: Optional[str] = None
    variables: Optional[Dict[str, Any]] = None


@router.post("/{evaluator_uuid}/preview-prompt")
async def preview_prompt(
    evaluator_uuid: str,
    payload: PromptPreviewRequest,
    ctx: OrgContext = Depends(get_current_org),
):
    evaluator = _visible_or_404(get_evaluator(evaluator_uuid), ctx.org_uuid)
    version_uuid = payload.version_uuid or evaluator.get("live_version_id")
    if not version_uuid:
        raise HTTPException(status_code=400, detail="Evaluator has no live version")
    version = get_evaluator_version(version_uuid)
    if not version or version["evaluator_id"] != evaluator_uuid:
        raise HTTPException(status_code=404, detail="Version not found")
    rendered = render_template(version["system_prompt"], payload.variables or {})
    return {"rendered_system_prompt": rendered}
