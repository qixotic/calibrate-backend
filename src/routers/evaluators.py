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

from auth_utils import get_current_user_id
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


EvaluatorTypeLiteral = Literal["tts", "stt", "llm", "simulation"]
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


class EvaluatorResponse(BaseModel):
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
    live_version: Optional[EvaluatorVersionResponse] = None


class EvaluatorDetailResponse(EvaluatorResponse):
    versions: List[EvaluatorVersionResponse]


class EvaluatorCreateResponse(BaseModel):
    uuid: str
    version_uuid: str


class VersionCreateResponse(BaseModel):
    version_uuid: str
    version_number: int


class SetLiveVersionRequest(BaseModel):
    version_uuid: str


# ============ Helpers ============


def _owner_check(evaluator: Dict[str, Any], user_id: str) -> None:
    """Defaults (owner_user_id IS NULL) are visible to everyone but mutable by no one."""
    if evaluator.get("owner_user_id") is None:
        raise HTTPException(status_code=403, detail="Default evaluators cannot be modified")
    if evaluator.get("owner_user_id") != user_id:
        raise HTTPException(status_code=404, detail="Evaluator not found")


def _visible_or_404(evaluator: Optional[Dict[str, Any]], user_id: str) -> Dict[str, Any]:
    if not evaluator:
        raise HTTPException(status_code=404, detail="Evaluator not found")
    if evaluator.get("owner_user_id") is not None and evaluator["owner_user_id"] != user_id:
        raise HTTPException(status_code=404, detail="Evaluator not found")
    return evaluator


def _ensure_unique_evaluator_name(
    name: str,
    user_id: str,
    exclude_uuid: Optional[str] = None,
) -> None:
    if evaluator_name_exists(name, owner_user_id=user_id, exclude_uuid=exclude_uuid):
        raise HTTPException(status_code=409, detail="Evaluator name already exists")


def _version_dict(version: Dict[str, Any]) -> Dict[str, Any]:
    """Shape an evaluator_versions row for the API response."""
    return {
        "uuid": version["uuid"],
        "version_number": version["version_number"],
        "judge_model": version["judge_model"],
        "system_prompt": version["system_prompt"],
        "output_config": version.get("output_config"),
        "variables": version.get("variables"),
        "created_at": version["created_at"],
    }


def _evaluator_response(evaluator: Dict[str, Any]) -> EvaluatorResponse:
    live_version = None
    if evaluator.get("live_version_id"):
        v = get_evaluator_version(evaluator["live_version_id"])
        if v:
            live_version = EvaluatorVersionResponse(**_version_dict(v))
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
    payload: EvaluatorCreate, user_id: str = Depends(get_current_user_id)
):
    _ensure_unique_evaluator_name(payload.name, user_id)
    with name_uniqueness_guard("Evaluator"):
        evaluator_uuid = create_evaluator(
            name=payload.name,
            description=payload.description,
            evaluator_type=payload.evaluator_type,
            data_type=payload.data_type,
            kind=payload.kind,
            output_type=payload.output_type,
            owner_user_id=user_id,
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
    create-evaluator form. The `name` field is null for `purpose=simulation` because there's
    no seeded simulation evaluator — the prompt is just a template."""

    purpose: Literal["llm", "stt", "tts", "simulation"]
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
    purpose: Literal["llm", "stt", "tts", "simulation"],
    _user_id: str = Depends(get_current_user_id),
):
    """Return the canonical default prompt + suggested config for a given purpose.

    For `llm`, `stt`, `tts` this matches the seeded default evaluator. For `simulation`
    there is no seeded evaluator — the response gives a template the frontend can use
    to prefill a "create simulation evaluator" form (the user replaces the literal
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
    user_id: str = Depends(get_current_user_id),
):
    evaluators = get_all_evaluators(
        user_id=user_id,
        include_defaults=include_defaults,
        evaluator_type=evaluator_type,
        data_type=data_type,
    )
    return [_evaluator_response(e) for e in evaluators]


@router.get("/{evaluator_uuid}", response_model=EvaluatorDetailResponse)
async def get_evaluator_endpoint(
    evaluator_uuid: str, user_id: str = Depends(get_current_user_id)
):
    evaluator = _visible_or_404(get_evaluator(evaluator_uuid), user_id)
    base = _evaluator_response(evaluator)
    versions = [EvaluatorVersionResponse(**_version_dict(v)) for v in get_evaluator_versions(evaluator_uuid)]
    return EvaluatorDetailResponse(**base.model_dump(), versions=versions)


@router.put("/{evaluator_uuid}", response_model=EvaluatorResponse)
async def update_evaluator_endpoint(
    evaluator_uuid: str,
    payload: EvaluatorUpdate,
    user_id: str = Depends(get_current_user_id),
):
    existing = _visible_or_404(get_evaluator(evaluator_uuid), user_id)
    _owner_check(existing, user_id)
    if payload.name is not None:
        _ensure_unique_evaluator_name(payload.name, user_id, exclude_uuid=evaluator_uuid)
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
    evaluator_uuid: str, user_id: str = Depends(get_current_user_id)
):
    existing = _visible_or_404(get_evaluator(evaluator_uuid), user_id)
    _owner_check(existing, user_id)
    if not delete_evaluator(evaluator_uuid):
        raise HTTPException(status_code=404, detail="Evaluator not found")
    return {"message": "Evaluator deleted"}


@router.post("/{evaluator_uuid}/duplicate", response_model=EvaluatorCreateResponse)
async def duplicate_evaluator_endpoint(
    evaluator_uuid: str,
    payload: EvaluatorDuplicateRequest,
    user_id: str = Depends(get_current_user_id),
):
    _visible_or_404(get_evaluator(evaluator_uuid), user_id)
    _ensure_unique_evaluator_name(payload.name, user_id)
    with name_uniqueness_guard("Evaluator"):
        new_uuid = duplicate_evaluator(
            evaluator_uuid, new_name=payload.name, owner_user_id=user_id
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
    evaluator_uuid: str, user_id: str = Depends(get_current_user_id)
):
    _visible_or_404(get_evaluator(evaluator_uuid), user_id)
    return [EvaluatorVersionResponse(**_version_dict(v)) for v in get_evaluator_versions(evaluator_uuid)]


@router.post("/{evaluator_uuid}/versions", response_model=VersionCreateResponse)
async def create_version(
    evaluator_uuid: str,
    payload: EvaluatorVersionCreateRequest,
    user_id: str = Depends(get_current_user_id),
):
    existing = _visible_or_404(get_evaluator(evaluator_uuid), user_id)
    _owner_check(existing, user_id)
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
    user_id: str = Depends(get_current_user_id),
):
    existing = _visible_or_404(get_evaluator(evaluator_uuid), user_id)
    _owner_check(existing, user_id)
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
    user_id: str = Depends(get_current_user_id),
):
    evaluator = _visible_or_404(get_evaluator(evaluator_uuid), user_id)
    version_uuid = payload.version_uuid or evaluator.get("live_version_id")
    if not version_uuid:
        raise HTTPException(status_code=400, detail="Evaluator has no live version")
    version = get_evaluator_version(version_uuid)
    if not version or version["evaluator_id"] != evaluator_uuid:
        raise HTTPException(status_code=404, detail="Version not found")
    rendered = render_template(version["system_prompt"], payload.variables or {})
    return {"rendered_system_prompt": rendered}
