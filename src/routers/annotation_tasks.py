from typing import Optional, List, Dict, Any
from fastapi import APIRouter, HTTPException, Depends, Query
from pydantic import BaseModel

import logging
import secrets

logger = logging.getLogger(__name__)

from db import (
    ANNOTATION_TASK_TYPES,
    create_annotation_task,
    ensure_name_unique,
    get_annotation_task,
    get_all_annotation_tasks,
    update_annotation_task,
    delete_annotation_task,
    add_evaluator_to_annotation_task,
    remove_evaluator_from_annotation_task,
    reorder_evaluators_for_annotation_task,
    get_evaluators_for_annotation_task,
    get_evaluator,
    get_evaluator_version,
    get_annotations_for_task,
    get_annotations_for_slots,
    create_annotation_items,
    bulk_update_annotation_items,
    soft_delete_annotation_items,
    soft_delete_annotation_job,
    bulk_soft_delete_annotation_jobs,
    get_annotation_items_for_task,
    get_annotation_item,
    create_annotation_job,
    get_annotation_job,
    get_jobs_for_task,
    get_jobs_for_task_detailed,
    get_job_items,
    get_evaluator_ids_for_job,
    update_annotation_job_status,
    update_annotation_job_visibility,
    upsert_annotation,
    get_annotations_for_item,
    get_annotated_item_ids,
    get_annotator,
    get_annotators_by_uuids,
    create_job,
    get_job,
    snapshot_eval_job_items,
    get_eval_job_items,
    get_generic_jobs_for_task,
    soft_delete_job,
    update_job,
    update_job_visibility,
    get_evaluator_runs_for_job,
    get_evaluator_runs_for_item,
    get_evaluator_runs_for_task,
    clear_evaluator_runs_for_job,
)
from annotation_eval_runner import (
    ANNOTATION_EVAL_JOB_TYPE,
    EVAL_JOB_TYPES,
    SUPPORTED_EVAL_TASK_TYPES,
    EvaluatorResolutionError,
    DatasetBuildError,
    _resolve_evaluator_dicts,
    build_dataset_for_task_type,
    start_annotation_eval_job,
)
from utils import (
    TaskStatus,
    can_start_job,
    compute_share_token_toggle,
    try_start_queued_job,
)
from auth_utils import get_current_org, OrgContext
from pagination import PaginationParams, make_search_params, make_sort_params

# Per-endpoint sort/search allowlists for the summary view. Built at module
# load time so FastAPI's dependency-graph introspection sees stable types.
_SummarySort = make_sort_params(
    sortable=["created_at", "updated_at"],
    default="created_at",
    default_order="desc",
)
_SummarySearch = make_search_params(searchable=["payload.name"])
from annotation_metrics import (
    aggregate_agreement,
    aggregate_human_evaluator_agreement,
    evaluator_human_pair_agreement,
    per_item_agreement,
    trend_series,
    trend_series_human_evaluator,
    filter_runs_to_live_versions,
    _pairwise_agreement,
    _scalar,
    _round_agreement,
)


def _live_version_map(evaluators: List[Dict[str, Any]]) -> Dict[str, Optional[str]]:
    return {e["uuid"]: e.get("live_version_id") for e in evaluators}


# Re-exported for tests; canonical home is llm_judge so agent-tests/STT/TTS can
# share the same scalar→label mapping.
from llm_judge import evaluator_value_name as _evaluator_value_name  # noqa: E402


def _enrich_evaluators_with_live_version(
    evaluators: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Mutate `evaluators` in place to add live-version fields the FE
    needs to render the labelling form / display values against the
    correct rubric: `output_config`, `scale_min`, `scale_max`,
    `variables`. Mirrors the public-form enrichment in `routers/public.py`
    so the owner-side and annotator-side responses match. Versions are
    fetched via a tiny per-call cache so a task with N evaluators that
    happen to share a live version (rare) doesn't issue N reads."""
    from llm_judge import _scale_bounds  # local to avoid module-load cycle

    version_cache: Dict[str, Optional[Dict[str, Any]]] = {}
    for ev in evaluators:
        live_version_id = ev.get("live_version_id")
        if live_version_id and live_version_id not in version_cache:
            version_cache[live_version_id] = get_evaluator_version(live_version_id)
        version = version_cache.get(live_version_id) if live_version_id else None
        output_config = version.get("output_config") if version else None
        scale_min, scale_max = _scale_bounds(output_config)
        ev["output_config"] = output_config
        ev["scale_min"] = scale_min
        ev["scale_max"] = scale_max
        ev["variables"] = version.get("variables") if version else None
    return evaluators


router = APIRouter(prefix="/annotation-tasks", tags=["annotation-tasks"])


class AnnotationTaskCreate(BaseModel):
    name: str
    type: str
    description: Optional[str] = None
    evaluator_ids: Optional[List[str]] = None


class AnnotationTaskUpdate(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None


class AnnotationTaskResponse(BaseModel):
    uuid: str
    name: str
    type: str
    description: Optional[str] = None
    created_at: str
    updated_at: str
    evaluators: List[Dict[str, Any]] = []
    item_count: Optional[int] = None
    # Inlined on the single-task fetch only; the list endpoint leaves these
    # empty (use the dedicated /items and /jobs endpoints for those views).
    items: List[Dict[str, Any]] = []
    jobs: List[Dict[str, Any]] = []


class AnnotationTaskCreateResponse(BaseModel):
    uuid: str
    message: str


class EvaluatorLinkRequest(BaseModel):
    evaluator_id: str


class EvaluatorOrderRequest(BaseModel):
    # Full ordered list of currently-linked evaluator UUIDs. Must match the
    # active linked set exactly — this endpoint reorders, it does not
    # link/unlink. Send `[]` only if the task has no linked evaluators.
    evaluator_ids: List[str]


def _ensure_owned_task(task_uuid: str, org_uuid: str) -> Dict[str, Any]:
    task = get_annotation_task(task_uuid)
    if not task or task.get("org_uuid") != org_uuid:
        # 404 (not 403) — avoid leaking existence
        raise HTTPException(status_code=404, detail="Annotation task not found")
    return task


def _ensure_owned_evaluator(evaluator_uuid: str, org_uuid: str) -> Dict[str, Any]:
    evaluator = get_evaluator(evaluator_uuid)
    if not evaluator:
        raise HTTPException(status_code=404, detail="Evaluator not found")
    owner_org = evaluator.get("org_uuid")
    # org_uuid IS NULL ⇒ seeded default (visible to every org)
    if owner_org is not None and owner_org != org_uuid:
        raise HTTPException(status_code=404, detail="Evaluator not found")
    return evaluator


@router.post("", response_model=AnnotationTaskCreateResponse)
async def create_annotation_task_endpoint(
    payload: AnnotationTaskCreate,
    ctx: OrgContext = Depends(get_current_org),
):
    """Create a new annotation task. Optionally link evaluators in the same call."""
    if payload.type not in ANNOTATION_TASK_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"type must be one of {list(ANNOTATION_TASK_TYPES)}",
        )
    if payload.evaluator_ids:
        for evaluator_id in payload.evaluator_ids:
            _ensure_owned_evaluator(evaluator_id, ctx.org_uuid)

    with ensure_name_unique(
        "annotation_tasks", payload.name, ctx.org_uuid, entity="Annotation task"
    ):
        task_uuid = create_annotation_task(
            name=payload.name,
            description=payload.description,
            type=payload.type,
            org_uuid=ctx.org_uuid,
            user_id=ctx.user_id,
        )

    if payload.evaluator_ids:
        for evaluator_id in payload.evaluator_ids:
            add_evaluator_to_annotation_task(task_uuid, evaluator_id)

    return AnnotationTaskCreateResponse(
        uuid=task_uuid, message="Annotation task created successfully"
    )


@router.get("", response_model=List[AnnotationTaskResponse])
async def list_annotation_tasks(ctx: OrgContext = Depends(get_current_org)):
    """List all annotation tasks owned by the authenticated user."""
    tasks = get_all_annotation_tasks(org_uuid=ctx.org_uuid)
    for task in tasks:
        task["evaluators"] = get_evaluators_for_annotation_task(task["uuid"])
    return tasks


@router.get("/{task_uuid}", response_model=AnnotationTaskResponse)
async def get_annotation_task_endpoint(
    task_uuid: str, ctx: OrgContext = Depends(get_current_org)
):
    """Get an annotation task by UUID, including its linked evaluators,
    all items (each annotated with per-item agreement stats), and all jobs."""
    task = _ensure_owned_task(task_uuid, ctx.org_uuid)
    evaluators = _enrich_evaluators_with_live_version(
        get_evaluators_for_annotation_task(task_uuid)
    )
    task["evaluators"] = evaluators
    task["jobs"] = get_jobs_for_task_detailed(task_uuid)

    items = get_annotation_items_for_task(task_uuid)
    # Pre-fetch annotations + evaluator_runs once and bucket by item to avoid
    # an N+1 query pattern on the per-item agreement computation. Per-item
    # agreement uses whichever evaluator version actually ran on each slot —
    # the live-version filter is reserved for AGGREGATED agreement (task-level
    # `/agreement` and account-level `/annotation-agreement/trend`).
    all_annotations = get_annotations_for_task(task_uuid)
    all_runs = get_evaluator_runs_for_task(task_uuid)
    annotations_by_item: Dict[str, List[Dict[str, Any]]] = {}
    for a in all_annotations:
        annotations_by_item.setdefault(a["item_id"], []).append(a)
    runs_by_item: Dict[str, List[Dict[str, Any]]] = {}
    for r in all_runs:
        runs_by_item.setdefault(r["item_id"], []).append(r)
    evaluator_ids = [e["uuid"] for e in evaluators]
    for item in items:
        item["agreement"] = per_item_agreement(
            annotations_by_item.get(item["uuid"], []),
            runs_by_item.get(item["uuid"], []),
            evaluator_ids,
        )
    task["items"] = items
    return task


@router.put("/{task_uuid}", response_model=AnnotationTaskResponse)
async def update_annotation_task_endpoint(
    task_uuid: str,
    payload: AnnotationTaskUpdate,
    ctx: OrgContext = Depends(get_current_org),
):
    _ensure_owned_task(task_uuid, ctx.org_uuid)
    with ensure_name_unique(
        "annotation_tasks",
        payload.name,
        ctx.org_uuid,
        entity="Annotation task",
        exclude_uuid=task_uuid,
    ):
        updated = update_annotation_task(
            task_uuid=task_uuid,
            name=payload.name,
            description=payload.description,
        )
    if not updated:
        raise HTTPException(status_code=400, detail="No fields to update")
    task = get_annotation_task(task_uuid)
    task["evaluators"] = get_evaluators_for_annotation_task(task_uuid)
    return task


@router.delete("/{task_uuid}")
async def delete_annotation_task_endpoint(
    task_uuid: str, ctx: OrgContext = Depends(get_current_org)
):
    _ensure_owned_task(task_uuid, ctx.org_uuid)
    deleted = delete_annotation_task(task_uuid)
    if not deleted:
        raise HTTPException(status_code=404, detail="Annotation task not found")
    return {"message": "Annotation task deleted successfully"}


# ============ Evaluator linking ============


@router.get("/{task_uuid}/evaluators")
async def list_task_evaluators(
    task_uuid: str, ctx: OrgContext = Depends(get_current_org)
):
    """Return each evaluator linked to this task in the same shape as
    `GET /evaluators/{uuid}` — i.e. full `EvaluatorDetailResponse` with
    `live_version` and the complete `versions[]` history. Lets the FE
    render the per-evaluator detail (rubric, prompt, judge model,
    variable specs) without an N+1 fan-out of `/evaluators/{uuid}` calls.
    """
    # Lazy import to avoid a circular module-load between the two router
    # files (annotation_tasks ↔ evaluators).
    from routers.evaluators import (
        EvaluatorDetailResponse,
        EvaluatorVersionResponse,
        _evaluator_response,
        _live_version_index,
        _version_dict,
    )
    from db import get_evaluator_versions

    _ensure_owned_task(task_uuid, ctx.org_uuid)
    # `get_evaluators_for_annotation_task` projects a slim column set with a
    # `linked_at` alias on the pivot — it omits the evaluator row's own
    # `created_at`/`updated_at`. Refetch the canonical evaluator row so
    # `_evaluator_response` has every field it expects.
    linked = get_evaluators_for_annotation_task(task_uuid)
    out: List[EvaluatorDetailResponse] = []
    for stub in linked:
        ev = get_evaluator(stub["uuid"])
        if not ev:
            continue
        base = _evaluator_response(ev)
        ev_output_type = ev.get("output_type", "binary")
        versions = [
            EvaluatorVersionResponse(**_version_dict(v, ev_output_type))
            for v in get_evaluator_versions(ev["uuid"])
        ]
        out.append(
            EvaluatorDetailResponse(
                **base.model_dump(exclude={"live_version"}),
                versions=versions,
                live_version_index=_live_version_index(
                    versions, base.live_version_id
                ),
            )
        )
    return out


@router.post("/{task_uuid}/evaluators")
async def link_evaluator_to_task(
    task_uuid: str,
    payload: EvaluatorLinkRequest,
    ctx: OrgContext = Depends(get_current_org),
):
    _ensure_owned_task(task_uuid, ctx.org_uuid)
    _ensure_owned_evaluator(payload.evaluator_id, ctx.org_uuid)
    add_evaluator_to_annotation_task(task_uuid, payload.evaluator_id)
    return {"message": "Evaluator linked to annotation task"}


@router.put("/{task_uuid}/evaluators/order")
async def reorder_task_evaluators(
    task_uuid: str,
    payload: EvaluatorOrderRequest,
    ctx: OrgContext = Depends(get_current_org),
):
    """Re-number the display order of evaluators linked to a task.

    `evaluator_ids` MUST be the full ordered list of currently-active
    evaluators on the task — same set, no duplicates. This endpoint reorders
    only; it does not link or unlink. Mismatch returns 400.

    The new order is read by every surface that lists task evaluators
    (`GET /annotation-tasks/{uuid}`, `GET /annotation-tasks/{uuid}/evaluators`,
    per-item summary, `/agreement`, `/summary`, item-edit). Existing job
    snapshots are NOT re-ordered — by design, since a job's evaluator order is
    frozen at creation time.
    """
    _ensure_owned_task(task_uuid, ctx.org_uuid)
    try:
        reorder_evaluators_for_annotation_task(task_uuid, payload.evaluator_ids)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return {
        "message": "Evaluator order updated",
        "evaluators": get_evaluators_for_annotation_task(task_uuid),
    }


# ============ Items ============



class AnnotationItemPayload(BaseModel):
    # `payload` is a free-form JSON value whose shape is owned by the
    # task `type`. The backend doesn't validate the shape — frontend +
    # downstream consumers (evaluator runs, agreement, etc.) interpret it.
    # For all task types `payload["name"]` is required and must be unique
    # within the task.
    payload: Any
    # Optional human annotations to seed alongside the item. Keys are
    # evaluator UUIDs that must be currently linked to the task; values
    # follow the canonical annotation shape used by the public form and
    # the agreement math: `{"value": <bool|number|string>, "reasoning"?:
    # str}` for EVERY output_type — binary uses a bool in `value`,
    # rating uses a number in `value`. The agreement helpers in
    # `annotation_metrics._scalar` only recognise the keys `value`,
    # `score`, `rating`, `label`, `binary`; using `pass` (or any other
    # custom key) stores fine but silently zeroes out of the agreement
    # aggregates. When any item carries this, `BulkItemsRequest.annotator_id`
    # is required.
    annotations: Optional[Dict[str, Any]] = None


class BulkItemsRequest(BaseModel):
    items: List[AnnotationItemPayload]
    # Required only if any item in `items` carries `annotations`. The
    # annotator must be owned by the requesting user. All seeded
    # annotations are attributed to one synthesised completed job for
    # this annotator.
    annotator_id: Optional[str] = None


@router.get("/{task_uuid}/items")
async def list_task_items(
    task_uuid: str, ctx: OrgContext = Depends(get_current_org)
):
    _ensure_owned_task(task_uuid, ctx.org_uuid)
    return get_annotation_items_for_task(task_uuid)


class AnnotatedItemsCheckRequest(BaseModel):
    annotator_id: str
    names: List[str]


@router.post("/{task_uuid}/items/annotated-check")
async def check_annotated_items(
    task_uuid: str,
    payload: AnnotatedItemsCheckRequest,
    ctx: OrgContext = Depends(get_current_org),
):
    """Pre-upload check: given a list of item names (in upload row order),
    return which rows already exist in the task and whether the annotator
    has previously annotated them.

    Response shape:
      {
        "all_new": bool,
        "existing_with_annotations": [{"index": int, "name": str}],
        "existing_without_annotations": [{"index": int, "name": str}]
      }

    Rows whose name doesn't exist in the task are omitted from both lists.
    """
    _ensure_owned_task(task_uuid, ctx.org_uuid)
    if not payload.names:
        raise HTTPException(status_code=400, detail="names must be non-empty")
    annotator = get_annotator(payload.annotator_id)
    if not annotator or annotator.get("org_uuid") != ctx.org_uuid:
        raise HTTPException(status_code=404, detail="Annotator not found")

    existing_items = get_annotation_items_for_task(task_uuid)
    name_to_uuid = {
        it["payload"]["name"]: it["uuid"]
        for it in existing_items
        if isinstance(it.get("payload"), dict) and it["payload"].get("name")
    }

    matched = {
        i: name_to_uuid[name]
        for i, name in enumerate(payload.names)
        if name in name_to_uuid
    }

    annotated_ids = set(
        get_annotated_item_ids(payload.annotator_id, list(matched.values()))
    ) if matched else set()

    existing_with_annotations = [
        {"index": i, "name": payload.names[i]}
        for i, item_id in matched.items()
        if item_id in annotated_ids
    ]
    existing_without_annotations = [
        {"index": i, "name": payload.names[i]}
        for i, item_id in matched.items()
        if item_id not in annotated_ids
    ]

    return {
        "all_new": not matched,
        "existing_with_annotations": existing_with_annotations,
        "existing_without_annotations": existing_without_annotations,
    }


@router.post("/{task_uuid}/items")
async def bulk_create_items(
    task_uuid: str,
    payload: BulkItemsRequest,
    ctx: OrgContext = Depends(get_current_org),
):
    """Bulk-insert annotation items. Order of insertion is preserved by `id`.

    Optionally seeds human annotations alongside the items: when any item
    in the request carries `annotations`, `annotator_id` must be supplied
    and one completed `annotation_jobs` row is synthesised for that
    annotator covering every newly-inserted item. Annotations are upserted
    as if the annotator had submitted them through the public form.
    Annotations are validated against the task's *currently linked*
    evaluator set; the task type (`stt | llm | llm-general | conversation |
    tts`) does not affect the contract. Value shape is uniform across output types:
    `{"value": <bool|number|string>, "reasoning"?: str}` — binary uses a
    bool, rating uses a number. This matches what the public form writes
    via `upsert_annotation`, and is the only shape `annotation_metrics`
    will count toward agreement aggregates."""
    task = _ensure_owned_task(task_uuid, ctx.org_uuid)
    if not payload.items:
        raise HTTPException(status_code=400, detail="items must be non-empty")

    missing = [
        i
        for i, it in enumerate(payload.items)
        if not (isinstance(it.payload, dict) and isinstance(it.payload.get("name"), str) and it.payload["name"])
    ]
    if missing:
        raise HTTPException(
            status_code=400,
            detail=(
                f"`payload.name` is required for {task['type']} task items. "
                f"Missing on items at index(es): {missing}"
            ),
        )
    names_in_batch = [it.payload["name"] for it in payload.items]
    if len(names_in_batch) != len(set(names_in_batch)):
        seen: set = set()
        dupes = sorted({n for n in names_in_batch if n in seen or seen.add(n)})  # type: ignore[func-returns-value]
        raise HTTPException(
            status_code=409,
            detail={
                "code": "ITEM_NAME_DUPLICATE_IN_REQUEST",
                "message": f"Duplicate `payload.name` value(s) within request: {dupes}",
                "conflicting_names": dupes,
            },
        )
    existing_name_to_uuid: Dict[str, str] = {}
    if task.get("item_count", 0) > 0:
        existing_items = get_annotation_items_for_task(task_uuid)
        existing_name_to_uuid = {
            it["payload"]["name"]: it["uuid"]
            for it in existing_items
            if isinstance(it.get("payload"), dict) and it["payload"].get("name")
        }
    # matched_existing: request index → UUID of the pre-existing item with
    # the same payload.name.
    matched_existing: Dict[int, str] = {
        i: existing_name_to_uuid[it.payload["name"]]
        for i, it in enumerate(payload.items)
        if it.payload["name"] in existing_name_to_uuid
    }

    items_with_annotations = [
        it for it in payload.items if it.annotations is not None
    ]
    if items_with_annotations and not payload.annotator_id:
        raise HTTPException(
            status_code=400,
            detail="annotator_id is required when any item carries `annotations`",
        )

    # Reject name conflicts only when no annotations are being supplied.
    # When annotations are present, existing items are folded into the new
    # job instead (see below).
    if matched_existing and not items_with_annotations:
        conflicts = sorted(
            payload.items[i].payload["name"] for i in matched_existing
        )
        raise HTTPException(
            status_code=409,
            detail={
                "code": "ITEM_NAME_CONFLICT",
                "message": f"`payload.name` already exists in this task: {conflicts}",
                "conflicting_names": conflicts,
            },
        )

    annotator: Optional[Dict[str, Any]] = None
    linked_evaluator_ids: set = set()
    if items_with_annotations:
        annotator = get_annotator(payload.annotator_id)
        if not annotator or annotator.get("org_uuid") != ctx.org_uuid:
            # 404 (not 403) — avoid leaking existence
            raise HTTPException(status_code=404, detail="Annotator not found")
        linked_evaluator_ids = {
            e["uuid"] for e in get_evaluators_for_annotation_task(task_uuid)
        }
        for idx, it in enumerate(items_with_annotations):
            if not isinstance(it.annotations, dict):
                raise HTTPException(
                    status_code=400,
                    detail=f"items[{idx}].annotations must be an object keyed by evaluator UUID",
                )
            unknown = [
                ev_id
                for ev_id in it.annotations.keys()
                if ev_id not in linked_evaluator_ids
            ]
            if unknown:
                raise HTTPException(
                    status_code=400,
                    detail=(
                        f"Evaluator(s) not linked to this task: {unknown}. "
                        f"Link them via POST /annotation-tasks/{task_uuid}/evaluators "
                        f"before seeding annotations."
                    ),
                )
            # Validate the value shape on every entry so a malformed dict
            # can't slip through and silently zero out of the agreement
            # aggregates (`annotation_metrics._scalar` only recognises
            # the keys `value`, `score`, `rating`, `label`, `binary`).
            # Bulk uploads are the only ingress path that doesn't go
            # through the public form's typed widget, so the canonical
            # check lives here.
            for ev_id, raw in it.annotations.items():
                if not isinstance(raw, dict):
                    raise HTTPException(
                        status_code=400,
                        detail=(
                            f"items[{idx}].annotations[{ev_id!r}] must be an object "
                            f"like {{\"value\": <bool|number|string>, \"reasoning\"?: str}}; "
                            f"got {type(raw).__name__}"
                        ),
                    )
                if "value" not in raw:
                    raise HTTPException(
                        status_code=400,
                        detail=(
                            f"items[{idx}].annotations[{ev_id!r}] is missing required key "
                            f"`value`. Use {{\"value\": <bool|number|string>, "
                            f"\"reasoning\"?: str}} for every output_type — binary uses a "
                            f"bool in `value`, rating uses a number. (The keys `pass`, "
                            f"`score`, `rating`, `label`, `binary` will round-trip on "
                            f"reads but won't count toward agreement aggregates.)"
                        ),
                    )

    # Create only the items that don't already exist by name.
    new_uuids = create_annotation_items(
        task_uuid,
        [
            {"payload": it.payload}
            for i, it in enumerate(payload.items)
            if i not in matched_existing
        ],
    )

    # Build a per-request-index UUID map: new items get a freshly created
    # UUID; name-matched items reuse their existing UUID.
    new_uuid_iter = iter(new_uuids)
    item_uuid_by_index: Dict[int, str] = {
        i: matched_existing[i] if i in matched_existing else next(new_uuid_iter)
        for i in range(len(payload.items))
    }
    all_item_uuids = list(item_uuid_by_index.values())

    if items_with_annotations:
        # One synthesised job covers every item (new + existing), so the
        # annotator shows up exactly once in agreement aggregates per
        # bulk upload (rather than fragmenting across N tiny jobs).
        # Items without `annotations` are still included in the job's
        # snapshot — leaving their slots blank, which the auto-complete
        # check at job-status-time treats the same as a partial form.
        public_token = secrets.token_urlsafe(24)
        job_uuid = create_annotation_job(
            task_id=task_uuid,
            annotator_id=payload.annotator_id,
            item_uuids=all_item_uuids,
            public_token=public_token,
            status="pending",
        )
        # Re-validate every requested evaluator_id against the job's own
        # snapshot, not the pre-creation linked set. Concurrent
        # link/unlink between the upstream check and `create_annotation_job`
        # can shift the snapshot under us; without this gate we'd persist
        # annotations on slots the job doesn't own, polluting downstream
        # `annotations`-by-task reads. Same contract enforced by the
        # public-form upsert endpoint in `routers/public.py`.
        snapshot_evaluator_ids = set(get_evaluator_ids_for_job(job_uuid))
        snapshot_mismatch: List[str] = []
        for it in payload.items:
            if not it.annotations:
                continue
            for evaluator_id in it.annotations.keys():
                if evaluator_id not in snapshot_evaluator_ids:
                    snapshot_mismatch.append(evaluator_id)
        if snapshot_mismatch:
            # Roll back only the newly-created items and the job. Existing
            # items that were matched by name must not be soft-deleted.
            if new_uuids:
                try:
                    soft_delete_annotation_items(task_uuid, new_uuids)
                except Exception as e:
                    logger.warning(
                        f"[bulk-create-items] rollback: failed to soft-delete "
                        f"items {new_uuids} after snapshot mismatch: {e}"
                    )
            try:
                soft_delete_annotation_job(job_uuid)
            except Exception as e:
                logger.warning(
                    f"[bulk-create-items] rollback: failed to soft-delete "
                    f"job {job_uuid} after snapshot mismatch: {e}"
                )
            raise HTTPException(
                status_code=409,
                detail=(
                    f"Evaluator(s) were unlinked from this task between "
                    f"validation and job creation: {sorted(set(snapshot_mismatch))}. "
                    f"Re-link them and retry, or drop them from the request. "
                    f"(The created items and job have been rolled back; "
                    f"safe to retry as-is.)"
                ),
            )
        any_annotation_written = False
        for i, it in enumerate(payload.items):
            if not it.annotations:
                continue
            item_uuid = item_uuid_by_index[i]
            for evaluator_id, value in it.annotations.items():
                upsert_annotation(
                    job_id=job_uuid,
                    item_id=item_uuid,
                    value=value,
                    evaluator_id=evaluator_id,
                )
                any_annotation_written = True
        # Auto-complete contract: every item × every evaluator IN THE JOB
        # SNAPSHOT must have a row. Same source of truth as the public-form
        # auto-complete path (`get_evaluator_ids_for_job`) — see also the
        # snapshot-mismatch gate above which uses the same set.
        items_fully_annotated = all(
            it.annotations
            and snapshot_evaluator_ids.issubset(set(it.annotations.keys()))
            for it in payload.items
        )
        if any_annotation_written and items_fully_annotated:
            update_annotation_job_status(
                job_uuid, status="completed", set_completed_at=True
            )
        elif any_annotation_written:
            # Partial fill: some slots filled, others left for the
            # annotator to finish via the public form. Mirror the public
            # upsert endpoint's "first save flips pending -> in_progress"
            # transition so status-based consumers (jobs list, dashboards)
            # don't see a job with real annotation data still labelled
            # `pending`.
            update_annotation_job_status(job_uuid, status="in_progress")
        return {
            "item_ids": all_item_uuids,
            "new_item_ids": new_uuids,
            "existing_item_ids": list(matched_existing.values()),
            "count": len(all_item_uuids),
            "annotation_job_id": job_uuid,
        }

    return {"item_ids": new_uuids, "count": len(new_uuids)}


class ItemUpdatePayload(BaseModel):
    uuid: str
    payload: Any


class BulkUpdateItemsRequest(BaseModel):
    updates: List[ItemUpdatePayload]


@router.put("/{task_uuid}/items")
async def bulk_update_items(
    task_uuid: str,
    payload: BulkUpdateItemsRequest,
    ctx: OrgContext = Depends(get_current_org),
):
    """Bulk-update item `payload`s in a task.

    Updates not in this task (or referencing deleted items) are skipped
    silently; `updated_count` reflects rows actually changed.
    """
    task = _ensure_owned_task(task_uuid, ctx.org_uuid)
    if not payload.updates:
        raise HTTPException(status_code=400, detail="updates must be non-empty")

    incoming_names = [
        (u.uuid, u.payload["name"])
        for u in payload.updates
        if isinstance(u.payload, dict) and isinstance(u.payload.get("name"), str) and u.payload["name"]
    ]
    if incoming_names:
        names_in_batch = [n for _, n in incoming_names]
        if len(names_in_batch) != len(set(names_in_batch)):
            seen: set = set()
            dupes = sorted({n for n in names_in_batch if n in seen or seen.add(n)})  # type: ignore[func-returns-value]
            raise HTTPException(
                status_code=409,
                detail={
                    "code": "ITEM_NAME_DUPLICATE_IN_REQUEST",
                    "message": f"Duplicate `payload.name` value(s) within request: {dupes}",
                    "conflicting_names": dupes,
                },
            )
        updating_uuids = {item_uuid for item_uuid, _ in incoming_names}
        existing_items = get_annotation_items_for_task(task_uuid)
        names_set = set(names_in_batch)
        conflicts = [
            n
            for it in existing_items
            if it["uuid"] not in updating_uuids
            and isinstance(it.get("payload"), dict)
            for n in [it["payload"].get("name")]
            if n in names_set
        ]
        if conflicts:
            deduped = sorted(set(conflicts))
            raise HTTPException(
                status_code=409,
                detail={
                    "code": "ITEM_NAME_CONFLICT",
                    "message": f"`payload.name` already exists in this task: {deduped}",
                    "conflicting_names": deduped,
                },
            )

    try:
        updated_count = bulk_update_annotation_items(
            task_uuid, [u.dict() for u in payload.updates]
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return {"updated_count": updated_count}


def _resolve_target_item_ids(
    task_uuid: str,
    *,
    select_all: bool,
    item_ids: List[str],
    q: Optional[str],
    items: Optional[List[Dict[str, Any]]] = None,
) -> List[str]:
    """Resolve the target item set for a bulk action that supports a
    `select_all` toggle.

    - `select_all=True`: returns every non-deleted item UUID in the task,
      optionally filtered by case-insensitive substring on `payload.name`
      (same field/match rule as the summary endpoint's `?q=`). The explicit
      `item_ids` list is ignored — `select_all` is the source of truth so
      stale checkboxes can't sneak through.
    - `select_all=False`: returns `item_ids` verbatim; `q` is ignored.

    Pass `items` to reuse an already-loaded task item list (avoids a second
    `get_annotation_items_for_task` round-trip); omitted ⇒ fetched lazily and
    only when `select_all=True`.

    Returns the raw resolved list (may be empty). Callers decide whether
    "empty" is a 400 or a no-op in their context.
    """
    if not select_all:
        return list(item_ids)
    if items is None:
        items = get_annotation_items_for_task(task_uuid)
    if q and q.strip():
        needle = q.strip().lower()
        items = [
            it
            for it in items
            if isinstance((it.get("payload") or {}).get("name"), str)
            and needle in it["payload"]["name"].lower()
        ]
    return [it["uuid"] for it in items]


class BulkDeleteItemsRequest(BaseModel):
    # When `select_all=True`, `item_ids` is ignored and the target set is
    # derived from the task (optionally filtered by `q` on `payload.name`).
    item_ids: List[str] = []
    select_all: bool = False
    q: Optional[str] = None


@router.delete("/{task_uuid}/items")
async def bulk_delete_items(
    task_uuid: str,
    payload: BulkDeleteItemsRequest,
    ctx: OrgContext = Depends(get_current_org),
):
    """Soft-delete one or more items in a task.

    Items not in this task (or already deleted) are skipped silently;
    `deleted_count` reflects how many rows actually transitioned to deleted.
    Items linked to existing jobs remain referenced by those jobs and their
    annotations — they just stop appearing in `GET /items`.

    Use `select_all=True` (optionally with `q`) to act on every item in the
    task matching the current search filter; in that mode `item_ids` is
    ignored.
    """
    _ensure_owned_task(task_uuid, ctx.org_uuid)
    target_ids = _resolve_target_item_ids(
        task_uuid,
        select_all=payload.select_all,
        item_ids=payload.item_ids,
        q=payload.q,
    )
    if not target_ids:
        raise HTTPException(
            status_code=400,
            detail=(
                "no items selected (provide item_ids, or select_all=true with "
                "a filter that matches at least one item)"
            ),
        )
    deleted_count = soft_delete_annotation_items(task_uuid, target_ids)
    return {"deleted_count": deleted_count}


@router.get("/{task_uuid}/items/{item_uuid}")
async def get_item(
    task_uuid: str, item_uuid: str, ctx: OrgContext = Depends(get_current_org)
):
    _ensure_owned_task(task_uuid, ctx.org_uuid)
    item = get_annotation_item(item_uuid)
    if not item or item.get("task_id") != task_uuid:
        raise HTTPException(status_code=404, detail="Item not found")
    return item


@router.get("/{task_uuid}/items/{item_uuid}/annotations")
async def list_item_annotations(
    task_uuid: str, item_uuid: str, ctx: OrgContext = Depends(get_current_org)
):
    """All human annotations across every job for one item. Sibling of
    `/items/{item_uuid}/evaluator-runs`."""
    _ensure_owned_task(task_uuid, ctx.org_uuid)
    item = get_annotation_item(item_uuid)
    if not item or item.get("task_id") != task_uuid:
        raise HTTPException(status_code=404, detail="Item not found")
    return get_annotations_for_item(item_uuid)


# ============ Jobs ============


class CreateJobsRequest(BaseModel):
    annotator_ids: List[str]
    # When `select_all=True`, `item_ids` is ignored and the target set is
    # derived from the task (optionally filtered by `q` on `payload.name`).
    item_ids: List[str] = []
    select_all: bool = False
    q: Optional[str] = None


@router.get("/{task_uuid}/jobs")
async def list_task_jobs(
    task_uuid: str, ctx: OrgContext = Depends(get_current_org)
):
    _ensure_owned_task(task_uuid, ctx.org_uuid)
    return get_jobs_for_task(task_uuid)


@router.post("/{task_uuid}/jobs")
async def create_jobs(
    task_uuid: str,
    payload: CreateJobsRequest,
    ctx: OrgContext = Depends(get_current_org),
):
    """Assign a set of items to one or more annotators. Creates ONE job per
    annotator — each with its own unique public_token. Job item sets are
    frozen after creation.

    Use `select_all=True` (optionally with `q`) to assign every item in the
    task matching the current search filter; in that mode `item_ids` is
    ignored."""
    _ensure_owned_task(task_uuid, ctx.org_uuid)
    if not payload.annotator_ids:
        raise HTTPException(
            status_code=400, detail="annotator_ids must be non-empty"
        )
    target_ids = _resolve_target_item_ids(
        task_uuid,
        select_all=payload.select_all,
        item_ids=payload.item_ids,
        q=payload.q,
    )
    if not target_ids:
        raise HTTPException(
            status_code=400,
            detail=(
                "no items selected (provide item_ids, or select_all=true with "
                "a filter that matches at least one item)"
            ),
        )
    if len(target_ids) != len(set(target_ids)):
        # Duplicate item_ids would violate UNIQUE(job_id, item_id) on
        # annotation_job_items and surface as a 500. Surface as a clean 400.
        # `select_all` expansion can't produce dupes (it scans DISTINCT rows),
        # so this only fires on caller-supplied lists.
        seen: set = set()
        duplicates: List[str] = []
        for i in target_ids:
            if i in seen:
                duplicates.append(i)
            else:
                seen.add(i)
        raise HTTPException(
            status_code=400,
            detail=f"Duplicate item_ids in request: {sorted(set(duplicates))}",
        )

    # Validate annotators (all up front, before any insert).
    annotators_by_id: Dict[str, Dict[str, Any]] = {}
    for annotator_id in payload.annotator_ids:
        annotator = get_annotator(annotator_id)
        if not annotator or annotator.get("org_uuid") != ctx.org_uuid:
            raise HTTPException(
                status_code=404,
                detail=f"Annotator not found: {annotator_id}",
            )
        annotators_by_id[annotator_id] = annotator

    # Validate items (all must belong to this task). `select_all` expansion
    # is already scoped to this task, so this only fires on caller-supplied
    # lists — but the check stays for both paths to keep one error shape.
    valid_item_ids = {
        it["uuid"] for it in get_annotation_items_for_task(task_uuid)
    }
    invalid = [i for i in target_ids if i not in valid_item_ids]
    if invalid:
        raise HTTPException(
            status_code=400,
            detail=f"item_ids not in this task: {invalid}",
        )

    jobs_created = []
    for annotator_id in payload.annotator_ids:
        public_token = secrets.token_urlsafe(24)
        job_uuid = create_annotation_job(
            task_id=task_uuid,
            annotator_id=annotator_id,
            item_uuids=target_ids,
            public_token=public_token,
        )
        jobs_created.append(
            {
                "uuid": job_uuid,
                "public_token": public_token,
                "annotator_id": annotator_id,
                "annotator_name": annotators_by_id[annotator_id]["name"],
                "item_ids": target_ids,
                "item_count": len(target_ids),
                "status": "pending",
            }
        )
    return {"jobs": jobs_created, "count": len(jobs_created)}


@router.get("/{task_uuid}/jobs/{job_uuid}")
async def get_annotation_job_endpoint(
    task_uuid: str,
    job_uuid: str,
    ctx: OrgContext = Depends(get_current_org),
):
    _ensure_owned_task(task_uuid, ctx.org_uuid)
    job = get_annotation_job(job_uuid)
    if not job or job.get("task_id") != task_uuid:
        raise HTTPException(status_code=404, detail="Job not found")
    job["items"] = get_job_items(job_uuid)
    return job


class BulkDeleteJobsRequest(BaseModel):
    job_uuids: List[str]


@router.delete("/{task_uuid}/jobs")
async def bulk_delete_annotation_jobs_endpoint(
    task_uuid: str,
    payload: BulkDeleteJobsRequest,
    ctx: OrgContext = Depends(get_current_org),
):
    """Soft-delete one or more labelling jobs in a task. UUIDs not in this
    task (or already deleted) are skipped silently; `deleted_count` reflects
    how many rows actually transitioned. Cascade matches the single-delete
    sibling: each deleted job's annotations drop out of every annotation read
    via the `j.deleted_at IS NULL` join filter."""
    _ensure_owned_task(task_uuid, ctx.org_uuid)
    if not payload.job_uuids:
        raise HTTPException(status_code=400, detail="job_uuids must be non-empty")
    deleted_count = bulk_soft_delete_annotation_jobs(task_uuid, payload.job_uuids)
    return {"deleted_count": deleted_count}


@router.delete("/{task_uuid}/jobs/{job_uuid}")
async def delete_annotation_job_endpoint(
    task_uuid: str,
    job_uuid: str,
    ctx: OrgContext = Depends(get_current_org),
):
    """Soft-delete one annotator's labelling job. The annotations stay in
    place but stop appearing in every downstream read (list, agreement,
    evaluator-run human columns) because all those queries filter
    `annotation_jobs.deleted_at IS NULL` at the join. Eval-run jobs
    (separate `jobs` table) are NOT cascaded — delete them via
    `DELETE /{task_uuid}/evaluator-runs/{job_uuid}` if needed."""
    _ensure_owned_task(task_uuid, ctx.org_uuid)
    job = get_annotation_job(job_uuid)
    if not job or job.get("task_id") != task_uuid:
        raise HTTPException(status_code=404, detail="Job not found")
    if not soft_delete_annotation_job(job_uuid):
        raise HTTPException(status_code=404, detail="Job not found")
    return {"message": "Annotation job deleted successfully"}


class AnnotationJobVisibilityRequest(BaseModel):
    is_public: bool


class AnnotationJobVisibilityResponse(BaseModel):
    is_public: bool
    view_token: Optional[str] = None


@router.patch(
    "/{task_uuid}/jobs/{job_uuid}/visibility",
    response_model=AnnotationJobVisibilityResponse,
)
async def update_annotation_job_visibility_endpoint(
    task_uuid: str,
    job_uuid: str,
    body: AnnotationJobVisibilityRequest,
    ctx: OrgContext = Depends(get_current_org),
):
    """Toggle a *read-only* public viewer link for one annotator's labelling
    job. The annotator's own `public_token` (read+write) is unaffected — this
    only opts into a separate `view_token` served at
    `GET /public/annotation-jobs/view/{view_token}`.

    Gated on `status == "completed"` so half-done jobs can't be shared. The
    `view_token` is reused across off→on→off cycles so previously-distributed
    links keep working when re-enabled."""
    _ensure_owned_task(task_uuid, ctx.org_uuid)
    job = get_annotation_job(job_uuid)
    if not job or job.get("task_id") != task_uuid:
        raise HTTPException(status_code=404, detail="Job not found")

    if body.is_public and job.get("status") != "completed":
        raise HTTPException(
            status_code=400,
            detail="Only completed labelling jobs can be shared publicly.",
        )

    token_to_persist, token_to_return = compute_share_token_toggle(
        job,
        body.is_public,
        token_field="view_token",
        token_factory=lambda: secrets.token_urlsafe(24),
    )
    update_annotation_job_visibility(job_uuid, body.is_public, token_to_persist)
    return AnnotationJobVisibilityResponse(
        is_public=body.is_public, view_token=token_to_return
    )


# ============ Annotations (judgements) ============


class AnnotationUpsertRequest(BaseModel):
    job_id: str
    item_id: str
    evaluator_id: Optional[str] = None  # None = row-level overall annotation
    value: Optional[Dict[str, Any]] = None


@router.post("/{task_uuid}/annotations")
async def upsert_annotation_endpoint(
    task_uuid: str,
    payload: AnnotationUpsertRequest,
    ctx: OrgContext = Depends(get_current_org),
):
    """Owner-side upsert of a single annotation.

    Both the item and the (non-null) evaluator must be in the job's
    snapshot — same contract as the public token flow. Without these
    checks, owner-authenticated callers could write annotations against
    items/evaluators that were never assigned to this specific job (just
    same task), polluting `completed_item_count`, agreement aggregates,
    and the summary view.
    """
    _ensure_owned_task(task_uuid, ctx.org_uuid)
    job = get_annotation_job(payload.job_id)
    if not job or job.get("task_id") != task_uuid:
        raise HTTPException(status_code=404, detail="Job not found")

    # Validate against the job's snapshotted items, not the source items
    # table — the source may have been edited or soft-deleted, but the
    # snapshot is what this job is contracted to label.
    job_item_ids = {it["uuid"] for it in get_job_items(payload.job_id)}
    if payload.item_id not in job_item_ids:
        raise HTTPException(status_code=404, detail="Item not in this job")

    # Validate against the job's snapshotted evaluator set. `evaluator_id IS
    # NULL` is the row-level overall annotation case and is always allowed.
    if payload.evaluator_id is not None:
        snapshot_evaluator_ids = set(get_evaluator_ids_for_job(payload.job_id))
        if payload.evaluator_id not in snapshot_evaluator_ids:
            raise HTTPException(
                status_code=400,
                detail=f"Evaluator not in this job: {payload.evaluator_id}",
            )

    annotation_uuid = upsert_annotation(
        job_id=payload.job_id,
        item_id=payload.item_id,
        evaluator_id=payload.evaluator_id,
        value=payload.value,
    )
    if job.get("status") == "pending":
        update_annotation_job_status(payload.job_id, "in_progress")
    return {"uuid": annotation_uuid, "message": "Annotation saved"}


# ============ Evaluator runs (run linked evaluators on all items) ============


class EvaluatorRunRequestEntry(BaseModel):
    evaluator_id: str
    evaluator_version_id: Optional[str] = None  # defaults to evaluator's live version


class EvaluatorRunStartRequest(BaseModel):
    evaluators: List[EvaluatorRunRequestEntry]
    # `select_all=True` (optionally filtered by `q` on `payload.name`) is the
    # explicit "run on every matching item" toggle — mirrors the same flag on
    # DELETE /items and POST /jobs so the FE has one shape across bulk
    # actions. When False, `item_ids` must be non-empty.
    item_ids: List[str] = []
    select_all: bool = False
    q: Optional[str] = None


@router.post("/{task_uuid}/evaluator-runs")
async def start_evaluator_run(
    task_uuid: str,
    payload: EvaluatorRunStartRequest,
    ctx: OrgContext = Depends(get_current_org),
):
    """Run one or more evaluators on every item in this task (or a subset).
    Returns a job UUID; the actual evaluation runs asynchronously via the
    calibrate CLI's `--eval-only` mode. Poll
    `GET /evaluator-runs/{job_uuid}` for status.

    Supported task types: `stt`, `llm`, `llm-general`, `conversation`. (Voice
    simulations and TTS are not supported in eval-only mode.)"""
    task = _ensure_owned_task(task_uuid, ctx.org_uuid)
    if task.get("type") not in SUPPORTED_EVAL_TASK_TYPES:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Evaluator runs are not supported for task type "
                f"{task.get('type')!r}. Supported: "
                f"{list(SUPPORTED_EVAL_TASK_TYPES)}."
            ),
        )
    if not payload.evaluators:
        raise HTTPException(status_code=400, detail="evaluators must be non-empty")

    all_items = get_annotation_items_for_task(task_uuid)
    if not all_items:
        raise HTTPException(status_code=400, detail="task has no items")

    # Resolve the item subset.
    #
    # `select_all=True` snapshots the live (q-filtered) set at submission
    # time. Storing the resolved UUIDs (instead of leaving null) ensures
    # recovery after a crash re-runs the same items the user originally
    # submitted, even if items were added or deleted in the meantime.
    if payload.select_all:
        target_ids = _resolve_target_item_ids(
            task_uuid,
            select_all=True,
            item_ids=[],
            q=payload.q,
            items=all_items,  # reuse the list already fetched above
        )
        if not target_ids:
            raise HTTPException(
                status_code=400,
                detail=(
                    "no items selected (select_all=true matched no items — "
                    "check the q filter)"
                ),
            )
        items_by_id = {it["uuid"]: it for it in all_items}
        items = [items_by_id[i] for i in target_ids]
        item_ids_persisted: List[str] = target_ids
    else:
        if not payload.item_ids:
            raise HTTPException(
                status_code=400,
                detail=(
                    "item_ids must be non-empty when select_all=false "
                    "(or pass select_all=true to run on every matching item)"
                ),
            )
        valid_ids = {it["uuid"] for it in all_items}
        invalid = [i for i in payload.item_ids if i not in valid_ids]
        if invalid:
            raise HTTPException(
                status_code=400,
                detail=f"item_ids not in this task: {invalid}",
            )
        # Preserve request order; drop accidental dupes.
        seen: set = set()
        ordered_subset_ids: List[str] = []
        for i in payload.item_ids:
            if i not in seen:
                seen.add(i)
                ordered_subset_ids.append(i)
        items_by_id = {it["uuid"]: it for it in all_items}
        items = [items_by_id[i] for i in ordered_subset_ids]
        item_ids_persisted = ordered_subset_ids

    linked = {
        e["uuid"] for e in get_evaluators_for_annotation_task(task_uuid)
    }
    try:
        resolved = _resolve_evaluator_dicts(
            [e.dict() for e in payload.evaluators], linked
        )
    except EvaluatorResolutionError as e:
        raise HTTPException(status_code=400, detail=str(e))

    # Validate item payload shape early so we 400 instead of failing async.
    try:
        build_dataset_for_task_type(task["type"], items, resolved)
    except DatasetBuildError as e:
        raise HTTPException(status_code=400, detail=str(e))

    # Decide queue vs immediate start (shared eval queue with stt-eval/tts-eval).
    can_start = can_start_job(EVAL_JOB_TYPES, ctx.org_uuid)
    initial_status = (
        TaskStatus.IN_PROGRESS.value if can_start else TaskStatus.QUEUED.value
    )

    job_uuid = create_job(
        job_type=ANNOTATION_EVAL_JOB_TYPE,
        org_uuid=ctx.org_uuid,
        user_id=ctx.user_id,
        status=initial_status,
        details={
            "task_id": task_uuid,
            "evaluators": [
                {
                    "evaluator_id": ev["uuid"],
                    "evaluator_version_id": ev["_evaluator_version_id"],
                    "name": ev["name"],
                }
                for ev in resolved
            ],
            "item_count": len(items),
            "item_ids": item_ids_persisted,
        },
    )
    # Snapshot the resolved item set onto the job so the runner reads
    # frozen payloads regardless of any subsequent edit / soft-delete on
    # the source `annotation_items` row. Order matches submission order
    # (preserved via the loop above) so reproducibility extends to the
    # exact byte sequence calibrate sees.
    snapshot_eval_job_items(job_uuid, items)

    if can_start:
        start_annotation_eval_job(
            job_uuid=job_uuid,
            task_uuid=task_uuid,
            user_id=ctx.user_id,
            evaluators_resolved=resolved,
            item_ids=item_ids_persisted,
        )
    return {
        "job_uuid": job_uuid,
        "status": initial_status,
        "evaluator_count": len(resolved),
        "item_count": len(items),
    }


def _enrich_runs_with_live_evaluator(
    runs: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Join each `evaluator_runs` row with the **current** evaluator metadata
    (name, description, output_type) so renames / description edits show up
    in API reads. The persisted `evaluator_id` and `evaluator_version_id`
    stay as the source of truth — we just bolt on a fresh `evaluator` block.

    For rating evaluators, the version's `output_config` rubric is also
    surfaced (along with derived `scale_min` / `scale_max`) so the FE can
    render the value against the right scale without a second roundtrip."""
    if not runs:
        return runs
    from llm_judge import _scale_bounds  # local to avoid cycle on module load

    eval_cache: Dict[str, Optional[Dict[str, Any]]] = {}
    version_cache: Dict[str, Optional[Dict[str, Any]]] = {}
    out: List[Dict[str, Any]] = []
    for run in runs:
        ev_id = run.get("evaluator_id")
        if ev_id and ev_id not in eval_cache:
            eval_cache[ev_id] = get_evaluator(ev_id)
        ev = eval_cache.get(ev_id) if ev_id else None
        version_id = run.get("evaluator_version_id")
        if version_id and version_id not in version_cache:
            version_cache[version_id] = get_evaluator_version(version_id)
        version = version_cache.get(version_id) if version_id else None
        enriched = dict(run)
        enriched["evaluator"] = (
            {
                "uuid": ev["uuid"],
                "name": ev.get("name"),
                "description": ev.get("description"),
                "output_type": ev.get("output_type"),
                "evaluator_type": ev.get("evaluator_type"),
                "data_type": ev.get("data_type"),
            }
            if ev
            else None
        )
        if version:
            output_config = version.get("output_config")
            scale_min, scale_max = _scale_bounds(output_config)
            enriched["evaluator_version"] = {
                "uuid": version["uuid"],
                "version_number": version.get("version_number"),
                "judge_model": version.get("judge_model"),
                "output_config": output_config,
                "scale_min": scale_min,
                "scale_max": scale_max,
            }
        else:
            enriched["evaluator_version"] = None
        out.append(enriched)
    return out


def _build_evaluators_block_for_eval_job(
    job_details: Optional[Dict[str, Any]],
    raw_runs: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Build the top-level `evaluators[]` block returned alongside an
    evaluator-run job detail response. Mirrors the shape used by the
    labelling-job viewer (`GET /public/annotation-jobs/view/{token}`) so the
    FE can read `output_config.scale` from one consistent place across both
    surfaces. Each entry pins the evaluator-VERSION the job actually ran
    against (from `details.evaluators` snapshot), not the live version —
    rubric edits after the run don't retroactively rewrite what the run
    was measured by.

    Also seeds entries from the runs themselves so jobs predating the
    snapshot (legacy `details.evaluators` absent) still get a populated
    block.
    """
    from llm_judge import _scale_bounds, default_output_config  # local to avoid module-load cycle

    snapshot = (job_details or {}).get("evaluators") or []
    # Dedupe (evaluator_id, evaluator_version_id) slots across snapshot
    # and runs so legacy/snapshot-less jobs still emit one entry per pinned
    # version actually used.
    slots: List[tuple] = []
    seen: set = set()
    for entry in snapshot:
        if not isinstance(entry, dict):
            continue
        slot = (entry.get("evaluator_id"), entry.get("evaluator_version_id"))
        if slot[0] and slot not in seen:
            seen.add(slot)
            slots.append(slot)
    for r in raw_runs or []:
        slot = (r.get("evaluator_id"), r.get("evaluator_version_id"))
        if slot[0] and slot not in seen:
            seen.add(slot)
            slots.append(slot)

    # Look up the slim snapshot name as a final fallback for soft-deleted
    # evaluators (get_evaluator filters deleted_at IS NULL).
    snapshot_name_by_uuid: Dict[str, str] = {}
    for entry in snapshot:
        if isinstance(entry, dict) and entry.get("evaluator_id"):
            snapshot_name_by_uuid.setdefault(
                entry["evaluator_id"], entry.get("name") or ""
            )

    eval_cache: Dict[str, Optional[Dict[str, Any]]] = {}
    version_cache: Dict[str, Optional[Dict[str, Any]]] = {}
    out: List[Dict[str, Any]] = []
    for ev_id, version_id in slots:
        if ev_id not in eval_cache:
            eval_cache[ev_id] = get_evaluator(ev_id)
        ev = eval_cache.get(ev_id)
        # If the evaluator was soft-deleted we still want a stub entry so
        # rows[] consumers can resolve the slot — otherwise the FE sees
        # `evaluator_id` references with no matching block entry. Fields
        # we can't recover (description, output_type, rubric) stay null.
        if version_id and version_id not in version_cache:
            version_cache[version_id] = get_evaluator_version(version_id)
        version = version_cache.get(version_id) if version_id else None
        output_config = version.get("output_config") if version else None
        # Apply the Correct/Wrong fallback for binary evaluators whose
        # pinned version has a null rubric — consistent with the other
        # evaluator-returning endpoints in this PR (evaluator detail,
        # versions list, summary, agent-test block builder).
        ev_output_type = ev.get("output_type") if ev else None
        if output_config is None:
            output_config = default_output_config(ev_output_type)
        scale_min, scale_max = _scale_bounds(output_config)
        out.append(
            {
                "uuid": ev_id,
                "name": (ev.get("name") if ev else None)
                or snapshot_name_by_uuid.get(ev_id),
                "description": ev.get("description") if ev else None,
                "output_type": ev.get("output_type") if ev else None,
                "evaluator_type": ev.get("evaluator_type") if ev else None,
                "data_type": ev.get("data_type") if ev else None,
                "evaluator_version_id": version_id,
                "version_number": (
                    version.get("version_number") if version else None
                ),
                "judge_model": version.get("judge_model") if version else None,
                "output_config": output_config,
                "scale_min": scale_min,
                "scale_max": scale_max,
                "variables": version.get("variables") if version else None,
                "deleted": ev is None,
            }
        )
    return out


def _strip_run_evaluator_blocks(
    runs: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Return the runs without the per-row `evaluator` / `evaluator_version`
    blobs. Callers surface that metadata via the top-level `evaluators[]`
    block instead, keyed by `(evaluator_id, evaluator_version_id)` on each
    run row."""
    if not runs:
        return runs
    return [
        {k: v for k, v in r.items() if k not in ("evaluator", "evaluator_version")}
        for r in runs
    ]


def _strip_details_evaluators(shaped: Dict[str, Any]) -> Dict[str, Any]:
    """Drop the slim `evaluators` snapshot from `details` — promoted to the
    response's top-level `evaluators[]` block (with rubric)."""
    details = shaped.get("details")
    if isinstance(details, dict) and "evaluators" in details:
        shaped["details"] = {k: v for k, v in details.items() if k != "evaluators"}
    return shaped


def _shape_eval_job_for_response(job: Dict[str, Any]) -> Dict[str, Any]:
    """Adapt a generic-jobs row into the annotation-eval job response shape.
    Lifts task_id from `details` and exposes `error`/`completed_at` at the
    top level. Translates the internal `"done"` status (shared across every
    eval flow in the codebase) to `"completed"` for this feature's API."""
    out = dict(job)
    details = out.get("details") or {}
    out["task_id"] = details.get("task_id")
    out["completed_at"] = details.get("completed_at")
    results = out.get("results") or {}
    out["error"] = results.get("error") if isinstance(results, dict) else None
    if out.get("status") == "done":
        out["status"] = "completed"
    out["is_public"] = bool(out.get("is_public"))
    out["share_token"] = out.get("share_token")
    return out


@router.get("/{task_uuid}/evaluator-runs")
async def list_evaluator_run_jobs(
    task_uuid: str, ctx: OrgContext = Depends(get_current_org)
):
    """Slim list of evaluator-run jobs for the task. Each row carries
    just the identifiers + status + counts + updated_at; the FE looks up
    each row's evaluators via the top-level `evaluators[]` block by
    `(evaluator_id, evaluator_version_id)`. Status `"done"` is normalized
    to `"completed"` to match the detail endpoint."""
    _ensure_owned_task(task_uuid, ctx.org_uuid)
    jobs = get_generic_jobs_for_task(task_uuid, ANNOTATION_EVAL_JOB_TYPE)

    # Aggregate the per-job snapshots into one combined block so the FE
    # only needs to fetch evaluator/version metadata once per unique
    # (evaluator_id, evaluator_version_id) pair across the list.
    combined_snapshot: List[Dict[str, Any]] = []
    seen_slots: set = set()
    for j in jobs:
        for entry in (j.get("details") or {}).get("evaluators") or []:
            if not isinstance(entry, dict):
                continue
            slot = (entry.get("evaluator_id"), entry.get("evaluator_version_id"))
            if slot[0] and slot not in seen_slots:
                seen_slots.add(slot)
                combined_snapshot.append(entry)
    evaluators_block = _build_evaluators_block_for_eval_job(
        {"evaluators": combined_snapshot}, []
    )

    runs: List[Dict[str, Any]] = []
    for j in jobs:
        details = j.get("details") or {}
        status = j.get("status")
        if status == "done":
            status = "completed"
        row_evals = [
            {
                "evaluator_id": e.get("evaluator_id"),
                "evaluator_version_id": e.get("evaluator_version_id"),
            }
            for e in (details.get("evaluators") or [])
            if isinstance(e, dict) and e.get("evaluator_id")
        ]
        runs.append(
            {
                "uuid": j["uuid"],
                "status": status,
                "item_count": details.get("item_count"),
                "updated_at": j.get("updated_at"),
                "evaluators": row_evals,
            }
        )

    return {"evaluators": evaluators_block, "runs": runs}


def _human_agreement_for_run(
    task_uuid: str, job_runs: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """Build the `human_agreement` block returned alongside an evaluator-run
    job. Restricted to slots (item_id, evaluator_id) actually exercised by
    THIS job's runs — annotations on items/evaluators not in the run are
    ignored even if present elsewhere on the task.

    Shape:
        {
          "evaluators": [
            { "evaluator_id": str, "evaluator_version_id": str|None,
              "agreement": float|None, "pair_count": int, "item_count": int }
          ],
          "items": [
            { "item_id": str, "annotator_count": int,
              "evaluators": [{evaluator_id, agreement, pair_count}] }
          ]
        }

    `evaluators[].agreement` reuses `aggregate_human_evaluator_agreement` —
    so it agrees with the task-level alignment block by construction.
    `items[]` only includes items where at least one human annotation exists
    on an evaluator that this job ran (no humans → no row, by design)."""
    if not job_runs:
        return {"evaluators": [], "items": []}

    item_ids_in_run = {r["item_id"] for r in job_runs if r.get("item_id")}
    evaluator_ids_in_run = list(
        {r["evaluator_id"] for r in job_runs if r.get("evaluator_id")}
    )
    if not item_ids_in_run or not evaluator_ids_in_run:
        return {"evaluators": [], "items": []}

    # Constrain the read to this run's slots at the DB layer rather than
    # pulling the task's entire annotation history and filtering in
    # Python — on a large task, the run is a tiny window and the filter
    # would dominate request latency. `include_deleted_items=True`
    # preserves annotations on items soft-deleted AFTER this run
    # completed (the run's `evaluator_runs` rows survive item delete; the
    # human side has to survive too or the agreement number on the
    # run-detail view silently shrinks).
    relevant_annotations = get_annotations_for_slots(
        task_uuid,
        item_ids=list(item_ids_in_run),
        evaluator_ids=evaluator_ids_in_run,
        include_deleted_items=True,
    )

    # Per-evaluator aggregate (across every item this job touched).
    evaluator_blocks: List[Dict[str, Any]] = []
    # Pre-compute version pin per evaluator from the run rows so the FE can
    # show "agreement on evaluator X version Y vs humans" without an extra
    # lookup. A job runs one version per evaluator by construction.
    version_by_evaluator: Dict[str, Optional[str]] = {}
    for r in job_runs:
        ev_id = r.get("evaluator_id")
        if ev_id and ev_id not in version_by_evaluator:
            version_by_evaluator[ev_id] = r.get("evaluator_version_id")
    for ev_id in evaluator_ids_in_run:
        agreement, pair_count = aggregate_human_evaluator_agreement(
            relevant_annotations, job_runs, ev_id
        )
        # Items with at least one human annotation on this evaluator.
        item_count = len(
            {
                a["item_id"]
                for a in relevant_annotations
                if a.get("evaluator_id") == ev_id
            }
        )
        evaluator_blocks.append(
            {
                "evaluator_id": ev_id,
                "evaluator_version_id": version_by_evaluator.get(ev_id),
                "agreement": agreement,
                "pair_count": pair_count,
                "item_count": item_count,
            }
        )

    # Per-item agreement, restricted to items that have at least one human
    # annotation on an evaluator this job ran. The whole point of this view
    # is "where do machines and humans disagree on this run" — items with
    # zero human signal contribute nothing.
    annotations_by_item: Dict[str, List[Dict[str, Any]]] = {}
    for a in relevant_annotations:
        annotations_by_item.setdefault(a["item_id"], []).append(a)
    runs_by_item: Dict[str, List[Dict[str, Any]]] = {}
    for r in job_runs:
        if r.get("item_id"):
            runs_by_item.setdefault(r["item_id"], []).append(r)

    # Resolve annotator names once for every annotator that contributed to
    # any of the relevant items, so per-item entries can label each value.
    annotator_uuids = sorted(
        {
            a.get("annotator_id")
            for a in relevant_annotations
            if a.get("annotator_id")
        }
    )
    annotators_by_uuid = (
        get_annotators_by_uuids(annotator_uuids) if annotator_uuids else {}
    )

    item_blocks: List[Dict[str, Any]] = []
    for item_id, item_annotations in annotations_by_item.items():
        per_item = per_item_agreement(
            item_annotations,
            runs_by_item.get(item_id, []),
            evaluator_ids_in_run,
        )
        # Drop evaluator slots with zero human pair count so the FE only
        # renders cells that actually compare to a human.
        ev_entries = [
            e for e in per_item.get("evaluators", []) if e.get("pair_count")
        ]
        if not ev_entries:
            continue
        # Bucket the raw human annotations on this item by evaluator so the
        # FE can show every annotator's exact value alongside the agreement
        # number. Annotations are already filtered to the run's slot set.
        #
        # **Latest-wins per (evaluator, annotator)** — matches the summary
        # endpoint's semantics ([`task_summary`](annotation_tasks.py)). If
        # the same annotator labeled the same slot across multiple
        # annotation jobs, only the most recent submission surfaces in
        # `human_annotations[]`. Input is sorted by `updated_at ASC`
        # (`get_annotations_for_slots`), so dict-overwrite per
        # `(ev_id, annotator_id)` gives latest-wins.
        latest_by_slot: Dict[tuple, Dict[str, Any]] = {}
        for a in item_annotations:
            ev_id = a.get("evaluator_id")
            annotator_id = a.get("annotator_id")
            if not ev_id or not annotator_id:
                continue
            latest_by_slot[(ev_id, annotator_id)] = a

        annotations_by_evaluator: Dict[str, List[Dict[str, Any]]] = {}
        for (ev_id, annotator_id), a in latest_by_slot.items():
            annotator = annotators_by_uuid.get(annotator_id) if annotator_id else None
            raw_value = a.get("value")
            reasoning = (
                raw_value.get("reasoning")
                if isinstance(raw_value, dict)
                else None
            )
            annotations_by_evaluator.setdefault(ev_id, []).append(
                {
                    "annotation_id": a.get("uuid"),
                    "annotator_id": annotator_id,
                    "annotator_name": (
                        annotator.get("name") if annotator else None
                    ),
                    "job_id": a.get("job_id"),
                    "value": raw_value,
                    "reasoning": reasoning,
                    "updated_at": a.get("updated_at"),
                }
            )
        # Sort each evaluator's annotations deterministically (oldest first)
        # so the FE can render them in a stable order.
        for entries in annotations_by_evaluator.values():
            entries.sort(key=lambda e: (e.get("updated_at") or "", e.get("annotation_id") or ""))
        # Inline the raw annotations onto each evaluator block keyed by
        # evaluator_id, so the agreement cell and the underlying values
        # render together without an extra join on the FE.
        for entry in ev_entries:
            entry["human_annotations"] = annotations_by_evaluator.get(
                entry["evaluator_id"], []
            )
        item_blocks.append(
            {
                "item_id": item_id,
                "annotator_count": len(
                    {
                        a.get("annotator_id")
                        for a in item_annotations
                        if a.get("annotator_id")
                    }
                ),
                "evaluators": ev_entries,
            }
        )
    # Stable order: items in run-row order so the FE can scroll predictably.
    item_order = [r["item_id"] for r in job_runs if r.get("item_id")]
    seen: set = set()
    ordered_item_ids: List[str] = []
    for i in item_order:
        if i not in seen:
            seen.add(i)
            ordered_item_ids.append(i)
    pos = {i: idx for idx, i in enumerate(ordered_item_ids)}
    item_blocks.sort(key=lambda b: pos.get(b["item_id"], len(pos)))

    return {"evaluators": evaluator_blocks, "items": item_blocks}


@router.get("/{task_uuid}/evaluator-runs/{job_uuid}")
async def get_evaluator_run_job(
    task_uuid: str,
    job_uuid: str,
    ctx: OrgContext = Depends(get_current_org),
):
    """Single evaluator-run job, with raw runs and human-agreement summary.

    `human_agreement` is computed only on slots (item × evaluator) that this
    job actually exercised AND that have at least one human annotation —
    so a fresh task with no humans yet returns empty arrays (not zeros)
    and the FE can fall back to the regular runs view."""
    _ensure_owned_task(task_uuid, ctx.org_uuid)
    job = get_job(job_uuid, org_uuid=ctx.org_uuid)
    if (
        not job
        or job.get("type") != ANNOTATION_EVAL_JOB_TYPE
        or (job.get("details") or {}).get("task_id") != task_uuid
    ):
        raise HTTPException(status_code=404, detail="Job not found")
    shaped = _shape_eval_job_for_response(job)
    raw_runs = get_evaluator_runs_for_job(job_uuid)
    # Top-level evaluators[] mirrors the labelling-job viewer shape so the
    # FE reads `output_config.scale` from one consistent place across the
    # two surfaces. Each entry pins the version the job ran against.
    shaped["evaluators"] = _build_evaluators_block_for_eval_job(
        job.get("details"), raw_runs
    )
    _strip_details_evaluators(shaped)
    # Per-run `evaluator` / `evaluator_version` are intentionally NOT
    # surfaced here — `(evaluator_id, evaluator_version_id)` on each run
    # row keys back into the top-level evaluators[] block.
    shaped["runs"] = _strip_run_evaluator_blocks(raw_runs)
    shaped["human_agreement"] = _human_agreement_for_run(task_uuid, raw_runs)
    # Frozen item snapshot — what calibrate actually saw, regardless of
    # any post-submit edits / soft-deletes on the source annotation_items.
    # Empty for legacy jobs created before snapshotting (those will be
    # backfilled on first run; see annotation_eval_runner._run_job).
    shaped["items"] = get_eval_job_items(job_uuid)
    return shaped


@router.delete("/{task_uuid}/evaluator-runs/{job_uuid}")
async def delete_evaluator_run_job(
    task_uuid: str,
    job_uuid: str,
    ctx: OrgContext = Depends(get_current_org),
):
    """Soft-delete an evaluator-run job and all its `evaluator_runs` rows.

    In-flight jobs (status = 'in_progress') are not allowed to be deleted —
    let them finish (or fail) first, then delete. Queued jobs CAN be deleted
    (they were never started)."""
    _ensure_owned_task(task_uuid, ctx.org_uuid)
    job = get_job(job_uuid, org_uuid=ctx.org_uuid)
    if (
        not job
        or job.get("type") != ANNOTATION_EVAL_JOB_TYPE
        or (job.get("details") or {}).get("task_id") != task_uuid
    ):
        raise HTTPException(status_code=404, detail="Job not found")
    if job.get("status") == TaskStatus.IN_PROGRESS.value:
        raise HTTPException(
            status_code=400,
            detail=(
                "Cannot delete an in-progress evaluator-run job; wait for it "
                "to finish or fail before deleting."
            ),
        )
    soft_delete_job(job_uuid)
    runs_deleted = clear_evaluator_runs_for_job(job_uuid)
    # If we just deleted a queued job, nothing changed for the running set,
    # but draining is cheap and harmless if there's already capacity.
    try:
        try_start_queued_job(EVAL_JOB_TYPES)
    except Exception:
        pass
    return {"deleted_runs": runs_deleted}


class EvaluatorRunVisibilityRequest(BaseModel):
    is_public: bool


class EvaluatorRunVisibilityResponse(BaseModel):
    is_public: bool
    share_token: Optional[str] = None


@router.patch(
    "/{task_uuid}/evaluator-runs/{job_uuid}/visibility",
    response_model=EvaluatorRunVisibilityResponse,
)
async def update_evaluator_run_visibility(
    task_uuid: str,
    job_uuid: str,
    body: EvaluatorRunVisibilityRequest,
    ctx: OrgContext = Depends(get_current_org),
):
    """Toggle public sharing for a completed annotation evaluator-run job.

    Only completed (`done`) jobs can be shared — sharing an in-flight or
    failed job would expose partial / error state behind a stable URL.
    Re-flipping `is_public=True` reuses the existing share_token so
    previously-distributed links keep working."""
    _ensure_owned_task(task_uuid, ctx.org_uuid)
    job = get_job(job_uuid, org_uuid=ctx.org_uuid)
    if (
        not job
        or job.get("type") != ANNOTATION_EVAL_JOB_TYPE
        or (job.get("details") or {}).get("task_id") != task_uuid
    ):
        raise HTTPException(status_code=404, detail="Job not found")

    if body.is_public and job.get("status") != TaskStatus.DONE.value:
        raise HTTPException(
            status_code=400,
            detail="Only completed evaluator-run jobs can be shared publicly.",
        )

    token_to_persist, token_to_return = compute_share_token_toggle(
        job, body.is_public
    )
    update_job_visibility(job_uuid, body.is_public, token_to_persist)
    return EvaluatorRunVisibilityResponse(
        is_public=body.is_public, share_token=token_to_return
    )


@router.get("/{task_uuid}/items/{item_uuid}/evaluator-runs")
async def list_item_evaluator_runs(
    task_uuid: str,
    item_uuid: str,
    ctx: OrgContext = Depends(get_current_org),
):
    _ensure_owned_task(task_uuid, ctx.org_uuid)
    item = get_annotation_item(item_uuid)
    if not item or item.get("task_id") != task_uuid:
        raise HTTPException(status_code=404, detail="Item not found")
    return _enrich_runs_with_live_evaluator(get_evaluator_runs_for_item(item_uuid))


# ============ Agreement (human-vs-human) ============


def _evaluator_alignment_block(
    annotations: List[Dict[str, Any]],
    evaluator_runs: List[Dict[str, Any]],
    linked_evaluators: List[Dict[str, Any]],
    bucket: str,
    days: int,
) -> List[Dict[str, Any]]:
    """For each evaluator linked to the task, compute current agreement vs
    humans + a cumulative trend series. Always returns one entry per linked
    evaluator (with `current=None, pair_count=0` when there's no overlap yet)."""
    evaluator_ids = [e["uuid"] for e in linked_evaluators]
    series_by_id = trend_series_human_evaluator(
        annotations, evaluator_runs, evaluator_ids, bucket=bucket, days=days
    )
    out: List[Dict[str, Any]] = []
    for ev in linked_evaluators:
        ev_id = ev["uuid"]
        cur, pairs = aggregate_human_evaluator_agreement(
            annotations, evaluator_runs, ev_id
        )
        out.append(
            {
                "evaluator_id": ev_id,
                "name": ev.get("name"),
                "current": cur,
                "pair_count": pairs,
                "series": series_by_id.get(ev_id, []),
            }
        )
    return out


@router.get("/{task_uuid}/agreement")
async def task_agreement(
    task_uuid: str,
    bucket: str = Query("week", pattern="^(week|month|year)$"),
    days: int = Query(90, ge=1, le=3650),
    ctx: OrgContext = Depends(get_current_org),
):
    """Human-vs-human agreement for a single task plus per-evaluator
    human-vs-evaluator alignment.

    Returns:
      - `human_human`: `{ current, pair_count, series }` (same shape as before
        but moved under a sub-key so the evaluators block is parallel).
      - `evaluators`: list of `{ evaluator_id, name, current, pair_count, series }`,
        one per evaluator linked to the task. `current`/`pair_count` use ALL
        data; each `series` is cumulative as-of-bucket-end.

    `current` numerics are mean pairwise agreement in `[0, 1]`; `null` when no
    comparable pairs exist yet.
    """
    _ensure_owned_task(task_uuid, ctx.org_uuid)
    annotations = get_annotations_for_task(task_uuid)
    linked = get_evaluators_for_annotation_task(task_uuid)
    runs = filter_runs_to_live_versions(
        get_evaluator_runs_for_task(task_uuid), _live_version_map(linked)
    )

    hh_current, hh_pairs = aggregate_agreement(annotations)
    hh_series = trend_series(annotations, bucket=bucket, days=days)

    evaluators_block = _evaluator_alignment_block(
        annotations, runs, linked, bucket, days
    )

    return {
        "task_id": task_uuid,
        "bucket": bucket,
        "days": days,
        "human_human": {
            "current": hh_current,
            "pair_count": hh_pairs,
            "series": hh_series,
        },
        "evaluators": evaluators_block,
    }


@router.get("/{task_uuid}/summary")
async def task_summary(
    task_uuid: str,
    item_id: Optional[str] = Query(
        None,
        description="Filter rows to a single item. The full task-wide annotator union is still returned in `annotators`.",
    ),
    live_only: bool = Query(
        False,
        description="When true, emit only one row per (item, evaluator) using the evaluator's live version. Non-live versions that have runs are excluded.",
    ),
    ctx: OrgContext = Depends(get_current_org),
    search: _SummarySearch = Depends(),
    sort: _SummarySort = Depends(),
    pagination: PaginationParams = Depends(),
):
    """Single denormalized view for the table. By default emits one row per
    `(item × evaluator × version)` so re-running an evaluator on a new
    version doesn't hide earlier-version results. Pass `live_only=true` to
    collapse to one row per `(item × evaluator)` using only the evaluator's
    live version. Each row carries the latest evaluator-run value and one
    annotation cell per annotator.

    Response shape:
      {
        "task_id": str,
        "task_type": "stt" | "llm" | "conversation",
        "evaluators": [
          {
            "uuid": str,
            "name": str, "description": str|null,
            "output_type": "binary" | "rating",
            "evaluator_type": str, "data_type": str,
            "live_version_id": str|null,
            "live_version_index": int|null,         # array position of the
                                                    # live version inside
                                                    # `versions[]`, or null
                                                    # when no live version
                                                    # / no match
            "versions": [
              {
                "uuid": str|null,                   # null only for the
                                                    # placeholder slot when
                                                    # the evaluator has no
                                                    # runs + no live version
                "version_number": int|null,
                "output_config": {scale: [...]}|null,  # binary defaults to
                                                       # Correct/Wrong when
                                                       # unset; rating stays
                                                       # null
                "scale_min": float|null,
                "scale_max": float|null,
                "is_live": bool
              }
            ],
            "run_count": int,                       # total evaluator-runs
                                                    # across ALL versions,
                                                    # restricted to items in
                                                    # scope (honors
                                                    # `item_id` and `q`;
                                                    # ignores `live_only`)
          }
        ],
        "annotators": [{uuid, name}],               # union of annotators with
                                                    # ≥1 (item, evaluator)
                                                    # annotation in this task
        "rows": [
          {
            "item_id": str,
            "payload": <item.payload>,              # FE derives display per task_type
            "evaluator_id": str,                    # FK into evaluators[].uuid
            "evaluator_version_id": str|null,       # FK into evaluators[].versions[].uuid
            "evaluator_value": <scalar|null>,       # latest run on this slot
            "evaluator_value_name": str|null,       # resolved label per row
                                                    # (FE could re-derive
                                                    # from output_config but
                                                    # we precompute it)
            "evaluator_reasoning": str|null,
            "annotations": {
              "<annotator_uuid>": {"value": <scalar>, "reasoning": str|null} | null,
              ...
            },
            "human_agreement": float|null,
            "evaluator_agreement": float|null
          }
        ],
        "item_comments": {                          # Scoped to the items on
                                                    # the CURRENT PAGE (same
                                                    # set as `rows`). For an
                                                    # export, pass
                                                    # `?limit=<total>` to
                                                    # collect every page in
                                                    # one shot.
          "<item_uuid>": {
            "<annotator_uuid>": str   # latest free-text comment
          }
        },
        "pagination": {                             # Pagination is at the
                                                    # ITEM level — each item
                                                    # expands into >=1 row
                                                    # (one per evaluator x
                                                    # version slot).
          "total": int,                             # total items in scope
                                                    # (honors item_id + q;
                                                    # NOT the annotator
                                                    # union, which stays
                                                    # task-wide so column
                                                    # headers don't shift)
          "limit": int,
          "offset": int
        }
      }

    Cell rules:
      - Default: one row per `(item, evaluator, version)` — a version row
        appears for every distinct version that has runs for this evaluator
        in the task, plus one for the live version (with `null` value if it
        hasn't run yet). With `live_only=true`, only the live-version row is
        emitted per `(item, evaluator)`.
      - `evaluator_value` is the latest evaluator-run for THAT specific
        version slot, regardless of which evaluator-run job produced it.
      - To check whether a row is on the live version, compare
        `evaluator_version_id` to the matching evaluator's `live_version_id`
        (or use `live_version_index` to index `versions[]` and read `is_live`).
        That flag is intentionally NOT duplicated on every row.
      - Each annotator cell is that annotator's latest annotation for the slot,
        across ALL annotation jobs (matches the agreement aggregator's
        latest-wins-per-annotator semantics). `null` if they haven't annotated it.
      - Row-level overall annotations (`evaluator_id IS NULL`) are surfaced
        in the top-level `item_comments` block — one free-text string per
        (item, annotator) pulled from `value["comment"]`. They are
        orthogonal to evaluators, so they live outside `rows[]`. Only
        items/annotators with a non-empty comment appear (the block is
        sparse). Latest-wins per (item, annotator) on `updated_at`. The
        `annotators[]` union includes annotators that contributed comments
        even if they never wrote a per-evaluator annotation — the union
        stays task-wide (not page-scoped) so column headers don't shift
        between pages. The `item_comments` block itself, however, IS scoped
        to the current page (matches `rows`); for export, pass
        `?limit=<total>`.
    """
    task = _ensure_owned_task(task_uuid, ctx.org_uuid)
    items = get_annotation_items_for_task(task_uuid)
    evaluators = get_evaluators_for_annotation_task(task_uuid)
    # Per-row view: latest run wins regardless of version, and per-row
    # `evaluator_agreement` compares annotators against THAT run. Aggregated
    # agreement (task-level / account-level) is the only place we restrict to
    # live-version runs — see `/annotation-tasks/{uuid}/agreement` and
    # `/annotation-agreement/trend`.
    runs = get_evaluator_runs_for_task(task_uuid)
    annotations = get_annotations_for_task(task_uuid)

    # Optional single-item filter. Validate it belongs to the task before
    # narrowing so a bad id 404s instead of silently returning empty rows.
    if item_id is not None:
        if not any(it["uuid"] == item_id for it in items):
            raise HTTPException(
                status_code=404, detail="Item not found in this task"
            )
        items = [it for it in items if it["uuid"] == item_id]

    # Substring search on `payload.name` (the only required string field on
    # items — enforced by the items POST validator). Applied AFTER `item_id`
    # so a single-item lookup that also passes `q` still narrows correctly.
    # Mechanics live in `pagination.make_search_params`.
    items = search.apply(items)

    # `items` is now the full scope (task-wide or filtered by item_id/q).
    # Keep it untouched for scoped_item_ids / annotator union / run_count
    # so the top-level evaluators[] and annotators[] blocks stay stable
    # across pages (consistent column headers in the FE table). Pagination
    # only slices which items get expanded into `rows`.
    total_items = len(items)

    # Sort the in-scope items before pagination so paging is stable across
    # requests. Mechanics live in `pagination.make_sort_params`.
    #
    # Tiebreaker is the autoincrement `id`, NOT the default `uuid`. Reason:
    # `POST /annotation-tasks/{uuid}/items` bulk-inserts whole batches in a
    # single second, so `created_at` collides across the entire batch (sqlite
    # CURRENT_TIMESTAMP is second-resolution). A `uuid` tiebreaker would
    # shuffle the batch into arbitrary order; `id` preserves insertion order,
    # which matches `get_annotation_items_for_task`'s historical
    # `ORDER BY id DESC` and is what users actually expect after a bulk add.
    items = sort.apply(items, secondary_key="id")
    paged_items = items[pagination.offset : pagination.offset + pagination.limit]

    # Latest evaluator_run per (item, evaluator, version). One row in the
    # response per distinct version that has run, so re-running on a new
    # version doesn't hide the previous version's results.
    latest_run: Dict[tuple, Dict[str, Any]] = {}
    latest_run_ts: Dict[tuple, str] = {}
    versions_by_evaluator: Dict[str, set] = {}
    # Total runs per evaluator across ALL versions, restricted to the items
    # currently in scope (honors `item_id` and `q`; ignores pagination so the
    # count reflects the full filtered scope, not just the current page).
    # Surfaced on each entry in the top-level `evaluators` list so the FE
    # can show the count even when `live_only=true` hides non-live version
    # rows.
    scoped_item_ids = {it["uuid"] for it in items}
    run_count_by_evaluator: Dict[str, int] = {}
    for r in runs:
        ev_id = r.get("evaluator_id")
        r_item_id = r.get("item_id")
        v_id = r.get("evaluator_version_id")
        if not ev_id or not r_item_id:
            continue
        if v_id:
            versions_by_evaluator.setdefault(ev_id, set()).add(v_id)
        # Only count runs that actually produced a label. Failed runs leave
        # `value` NULL and would otherwise inflate the count beyond what an
        # annotator/FE would think of as a "label".
        if r_item_id in scoped_item_ids and r.get("value") is not None:
            run_count_by_evaluator[ev_id] = (
                run_count_by_evaluator.get(ev_id, 0) + 1
            )
        ts = r.get("completed_at") or r.get("created_at") or ""
        slot = (r_item_id, ev_id, v_id)
        if slot not in latest_run_ts or ts > latest_run_ts[slot]:
            latest_run[slot] = r
            latest_run_ts[slot] = ts

    # Always include the live version of each linked evaluator, even if it
    # hasn't run yet — that keeps a baseline "current version" row in the
    # table.
    for ev in evaluators:
        live_v = ev.get("live_version_id")
        if live_v:
            versions_by_evaluator.setdefault(ev["uuid"], set()).add(live_v)
        else:
            versions_by_evaluator.setdefault(ev["uuid"], set())

    # Latest annotation per (item, evaluator, annotator). Input is sorted by
    # updated_at ASC so overwrite gives latest-wins.
    latest_ann: Dict[tuple, Dict[str, Any]] = {}
    # Row-level (evaluator_id IS NULL) annotations carry per-(item, annotator)
    # free-text comments. Same latest-wins-on-updated_at semantics as
    # `latest_ann`, but crucially: a newer cleared/invalid comment ERASES an
    # older valid one. This matters when the same annotator has multiple
    # jobs on the same task — the unique row key is (job_id, item_id,
    # evaluator_id), so each job keeps its own row and a "clear" in the
    # newer job must not let the older job's "looks bad" survive in the
    # response. We read the string out of `value["comment"]` and treat
    # anything that isn't a non-empty string as a clear; this also guards
    # against future shape changes that would otherwise crash the response.
    # Built task-wide so the annotator union stays consistent with the
    # per-evaluator path — the per-item filter is applied later, only to
    # the response block.
    all_item_comments: Dict[str, Dict[str, str]] = {}
    for a in annotations:
        annotator_id = a.get("annotator_id")
        ev_id = a.get("evaluator_id")
        a_item_id = a.get("item_id")
        if not annotator_id or not a_item_id:
            continue
        if ev_id is None:
            value = a.get("value")
            comment: Optional[str] = None
            if isinstance(value, dict):
                raw = value.get("comment")
                if isinstance(raw, str) and raw:
                    comment = raw
            if comment is not None:
                all_item_comments.setdefault(a_item_id, {})[annotator_id] = comment
            else:
                # Newer cleared/invalid comment wipes any older one so the
                # response reflects the annotator's latest intent.
                cells = all_item_comments.get(a_item_id)
                if cells is not None:
                    cells.pop(annotator_id, None)
                    if not cells:
                        all_item_comments.pop(a_item_id, None)
            continue
        latest_ann[(a_item_id, ev_id, annotator_id)] = a

    # Annotator union — those with ≥1 (item, evaluator) annotation OR ≥1
    # free-text comment anywhere in this task. Stays task-wide even when
    # `item_id` is set, matching the docstring contract. Stable ordering by
    # name then uuid. Single bulk lookup replaces the per-annotator
    # `get_annotator(aid)` round-trips.
    annotator_ids = list(
        {key[2] for key in latest_ann.keys()}
        | {aid for cells in all_item_comments.values() for aid in cells.keys()}
    )
    annotator_rows = get_annotators_by_uuids(annotator_ids)
    annotators: List[Dict[str, Any]] = [
        {"uuid": a["uuid"], "name": a.get("name")}
        for a in annotator_rows.values()
    ]
    annotators.sort(key=lambda x: ((x.get("name") or "").lower(), x["uuid"]))

    version_cache: Dict[str, Optional[Dict[str, Any]]] = {}

    from llm_judge import _scale_bounds  # local to avoid module-load cycle

    def _version_meta(version_id: Optional[str]) -> Optional[Dict[str, Any]]:
        if not version_id:
            return None
        if version_id in version_cache:
            return version_cache[version_id]
        v = get_evaluator_version(version_id)
        if v:
            scale_min, scale_max = _scale_bounds(v.get("output_config"))
            meta = {
                "uuid": v["uuid"],
                "version_number": v.get("version_number"),
                "scale_min": scale_min,
                "scale_max": scale_max,
                "output_config": v.get("output_config"),
            }
        else:
            meta = None
        version_cache[version_id] = meta
        return meta


    def _scalar_and_reasoning(value: Any) -> tuple:
        if isinstance(value, dict):
            return value.get("value"), value.get("reasoning")
        return value, None

    def _version_row_keys(ev_id: str, live_v: Optional[str]) -> List[Optional[str]]:
        if live_only:
            # Single row per (item, evaluator) using the live version. If the
            # evaluator has no live version, emit a null-version placeholder
            # row so the evaluator stays visible (consistent with the
            # non-filtered behavior below).
            return [live_v] if live_v else [None]
        versions = list(versions_by_evaluator.get(ev_id, set()))
        if not versions:
            # Evaluator linked but has no live version and no runs anywhere.
            # Emit a single null-version row so the evaluator stays visible.
            return [None]
        # Stable ordering: live version first, then remaining versions by
        # version_number ascending (None last).
        def _sort_key(v_id: str) -> tuple:
            meta = _version_meta(v_id) or {}
            num = meta.get("version_number")
            return (0 if v_id == live_v else 1, num if num is not None else 1 << 30)
        return sorted(versions, key=_sort_key)

    rows: List[Dict[str, Any]] = []
    for item in paged_items:
        # Annotations are not version-scoped (the table has no
        # `evaluator_version_id`), so the same per-evaluator annotation cells
        # are reused across every version row for that (item, evaluator).
        for ev in evaluators:
            ev_id = ev["uuid"]
            live_v = ev.get("live_version_id")

            ann_cells: Dict[str, Optional[Dict[str, Any]]] = {}
            slot_human_scalars: List[Any] = []
            for annotator in annotators:
                a = latest_ann.get((item["uuid"], ev_id, annotator["uuid"]))
                if a is None:
                    ann_cells[annotator["uuid"]] = None
                    continue
                val, reasoning = _scalar_and_reasoning(a.get("value"))
                ann_cells[annotator["uuid"]] = {
                    "value": val,
                    "reasoning": reasoning,
                }
                scalar = _scalar(a.get("value"))
                if scalar is not None:
                    slot_human_scalars.append(scalar)

            hh_mean, hh_pairs = _pairwise_agreement(slot_human_scalars)
            human_agreement = (
                _round_agreement(hh_mean) if hh_pairs > 0 else None
            )

            for version_id in _version_row_keys(ev_id, live_v):
                run = latest_run.get((item["uuid"], ev_id, version_id))
                run_value = run.get("value") if run else None
                ev_value, ev_reasoning = _scalar_and_reasoning(run_value)
                version_meta = _version_meta(version_id)

                # Per-row evaluator agreement: pairs THIS version's run value
                # with every human annotation on the (item, evaluator) slot.
                # Per-version, so each version row gets its own number. Shares
                # `evaluator_human_pair_agreement` with the task-level rollup
                # so the two endpoints stay consistent.
                eval_scalar = (
                    _scalar(run_value) if run_value is not None else None
                )
                evaluator_agreement: Optional[float] = None
                if eval_scalar is not None and slot_human_scalars:
                    total, pairs = evaluator_human_pair_agreement(
                        eval_scalar, slot_human_scalars
                    )
                    if pairs > 0:
                        evaluator_agreement = _round_agreement(total / pairs)

                rows.append(
                    {
                        "item_id": item["uuid"],
                        "payload": item.get("payload"),
                        # Reference into the top-level `evaluators[]` block
                        # — name/description/output_type and the version's
                        # rubric (output_config, scale_min/max, is_live)
                        # live there, not duplicated on every row.
                        "evaluator_id": ev_id,
                        "evaluator_version_id": version_id,
                        "evaluator_value": ev_value,
                        "evaluator_value_name": _evaluator_value_name(
                            ev_value,
                            ev.get("output_type"),
                            (version_meta or {}).get("output_config"),
                        ),
                        "evaluator_reasoning": ev_reasoning,
                        "annotations": ann_cells,
                        "human_agreement": human_agreement,
                        "evaluator_agreement": evaluator_agreement,
                    }
                )

    # Build the enriched top-level `evaluators[]` block — one entry per
    # linked evaluator with the per-version rubric inlined so the FE has
    # everything to render row labels without joining per-row metadata.
    from llm_judge import default_output_config

    evaluators_block: List[Dict[str, Any]] = []
    for ev in evaluators:
        ev_id = ev["uuid"]
        live_v = ev.get("live_version_id")
        version_ids = _version_row_keys(ev_id, live_v)
        versions_payload: List[Dict[str, Any]] = []
        live_version_index: Optional[int] = None
        for v_id in version_ids:
            if v_id is None:
                # Placeholder for evaluators with no live version + no runs
                # — keep the slot so the table can still render an empty
                # evaluator column.
                versions_payload.append(
                    {
                        "uuid": None,
                        "version_number": None,
                        "output_config": default_output_config(
                            ev.get("output_type")
                        ),
                        "scale_min": None,
                        "scale_max": None,
                        "is_live": False,
                    }
                )
                continue
            meta = _version_meta(v_id) or {}
            output_config = meta.get("output_config")
            if output_config is None:
                output_config = default_output_config(ev.get("output_type"))
            is_live = v_id == live_v
            if is_live:
                live_version_index = len(versions_payload)
            versions_payload.append(
                {
                    "uuid": v_id,
                    "version_number": meta.get("version_number"),
                    "output_config": output_config,
                    "scale_min": meta.get("scale_min"),
                    "scale_max": meta.get("scale_max"),
                    "is_live": is_live,
                }
            )
        evaluators_block.append(
            {
                "uuid": ev_id,
                "name": ev.get("name"),
                "description": ev.get("description"),
                "output_type": ev.get("output_type"),
                "evaluator_type": ev.get("evaluator_type"),
                "data_type": ev.get("data_type"),
                "live_version_id": live_v,
                "live_version_index": live_version_index,
                "versions": versions_payload,
                "run_count": run_count_by_evaluator.get(ev_id, 0),
            }
        )

    # Filter to surviving annotators (soft-deleted ones are dropped from
    # `annotators[]` by `get_annotators_by_uuids`, so emitting their UUIDs
    # in `item_comments` would create orphans the FE has no name for).
    # Scope is `paged_items` (NOT the broader `scoped_item_ids`) so the
    # comments block tracks the rows on the current page rather than
    # shipping comments for off-page items the FE wouldn't render. To
    # collect comments for the full filtered scope (e.g. CSV export),
    # request `?limit=<total>` — the cap is set high enough for that.
    paged_item_ids = {it["uuid"] for it in paged_items}
    surviving_annotator_ids = {a["uuid"] for a in annotators}
    item_comments: Dict[str, Dict[str, str]] = {}
    for cmt_item_id, cells in all_item_comments.items():
        if cmt_item_id not in paged_item_ids:
            continue
        surviving_cells = {
            aid: text
            for aid, text in cells.items()
            if aid in surviving_annotator_ids
        }
        if surviving_cells:
            item_comments[cmt_item_id] = surviving_cells

    return {
        "task_id": task_uuid,
        "task_type": task["type"],
        "evaluators": evaluators_block,
        "annotators": annotators,
        "rows": rows,
        "item_comments": item_comments,
        "pagination": {
            "total": total_items,
            "limit": pagination.limit,
            "offset": pagination.offset,
        },
    }


@router.delete("/{task_uuid}/evaluators/{evaluator_uuid}")
async def unlink_evaluator_from_task(
    task_uuid: str,
    evaluator_uuid: str,
    ctx: OrgContext = Depends(get_current_org),
):
    _ensure_owned_task(task_uuid, ctx.org_uuid)
    removed = remove_evaluator_from_annotation_task(task_uuid, evaluator_uuid)
    if not removed:
        raise HTTPException(
            status_code=404, detail="Evaluator is not linked to this task"
        )
    return {"message": "Evaluator unlinked from annotation task"}
