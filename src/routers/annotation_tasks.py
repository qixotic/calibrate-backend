from typing import Optional, List, Dict, Any
from fastapi import APIRouter, HTTPException, Depends, Query
from pydantic import BaseModel

import secrets

from db import (
    ANNOTATION_TASK_TYPES,
    create_annotation_task,
    get_annotation_task,
    get_all_annotation_tasks,
    update_annotation_task,
    delete_annotation_task,
    add_evaluator_to_annotation_task,
    remove_evaluator_from_annotation_task,
    get_evaluators_for_annotation_task,
    get_evaluator,
    get_evaluator_version,
    get_annotations_for_task,
    create_annotation_items,
    bulk_update_annotation_items,
    soft_delete_annotation_items,
    get_annotation_items_for_task,
    get_annotation_item,
    create_annotation_job,
    get_annotation_job,
    get_jobs_for_task,
    get_jobs_for_task_detailed,
    get_job_items,
    update_annotation_job_status,
    upsert_annotation,
    get_annotations_for_item,
    get_annotator,
    create_job,
    get_job,
    get_generic_jobs_for_task,
    soft_delete_job,
    update_job,
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
    try_start_queued_job,
)
from auth_utils import get_current_user_id
from annotation_metrics import (
    aggregate_agreement,
    aggregate_human_evaluator_agreement,
    per_item_agreement,
    trend_series,
    trend_series_human_evaluator,
)


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


def _ensure_owned_task(task_uuid: str, user_id: str) -> Dict[str, Any]:
    task = get_annotation_task(task_uuid)
    if not task or task.get("user_id") != user_id:
        # 404 (not 403) — avoid leaking existence
        raise HTTPException(status_code=404, detail="Annotation task not found")
    return task


def _ensure_owned_evaluator(evaluator_uuid: str, user_id: str) -> Dict[str, Any]:
    evaluator = get_evaluator(evaluator_uuid)
    if not evaluator:
        raise HTTPException(status_code=404, detail="Evaluator not found")
    owner = evaluator.get("owner_user_id")
    # owner_user_id IS NULL ⇒ seeded default (visible to everyone)
    if owner is not None and owner != user_id:
        raise HTTPException(status_code=404, detail="Evaluator not found")
    return evaluator


@router.post("", response_model=AnnotationTaskCreateResponse)
async def create_annotation_task_endpoint(
    payload: AnnotationTaskCreate,
    user_id: str = Depends(get_current_user_id),
):
    """Create a new annotation task. Optionally link evaluators in the same call."""
    if payload.type not in ANNOTATION_TASK_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"type must be one of {list(ANNOTATION_TASK_TYPES)}",
        )
    if payload.evaluator_ids:
        for evaluator_id in payload.evaluator_ids:
            _ensure_owned_evaluator(evaluator_id, user_id)

    task_uuid = create_annotation_task(
        name=payload.name,
        description=payload.description,
        type=payload.type,
        user_id=user_id,
    )

    if payload.evaluator_ids:
        for evaluator_id in payload.evaluator_ids:
            add_evaluator_to_annotation_task(task_uuid, evaluator_id)

    return AnnotationTaskCreateResponse(
        uuid=task_uuid, message="Annotation task created successfully"
    )


@router.get("", response_model=List[AnnotationTaskResponse])
async def list_annotation_tasks(user_id: str = Depends(get_current_user_id)):
    """List all annotation tasks owned by the authenticated user."""
    tasks = get_all_annotation_tasks(user_id=user_id)
    for task in tasks:
        task["evaluators"] = get_evaluators_for_annotation_task(task["uuid"])
    return tasks


@router.get("/{task_uuid}", response_model=AnnotationTaskResponse)
async def get_annotation_task_endpoint(
    task_uuid: str, user_id: str = Depends(get_current_user_id)
):
    """Get an annotation task by UUID, including its linked evaluators,
    all items (each annotated with per-item agreement stats), and all jobs."""
    task = _ensure_owned_task(task_uuid, user_id)
    evaluators = get_evaluators_for_annotation_task(task_uuid)
    task["evaluators"] = evaluators
    task["jobs"] = get_jobs_for_task_detailed(task_uuid)

    items = get_annotation_items_for_task(task_uuid)
    # Pre-fetch annotations + evaluator_runs once and bucket by item to avoid
    # an N+1 query pattern on the per-item agreement computation.
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
    user_id: str = Depends(get_current_user_id),
):
    _ensure_owned_task(task_uuid, user_id)
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
    task_uuid: str, user_id: str = Depends(get_current_user_id)
):
    _ensure_owned_task(task_uuid, user_id)
    deleted = delete_annotation_task(task_uuid)
    if not deleted:
        raise HTTPException(status_code=404, detail="Annotation task not found")
    return {"message": "Annotation task deleted successfully"}


# ============ Evaluator linking ============


@router.get("/{task_uuid}/evaluators", response_model=List[Dict[str, Any]])
async def list_task_evaluators(
    task_uuid: str, user_id: str = Depends(get_current_user_id)
):
    _ensure_owned_task(task_uuid, user_id)
    return get_evaluators_for_annotation_task(task_uuid)


@router.post("/{task_uuid}/evaluators")
async def link_evaluator_to_task(
    task_uuid: str,
    payload: EvaluatorLinkRequest,
    user_id: str = Depends(get_current_user_id),
):
    _ensure_owned_task(task_uuid, user_id)
    _ensure_owned_evaluator(payload.evaluator_id, user_id)
    add_evaluator_to_annotation_task(task_uuid, payload.evaluator_id)
    return {"message": "Evaluator linked to annotation task"}


# ============ Items ============


class AnnotationItemPayload(BaseModel):
    # `payload` is a free-form JSON value whose shape is owned by the
    # task `type`. The backend doesn't validate the shape — frontend +
    # downstream consumers (evaluator runs, agreement, etc.) interpret it.
    payload: Any


class BulkItemsRequest(BaseModel):
    items: List[AnnotationItemPayload]


@router.get("/{task_uuid}/items")
async def list_task_items(
    task_uuid: str, user_id: str = Depends(get_current_user_id)
):
    _ensure_owned_task(task_uuid, user_id)
    return get_annotation_items_for_task(task_uuid)


@router.post("/{task_uuid}/items")
async def bulk_create_items(
    task_uuid: str,
    payload: BulkItemsRequest,
    user_id: str = Depends(get_current_user_id),
):
    """Bulk-insert annotation items. Order of insertion is preserved by `id`."""
    _ensure_owned_task(task_uuid, user_id)
    if not payload.items:
        raise HTTPException(status_code=400, detail="items must be non-empty")
    new_uuids = create_annotation_items(
        task_uuid, [it.dict() for it in payload.items]
    )
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
    user_id: str = Depends(get_current_user_id),
):
    """Bulk-update item `payload`s in a task.

    Updates not in this task (or referencing deleted items) are skipped
    silently; `updated_count` reflects rows actually changed.
    """
    _ensure_owned_task(task_uuid, user_id)
    if not payload.updates:
        raise HTTPException(status_code=400, detail="updates must be non-empty")
    try:
        updated_count = bulk_update_annotation_items(
            task_uuid, [u.dict() for u in payload.updates]
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return {"updated_count": updated_count}


class BulkDeleteItemsRequest(BaseModel):
    item_ids: List[str]


@router.delete("/{task_uuid}/items")
async def bulk_delete_items(
    task_uuid: str,
    payload: BulkDeleteItemsRequest,
    user_id: str = Depends(get_current_user_id),
):
    """Soft-delete one or more items in a task.

    Items not in this task (or already deleted) are skipped silently;
    `deleted_count` reflects how many rows actually transitioned to deleted.
    Items linked to existing jobs remain referenced by those jobs and their
    annotations — they just stop appearing in `GET /items`.
    """
    _ensure_owned_task(task_uuid, user_id)
    if not payload.item_ids:
        raise HTTPException(status_code=400, detail="item_ids must be non-empty")
    deleted_count = soft_delete_annotation_items(task_uuid, payload.item_ids)
    return {"deleted_count": deleted_count}


@router.get("/{task_uuid}/items/{item_uuid}")
async def get_item(
    task_uuid: str, item_uuid: str, user_id: str = Depends(get_current_user_id)
):
    _ensure_owned_task(task_uuid, user_id)
    item = get_annotation_item(item_uuid)
    if not item or item.get("task_id") != task_uuid:
        raise HTTPException(status_code=404, detail="Item not found")
    item["annotations"] = get_annotations_for_item(item_uuid)
    return item


# ============ Jobs ============


class CreateJobsRequest(BaseModel):
    annotator_ids: List[str]
    item_ids: List[str]


@router.get("/{task_uuid}/jobs")
async def list_task_jobs(
    task_uuid: str, user_id: str = Depends(get_current_user_id)
):
    _ensure_owned_task(task_uuid, user_id)
    return get_jobs_for_task(task_uuid)


@router.post("/{task_uuid}/jobs")
async def create_jobs(
    task_uuid: str,
    payload: CreateJobsRequest,
    user_id: str = Depends(get_current_user_id),
):
    """Assign a set of items to one or more annotators. Creates ONE job per
    annotator — each with its own unique public_token. Job item sets are
    frozen after creation."""
    _ensure_owned_task(task_uuid, user_id)
    if not payload.annotator_ids:
        raise HTTPException(
            status_code=400, detail="annotator_ids must be non-empty"
        )
    if not payload.item_ids:
        raise HTTPException(status_code=400, detail="item_ids must be non-empty")

    # Validate annotators (all up front, before any insert).
    annotators_by_id: Dict[str, Dict[str, Any]] = {}
    for annotator_id in payload.annotator_ids:
        annotator = get_annotator(annotator_id)
        if not annotator or annotator.get("user_id") != user_id:
            raise HTTPException(
                status_code=404,
                detail=f"Annotator not found: {annotator_id}",
            )
        annotators_by_id[annotator_id] = annotator

    # Validate items (all must belong to this task).
    valid_item_ids = {
        it["uuid"] for it in get_annotation_items_for_task(task_uuid)
    }
    invalid = [i for i in payload.item_ids if i not in valid_item_ids]
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
            item_uuids=payload.item_ids,
            public_token=public_token,
        )
        jobs_created.append(
            {
                "uuid": job_uuid,
                "public_token": public_token,
                "annotator_id": annotator_id,
                "annotator_name": annotators_by_id[annotator_id]["name"],
                "item_ids": payload.item_ids,
                "item_count": len(payload.item_ids),
                "status": "pending",
            }
        )
    return {"jobs": jobs_created, "count": len(jobs_created)}


@router.get("/{task_uuid}/jobs/{job_uuid}")
async def get_annotation_job_endpoint(
    task_uuid: str,
    job_uuid: str,
    user_id: str = Depends(get_current_user_id),
):
    _ensure_owned_task(task_uuid, user_id)
    job = get_annotation_job(job_uuid)
    if not job or job.get("task_id") != task_uuid:
        raise HTTPException(status_code=404, detail="Job not found")
    job["items"] = get_job_items(job_uuid)
    return job


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
    user_id: str = Depends(get_current_user_id),
):
    """Owner-side upsert of a single annotation. The public-link/token flow
    will be added in a later slice."""
    _ensure_owned_task(task_uuid, user_id)
    job = get_annotation_job(payload.job_id)
    if not job or job.get("task_id") != task_uuid:
        raise HTTPException(status_code=404, detail="Job not found")
    item = get_annotation_item(payload.item_id)
    if not item or item.get("task_id") != task_uuid:
        raise HTTPException(status_code=404, detail="Item not found")
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
    # Optional subset. Omit/null = run on every item in the task.
    # Empty array is rejected (400) — most likely an accidental empty submit.
    item_ids: Optional[List[str]] = None


@router.post("/{task_uuid}/evaluator-runs")
async def start_evaluator_run(
    task_uuid: str,
    payload: EvaluatorRunStartRequest,
    user_id: str = Depends(get_current_user_id),
):
    """Run one or more evaluators on every item in this task (or a subset).
    Returns a job UUID; the actual evaluation runs asynchronously via the
    calibrate CLI's `--eval-only` mode. Poll
    `GET /evaluator-runs/{job_uuid}` for status.

    Supported task types: `stt`, `llm`, `simulation`. (Voice simulations and
    TTS are not supported in eval-only mode.)"""
    task = _ensure_owned_task(task_uuid, user_id)
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
    if payload.item_ids is None:
        items = all_items
        item_ids_persisted: Optional[List[str]] = None  # null = "all items"
    else:
        if not payload.item_ids:
            raise HTTPException(
                status_code=400,
                detail="item_ids must be non-empty if provided (omit the field to run on all items)",
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
        build_dataset_for_task_type(
            task["type"], items, [ev["name"] for ev in resolved]
        )
    except DatasetBuildError as e:
        raise HTTPException(status_code=400, detail=str(e))

    # Decide queue vs immediate start (shared eval queue with stt-eval/tts-eval).
    can_start = can_start_job(EVAL_JOB_TYPES, user_id)
    initial_status = (
        TaskStatus.IN_PROGRESS.value if can_start else TaskStatus.QUEUED.value
    )

    job_uuid = create_job(
        job_type=ANNOTATION_EVAL_JOB_TYPE,
        user_id=user_id,
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

    if can_start:
        start_annotation_eval_job(
            job_uuid=job_uuid,
            task_uuid=task_uuid,
            user_id=user_id,
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
    stay as the source of truth — we just bolt on a fresh `evaluator` block."""
    if not runs:
        return runs
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
        enriched["evaluator_version"] = (
            {
                "uuid": version["uuid"],
                "version_number": version.get("version_number"),
                "judge_model": version.get("judge_model"),
            }
            if version
            else None
        )
        out.append(enriched)
    return out


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
    return out


@router.get("/{task_uuid}/evaluator-runs")
async def list_evaluator_run_jobs(
    task_uuid: str, user_id: str = Depends(get_current_user_id)
):
    _ensure_owned_task(task_uuid, user_id)
    jobs = get_generic_jobs_for_task(task_uuid, ANNOTATION_EVAL_JOB_TYPE)
    return [_shape_eval_job_for_response(j) for j in jobs]


@router.get("/{task_uuid}/evaluator-runs/{job_uuid}")
async def get_evaluator_run_job(
    task_uuid: str,
    job_uuid: str,
    user_id: str = Depends(get_current_user_id),
):
    _ensure_owned_task(task_uuid, user_id)
    job = get_job(job_uuid, user_id=user_id)
    if (
        not job
        or job.get("type") != ANNOTATION_EVAL_JOB_TYPE
        or (job.get("details") or {}).get("task_id") != task_uuid
    ):
        raise HTTPException(status_code=404, detail="Job not found")
    shaped = _shape_eval_job_for_response(job)
    shaped["runs"] = _enrich_runs_with_live_evaluator(
        get_evaluator_runs_for_job(job_uuid)
    )
    return shaped


@router.delete("/{task_uuid}/evaluator-runs/{job_uuid}")
async def delete_evaluator_run_job(
    task_uuid: str,
    job_uuid: str,
    user_id: str = Depends(get_current_user_id),
):
    """Soft-delete an evaluator-run job and all its `evaluator_runs` rows.

    In-flight jobs (status = 'in_progress') are not allowed to be deleted —
    let them finish (or fail) first, then delete. Queued jobs CAN be deleted
    (they were never started)."""
    _ensure_owned_task(task_uuid, user_id)
    job = get_job(job_uuid, user_id=user_id)
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


@router.get("/{task_uuid}/items/{item_uuid}/evaluator-runs")
async def list_item_evaluator_runs(
    task_uuid: str,
    item_uuid: str,
    user_id: str = Depends(get_current_user_id),
):
    _ensure_owned_task(task_uuid, user_id)
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
    user_id: str = Depends(get_current_user_id),
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
    _ensure_owned_task(task_uuid, user_id)
    annotations = get_annotations_for_task(task_uuid)
    runs = get_evaluator_runs_for_task(task_uuid)
    linked = get_evaluators_for_annotation_task(task_uuid)

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


@router.delete("/{task_uuid}/evaluators/{evaluator_uuid}")
async def unlink_evaluator_from_task(
    task_uuid: str,
    evaluator_uuid: str,
    user_id: str = Depends(get_current_user_id),
):
    _ensure_owned_task(task_uuid, user_id)
    removed = remove_evaluator_from_annotation_task(task_uuid, evaluator_uuid)
    if not removed:
        raise HTTPException(
            status_code=404, detail="Evaluator is not linked to this task"
        )
    return {"message": "Evaluator unlinked from annotation task"}
