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
    get_evaluator_ids_for_job,
    update_annotation_job_status,
    upsert_annotation,
    get_annotations_for_item,
    get_annotator,
    get_annotators_by_uuids,
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
    return item


@router.get("/{task_uuid}/items/{item_uuid}/annotations")
async def list_item_annotations(
    task_uuid: str, item_uuid: str, user_id: str = Depends(get_current_user_id)
):
    """All human annotations across every job for one item. Sibling of
    `/items/{item_uuid}/evaluator-runs`."""
    _ensure_owned_task(task_uuid, user_id)
    item = get_annotation_item(item_uuid)
    if not item or item.get("task_id") != task_uuid:
        raise HTTPException(status_code=404, detail="Item not found")
    return get_annotations_for_item(item_uuid)


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
    if len(payload.item_ids) != len(set(payload.item_ids)):
        # Duplicate item_ids would violate UNIQUE(job_id, item_id) on
        # annotation_job_items and surface as a 500. Surface as a clean 400.
        seen: set = set()
        duplicates: List[str] = []
        for i in payload.item_ids:
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
    """Owner-side upsert of a single annotation.

    Both the item and the (non-null) evaluator must be in the job's
    snapshot — same contract as the public token flow. Without these
    checks, owner-authenticated callers could write annotations against
    items/evaluators that were never assigned to this specific job (just
    same task), polluting `completed_item_count`, agreement aggregates,
    and the summary view.
    """
    _ensure_owned_task(task_uuid, user_id)
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
        # "All items" snapshots the live set at submission time. Storing the
        # resolved UUIDs (instead of leaving null) ensures recovery after a
        # crash re-runs the same items the user originally submitted, even if
        # items were added or deleted in the meantime.
        items = all_items
        item_ids_persisted: Optional[List[str]] = [it["uuid"] for it in all_items]
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
        build_dataset_for_task_type(task["type"], items, resolved)
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
    user_id: str = Depends(get_current_user_id),
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
        "task_type": "stt" | "llm" | "simulation",
        "evaluators": [{evaluator_id, name, output_type}],
        "annotators": [{uuid, name}],   # union of annotators with ≥1 (item, evaluator)
                                        # annotation in this task; column order
        "rows": [
          {
            "item_id": str,
            "payload": <item.payload>,             # FE derives display per task_type
            "evaluator_id": str,
            "evaluator_name": str,
            "output_type": "binary" | "rating",
            "evaluator_version_id": str | null,
            "evaluator_version_number": int | null,
            "evaluator_value": <scalar | null>,    # latest run on this slot
            "evaluator_reasoning": str | null,
            "annotations": {
              "<annotator_uuid>": {"value": <scalar>, "reasoning": str | null} | null,
              ...
            }
          }
        ]
      }

    Cell rules:
      - Default: one row per `(item, evaluator, version)` — a version row
        appears for every distinct version that has runs for this evaluator
        in the task, plus one for the live version (with `null` value if it
        hasn't run yet). With `live_only=true`, only the live-version row is
        emitted per `(item, evaluator)`.
      - `evaluator_value` is the latest evaluator-run for THAT specific
        version slot, regardless of which evaluator-run job produced it.
      - `is_live_version` flags rows whose `evaluator_version_id` matches
        the evaluator's current `live_version_id`.
      - Each annotator cell is that annotator's latest annotation for the slot,
        across ALL annotation jobs (matches the agreement aggregator's
        latest-wins-per-annotator semantics). `null` if they haven't annotated it.
      - Row-level overall annotations (`evaluator_id IS NULL`) are not surfaced
        here — this view is per-evaluator.
    """
    task = _ensure_owned_task(task_uuid, user_id)
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

    # Latest evaluator_run per (item, evaluator, version). One row in the
    # response per distinct version that has run, so re-running on a new
    # version doesn't hide the previous version's results.
    latest_run: Dict[tuple, Dict[str, Any]] = {}
    latest_run_ts: Dict[tuple, str] = {}
    versions_by_evaluator: Dict[str, set] = {}
    for r in runs:
        ev_id = r.get("evaluator_id")
        r_item_id = r.get("item_id")
        v_id = r.get("evaluator_version_id")
        if not ev_id or not r_item_id:
            continue
        if v_id:
            versions_by_evaluator.setdefault(ev_id, set()).add(v_id)
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
    for a in annotations:
        annotator_id = a.get("annotator_id")
        ev_id = a.get("evaluator_id")
        a_item_id = a.get("item_id")
        if not annotator_id or not ev_id or not a_item_id:
            continue
        latest_ann[(a_item_id, ev_id, annotator_id)] = a

    # Annotator union — only those with ≥1 (item, evaluator) annotation visible
    # in this view. Stable ordering by name then uuid. Single bulk lookup
    # replaces the per-annotator `get_annotator(aid)` round-trips.
    annotator_ids = list({key[2] for key in latest_ann.keys()})
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
    for item in items:
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
                        "evaluator_id": ev_id,
                        "evaluator_name": ev.get("name"),
                        "output_type": ev.get("output_type"),
                        "evaluator_version_id": version_id,
                        "evaluator_version_number": (
                            version_meta.get("version_number")
                            if version_meta
                            else None
                        ),
                        "scale_min": (
                            version_meta.get("scale_min")
                            if version_meta
                            else None
                        ),
                        "scale_max": (
                            version_meta.get("scale_max")
                            if version_meta
                            else None
                        ),
                        "is_live_version": (
                            version_id == live_v if version_id else False
                        ),
                        "evaluator_value": ev_value,
                        "evaluator_reasoning": ev_reasoning,
                        "annotations": ann_cells,
                        "human_agreement": human_agreement,
                        "evaluator_agreement": evaluator_agreement,
                    }
                )

    return {
        "task_id": task_uuid,
        "task_type": task["type"],
        "evaluators": [
            {
                "evaluator_id": e["uuid"],
                "name": e.get("name"),
                "output_type": e.get("output_type"),
            }
            for e in evaluators
        ],
        "annotators": annotators,
        "rows": rows,
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
