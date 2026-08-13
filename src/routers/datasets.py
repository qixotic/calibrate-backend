import logging
from typing import List, Literal, Optional

from fastapi import APIRouter, HTTPException, Depends, Path, Query
from pydantic import BaseModel, Field

from auth_utils import get_current_org, OrgContext
from utils import presign_audio_path
from db import (
    ensure_name_unique,
    create_dataset,
    get_dataset,
    get_all_datasets,
    get_dataset_item_counts,
    get_dataset_eval_counts,
    update_dataset_name,
    delete_dataset,
    add_dataset_items,
    get_dataset_item,
    get_dataset_items,
    get_dataset_items_by_uuids,
    update_dataset_item,
    delete_dataset_item,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/datasets", tags=["datasets"])

_EXAMPLE_DATASET_UUID = "f47ac10b-58cc-4372-a567-0e02b2c3d479"
_EXAMPLE_ITEM_UUID = "6ba7b810-9dad-11d1-80b4-00c04fd430c8"


# ── Request / Response models ────────────────────────────────────────────────


class DatasetCreateRequest(BaseModel):
    name: str = Field(description="Dataset name, unique within the workspace")
    dataset_type: Literal["stt", "tts"] = Field(
        description="`stt` items carry audio + ground-truth text. `tts` items carry text only"
    )


class DatasetRenameRequest(BaseModel):
    name: str = Field(description="New dataset name, unique within the workspace")


class DatasetItemIn(BaseModel):
    audio_path: Optional[str] = Field(
        None,
        description="Audio location as an `s3://bucket/key` URI. **Required for STT datasets. Must be omitted for TTS datasets**",
    )
    text: str = Field(
        description="For STT, the ground-truth transcript. For TTS, the text to synthesize"
    )


class DatasetItemUpdate(BaseModel):
    audio_path: Optional[str] = Field(
        None,
        description="New audio `s3://bucket/key` URI. Omit the field to leave audio unchanged. **STT requires a value, TTS forbids one**",
    )
    text: Optional[str] = Field(
        None, description="New item text. Omit to leave text unchanged"
    )


class DatasetItemResponse(BaseModel):
    uuid: str = Field(
        min_length=36,
        max_length=36,
        description="Dataset item ID",
        examples=[_EXAMPLE_ITEM_UUID],
    )
    audio_path: Optional[str] = Field(
        None,
        description="Presigned download URL for the item's audio",
    )
    text: str = Field(description="Ground-truth transcript (STT) or synthesis text (TTS)")
    order_index: int = Field(description="Position of the item within the dataset. The first item is 0")
    created_at: str = Field(description="When the dataset item was created (ISO 8601 UTC)")
    updated_at: Optional[str] = Field(
        None, description="When the dataset item was last updated (ISO 8601 UTC)"
    )


class DatasetResponse(BaseModel):
    uuid: str = Field(
        min_length=36,
        max_length=36,
        description="Dataset ID",
        examples=[_EXAMPLE_DATASET_UUID],
    )
    name: str = Field(description="Dataset name")
    dataset_type: str = Field(description="Dataset type (`stt` or `tts`)")
    item_count: int = Field(description="Number of items in the dataset")
    eval_count: int = Field(description="Number of evaluation jobs that used this dataset")
    created_at: str = Field(description="When the dataset was created (ISO 8601 UTC)")
    updated_at: str = Field(description="When the dataset was last updated (ISO 8601 UTC)")


class DatasetDetailResponse(DatasetResponse):
    items: List[DatasetItemResponse] = Field(
        description="All items in the dataset, ordered by `order_index`"
    )


# ── Helpers ──────────────────────────────────────────────────────────────────


def _validate_items_for_type(dataset_type: str, items: List[DatasetItemIn]) -> None:
    """Raise HTTPException if items are inconsistent with the dataset type."""
    for item in items:
        if dataset_type == "stt" and not item.audio_path:
            raise HTTPException(
                status_code=400,
                detail="STT dataset items must include audio_path",
            )
        if dataset_type == "tts" and item.audio_path:
            raise HTTPException(
                status_code=400,
                detail="TTS dataset items must not include audio_path",
            )


def _item_row_to_response(row: dict) -> DatasetItemResponse:
    return DatasetItemResponse(
        uuid=row["uuid"],
        audio_path=presign_audio_path(row.get("audio_path")),
        text=row["text"],
        order_index=row["order_index"],
        created_at=row["created_at"],
        updated_at=row.get("updated_at"),
    )


def _dataset_row_to_response(
    row: dict, item_count: int, eval_count: int = 0
) -> DatasetResponse:
    return DatasetResponse(
        uuid=row["uuid"],
        name=row["name"],
        dataset_type=row["type"],
        item_count=item_count,
        eval_count=eval_count,
        created_at=row["created_at"],
        updated_at=row["updated_at"],
    )


# ── Routes ───────────────────────────────────────────────────────────────────


@router.post("", response_model=DatasetResponse, status_code=201, summary="Create dataset")
def create_new_dataset(
    request: DatasetCreateRequest,
    ctx: OrgContext = Depends(get_current_org),
):
    """Create a new empty dataset. Add items separately"""
    with ensure_name_unique("datasets", request.name, ctx.org_uuid, entity="Dataset"):
        dataset_uuid = create_dataset(
            name=request.name,
            dataset_type=request.dataset_type,
            org_uuid=ctx.org_uuid,
            user_id=ctx.user_id,
        )
    row = get_dataset(dataset_uuid, org_uuid=ctx.org_uuid)
    return _dataset_row_to_response(row, item_count=0, eval_count=0)


@router.get("", response_model=List[DatasetResponse], summary="List datasets")
def list_datasets(
    dataset_type: Optional[str] = Query(
        None, description="Filter by dataset type (`stt` or `tts`). Omit to return all types"
    ),
    ctx: OrgContext = Depends(get_current_org),
):
    """List all datasets, optionally filtered by type"""
    if dataset_type and dataset_type not in ("stt", "tts"):
        raise HTTPException(
            status_code=400, detail="dataset_type must be 'stt' or 'tts'"
        )

    rows = get_all_datasets(org_uuid=ctx.org_uuid, dataset_type=dataset_type)
    uuids = [row["uuid"] for row in rows]
    counts = get_dataset_item_counts(uuids)
    eval_counts = get_dataset_eval_counts(uuids)
    return [
        _dataset_row_to_response(
            row,
            item_count=counts.get(row["uuid"], 0),
            eval_count=eval_counts.get(row["uuid"], 0),
        )
        for row in rows
    ]


@router.get("/{dataset_id}", response_model=DatasetDetailResponse, summary="Get dataset")
def get_dataset_detail(
    dataset_id: str = Path(
        description="The dataset to retrieve",
        examples=["f47ac10b-58cc-4372-a567-0e02b2c3d479"],
    ),
    ctx: OrgContext = Depends(get_current_org),
):
    """Get a dataset with all of its items"""
    row = get_dataset(dataset_id, org_uuid=ctx.org_uuid)
    if not row:
        raise HTTPException(status_code=404, detail="Dataset not found")

    items = get_dataset_items(dataset_id)
    eval_counts = get_dataset_eval_counts([dataset_id])
    return DatasetDetailResponse(
        uuid=row["uuid"],
        name=row["name"],
        dataset_type=row["type"],
        item_count=len(items),
        eval_count=eval_counts.get(dataset_id, 0),
        created_at=row["created_at"],
        updated_at=row["updated_at"],
        items=[_item_row_to_response(i) for i in items],
    )


@router.patch("/{dataset_id}", response_model=DatasetResponse, summary="Update dataset")
def rename_dataset(
    request: DatasetRenameRequest,
    dataset_id: str = Path(
        description="The dataset to rename",
        examples=["f47ac10b-58cc-4372-a567-0e02b2c3d479"],
    ),
    ctx: OrgContext = Depends(get_current_org),
):
    """Rename a dataset"""
    row = get_dataset(dataset_id, org_uuid=ctx.org_uuid)
    if not row:
        raise HTTPException(status_code=404, detail="Dataset not found")

    with ensure_name_unique(
        "datasets", request.name, ctx.org_uuid, entity="Dataset", exclude_uuid=dataset_id
    ):
        update_dataset_name(dataset_id, org_uuid=ctx.org_uuid, name=request.name)
    row = get_dataset(dataset_id, org_uuid=ctx.org_uuid)
    counts = get_dataset_item_counts([dataset_id])
    eval_counts = get_dataset_eval_counts([dataset_id])
    return _dataset_row_to_response(
        row,
        item_count=counts.get(dataset_id, 0),
        eval_count=eval_counts.get(dataset_id, 0),
    )


@router.delete("/{dataset_id}", status_code=204, summary="Delete dataset")
def remove_dataset(
    dataset_id: str = Path(
        description="The dataset to delete",
        examples=["f47ac10b-58cc-4372-a567-0e02b2c3d479"],
    ),
    ctx: OrgContext = Depends(get_current_org),
):
    """Delete a dataset and all of its items"""
    row = get_dataset(dataset_id, org_uuid=ctx.org_uuid)
    if not row:
        raise HTTPException(status_code=404, detail="Dataset not found")

    delete_dataset(dataset_id, org_uuid=ctx.org_uuid)


@router.post(
    "/{dataset_id}/items",
    response_model=List[DatasetItemResponse],
    status_code=201,
    summary="Bulk create dataset items",
)
def add_items(
    items: List[DatasetItemIn],
    dataset_id: str = Path(
        description="The dataset to add items to",
        examples=["f47ac10b-58cc-4372-a567-0e02b2c3d479"],
    ),
    ctx: OrgContext = Depends(get_current_org),
):
    """Append items to a dataset"""
    if not items:
        raise HTTPException(status_code=400, detail="items list cannot be empty")
    if len(items) > 1000:
        raise HTTPException(
            status_code=400, detail="Cannot add more than 1000 items per request"
        )

    row = get_dataset(dataset_id, org_uuid=ctx.org_uuid)
    if not row:
        raise HTTPException(status_code=404, detail="Dataset not found")

    _validate_items_for_type(row["type"], items)

    item_dicts = [{"audio_path": i.audio_path, "text": i.text} for i in items]
    new_uuids = add_dataset_items(dataset_id, item_dicts)

    return [_item_row_to_response(i) for i in get_dataset_items_by_uuids(new_uuids)]


@router.patch(
    "/{dataset_id}/items/{item_uuid}",
    response_model=DatasetItemResponse,
    summary="Update dataset item",
)
def update_item(
    request: DatasetItemUpdate,
    dataset_id: str = Path(
        description="The dataset containing the item",
        examples=["f47ac10b-58cc-4372-a567-0e02b2c3d479"],
    ),
    item_uuid: str = Path(
        description="The dataset item to update",
        examples=["a3b2c1d0-e5f4-3210-abcd-ef1234567890"],
    ),
    ctx: OrgContext = Depends(get_current_org),
):
    """Update a dataset item's audio or ground-truth text"""
    row = get_dataset(dataset_id, org_uuid=ctx.org_uuid)
    if not row:
        raise HTTPException(status_code=404, detail="Dataset not found")

    if not {"text", "audio_path"} & request.model_fields_set:
        raise HTTPException(status_code=400, detail="Nothing to update")

    if "audio_path" in request.model_fields_set:
        if row["type"] == "stt" and not request.audio_path:
            raise HTTPException(
                status_code=400,
                detail="STT dataset items must include audio_path",
            )
        if row["type"] == "tts" and request.audio_path:
            raise HTTPException(
                status_code=400,
                detail="TTS dataset items must not include audio_path",
            )

    updated = update_dataset_item(
        item_uuid,
        dataset_id,
        text=request.text if "text" in request.model_fields_set else None,
        audio_path=(
            request.audio_path if "audio_path" in request.model_fields_set else ...
        ),
    )
    if not updated:
        raise HTTPException(status_code=404, detail="Item not found")

    item = get_dataset_item(item_uuid, dataset_id)
    if not item:
        raise HTTPException(status_code=404, detail="Item not found")
    return _item_row_to_response(item)


@router.delete(
    "/{dataset_id}/items/{item_uuid}", status_code=204, summary="Delete dataset item"
)
def remove_item(
    dataset_id: str = Path(
        description="The dataset containing the item",
        examples=["f47ac10b-58cc-4372-a567-0e02b2c3d479"],
    ),
    item_uuid: str = Path(
        description="The dataset item to delete",
        examples=["a3b2c1d0-e5f4-3210-abcd-ef1234567890"],
    ),
    ctx: OrgContext = Depends(get_current_org),
):
    """Delete a dataset item from its dataset"""
    row = get_dataset(dataset_id, org_uuid=ctx.org_uuid)
    if not row:
        raise HTTPException(status_code=404, detail="Dataset not found")

    deleted = delete_dataset_item(item_uuid, dataset_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Item not found")
