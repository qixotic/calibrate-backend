"""Shared helpers for resolving dataset inputs used by STT and TTS evaluation routers."""

from dataclasses import dataclass
from typing import Dict, List, Optional

from fastapi import HTTPException

from db import get_dataset, get_dataset_items, create_dataset, add_dataset_items


@dataclass
class ResolvedDatasetInputs:
    texts: List[str]
    audio_paths: Optional[List[str]]
    dataset_id: Optional[str]
    dataset_name: Optional[str]
    item_ids: Optional[List[str]]


def resolve_dataset_inputs(
    *,
    dataset_id: Optional[str],
    org_uuid: str,
    expected_type: str,
    texts: Optional[List[str]] = None,
    audio_paths: Optional[List[str]] = None,
    dataset_name: Optional[str] = None,
    user_id: Optional[str] = None,
) -> ResolvedDatasetInputs:
    """Resolve evaluation inputs from an existing dataset or inline data.

    For STT (expected_type="stt") both audio_paths and texts are required inline.
    For TTS (expected_type="tts") only texts are required inline.

    When *dataset_name* is provided with inline data, a new dataset is created
    and persisted before the evaluation starts. Ignored when *dataset_id* is set.
    """
    if dataset_id:
        dataset = get_dataset(dataset_id, org_uuid=org_uuid)
        if not dataset:
            raise HTTPException(status_code=404, detail="Dataset not found")
        if dataset["type"] != expected_type:
            raise HTTPException(
                status_code=400,
                detail=f"Dataset type must be '{expected_type}' for {expected_type.upper()} evaluation",
            )
        items = get_dataset_items(dataset_id)
        if not items:
            raise HTTPException(status_code=400, detail="Dataset has no items")

        resolved_texts = [i["text"] for i in items]
        resolved_audio = (
            [i["audio_path"] for i in items] if expected_type == "stt" else None
        )
        return ResolvedDatasetInputs(
            texts=resolved_texts,
            audio_paths=resolved_audio,
            dataset_id=dataset_id,
            dataset_name=dataset.get("name"),
            item_ids=[i["uuid"] for i in items],
        )

    resolved_texts = texts or []
    resolved_audio = (audio_paths or []) if expected_type == "stt" else None

    if expected_type == "stt":
        if not resolved_audio:
            raise HTTPException(
                status_code=400,
                detail="Provide either dataset_id or audio_paths + texts",
            )
        if len(resolved_audio) != len(resolved_texts):
            raise HTTPException(
                status_code=400,
                detail="Number of audio paths must match number of ground truth texts",
            )
    else:
        if not resolved_texts:
            raise HTTPException(
                status_code=400,
                detail="Provide either dataset_id or texts",
            )

    resolved_dataset_id: Optional[str] = None
    resolved_item_ids: Optional[List[str]] = None
    if dataset_name:
        resolved_dataset_id = create_dataset(
            name=dataset_name,
            dataset_type=expected_type,
            org_uuid=org_uuid,
            user_id=user_id,
        )
        item_dicts: List[Dict] = []
        if expected_type == "stt" and resolved_audio:
            item_dicts = [
                {"audio_path": ap, "text": t}
                for ap, t in zip(resolved_audio, resolved_texts)
            ]
        else:
            item_dicts = [{"text": t} for t in resolved_texts]
        resolved_item_ids = add_dataset_items(resolved_dataset_id, item_dicts)

    return ResolvedDatasetInputs(
        texts=resolved_texts,
        audio_paths=resolved_audio,
        dataset_id=resolved_dataset_id,
        dataset_name=dataset_name,
        item_ids=resolved_item_ids,
    )


def resolve_eval_rerun_inputs_from_job_details(
    details: dict,
    *,
    org_uuid: str,
    expected_type: str,
) -> ResolvedDatasetInputs:
    """Resolve inputs for an in-place STT/TTS eval retry.

    Dataset-backed jobs re-read live dataset items so post-failure dataset fixes
    are picked up. Inline-only jobs reuse the stored snapshot.
    """
    dataset_id = details.get("dataset_id")
    if dataset_id:
        return resolve_dataset_inputs(
            dataset_id=dataset_id,
            org_uuid=org_uuid,
            expected_type=expected_type,
        )
    return ResolvedDatasetInputs(
        texts=details.get("texts") or [],
        audio_paths=details.get("audio_paths") if expected_type == "stt" else None,
        dataset_id=None,
        dataset_name=details.get("dataset_name"),
        item_ids=details.get("dataset_item_ids"),
    )
