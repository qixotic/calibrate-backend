"""Tests for `src/dataset_utils.py`."""

from __future__ import annotations

import uuid

import pytest
from fastapi import HTTPException

import db
from dataset_utils import resolve_dataset_inputs, resolve_eval_rerun_inputs_from_job_details


@pytest.fixture
def user():
    """Return `(user_uuid, org_uuid)` for a freshly-created user.

    Post-multi-tenant migration the dataset helpers scope by org rather than
    user, but most tests don't care about the user UUID; expose both as a
    tuple so callers can pick what they need.
    """
    email = f"ds-{uuid.uuid4().hex[:8]}@example.com"
    user_uuid = db.create_user("D", "U", email)
    org = db.get_personal_org_for_user(user_uuid)
    return {"user_id": user_uuid, "org_uuid": org["uuid"]}


def test_resolve_dataset_inputs_missing_dataset(user):
    with pytest.raises(HTTPException) as ex:
        resolve_dataset_inputs(
            dataset_id="missing", org_uuid=user["org_uuid"], expected_type="stt"
        )
    assert ex.value.status_code == 404


def test_resolve_dataset_inputs_type_mismatch(user):
    ds_uuid = db.create_dataset(
        name=f"ds-{uuid.uuid4().hex[:6]}",
        dataset_type="tts",
        org_uuid=user["org_uuid"],
        user_id=user["user_id"],
    )
    db.add_dataset_items(ds_uuid, [{"text": "hi"}])
    with pytest.raises(HTTPException) as ex:
        resolve_dataset_inputs(
            dataset_id=ds_uuid, org_uuid=user["org_uuid"], expected_type="stt"
        )
    assert ex.value.status_code == 400


def test_resolve_dataset_inputs_empty(user):
    ds_uuid = db.create_dataset(
        name=f"ds-{uuid.uuid4().hex[:6]}",
        dataset_type="stt",
        org_uuid=user["org_uuid"],
        user_id=user["user_id"],
    )
    with pytest.raises(HTTPException) as ex:
        resolve_dataset_inputs(
            dataset_id=ds_uuid, org_uuid=user["org_uuid"], expected_type="stt"
        )
    assert ex.value.status_code == 400


def test_resolve_dataset_inputs_stt_dataset(user):
    ds_uuid = db.create_dataset(
        name=f"ds-{uuid.uuid4().hex[:6]}",
        dataset_type="stt",
        org_uuid=user["org_uuid"],
        user_id=user["user_id"],
    )
    db.add_dataset_items(
        ds_uuid,
        [{"text": "hi", "audio_path": "s3://b/k1"}, {"text": "bye", "audio_path": "s3://b/k2"}],
    )
    resolved = resolve_dataset_inputs(
        dataset_id=ds_uuid, org_uuid=user["org_uuid"], expected_type="stt"
    )
    assert resolved.texts == ["hi", "bye"]
    assert resolved.audio_paths == ["s3://b/k1", "s3://b/k2"]
    assert resolved.dataset_id == ds_uuid
    assert resolved.item_ids and len(resolved.item_ids) == 2


def test_resolve_dataset_inputs_tts_dataset(user):
    ds_uuid = db.create_dataset(
        name=f"ds-{uuid.uuid4().hex[:6]}",
        dataset_type="tts",
        org_uuid=user["org_uuid"],
        user_id=user["user_id"],
    )
    db.add_dataset_items(ds_uuid, [{"text": "hi"}])
    resolved = resolve_dataset_inputs(
        dataset_id=ds_uuid, org_uuid=user["org_uuid"], expected_type="tts"
    )
    assert resolved.audio_paths is None
    assert resolved.texts == ["hi"]


def test_resolve_dataset_inputs_inline_stt_creates_new(user):
    resolved = resolve_dataset_inputs(
        dataset_id=None,
        org_uuid=user["org_uuid"],
        user_id=user["user_id"],
        expected_type="stt",
        texts=["a", "b"],
        audio_paths=["s3://b/1", "s3://b/2"],
        dataset_name="brand-new-stt",
    )
    assert resolved.dataset_id is not None
    assert resolved.item_ids and len(resolved.item_ids) == 2
    assert db.get_dataset(resolved.dataset_id, user["org_uuid"])["type"] == "stt"


def test_resolve_dataset_inputs_inline_tts_creates_new(user):
    resolved = resolve_dataset_inputs(
        dataset_id=None,
        org_uuid=user["org_uuid"],
        user_id=user["user_id"],
        expected_type="tts",
        texts=["a"],
        dataset_name="brand-new-tts",
    )
    assert resolved.dataset_id is not None


def test_resolve_dataset_inputs_inline_stt_validation():
    # No audio paths
    with pytest.raises(HTTPException):
        resolve_dataset_inputs(
            dataset_id=None,
            org_uuid="org",
            expected_type="stt",
            texts=["hi"],
            audio_paths=None,
        )
    # Length mismatch
    with pytest.raises(HTTPException):
        resolve_dataset_inputs(
            dataset_id=None,
            org_uuid="org",
            expected_type="stt",
            texts=["hi"],
            audio_paths=["a", "b"],
        )


def test_resolve_dataset_inputs_inline_tts_requires_texts():
    with pytest.raises(HTTPException):
        resolve_dataset_inputs(
            dataset_id=None,
            org_uuid="org",
            expected_type="tts",
            texts=None,
        )


def test_resolve_eval_rerun_inputs_rereads_dataset(user):
    ds_uuid = db.create_dataset(
        name=f"ds-{uuid.uuid4().hex[:6]}",
        dataset_type="stt",
        org_uuid=user["org_uuid"],
        user_id=user["user_id"],
    )
    item_ids = db.add_dataset_items(
        ds_uuid,
        [{"text": "hi", "audio_path": "s3://b/stale.wav"}],
    )
    db.update_dataset_item(item_ids[0], ds_uuid, audio_path="s3://b/fixed.wav")

    stale_details = {
        "dataset_id": ds_uuid,
        "audio_paths": ["s3://b/stale.wav"],
        "texts": ["hi"],
        "dataset_item_ids": item_ids,
    }
    resolved = resolve_eval_rerun_inputs_from_job_details(
        stale_details,
        org_uuid=user["org_uuid"],
        expected_type="stt",
    )
    assert resolved.audio_paths == ["s3://b/fixed.wav"]
    assert resolved.texts == ["hi"]
    assert resolved.item_ids == item_ids


def test_resolve_eval_rerun_inputs_keeps_inline_snapshot(user):
    inline_details = {
        "audio_paths": ["s3://b/inline.wav"],
        "texts": ["hello"],
        "dataset_item_ids": None,
    }
    resolved = resolve_eval_rerun_inputs_from_job_details(
        inline_details,
        org_uuid=user["org_uuid"],
        expected_type="stt",
    )
    assert resolved.audio_paths == ["s3://b/inline.wav"]
    assert resolved.texts == ["hello"]
    assert resolved.dataset_id is None


def test_resolve_eval_rerun_inputs_tts_dataset(user):
    ds_uuid = db.create_dataset(
        name=f"ds-{uuid.uuid4().hex[:6]}",
        dataset_type="tts",
        org_uuid=user["org_uuid"],
        user_id=user["user_id"],
    )
    item_ids = db.add_dataset_items(ds_uuid, [{"text": "old"}])
    db.update_dataset_item(item_ids[0], ds_uuid, text="new")

    resolved = resolve_eval_rerun_inputs_from_job_details(
        {"dataset_id": ds_uuid, "texts": ["old"], "dataset_item_ids": item_ids},
        org_uuid=user["org_uuid"],
        expected_type="tts",
    )
    assert resolved.texts == ["new"]
