"""Integration tests for /annotation-tasks and /annotation-agreement routers."""

from __future__ import annotations

import uuid
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient


@pytest.fixture(scope="module")
def app():
    import main as main_mod

    return main_mod.app


@pytest.fixture(scope="module")
def client(app):
    with patch("main.recover_pending_jobs"):
        with TestClient(app) as c:
            yield c


def _signup(client):
    suffix = uuid.uuid4().hex[:8]
    body = client.post(
        "/auth/signup",
        json={
            "first_name": "A",
            "last_name": "U",
            "email": f"ann-{suffix}@example.com",
            "password": "passw0rd",
        },
    ).json()
    return {
        "headers": {"Authorization": f"Bearer {body['access_token']}"},
        "user_uuid": body["user"]["uuid"],
    }


def _llm_evaluator(client, h):
    evaluators = client.get("/evaluators", headers=h).json()
    return next(e for e in evaluators if e.get("evaluator_type") == "llm")


# ---------------------------------------------------------------------------
# CRUD
# ---------------------------------------------------------------------------


def test_annotation_task_crud(client):
    auth = _signup(client)
    h = auth["headers"]
    llm_ev = _llm_evaluator(client, h)

    # invalid type
    bad_type = client.post(
        "/annotation-tasks",
        json={"name": "x", "type": "bogus"},
        headers=h,
    )
    assert bad_type.status_code == 400

    # invalid evaluator
    bad_ev = client.post(
        "/annotation-tasks",
        json={"name": "x", "type": "llm", "evaluator_ids": ["missing"]},
        headers=h,
    )
    assert bad_ev.status_code == 404

    # create
    name = f"task-{uuid.uuid4().hex[:6]}"
    create = client.post(
        "/annotation-tasks",
        json={
            "name": name,
            "type": "llm",
            "description": "d",
            "evaluator_ids": [llm_ev["uuid"]],
        },
        headers=h,
    )
    assert create.status_code == 200
    task_uuid = create.json()["uuid"]

    # list
    listing = client.get("/annotation-tasks", headers=h)
    assert listing.status_code == 200
    assert any(t["uuid"] == task_uuid for t in listing.json())

    # detail
    detail = client.get(f"/annotation-tasks/{task_uuid}", headers=h)
    assert detail.status_code == 200
    assert detail.json()["item_count"] == 0

    # update
    upd = client.put(
        f"/annotation-tasks/{task_uuid}",
        json={"name": f"{name}-new", "description": "nd"},
        headers=h,
    )
    assert upd.status_code == 200

    # update with no fields
    no_op = client.put(f"/annotation-tasks/{task_uuid}", json={}, headers=h)
    assert no_op.status_code == 400

    # missing task
    assert (
        client.get("/annotation-tasks/missing", headers=h).status_code == 404
    )
    assert (
        client.put(
            "/annotation-tasks/missing", json={"name": "x"}, headers=h
        ).status_code
        == 404
    )

    # other user denied (404)
    other = _signup(client)
    assert (
        client.get(
            f"/annotation-tasks/{task_uuid}", headers=other["headers"]
        ).status_code
        == 404
    )

    # list task evaluators
    list_ev = client.get(f"/annotation-tasks/{task_uuid}/evaluators", headers=h)
    assert list_ev.status_code == 200

    # unlink evaluator
    unlink = client.delete(
        f"/annotation-tasks/{task_uuid}/evaluators/{llm_ev['uuid']}", headers=h
    )
    assert unlink.status_code == 200
    # again → 404
    assert (
        client.delete(
            f"/annotation-tasks/{task_uuid}/evaluators/{llm_ev['uuid']}",
            headers=h,
        ).status_code
        == 404
    )

    # Now link via the dedicated endpoint (after unlink)
    relink = client.post(
        f"/annotation-tasks/{task_uuid}/evaluators",
        json={"evaluator_id": llm_ev["uuid"]},
        headers=h,
    )
    assert relink.status_code == 200

    # delete task
    deleted = client.delete(f"/annotation-tasks/{task_uuid}", headers=h)
    assert deleted.status_code == 200
    assert (
        client.delete(f"/annotation-tasks/{task_uuid}", headers=h).status_code == 404
    )


def test_annotation_items_crud(client):
    auth = _signup(client)
    h = auth["headers"]
    llm_ev = _llm_evaluator(client, h)
    create = client.post(
        "/annotation-tasks",
        json={
            "name": f"t-{uuid.uuid4().hex[:6]}",
            "type": "llm",
            "evaluator_ids": [llm_ev["uuid"]],
        },
        headers=h,
    )
    task_uuid = create.json()["uuid"]

    # empty list
    empty = client.post(
        f"/annotation-tasks/{task_uuid}/items", json={"items": []}, headers=h
    )
    assert empty.status_code == 400

    # missing payload.name
    bad = client.post(
        f"/annotation-tasks/{task_uuid}/items",
        json={"items": [{"payload": {}}]},
        headers=h,
    )
    assert bad.status_code == 400

    # duplicate names in request
    dup = client.post(
        f"/annotation-tasks/{task_uuid}/items",
        json={
            "items": [
                {"payload": {"name": "x"}},
                {"payload": {"name": "x"}},
            ]
        },
        headers=h,
    )
    assert dup.status_code == 409

    # create items
    created = client.post(
        f"/annotation-tasks/{task_uuid}/items",
        json={"items": [{"payload": {"name": "i1"}}, {"payload": {"name": "i2"}}]},
        headers=h,
    )
    assert created.status_code == 200
    item_ids = created.json()["item_ids"]

    # conflict on duplicate name
    conflict = client.post(
        f"/annotation-tasks/{task_uuid}/items",
        json={"items": [{"payload": {"name": "i1"}}]},
        headers=h,
    )
    assert conflict.status_code == 409

    # GET items
    listed = client.get(f"/annotation-tasks/{task_uuid}/items", headers=h)
    assert listed.status_code == 200
    assert len(listed.json()) >= 2

    # GET item detail
    detail = client.get(
        f"/annotation-tasks/{task_uuid}/items/{item_ids[0]}", headers=h
    )
    assert detail.status_code == 200
    missing = client.get(
        f"/annotation-tasks/{task_uuid}/items/missing", headers=h
    )
    assert missing.status_code == 404

    # Item annotations list
    anns = client.get(
        f"/annotation-tasks/{task_uuid}/items/{item_ids[0]}/annotations",
        headers=h,
    )
    assert anns.status_code == 200
    # Missing item
    missing_anns = client.get(
        f"/annotation-tasks/{task_uuid}/items/missing/annotations",
        headers=h,
    )
    assert missing_anns.status_code == 404

    # Item evaluator-runs list
    eval_runs = client.get(
        f"/annotation-tasks/{task_uuid}/items/{item_ids[0]}/evaluator-runs",
        headers=h,
    )
    assert eval_runs.status_code == 200

    # Bulk-update — empty
    bu_empty = client.put(
        f"/annotation-tasks/{task_uuid}/items",
        json={"updates": []},
        headers=h,
    )
    assert bu_empty.status_code == 400

    # Bulk-update — duplicate names
    bu_dup = client.put(
        f"/annotation-tasks/{task_uuid}/items",
        json={
            "updates": [
                {"uuid": item_ids[0], "payload": {"name": "dup"}},
                {"uuid": item_ids[1], "payload": {"name": "dup"}},
            ]
        },
        headers=h,
    )
    assert bu_dup.status_code == 409

    # Bulk-update
    bu = client.put(
        f"/annotation-tasks/{task_uuid}/items",
        json={"updates": [{"uuid": item_ids[0], "payload": {"name": "renamed"}}]},
        headers=h,
    )
    assert bu.status_code == 200

    # Bulk-delete — empty
    bd_empty = client.request(
        "DELETE",
        f"/annotation-tasks/{task_uuid}/items",
        json={"item_ids": []},
        headers=h,
    )
    assert bd_empty.status_code == 400

    # Bulk-delete
    bd = client.request(
        "DELETE",
        f"/annotation-tasks/{task_uuid}/items",
        json={"item_ids": [item_ids[0]]},
        headers=h,
    )
    assert bd.status_code == 200


def test_annotation_jobs_crud(client):
    auth = _signup(client)
    h = auth["headers"]
    llm_ev = _llm_evaluator(client, h)
    task_uuid = client.post(
        "/annotation-tasks",
        json={
            "name": f"t-{uuid.uuid4().hex[:6]}",
            "type": "llm",
            "evaluator_ids": [llm_ev["uuid"]],
        },
        headers=h,
    ).json()["uuid"]

    # Need items
    items = client.post(
        f"/annotation-tasks/{task_uuid}/items",
        json={"items": [{"payload": {"name": "i1"}}, {"payload": {"name": "i2"}}]},
        headers=h,
    ).json()["item_ids"]

    # Need an annotator
    annotator = client.post(
        "/annotators", json={"name": f"ann-{uuid.uuid4().hex[:6]}"}, headers=h
    ).json()

    # Empty annotator_ids
    empty_a = client.post(
        f"/annotation-tasks/{task_uuid}/jobs",
        json={"annotator_ids": [], "item_ids": items},
        headers=h,
    )
    assert empty_a.status_code == 400

    # Empty item_ids
    empty_i = client.post(
        f"/annotation-tasks/{task_uuid}/jobs",
        json={"annotator_ids": [annotator["uuid"]], "item_ids": []},
        headers=h,
    )
    assert empty_i.status_code == 400

    # Duplicate item_ids
    dup_i = client.post(
        f"/annotation-tasks/{task_uuid}/jobs",
        json={
            "annotator_ids": [annotator["uuid"]],
            "item_ids": [items[0], items[0]],
        },
        headers=h,
    )
    assert dup_i.status_code == 400

    # Bad annotator
    bad_a = client.post(
        f"/annotation-tasks/{task_uuid}/jobs",
        json={"annotator_ids": ["missing"], "item_ids": items},
        headers=h,
    )
    assert bad_a.status_code == 404

    # Bad item
    bad_i = client.post(
        f"/annotation-tasks/{task_uuid}/jobs",
        json={"annotator_ids": [annotator["uuid"]], "item_ids": ["missing"]},
        headers=h,
    )
    assert bad_i.status_code == 400

    # Create
    create = client.post(
        f"/annotation-tasks/{task_uuid}/jobs",
        json={"annotator_ids": [annotator["uuid"]], "item_ids": items},
        headers=h,
    )
    assert create.status_code == 200
    job_uuid = create.json()["jobs"][0]["uuid"]

    # List jobs
    listing = client.get(f"/annotation-tasks/{task_uuid}/jobs", headers=h)
    assert listing.status_code == 200

    # Detail
    detail = client.get(
        f"/annotation-tasks/{task_uuid}/jobs/{job_uuid}", headers=h
    )
    assert detail.status_code == 200
    missing = client.get(
        f"/annotation-tasks/{task_uuid}/jobs/missing", headers=h
    )
    assert missing.status_code == 404

    # Cannot share an incomplete job
    bad_share = client.patch(
        f"/annotation-tasks/{task_uuid}/jobs/{job_uuid}/visibility",
        json={"is_public": True},
        headers=h,
    )
    assert bad_share.status_code == 400

    # Visibility off → 200 (always permitted)
    off = client.patch(
        f"/annotation-tasks/{task_uuid}/jobs/{job_uuid}/visibility",
        json={"is_public": False},
        headers=h,
    )
    assert off.status_code == 200

    # Annotation upsert against missing job
    bad_upsert = client.post(
        f"/annotation-tasks/{task_uuid}/annotations",
        json={"job_id": "missing", "item_id": items[0], "value": {"value": True}},
        headers=h,
    )
    assert bad_upsert.status_code == 404

    # Annotation upsert with bad item
    bad_item = client.post(
        f"/annotation-tasks/{task_uuid}/annotations",
        json={"job_id": job_uuid, "item_id": "missing", "value": {"value": True}},
        headers=h,
    )
    assert bad_item.status_code == 404

    # Annotation upsert with bad evaluator id
    bad_ev = client.post(
        f"/annotation-tasks/{task_uuid}/annotations",
        json={
            "job_id": job_uuid,
            "item_id": items[0],
            "evaluator_id": "missing",
            "value": {"value": True},
        },
        headers=h,
    )
    assert bad_ev.status_code == 400

    # Successful upsert
    upsert = client.post(
        f"/annotation-tasks/{task_uuid}/annotations",
        json={
            "job_id": job_uuid,
            "item_id": items[0],
            "evaluator_id": llm_ev["uuid"],
            "value": {"value": True},
        },
        headers=h,
    )
    assert upsert.status_code == 200


def test_delete_annotation_job(client):
    """Soft-delete one annotator's labelling job. Verify:
    - 404 when the job doesn't belong to the path's task
    - 404 when not owned by the caller
    - happy path: job vanishes from list + detail, its annotations stop
      contributing to inter-annotator agreement, and double-delete is 404."""
    auth = _signup(client)
    h = auth["headers"]
    llm_ev = _llm_evaluator(client, h)
    task_uuid = client.post(
        "/annotation-tasks",
        json={
            "name": f"t-{uuid.uuid4().hex[:6]}",
            "type": "llm",
            "evaluator_ids": [llm_ev["uuid"]],
        },
        headers=h,
    ).json()["uuid"]
    items = client.post(
        f"/annotation-tasks/{task_uuid}/items",
        json={"items": [{"payload": {"name": "i1"}}]},
        headers=h,
    ).json()["item_ids"]

    # Two annotators so we can observe agreement collapsing after delete.
    ann_a = client.post(
        "/annotators", json={"name": f"a-{uuid.uuid4().hex[:6]}"}, headers=h
    ).json()
    ann_b = client.post(
        "/annotators", json={"name": f"b-{uuid.uuid4().hex[:6]}"}, headers=h
    ).json()
    jobs = client.post(
        f"/annotation-tasks/{task_uuid}/jobs",
        json={
            "annotator_ids": [ann_a["uuid"], ann_b["uuid"]],
            "item_ids": items,
        },
        headers=h,
    ).json()["jobs"]
    job_a = next(j for j in jobs if j["annotator_id"] == ann_a["uuid"])
    job_b = next(j for j in jobs if j["annotator_id"] == ann_b["uuid"])

    # Both annotators agree on the same slot → pairwise agreement should be 1.0.
    for job in (job_a, job_b):
        client.post(
            f"/public/annotation-jobs/{job['public_token']}/annotations",
            json={
                "item_id": items[0],
                "annotations": [
                    {"evaluator_id": llm_ev["uuid"], "value": {"value": True}},
                ],
            },
        )

    # Sanity: pre-delete pairwise agreement is 1.0 (both annotators agreed).
    import db as db_mod
    from annotation_metrics import aggregate_agreement

    pre = db_mod.get_annotations_for_task(task_uuid)
    pre_pairwise, _ = aggregate_agreement(pre)
    assert pre_pairwise == 1.0

    # 404 when job UUID doesn't match path task
    other_task = client.post(
        "/annotation-tasks",
        json={
            "name": f"t-{uuid.uuid4().hex[:6]}",
            "type": "llm",
            "evaluator_ids": [llm_ev["uuid"]],
        },
        headers=h,
    ).json()["uuid"]
    mismatch = client.delete(
        f"/annotation-tasks/{other_task}/jobs/{job_a['uuid']}", headers=h
    )
    assert mismatch.status_code == 404

    # 404 when caller doesn't own the task
    other_auth = _signup(client)
    other = client.delete(
        f"/annotation-tasks/{task_uuid}/jobs/{job_a['uuid']}",
        headers=other_auth["headers"],
    )
    assert other.status_code == 404

    # Happy path
    ok = client.delete(
        f"/annotation-tasks/{task_uuid}/jobs/{job_a['uuid']}", headers=h
    )
    assert ok.status_code == 200

    # Job vanishes from list + detail
    listing = client.get(
        f"/annotation-tasks/{task_uuid}/jobs", headers=h
    ).json()
    assert all(j["uuid"] != job_a["uuid"] for j in listing)
    gone = client.get(
        f"/annotation-tasks/{task_uuid}/jobs/{job_a['uuid']}", headers=h
    )
    assert gone.status_code == 404

    # Annotator B's job survives
    survivor = client.get(
        f"/annotation-tasks/{task_uuid}/jobs/{job_b['uuid']}", headers=h
    )
    assert survivor.status_code == 200

    # Pairwise agreement: ann_a's contribution is gone, so the slot now has
    # only one annotator and contributes nothing — `aggregate_agreement`
    # returns None when there are no overlapping pairs.
    annotations = db_mod.get_annotations_for_task(task_uuid)
    pairwise, _ = aggregate_agreement(annotations)
    assert pairwise is None

    # Double-delete → 404
    again = client.delete(
        f"/annotation-tasks/{task_uuid}/jobs/{job_a['uuid']}", headers=h
    )
    assert again.status_code == 404


def test_bulk_delete_annotation_jobs(client):
    """Bulk soft-delete labelling jobs. Verify silent-skip for foreign /
    already-deleted / unknown UUIDs, non-empty enforcement, and that the
    cascade through `j.deleted_at IS NULL` removes the deleted jobs'
    annotations from agreement output."""
    auth = _signup(client)
    h = auth["headers"]
    llm_ev = _llm_evaluator(client, h)
    task_uuid = client.post(
        "/annotation-tasks",
        json={
            "name": f"t-{uuid.uuid4().hex[:6]}",
            "type": "llm",
            "evaluator_ids": [llm_ev["uuid"]],
        },
        headers=h,
    ).json()["uuid"]
    items = client.post(
        f"/annotation-tasks/{task_uuid}/items",
        json={"items": [{"payload": {"name": "i1"}}]},
        headers=h,
    ).json()["item_ids"]

    # Three annotators on the same slot, all agreeing → pre-delete agreement is 1.0
    ann_uuids = []
    for _ in range(3):
        a = client.post(
            "/annotators", json={"name": f"a-{uuid.uuid4().hex[:6]}"}, headers=h
        ).json()
        ann_uuids.append(a["uuid"])
    created = client.post(
        f"/annotation-tasks/{task_uuid}/jobs",
        json={"annotator_ids": ann_uuids, "item_ids": items},
        headers=h,
    ).json()["jobs"]
    for job in created:
        client.post(
            f"/public/annotation-jobs/{job['public_token']}/annotations",
            json={
                "item_id": items[0],
                "annotations": [
                    {"evaluator_id": llm_ev["uuid"], "value": {"value": True}},
                ],
            },
        )

    import db as db_mod
    from annotation_metrics import aggregate_agreement

    pre_pairwise, _ = aggregate_agreement(db_mod.get_annotations_for_task(task_uuid))
    assert pre_pairwise == 1.0

    # Empty payload → 400
    empty = client.request(
        "DELETE",
        f"/annotation-tasks/{task_uuid}/jobs",
        json={"job_uuids": []},
        headers=h,
    )
    assert empty.status_code == 400

    # Foreign task: ensure jobs in a different task are silently skipped
    other_task = client.post(
        "/annotation-tasks",
        json={
            "name": f"t-{uuid.uuid4().hex[:6]}",
            "type": "llm",
            "evaluator_ids": [llm_ev["uuid"]],
        },
        headers=h,
    ).json()["uuid"]
    other_items = client.post(
        f"/annotation-tasks/{other_task}/items",
        json={"items": [{"payload": {"name": "x"}}]},
        headers=h,
    ).json()["item_ids"]
    other_ann = client.post(
        "/annotators", json={"name": f"x-{uuid.uuid4().hex[:6]}"}, headers=h
    ).json()
    other_job = client.post(
        f"/annotation-tasks/{other_task}/jobs",
        json={"annotator_ids": [other_ann["uuid"]], "item_ids": other_items},
        headers=h,
    ).json()["jobs"][0]

    # Delete two of the three jobs in `task_uuid`, mixed in with an unknown
    # UUID and a foreign-task UUID. Expect deleted_count=2.
    targets = [created[0]["uuid"], created[1]["uuid"], "missing", other_job["uuid"]]
    bulk = client.request(
        "DELETE",
        f"/annotation-tasks/{task_uuid}/jobs",
        json={"job_uuids": targets},
        headers=h,
    )
    assert bulk.status_code == 200
    assert bulk.json()["deleted_count"] == 2

    # Foreign job is untouched
    assert (
        client.get(
            f"/annotation-tasks/{other_task}/jobs/{other_job['uuid']}", headers=h
        ).status_code
        == 200
    )

    # Only one annotator left → no pairwise overlap → None.
    post_pairwise, _ = aggregate_agreement(
        db_mod.get_annotations_for_task(task_uuid)
    )
    assert post_pairwise is None

    # Re-running with the same payload now transitions zero rows.
    again = client.request(
        "DELETE",
        f"/annotation-tasks/{task_uuid}/jobs",
        json={"job_uuids": targets},
        headers=h,
    )
    assert again.status_code == 200
    assert again.json()["deleted_count"] == 0

    # Non-owner gets 404 on the task scope (not 403).
    other_auth = _signup(client)
    forbidden = client.request(
        "DELETE",
        f"/annotation-tasks/{task_uuid}/jobs",
        json={"job_uuids": [created[2]["uuid"]]},
        headers=other_auth["headers"],
    )
    assert forbidden.status_code == 404


def test_annotated_check(client):
    auth = _signup(client)
    h = auth["headers"]
    llm_ev = _llm_evaluator(client, h)
    task_uuid = client.post(
        "/annotation-tasks",
        json={
            "name": f"t-{uuid.uuid4().hex[:6]}",
            "type": "llm",
            "evaluator_ids": [llm_ev["uuid"]],
        },
        headers=h,
    ).json()["uuid"]

    annotator = client.post(
        "/annotators", json={"name": f"ann-{uuid.uuid4().hex[:6]}"}, headers=h
    ).json()

    # Empty names → 400
    empty = client.post(
        f"/annotation-tasks/{task_uuid}/items/annotated-check",
        json={"annotator_id": annotator["uuid"], "names": []},
        headers=h,
    )
    assert empty.status_code == 400

    # Missing annotator
    bad = client.post(
        f"/annotation-tasks/{task_uuid}/items/annotated-check",
        json={"annotator_id": "missing", "names": ["x"]},
        headers=h,
    )
    assert bad.status_code == 404

    # Successful (all new)
    ok = client.post(
        f"/annotation-tasks/{task_uuid}/items/annotated-check",
        json={"annotator_id": annotator["uuid"], "names": ["never-seen"]},
        headers=h,
    )
    assert ok.status_code == 200
    assert ok.json()["all_new"] is True


def test_annotation_agreement_endpoints(client):
    auth = _signup(client)
    h = auth["headers"]

    # Account-wide trend (no tasks)
    trend = client.get("/annotation-agreement/trend", headers=h)
    assert trend.status_code == 200

    # With task_id filter (missing task)
    missing_task = client.get(
        "/annotation-agreement/trend", params={"task_id": "missing"}, headers=h
    )
    assert missing_task.status_code == 404

    # Per-evaluator trend
    llm_ev = _llm_evaluator(client, h)
    ev_trend = client.get(
        f"/annotation-agreement/evaluator/{llm_ev['uuid']}/trend", headers=h
    )
    assert ev_trend.status_code == 200

    # Per-evaluator trend with missing task
    ev_with_task = client.get(
        f"/annotation-agreement/evaluator/{llm_ev['uuid']}/trend",
        params={"task_id": "missing"},
        headers=h,
    )
    assert ev_with_task.status_code == 404

    # Per-evaluator trend with missing evaluator
    missing_ev = client.get(
        "/annotation-agreement/evaluator/missing/trend", headers=h
    )
    assert missing_ev.status_code == 404


def test_annotation_task_agreement_and_summary(client):
    auth = _signup(client)
    h = auth["headers"]
    llm_ev = _llm_evaluator(client, h)
    task_uuid = client.post(
        "/annotation-tasks",
        json={
            "name": f"t-{uuid.uuid4().hex[:6]}",
            "type": "llm",
            "evaluator_ids": [llm_ev["uuid"]],
        },
        headers=h,
    ).json()["uuid"]

    # Empty agreement
    agree = client.get(f"/annotation-tasks/{task_uuid}/agreement", headers=h)
    assert agree.status_code == 200
    # missing task
    assert (
        client.get("/annotation-tasks/missing/agreement", headers=h).status_code
        == 404
    )

    # Summary (no items)
    summary = client.get(f"/annotation-tasks/{task_uuid}/summary", headers=h)
    assert summary.status_code == 200

    # Add an item then filter by it
    item_uuid = client.post(
        f"/annotation-tasks/{task_uuid}/items",
        json={"items": [{"payload": {"name": "i1"}}]},
        headers=h,
    ).json()["item_ids"][0]
    summary_filtered = client.get(
        f"/annotation-tasks/{task_uuid}/summary",
        params={"item_id": item_uuid, "live_only": True},
        headers=h,
    )
    assert summary_filtered.status_code == 200
    # Missing item id → 404
    missing_summary = client.get(
        f"/annotation-tasks/{task_uuid}/summary",
        params={"item_id": "missing"},
        headers=h,
    )
    assert missing_summary.status_code == 404


def test_evaluator_runs_endpoints(client, monkeypatch):
    auth = _signup(client)
    h = auth["headers"]
    llm_ev = _llm_evaluator(client, h)
    task_uuid = client.post(
        "/annotation-tasks",
        json={
            "name": f"t-{uuid.uuid4().hex[:6]}",
            "type": "llm",
            "evaluator_ids": [llm_ev["uuid"]],
        },
        headers=h,
    ).json()["uuid"]

    # No items → 400
    no_items = client.post(
        f"/annotation-tasks/{task_uuid}/evaluator-runs",
        json={"evaluators": [{"evaluator_id": llm_ev["uuid"]}]},
        headers=h,
    )
    assert no_items.status_code == 400

    # Add items
    client.post(
        f"/annotation-tasks/{task_uuid}/items",
        json={"items": [{"payload": {"name": "i1"}}]},
        headers=h,
    )

    # Empty evaluators
    empty = client.post(
        f"/annotation-tasks/{task_uuid}/evaluator-runs",
        json={"evaluators": []},
        headers=h,
    )
    assert empty.status_code == 400

    # Empty item_ids (but provided)
    empty_items = client.post(
        f"/annotation-tasks/{task_uuid}/evaluator-runs",
        json={
            "evaluators": [{"evaluator_id": llm_ev["uuid"]}],
            "item_ids": [],
        },
        headers=h,
    )
    assert empty_items.status_code == 400

    # Invalid item id
    bad_items = client.post(
        f"/annotation-tasks/{task_uuid}/evaluator-runs",
        json={
            "evaluators": [{"evaluator_id": llm_ev["uuid"]}],
            "item_ids": ["missing"],
        },
        headers=h,
    )
    assert bad_items.status_code == 400

    # List evaluator runs (empty)
    listing = client.get(
        f"/annotation-tasks/{task_uuid}/evaluator-runs", headers=h
    )
    assert listing.status_code == 200
    # Missing task
    assert (
        client.get(
            "/annotation-tasks/missing/evaluator-runs", headers=h
        ).status_code
        == 404
    )

    # Get unknown evaluator-run job
    missing = client.get(
        f"/annotation-tasks/{task_uuid}/evaluator-runs/missing", headers=h
    )
    assert missing.status_code == 404

    # Delete unknown evaluator-run job
    assert (
        client.delete(
            f"/annotation-tasks/{task_uuid}/evaluator-runs/missing", headers=h
        ).status_code
        == 404
    )

    # Visibility on missing
    assert (
        client.patch(
            f"/annotation-tasks/{task_uuid}/evaluator-runs/missing/visibility",
            json={"is_public": True},
            headers=h,
        ).status_code
        == 404
    )


def test_annotation_eval_unsupported_task_type(client):
    auth = _signup(client)
    h = auth["headers"]
    # Try with a tts annotation task type if supported, else use a known unsupported
    # Use 'tts' which is in ANNOTATION_TASK_TYPES but not SUPPORTED_EVAL_TASK_TYPES
    # First, what types exist?
    import db as db_mod
    task_types = set(db_mod.ANNOTATION_TASK_TYPES)
    # SUPPORTED_EVAL_TASK_TYPES is {stt, llm, simulation}; tts is excluded
    if "tts" in task_types:
        llm_ev = _llm_evaluator(client, h)
        task_uuid = client.post(
            "/annotation-tasks",
            json={
                "name": f"t-{uuid.uuid4().hex[:6]}",
                "type": "tts",
                "evaluator_ids": [llm_ev["uuid"]],
            },
            headers=h,
        ).json()["uuid"]
        resp = client.post(
            f"/annotation-tasks/{task_uuid}/evaluator-runs",
            json={"evaluators": [{"evaluator_id": llm_ev["uuid"]}]},
            headers=h,
        )
        assert resp.status_code == 400
