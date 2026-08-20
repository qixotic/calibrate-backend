"""Tests for the trace-eval queue claim/settle functions in src/db.py.

Plain functions, driven by tests only -- no worker pool yet (that's PR 6).
"""

from __future__ import annotations

import threading
import time
import uuid

import db


def _org() -> str:
    return str(uuid.uuid4())


def _enqueue(
    org: str,
    *,
    trace_uuid: str | None = None,
    evaluator_uuid: str | None = None,
    evaluator_version_id: int = 1,
    agent_id: str = "agent-1",
    status: str = "pending",
    available_at: int = 0,
    attempts: int = 0,
) -> int:
    trace_uuid = trace_uuid or str(uuid.uuid4())
    evaluator_uuid = evaluator_uuid or str(uuid.uuid4())
    with db.get_db_connection() as conn:
        cur = conn.execute(
            "INSERT INTO trace_eval_queue "
            "(trace_uuid, evaluator_uuid, evaluator_version_id, org_uuid, "
            "agent_id, status, available_at, attempts) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (
                trace_uuid,
                evaluator_uuid,
                evaluator_version_id,
                org,
                agent_id,
                status,
                available_at,
                attempts,
            ),
        )
        conn.commit()
        return cur.lastrowid


def _queue_row(queue_id: int):
    with db.get_db_connection() as conn:
        return conn.execute(
            "SELECT * FROM trace_eval_queue WHERE id = ?", (queue_id,)
        ).fetchone()


def test_claim_returns_ready_rows_and_marks_processing():
    org = _org()
    queue_id = _enqueue(org, available_at=0)

    claimed = db.claim_trace_eval_queue_rows(batch_size=10)

    ids = {row["id"] for row in claimed}
    assert queue_id in ids
    row = next(r for r in claimed if r["id"] == queue_id)
    assert row["attempts"] == 1

    stored = _queue_row(queue_id)
    assert stored["status"] == "processing"
    assert stored["attempts"] == 1
    assert stored["available_at"] > int(time.time())


def test_claim_respects_batch_size():
    org = _org()
    for _ in range(3):
        _enqueue(org, available_at=0)

    claimed = db.claim_trace_eval_queue_rows(batch_size=2)

    assert len(claimed) == 2


def test_claim_skips_rows_not_yet_available():
    org = _org()
    future = int(time.time()) + 3600
    queue_id = _enqueue(org, available_at=future)

    claimed = db.claim_trace_eval_queue_rows(batch_size=10)

    assert queue_id not in {row["id"] for row in claimed}


def test_reclaiming_an_expired_lease_increments_attempts():
    org = _org()
    queue_id = _enqueue(org, available_at=0)

    first = db.claim_trace_eval_queue_rows(batch_size=10, lease_seconds=1)
    first_row = next(r for r in first if r["id"] == queue_id)
    assert first_row["attempts"] == 1

    # Simulate the lease expiring by forcing available_at into the past,
    # rather than sleeping in a test.
    with db.get_db_connection() as conn:
        conn.execute(
            "UPDATE trace_eval_queue SET available_at = 0 WHERE id = ?",
            (queue_id,),
        )
        conn.commit()

    second = db.claim_trace_eval_queue_rows(batch_size=10, lease_seconds=60)
    second_row = next(r for r in second if r["id"] == queue_id)
    assert second_row["attempts"] == 2


def test_two_concurrent_claimers_never_receive_the_same_row():
    org = _org()
    queue_ids = {_enqueue(org, available_at=0) for _ in range(20)}

    results: list[list[dict]] = [[], []]

    def _claim(slot: int):
        results[slot] = db.claim_trace_eval_queue_rows(batch_size=20)

    t1 = threading.Thread(target=_claim, args=(0,))
    t2 = threading.Thread(target=_claim, args=(1,))
    t1.start()
    t2.start()
    t1.join()
    t2.join()

    ids_a = {row["id"] for row in results[0]}
    ids_b = {row["id"] for row in results[1]}
    assert ids_a.isdisjoint(ids_b)
    assert (ids_a | ids_b) == queue_ids


def test_settle_success_upserts_score_and_deletes_queue_row():
    org = _org()
    trace_uuid = str(uuid.uuid4())
    evaluator_uuid = str(uuid.uuid4())
    queue_id = _enqueue(
        org, trace_uuid=trace_uuid, evaluator_uuid=evaluator_uuid, evaluator_version_id=1
    )

    db.settle_trace_eval_success(
        queue_id,
        trace_uuid=trace_uuid,
        evaluator_uuid=evaluator_uuid,
        evaluator_version_id=1,
        org_uuid=org,
        score=1.0,
        reasoning="good",
    )

    assert _queue_row(queue_id) is None
    with db.get_db_connection() as conn:
        scores = conn.execute(
            "SELECT * FROM trace_scores WHERE trace_uuid = ?", (trace_uuid,)
        ).fetchall()
    assert len(scores) == 1
    assert scores[0]["score"] == 1.0
    assert scores[0]["reasoning"] == "good"


def test_settle_success_retry_same_version_overwrites():
    org = _org()
    trace_uuid = str(uuid.uuid4())
    evaluator_uuid = str(uuid.uuid4())

    db.settle_trace_eval_success(
        _enqueue(org, trace_uuid=trace_uuid, evaluator_uuid=evaluator_uuid, evaluator_version_id=1),
        trace_uuid=trace_uuid,
        evaluator_uuid=evaluator_uuid,
        evaluator_version_id=1,
        org_uuid=org,
        score=0.0,
        reasoning="first attempt",
    )
    db.settle_trace_eval_success(
        _enqueue(org, trace_uuid=trace_uuid, evaluator_uuid=evaluator_uuid, evaluator_version_id=1),
        trace_uuid=trace_uuid,
        evaluator_uuid=evaluator_uuid,
        evaluator_version_id=1,
        org_uuid=org,
        score=1.0,
        reasoning="retried",
    )

    with db.get_db_connection() as conn:
        scores = conn.execute(
            "SELECT * FROM trace_scores WHERE trace_uuid = ?", (trace_uuid,)
        ).fetchall()
    assert len(scores) == 1
    assert scores[0]["score"] == 1.0
    assert scores[0]["reasoning"] == "retried"


def test_settle_success_new_version_coexists_with_old():
    org = _org()
    trace_uuid = str(uuid.uuid4())
    evaluator_uuid = str(uuid.uuid4())

    db.settle_trace_eval_success(
        _enqueue(org, trace_uuid=trace_uuid, evaluator_uuid=evaluator_uuid, evaluator_version_id=1),
        trace_uuid=trace_uuid,
        evaluator_uuid=evaluator_uuid,
        evaluator_version_id=1,
        org_uuid=org,
        score=0.0,
        reasoning="v1",
    )
    db.settle_trace_eval_success(
        _enqueue(org, trace_uuid=trace_uuid, evaluator_uuid=evaluator_uuid, evaluator_version_id=2),
        trace_uuid=trace_uuid,
        evaluator_uuid=evaluator_uuid,
        evaluator_version_id=2,
        org_uuid=org,
        score=1.0,
        reasoning="v2",
    )

    with db.get_db_connection() as conn:
        scores = conn.execute(
            "SELECT * FROM trace_scores WHERE trace_uuid = ? ORDER BY evaluator_version_id",
            (trace_uuid,),
        ).fetchall()
    assert [s["evaluator_version_id"] for s in scores] == [1, 2]


def test_settle_failure_defers_with_backoff_and_jitter():
    org = _org()
    queue_id = _enqueue(org, available_at=0)
    before = int(time.time())

    dead_lettered = db.settle_trace_eval_failure(
        queue_id,
        trace_uuid=str(uuid.uuid4()),
        evaluator_uuid=str(uuid.uuid4()),
        evaluator_version_id=1,
        org_uuid=org,
        attempts=1,
        error="judge timed out",
        max_attempts=5,
    )

    assert dead_lettered is False
    row = _queue_row(queue_id)
    assert row["status"] == "pending"
    assert row["available_at"] > before


def test_settle_failure_deferred_rows_land_on_distinct_timestamps():
    org = _org()
    queue_ids = [_enqueue(org, available_at=0) for _ in range(10)]

    for queue_id in queue_ids:
        db.settle_trace_eval_failure(
            queue_id,
            trace_uuid=str(uuid.uuid4()),
            evaluator_uuid=str(uuid.uuid4()),
            evaluator_version_id=1,
            org_uuid=org,
            attempts=1,
            error="judge timed out",
            max_attempts=5,
        )

    available_ats = {_queue_row(q)["available_at"] for q in queue_ids}
    # Jitter spreads a whole failed batch across distinct timestamps instead
    # of reassembling the identical batch on the next claim.
    assert len(available_ats) > 1


def test_settle_failure_dead_letters_past_max_attempts():
    org = _org()
    trace_uuid = str(uuid.uuid4())
    evaluator_uuid = str(uuid.uuid4())
    queue_id = _enqueue(
        org, trace_uuid=trace_uuid, evaluator_uuid=evaluator_uuid, evaluator_version_id=1
    )

    dead_lettered = db.settle_trace_eval_failure(
        queue_id,
        trace_uuid=trace_uuid,
        evaluator_uuid=evaluator_uuid,
        evaluator_version_id=1,
        org_uuid=org,
        attempts=5,
        error="judge kept failing",
        max_attempts=5,
    )

    assert dead_lettered is True
    assert _queue_row(queue_id) is None
    with db.get_db_connection() as conn:
        errors = conn.execute(
            "SELECT * FROM trace_eval_errors WHERE trace_uuid = ?", (trace_uuid,)
        ).fetchall()
    assert len(errors) == 1
    assert errors[0]["attempts"] == 5
    assert errors[0]["error"] == "judge kept failing"
