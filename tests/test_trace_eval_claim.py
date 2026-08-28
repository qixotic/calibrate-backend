"""Claim, hydrate, invoke, and settle tests for trace scoring.

No lifespan worker — these drive the plain functions directly. Subprocess
tests stub Popen rather than launching calibrate-agent.
"""

from __future__ import annotations

import json
import random
import shutil
import subprocess
import threading
import uuid
from pathlib import Path
from unittest.mock import patch

import pytest

import db
import llm_judge
import trace_scoring as ts


def _org() -> str:
    return str(uuid.uuid4())


def _insert_agent(org: str, *, interaction_type="conversation", auto_score=True):
    agent_uuid = str(uuid.uuid4())
    with db.get_db_connection() as conn:
        conn.execute(
            "INSERT INTO agents "
            "(uuid, org_uuid, name, config, interaction_type, auto_score_traces) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (
                agent_uuid,
                org,
                f"agent-{agent_uuid[:8]}",
                "{}",
                interaction_type,
                1 if auto_score else 0,
            ),
        )
        conn.commit()
    return db.get_agent(agent_uuid)


def _eligible_evaluator(
    org: str,
    evaluator_type="llm",
    *,
    output_type="binary",
    name=None,
    prompt="Judge it.",
):
    kwargs = {}
    if output_type == "rating":
        kwargs["output_config"] = {
            "scale": [
                {"value": 1, "name": "Bad"},
                {"value": 5, "name": "Good"},
            ]
        }
    ev = db.create_evaluator(
        name=name or f"eval-{uuid.uuid4().hex[:6]}",
        evaluator_type=evaluator_type,
        output_type=output_type,
        org_uuid=org,
        owner_user_id=str(uuid.uuid4()),
    )
    version = db.create_evaluator_version(ev, "openai/gpt-4.1", prompt, **kwargs)
    db.set_evaluator_live_version(ev, version["uuid"])
    return ev, version["uuid"]


def _ingest_scored(org: str, agent: dict, **overrides):
    payload = {
        "message_id": None,
        "conversation_id": "conv-1",
        "input": [{"role": "user", "content": "hi"}],
        "output": {"response": "hello", "tool_calls": None},
        "metadata": None,
    }
    payload.update(overrides)
    return db.create_trace_with_eval_run(org_uuid=org, agent=agent, **payload)


def _run_for_trace(trace_uuid: str) -> dict:
    with db.get_db_connection() as conn:
        row = conn.execute(
            "SELECT * FROM trace_evaluations WHERE trace_uuid = ?",
            (trace_uuid,),
        ).fetchone()
    return dict(row)


def _setup_pending(org: str, **ingest_overrides):
    agent = _insert_agent(org)
    ev, version_id = _eligible_evaluator(org)
    db.add_evaluator_to_agent(agent["uuid"], ev)
    trace = _ingest_scored(org, agent, **ingest_overrides)
    return agent, ev, version_id, trace, _run_for_trace(trace["uuid"])


def _isolate(run_uuids, at=0):
    """Park every other open run so a global claim cannot steal this test's work."""
    with db.get_db_connection() as conn:
        conn.execute(
            "UPDATE trace_evaluations SET available_at = 2000000000 "
            "WHERE status IN ('pending', 'processing')"
        )
        unique = [u for u in run_uuids if u]
        if unique:
            placeholders = ",".join("?" * len(unique))
            conn.execute(
                f"UPDATE trace_evaluations SET available_at = ? WHERE uuid IN ({placeholders})",
                [at, *unique],
            )
        conn.commit()


def _passing_invoke(config, dataset, **kwargs):
    evaluators = {ev["name"]: ev for ev in config["evaluators"]}
    results = []
    for item in dataset:
        test_case = item["test_case"]
        judge = {}
        for ref in test_case["evaluation"]["criteria"]:
            ev = evaluators[ref["name"]]
            judgement = {"reasoning": "ok", "evaluator_id": ev["id"]}
            if ev.get("type") == "rating":
                judgement["score"] = ev.get("scale_max", 5)
            else:
                judgement["match"] = True
            judge[ref["name"]] = judgement
        results.append(
            {
                "test_case_id": test_case["id"],
                "test_case": test_case,
                "metrics": {"passed": True, "judge_results": judge},
            }
        )
    return ts.EvalOnlyCliResult(returncode=0, timed_out=False, results=results)


class _FakePopen:
    """Popen stand-in: `poll()` is None while `returncode` is None."""

    def poll(self):
        return self.returncode

    def kill(self):
        if self.returncode is None:
            self.returncode = -9


def test_sqlite_returning_support_rejects_old_versions(monkeypatch):
    monkeypatch.setattr(db.sqlite3, "sqlite_version_info", (3, 34, 0))
    monkeypatch.setattr(db.sqlite3, "sqlite_version", "3.34.0")
    with pytest.raises(RuntimeError, match="3.35"):
        db.assert_sqlite_returning_support()


def test_claim_returns_oldest_first_and_stamps_lease():
    org = _org()
    _, _, _, _, older = _setup_pending(org)
    _, _, _, _, newer = _setup_pending(org)
    _isolate([older["uuid"], newer["uuid"]], at=20)
    with db.get_db_connection() as conn:
        conn.execute(
            "UPDATE trace_evaluations SET available_at = 10 WHERE uuid = ?",
            (older["uuid"],),
        )
        conn.execute(
            "UPDATE trace_evaluations SET available_at = 20 WHERE uuid = ?",
            (newer["uuid"],),
        )
        conn.commit()

    claimed = db.claim_trace_evaluations(now=100, lease_seconds=60, batch_size=1)
    assert len(claimed) == 1
    assert claimed[0]["uuid"] == older["uuid"]
    assert claimed[0]["attempts"] == 1
    stored = db.get_trace_evaluation(older["uuid"])
    assert stored["status"] == "processing"
    assert stored["available_at"] == 160
    assert stored["attempts"] == 1
    assert json.loads(claimed[0]["criteria"])["evaluators"]


def test_claim_skips_future_available_at():
    org = _org()
    _, _, _, _, run = _setup_pending(org)
    _isolate([run["uuid"]], at=10_000)
    assert db.claim_trace_evaluations(now=50, lease_seconds=60, batch_size=10) == []


def test_expired_lease_is_reclaimed_and_increments_attempts():
    org = _org()
    _, _, _, _, run = _setup_pending(org)
    _isolate([run["uuid"]], at=0)
    first = db.claim_trace_evaluations(now=10, lease_seconds=60, batch_size=10)
    assert first[0]["attempts"] == 1
    with db.get_db_connection() as conn:
        conn.execute(
            "UPDATE trace_evaluations SET available_at = 0 WHERE uuid = ?",
            (run["uuid"],),
        )
        conn.commit()
    second = db.claim_trace_evaluations(now=20, lease_seconds=60, batch_size=10)
    assert second[0]["uuid"] == run["uuid"]
    assert second[0]["attempts"] == 2


def test_two_concurrent_claimers_never_receive_the_same_run():
    org = _org()
    run_uuids = set()
    for _ in range(16):
        _, _, _, _, run = _setup_pending(org)
        run_uuids.add(run["uuid"])
    _isolate(run_uuids, at=0)
    results: list[list[dict]] = [[], []]

    def _claim(slot: int):
        results[slot] = db.claim_trace_evaluations(
            now=1, lease_seconds=60, batch_size=20
        )

    t1 = threading.Thread(target=_claim, args=(0,))
    t2 = threading.Thread(target=_claim, args=(1,))
    t1.start()
    t2.start()
    t1.join()
    t2.join()
    ids_a = {row["uuid"] for row in results[0]}
    ids_b = {row["uuid"] for row in results[1]}
    assert ids_a.isdisjoint(ids_b)
    assert ids_a | ids_b == run_uuids


def test_hydrate_uses_pinned_version_not_live():
    org = _org()
    agent = _insert_agent(org)
    ev, pinned = _eligible_evaluator(org, prompt="PINNED PROMPT")
    db.add_evaluator_to_agent(agent["uuid"], ev)
    trace = _ingest_scored(org, agent)
    run = _run_for_trace(trace["uuid"])
    snapshot = json.loads(run["criteria"])
    assert snapshot["evaluators"][0]["evaluator_version_id"] == pinned
    live = db.create_evaluator_version(ev, "openai/gpt-4.1", "LIVE PROMPT")
    db.set_evaluator_live_version(ev, live["uuid"])

    called = {"refresh": 0}

    def _boom(snapshot_evaluators):
        called["refresh"] += 1
        raise AssertionError("refresh_evaluators_to_live must not run")

    with patch.object(llm_judge, "refresh_evaluators_to_live", _boom):
        hydrated = ts.hydrate_pinned_evaluators(snapshot["evaluators"])
    assert called["refresh"] == 0
    assert hydrated[0]["system_prompt"] == "PINNED PROMPT"
    assert hydrated[0]["evaluator_version_id"] == pinned


def test_hydrate_includes_soft_deleted_historical_version_and_evaluator():
    org = _org()
    agent = _insert_agent(org)
    ev, pinned = _eligible_evaluator(org, prompt="historical")
    db.add_evaluator_to_agent(agent["uuid"], ev)
    live = db.create_evaluator_version(ev, "openai/gpt-4.1", "new live")
    db.set_evaluator_live_version(ev, live["uuid"])
    assert db.soft_delete_evaluator_version(ev, pinned) == "deleted"
    assert db.delete_evaluator(ev) is True
    hydrated = ts.hydrate_pinned_evaluators(
        [{"evaluator_uuid": ev, "evaluator_version_id": pinned}]
    )
    assert hydrated is not None
    assert hydrated[0]["system_prompt"] == "historical"
    assert db.get_evaluator(ev) is None


def test_missing_or_mismatched_pin_is_corrupt_snapshot():
    org = _org()
    agent = _insert_agent(org)
    ev, version_id = _eligible_evaluator(org)
    db.add_evaluator_to_agent(agent["uuid"], ev)
    other, other_version = _eligible_evaluator(org, name="other")
    db.add_evaluator_to_agent(agent["uuid"], other)
    trace = _ingest_scored(org, agent)

    assert (
        ts.hydrate_pinned_evaluators(
            [{"evaluator_uuid": ev, "evaluator_version_id": str(uuid.uuid4())}]
        )
        is None
    )
    assert (
        ts.hydrate_pinned_evaluators(
            [{"evaluator_uuid": ev, "evaluator_version_id": other_version}]
        )
        is None
    )

    run_uuid = _run_for_trace(trace["uuid"])["uuid"]
    _isolate([run_uuid], at=0)
    claimed = db.claim_trace_evaluations(now=1, lease_seconds=60, batch_size=1)
    claimed[0]["criteria"] = json.dumps(
        {
            "type": "response",
            "evaluators": [
                {"evaluator_uuid": ev, "evaluator_version_id": str(uuid.uuid4())}
            ],
        }
    )
    ts.process_claimed_runs(claimed, now=5, invoke=_passing_invoke)
    stored = db.get_trace_evaluation(claimed[0]["uuid"])
    assert stored["status"] == "failed"
    assert stored["error"] == "corrupt_snapshot"
    assert db.get_trace_scores(claimed[0]["uuid"]) == []


def test_claim_and_score_completes_binary_and_rating():
    org = _org()
    agent = _insert_agent(org)
    binary, binary_v = _eligible_evaluator(org, name="Bin")
    rating, rating_v = _eligible_evaluator(
        org, output_type="rating", name="Rate"
    )
    db.add_evaluator_to_agent(agent["uuid"], binary)
    db.add_evaluator_to_agent(agent["uuid"], rating)
    trace = _ingest_scored(org, agent)
    _isolate([_run_for_trace(trace["uuid"])["uuid"]], at=0)
    ts.claim_and_score_batch(
        now=10, batch_size=5, lease_seconds=60, invoke=_passing_invoke
    )
    run = _run_for_trace(trace["uuid"])
    assert run["status"] == "completed"
    scores = {row["evaluator_uuid"]: row for row in db.get_trace_scores(run["uuid"])}
    assert scores[binary]["match"] == 1
    assert scores[binary]["score"] is None
    assert scores[binary]["evaluator_version_id"] == binary_v
    assert scores[rating]["match"] is None
    assert scores[rating]["score"] == 5.0
    assert scores[rating]["evaluator_version_id"] == rating_v


def test_general_item_shape_and_mapping_by_test_case_id():
    org = _org()
    agent = _insert_agent(org, interaction_type="general")
    ev, _ = _eligible_evaluator(org, "llm-general", name="Gen")
    db.add_evaluator_to_agent(agent["uuid"], ev)
    trace = _ingest_scored(
        org,
        agent,
        input="Summarize this.",
        output={"response": "Summary.", "tool_calls": None},
    )
    seen = {}

    def invoke(config, dataset, **kwargs):
        seen["dataset"] = dataset
        seen["config"] = config
        assert dataset[0]["test_case"]["id"] == _run_for_trace(trace["uuid"])["uuid"]
        assert dataset[0]["test_case"]["input"] == "Summarize this."
        assert "history" not in dataset[0]["test_case"]
        assert "tool_calls" not in dataset[0]["output"]
        return _passing_invoke(config, dataset, **kwargs)

    _isolate([_run_for_trace(trace["uuid"])["uuid"]], at=0)
    ts.claim_and_score_batch(now=10, batch_size=5, lease_seconds=60, invoke=invoke)
    assert db.get_trace_evaluation(_run_for_trace(trace["uuid"])["uuid"])["status"] == "completed"
    assert seen["dataset"][0]["test_case"]["evaluation"]["type"] == "general"


def test_name_collision_suffix_and_uuid_dedupe_across_batch():
    org_a, org_b = _org(), _org()
    agent_a1 = _insert_agent(org_a)
    agent_a2 = _insert_agent(org_a)
    agent_b = _insert_agent(org_b)
    ev_a, _ = _eligible_evaluator(org_a, name="Same")
    ev_b, _ = _eligible_evaluator(org_b, name="Same")
    db.add_evaluator_to_agent(agent_a1["uuid"], ev_a)
    db.add_evaluator_to_agent(agent_a2["uuid"], ev_a)
    db.add_evaluator_to_agent(agent_b["uuid"], ev_b)
    traces = [
        _ingest_scored(org_a, agent_a1),
        _ingest_scored(org_a, agent_a2),
        _ingest_scored(org_b, agent_b),
    ]
    seen = {}

    def invoke(config, dataset, **kwargs):
        seen["names"] = [ev["name"] for ev in config["evaluators"]]
        seen["ids"] = [ev["id"] for ev in config["evaluators"]]
        seen["calls"] = seen.get("calls", 0) + 1
        return _passing_invoke(config, dataset, **kwargs)

    _isolate([_run_for_trace(t["uuid"])["uuid"] for t in traces], at=0)
    ts.claim_and_score_batch(now=10, batch_size=10, lease_seconds=60, invoke=invoke)
    assert seen["calls"] == 1
    assert len(seen["ids"]) == 2
    assert set(seen["ids"]) == {ev_a, ev_b}
    assert len(set(seen["names"])) == 2
    assert "Same" in seen["names"]
    assert any(name.startswith("Same-") for name in seen["names"])


def test_shuffled_results_still_map_and_incomplete_is_not_settled():
    org = _org()
    setups = [_setup_pending(org) for _ in range(2)]
    _isolate([s[4]["uuid"] for s in setups], at=0)
    claimed = db.claim_trace_evaluations(now=1, lease_seconds=60, batch_size=10)

    def shuffled(config, dataset, **kwargs):
        result = _passing_invoke(config, dataset, **kwargs)
        return ts.EvalOnlyCliResult(
            returncode=0, timed_out=False, results=list(reversed(result.results))
        )

    ts.process_claimed_runs(claimed, now=5, invoke=shuffled)
    for _, _, _, _, run in setups:
        assert db.get_trace_evaluation(run["uuid"])["status"] == "completed"


def test_partial_results_settle_finished_and_defer_the_rest():
    org = _org()
    first = _setup_pending(org)
    second = _setup_pending(org)
    _isolate([first[4]["uuid"], second[4]["uuid"]], at=0)
    claimed = db.claim_trace_evaluations(now=1, lease_seconds=60, batch_size=10)

    def partial(config, dataset, **kwargs):
        result = _passing_invoke(config, dataset, **kwargs)
        keep_id = first[4]["uuid"]
        kept = [row for row in result.results if row["test_case_id"] == keep_id]
        return ts.EvalOnlyCliResult(returncode=1, timed_out=False, results=kept)

    rng = random.Random(0)
    ts.process_claimed_runs(claimed, now=5, invoke=partial, rng=rng, max_attempts=5)
    assert db.get_trace_evaluation(first[4]["uuid"])["status"] == "completed"
    leftover = db.get_trace_evaluation(second[4]["uuid"])
    assert leftover["status"] == "pending"
    assert leftover["available_at"] > 5
    assert leftover["attempts"] == 1
    assert db.get_trace_scores(second[4]["uuid"]) == []


def test_timeout_parses_partial_and_defers_unfinished_with_jitter():
    org = _org()
    first = _setup_pending(org)
    second = _setup_pending(org)
    _isolate([first[4]["uuid"], second[4]["uuid"]], at=0)
    claimed = db.claim_trace_evaluations(now=1, lease_seconds=60, batch_size=10)

    def timed_out(config, dataset, **kwargs):
        result = _passing_invoke(config, dataset, **kwargs)
        keep_id = first[4]["uuid"]
        kept = [row for row in result.results if row["test_case_id"] == keep_id]
        return ts.EvalOnlyCliResult(
            returncode=-9, timed_out=True, results=kept, error="timed out"
        )

    rng = random.Random(1)
    expected_at = ts.backoff_available_at(1, 5, random.Random(1))
    ts.process_claimed_runs(claimed, now=5, invoke=timed_out, rng=rng, max_attempts=5)
    assert db.get_trace_evaluation(first[4]["uuid"])["status"] == "completed"
    leftover = db.get_trace_evaluation(second[4]["uuid"])
    assert leftover["status"] == "pending"
    assert leftover["available_at"] == expected_at
    assert "timed out" in leftover["error"]


def test_partial_unfinished_hits_attempt_ceiling_without_reassembling():
    org = _org()
    first = _setup_pending(org)
    second = _setup_pending(org)
    leftover_uuid = second[4]["uuid"]
    _isolate([first[4]["uuid"], leftover_uuid], at=0)
    claimed = db.claim_trace_evaluations(now=1, lease_seconds=60, batch_size=10)

    def partial(config, dataset, **kwargs):
        result = _passing_invoke(config, dataset, **kwargs)
        keep_id = first[4]["uuid"]
        kept = [row for row in result.results if row["test_case_id"] == keep_id]
        return ts.EvalOnlyCliResult(
            returncode=1, timed_out=False, results=kept, error="partial"
        )

    ts.process_claimed_runs(
        claimed, now=5, invoke=partial, rng=random.Random(2), max_attempts=2
    )
    assert db.get_trace_evaluation(first[4]["uuid"])["status"] == "completed"
    deferred = db.get_trace_evaluation(leftover_uuid)
    assert deferred["status"] == "pending"
    assert deferred["available_at"] > 5

    with db.get_db_connection() as conn:
        conn.execute(
            "UPDATE trace_evaluations SET available_at = 0, attempts = 2 WHERE uuid = ?",
            (leftover_uuid,),
        )
        conn.commit()
    _isolate([leftover_uuid], at=0)
    claimed = db.claim_trace_evaluations(now=200, lease_seconds=60, batch_size=1)
    assert claimed[0]["attempts"] == 3
    ts.process_claimed_runs(
        claimed, now=200, invoke=partial, rng=random.Random(3), max_attempts=2
    )
    failed = db.get_trace_evaluation(leftover_uuid)
    assert failed["status"] == "failed"
    assert failed["completed_at"] == 200
    assert db.get_trace_evaluation(first[4]["uuid"])["status"] == "completed"
    assert db.get_trace_scores(leftover_uuid) == []


def test_deleted_during_failed_invoke_skips_instead_of_deferring():
    org = _org()
    agent, _, _, trace, run = _setup_pending(org)
    _isolate([run["uuid"]], at=0)
    claimed = db.claim_trace_evaluations(now=1, lease_seconds=60, batch_size=1)

    def boom_and_delete(config, dataset, **kwargs):
        db.soft_delete_traces(org, trace_ids=[trace["uuid"]])
        raise RuntimeError("provider down")

    ts.process_claimed_runs(claimed, now=5, invoke=boom_and_delete, max_attempts=5)
    stored = db.get_trace_evaluation(run["uuid"])
    assert stored["status"] == "skipped"
    assert stored["error"] == "trace_deleted"
    assert db.get_trace_scores(run["uuid"]) == []

    org2 = _org()
    agent2, _, _, _, run2 = _setup_pending(org2)
    _isolate([run2["uuid"]], at=0)
    claimed = db.claim_trace_evaluations(now=1, lease_seconds=60, batch_size=1)

    def boom_and_delete_agent(config, dataset, **kwargs):
        db.delete_agent(agent2["uuid"])
        return ts.EvalOnlyCliResult(
            returncode=1, timed_out=False, results=[], error="judge crashed"
        )

    ts.process_claimed_runs(claimed, now=5, invoke=boom_and_delete_agent)
    stored2 = db.get_trace_evaluation(run2["uuid"])
    assert stored2["status"] == "skipped"
    assert stored2["error"] == "agent_deleted"
    assert db.get_trace_scores(run2["uuid"]) == []


def test_deleted_partial_leftover_skips_instead_of_deferring():
    org = _org()
    first = _setup_pending(org)
    second = _setup_pending(org)
    _isolate([first[4]["uuid"], second[4]["uuid"]], at=0)
    claimed = db.claim_trace_evaluations(now=1, lease_seconds=60, batch_size=10)

    def partial_then_delete(config, dataset, **kwargs):
        db.soft_delete_traces(org, trace_ids=[second[3]["uuid"]])
        result = _passing_invoke(config, dataset, **kwargs)
        keep_id = first[4]["uuid"]
        kept = [row for row in result.results if row["test_case_id"] == keep_id]
        return ts.EvalOnlyCliResult(returncode=1, timed_out=False, results=kept)

    ts.process_claimed_runs(claimed, now=5, invoke=partial_then_delete)
    assert db.get_trace_evaluation(first[4]["uuid"])["status"] == "completed"
    leftover = db.get_trace_evaluation(second[4]["uuid"])
    assert leftover["status"] == "skipped"
    assert leftover["error"] == "trace_deleted"
    assert db.get_trace_scores(second[4]["uuid"]) == []


def test_malformed_unknown_duplicate_output_is_not_settled():
    org = _org()
    _, ev, _, _, run = _setup_pending(org)
    _isolate([run["uuid"]], at=0)
    claimed = db.claim_trace_evaluations(now=1, lease_seconds=60, batch_size=1)

    def bad(config, dataset, **kwargs):
        run_id = dataset[0]["test_case"]["id"]
        return ts.EvalOnlyCliResult(
            returncode=0,
            timed_out=False,
            results=[
                {
                    "test_case_id": run_id,
                    "metrics": {
                        "judge_results": {
                            "Nope": {"match": True, "evaluator_id": str(uuid.uuid4())}
                        }
                    },
                },
                {
                    "test_case_id": run_id,
                    "metrics": {
                        "judge_results": {
                            "Bin": {"match": True, "evaluator_id": ev}
                        }
                    },
                },
            ],
        )

    ts.process_claimed_runs(claimed, now=5, invoke=bad, max_attempts=5)
    stored = db.get_trace_evaluation(run["uuid"])
    assert stored["status"] == "pending"
    assert db.get_trace_scores(run["uuid"]) == []


def test_whole_invocation_failure_defers_with_jitter_then_fails_at_ceiling():
    org = _org()
    _, _, _, _, run = _setup_pending(org)
    _isolate([run["uuid"]], at=0)
    claimed = db.claim_trace_evaluations(now=100, lease_seconds=60, batch_size=1)

    def boom(config, dataset, **kwargs):
        raise RuntimeError("provider down")

    rng = random.Random(1)
    ts.process_claimed_runs(claimed, now=100, invoke=boom, rng=rng, max_attempts=2)
    deferred = db.get_trace_evaluation(run["uuid"])
    assert deferred["status"] == "pending"
    assert deferred["available_at"] > 100
    assert "provider down" in deferred["error"]

    with db.get_db_connection() as conn:
        conn.execute(
            "UPDATE trace_evaluations SET available_at = 0, attempts = 2 WHERE uuid = ?",
            (run["uuid"],),
        )
        conn.commit()
    _isolate([run["uuid"]], at=0)
    claimed = db.claim_trace_evaluations(now=200, lease_seconds=60, batch_size=1)
    assert claimed[0]["attempts"] == 3
    ts.process_claimed_runs(claimed, now=200, invoke=boom, max_attempts=2)
    failed = db.get_trace_evaluation(run["uuid"])
    assert failed["status"] == "failed"
    assert failed["completed_at"] == 200


def test_deleted_trace_or_agent_skips_without_scores():
    org = _org()
    agent, _, _, trace, run = _setup_pending(org)
    db.soft_delete_traces(org, trace_ids=[trace["uuid"]])
    _isolate([run["uuid"]], at=0)
    claimed = db.claim_trace_evaluations(now=1, lease_seconds=60, batch_size=1)
    ts.process_claimed_runs(claimed, now=5, invoke=_passing_invoke)
    stored = db.get_trace_evaluation(run["uuid"])
    assert stored["status"] == "skipped"
    assert stored["error"] == "trace_deleted"
    assert db.get_trace_scores(run["uuid"]) == []

    org2 = _org()
    agent2, _, _, trace2, run2 = _setup_pending(org2)
    _isolate([run2["uuid"]], at=0)
    claimed = db.claim_trace_evaluations(now=1, lease_seconds=60, batch_size=1)
    db.delete_agent(agent2["uuid"])
    ts.process_claimed_runs(claimed, now=5, invoke=_passing_invoke)
    stored2 = db.get_trace_evaluation(run2["uuid"])
    assert stored2["status"] == "skipped"
    assert stored2["error"] == "agent_deleted"
    assert db.get_trace_scores(run2["uuid"]) == []


def test_liveness_flip_during_invoke_skips_and_writes_no_scores():
    org = _org()
    _, _, _, trace, run = _setup_pending(org)
    _isolate([run["uuid"]], at=0)
    claimed = db.claim_trace_evaluations(now=1, lease_seconds=60, batch_size=1)

    def invoke_then_delete(config, dataset, **kwargs):
        db.soft_delete_traces(org, trace_ids=[trace["uuid"]])
        return _passing_invoke(config, dataset, **kwargs)

    ts.process_claimed_runs(claimed, now=5, invoke=invoke_then_delete)
    stored = db.get_trace_evaluation(run["uuid"])
    assert stored["status"] == "skipped"
    assert stored["error"] == "trace_deleted"
    assert db.get_trace_scores(run["uuid"]) == []


def test_late_settlement_is_idempotent_noop():
    org = _org()
    _, ev, version_id, trace, run = _setup_pending(org)
    _isolate([run["uuid"]], at=0)
    claimed = db.claim_trace_evaluations(now=1, lease_seconds=60, batch_size=1)
    ts.process_claimed_runs(claimed, now=5, invoke=_passing_invoke)
    assert db.get_trace_evaluation(run["uuid"])["status"] == "completed"
    first_scores = db.get_trace_scores(run["uuid"])

    again = db.settle_trace_evaluation_completed(
        run["uuid"],
        [
            {
                "evaluator_uuid": ev,
                "evaluator_version_id": version_id,
                "match": 0,
                "score": None,
                "reasoning": "late",
            }
        ],
        now=9,
    )
    assert again == "noop"
    assert db.get_trace_scores(run["uuid"])[0]["match"] == first_scores[0]["match"]
    assert db.get_trace_evaluation(run["uuid"])["status"] == "completed"


def test_same_version_rescore_is_a_distinct_run():
    org = _org()
    agent, ev, version_id, trace, run = _setup_pending(org)
    _isolate([run["uuid"]], at=0)
    ts.claim_and_score_batch(
        now=10, batch_size=5, lease_seconds=60, invoke=_passing_invoke
    )
    first = _run_for_trace(trace["uuid"])
    assert first["status"] == "completed"
    now = 20
    second_uuid = str(uuid.uuid4())
    with db.get_db_connection() as conn:
        conn.execute(
            "INSERT INTO trace_evaluations "
            "(uuid, trace_uuid, org_uuid, agent_id, status, criteria, "
            "available_at, attempts, created_at, updated_at) "
            "VALUES (?, ?, ?, ?, 'pending', ?, ?, 0, ?, ?)",
            (
                second_uuid,
                trace["uuid"],
                org,
                agent["uuid"],
                first["criteria"],
                now,
                now,
                now,
            ),
        )
        conn.commit()
    _isolate([second_uuid], at=0)
    ts.claim_and_score_batch(
        now=now, batch_size=5, lease_seconds=60, invoke=_passing_invoke
    )
    with db.get_db_connection() as conn:
        runs = conn.execute(
            "SELECT uuid, status FROM trace_evaluations WHERE trace_uuid = ? "
            "ORDER BY created_at",
            (trace["uuid"],),
        ).fetchall()
        scores = conn.execute(
            "SELECT run_uuid, evaluator_uuid, evaluator_version_id FROM trace_scores "
            "WHERE trace_uuid = ?",
            (trace["uuid"],),
        ).fetchall()
    assert len(runs) == 2
    assert {r["status"] for r in runs} == {"completed"}
    assert {s["run_uuid"] for s in scores} == {r["uuid"] for r in runs}
    assert {s["evaluator_version_id"] for s in scores} == {version_id}


def test_invoke_eval_only_cli_uses_cli_seam_temp_files_and_new_session():
    captured = {}

    class FakeProc(_FakePopen):
        def __init__(self, cmd):
            self.cmd = cmd
            self.pid = 4242
            self.returncode = 0
            output_dir = Path(cmd[cmd.index("-o") + 1])
            dataset = json.loads(Path(cmd[cmd.index("--dataset") + 1]).read_text())
            config = json.loads(Path(cmd[cmd.index("-c") + 1]).read_text())
            ev = config["evaluators"][0]
            output_dir.mkdir(parents=True, exist_ok=True)
            (output_dir / "results.json").write_text(
                json.dumps(
                    [
                        {
                            "test_case_id": dataset[0]["test_case"]["id"],
                            "test_case": dataset[0]["test_case"],
                            "metrics": {
                                "judge_results": {
                                    ev["name"]: {
                                        "match": True,
                                        "evaluator_id": ev["id"],
                                        "reasoning": "fake",
                                    }
                                }
                            },
                        }
                    ]
                ),
                encoding="utf-8",
            )

        def wait(self, timeout=None):
            return 0

    def fake_popen(cmd, **kwargs):
        captured["cmd"] = cmd
        captured["kwargs"] = kwargs
        return FakeProc(cmd)

    with patch("trace_scoring.subprocess.Popen", fake_popen), patch(
        "trace_scoring.get_calibrate_agent_cli", return_value="calibrate-agent"
    ):
        result = ts.invoke_eval_only_cli(
            {"evaluators": [{"name": "Bin", "id": "e1", "type": "binary"}]},
            [
                {
                    "test_case": {
                        "id": "run-1",
                        "history": [],
                        "evaluation": {
                            "type": "response",
                            "criteria": [{"name": "Bin"}],
                        },
                    },
                    "output": {"response": "hi", "tool_calls": []},
                }
            ],
            timeout_seconds=30,
            parallel=2,
        )
    assert captured["kwargs"]["start_new_session"] is True
    assert captured["kwargs"]["stdout"].name.endswith("stdout.log")
    assert captured["kwargs"]["stderr"].name.endswith("stderr.log")
    assert captured["cmd"][:2] == ["calibrate-agent", "llm"]
    assert "--eval-only" in captured["cmd"]
    assert result.returncode == 0
    assert result.results[0]["test_case_id"] == "run-1"


def test_invoke_eval_only_cli_timeout_kills_process_group_and_parses_partial():
    killed = []

    class FakeProc(_FakePopen):
        def __init__(self, cmd):
            self.cmd = cmd
            self.pid = 99
            self.returncode = None
            output_dir = Path(cmd[cmd.index("-o") + 1])
            output_dir.mkdir(parents=True, exist_ok=True)
            (output_dir / "results.json").write_text("[]", encoding="utf-8")
            self._waited = 0

        def wait(self, timeout=None):
            self._waited += 1
            if self._waited == 1:
                raise subprocess.TimeoutExpired(self.cmd, timeout)
            self.returncode = -9
            return -9

    with patch("trace_scoring.subprocess.Popen", lambda cmd, **kwargs: FakeProc(cmd)), patch(
        "trace_scoring.get_calibrate_agent_cli", return_value="calibrate-agent"
    ), patch("trace_scoring.kill_process_group", lambda pid, job: killed.append((pid, job))):
        result = ts.invoke_eval_only_cli(
            {"evaluators": []},
            [],
            timeout_seconds=1,
        )
    assert result.timed_out is True
    assert killed == [(99, "trace-scoring")]
    assert result.results == []


def test_claim_empty_batch_size_is_noop():
    assert db.claim_trace_evaluations(now=1, lease_seconds=60, batch_size=0) == []
    assert db.claim_trace_evaluations(now=1, lease_seconds=60, batch_size=-1) == []


def test_process_claimed_runs_empty_is_noop():
    ts.process_claimed_runs([])


def test_mixed_response_and_general_share_one_invocation():
    conv_org = _org()
    gen_org = _org()
    conv_agent = _insert_agent(conv_org)
    gen_agent = _insert_agent(gen_org, interaction_type="general")
    conv_ev, _ = _eligible_evaluator(conv_org, name="Conv")
    gen_ev, _ = _eligible_evaluator(gen_org, "llm-general", name="Gen")
    db.add_evaluator_to_agent(conv_agent["uuid"], conv_ev)
    db.add_evaluator_to_agent(gen_agent["uuid"], gen_ev)
    conv_trace = _ingest_scored(conv_org, conv_agent)
    gen_trace = _ingest_scored(
        gen_org,
        gen_agent,
        input="Summarize this.",
        output={"response": "Summary.", "tool_calls": None},
    )
    seen = {}

    def invoke(config, dataset, **kwargs):
        seen["calls"] = seen.get("calls", 0) + 1
        seen["types"] = [row["test_case"]["evaluation"]["type"] for row in dataset]
        return _passing_invoke(config, dataset, **kwargs)

    _isolate(
        [
            _run_for_trace(conv_trace["uuid"])["uuid"],
            _run_for_trace(gen_trace["uuid"])["uuid"],
        ],
        at=0,
    )
    ts.claim_and_score_batch(now=10, batch_size=10, lease_seconds=60, invoke=invoke)
    assert seen["calls"] == 1
    assert set(seen["types"]) == {"response", "general"}


def test_unique_result_with_incomplete_judges_is_not_settled():
    org = _org()
    agent = _insert_agent(org)
    first, _ = _eligible_evaluator(org, name="One")
    second, _ = _eligible_evaluator(org, name="Two")
    db.add_evaluator_to_agent(agent["uuid"], first)
    db.add_evaluator_to_agent(agent["uuid"], second)
    trace = _ingest_scored(org, agent)
    run = _run_for_trace(trace["uuid"])
    _isolate([run["uuid"]], at=0)
    claimed = db.claim_trace_evaluations(now=1, lease_seconds=60, batch_size=1)

    def drop_one(config, dataset, **kwargs):
        result = _passing_invoke(config, dataset, **kwargs)
        judges = result.results[0]["metrics"]["judge_results"]
        judges.pop(next(iter(judges)))
        return result

    ts.process_claimed_runs(claimed, now=5, invoke=drop_one)
    stored = db.get_trace_evaluation(run["uuid"])
    assert stored["status"] == "pending"
    assert db.get_trace_scores(run["uuid"]) == []


def test_concurrent_settlers_only_one_writes_scores():
    org = _org()
    _, ev, version_id, _, run = _setup_pending(org)
    _isolate([run["uuid"]], at=0)
    claimed = db.claim_trace_evaluations(now=1, lease_seconds=60, batch_size=1)
    assert claimed
    scores = [
        {
            "evaluator_uuid": ev,
            "evaluator_version_id": version_id,
            "match": 1,
            "score": None,
            "reasoning": "ok",
        }
    ]
    outcomes: list[str] = []
    barrier = threading.Barrier(2)

    def settle():
        barrier.wait()
        outcomes.append(
            db.settle_trace_evaluation_completed(run["uuid"], scores, now=5)
        )

    t1 = threading.Thread(target=settle)
    t2 = threading.Thread(target=settle)
    t1.start()
    t2.start()
    t1.join()
    t2.join()
    assert sorted(outcomes) == ["completed", "noop"]
    stored = db.get_trace_scores(run["uuid"])
    assert len(stored) == 1
    assert stored[0]["match"] == 1


def test_concurrent_skip_settlers_only_one_wins():
    org = _org()
    _, _, _, trace, run = _setup_pending(org)
    _isolate([run["uuid"]], at=0)
    db.claim_trace_evaluations(now=1, lease_seconds=60, batch_size=1)
    db.soft_delete_traces(org, trace_ids=[trace["uuid"]])
    scores = [
        {
            "evaluator_uuid": str(uuid.uuid4()),
            "evaluator_version_id": str(uuid.uuid4()),
            "match": 1,
            "score": None,
            "reasoning": "late",
        }
    ]
    outcomes: list[str] = []
    barrier = threading.Barrier(2)

    def settle():
        barrier.wait()
        outcomes.append(
            db.settle_trace_evaluation_completed(run["uuid"], scores, now=5)
        )

    t1 = threading.Thread(target=settle)
    t2 = threading.Thread(target=settle)
    t1.start()
    t2.start()
    t1.join()
    t2.join()
    assert sorted(outcomes) == ["noop", "skipped"]
    assert db.get_trace_scores(run["uuid"]) == []


def test_rc0_incomplete_results_are_deferred():
    org = _org()
    _, _, _, _, run = _setup_pending(org)
    _isolate([run["uuid"]], at=0)
    claimed = db.claim_trace_evaluations(now=1, lease_seconds=60, batch_size=1)

    def incomplete(config, dataset, **kwargs):
        return ts.EvalOnlyCliResult(returncode=0, timed_out=False, results=[])

    ts.process_claimed_runs(claimed, now=5, invoke=incomplete)
    stored = db.get_trace_evaluation(run["uuid"])
    assert stored["status"] == "pending"
    assert stored["available_at"] > 5


def test_nonzero_exit_with_no_complete_results_defers():
    org = _org()
    _, _, _, _, run = _setup_pending(org)
    _isolate([run["uuid"]], at=0)
    claimed = db.claim_trace_evaluations(now=1, lease_seconds=60, batch_size=1)

    def boom(config, dataset, **kwargs):
        return ts.EvalOnlyCliResult(
            returncode=1, timed_out=False, results=[], error="judge crashed"
        )

    ts.process_claimed_runs(claimed, now=5, invoke=boom)
    stored = db.get_trace_evaluation(run["uuid"])
    assert stored["status"] == "pending"
    assert "judge crashed" in stored["error"]


def test_invalid_snapshot_json_fails_corrupt_snapshot():
    org = _org()
    _, _, _, _, run = _setup_pending(org)
    _isolate([run["uuid"]], at=0)
    claimed = db.claim_trace_evaluations(now=1, lease_seconds=60, batch_size=1)
    claimed[0]["criteria"] = "not-json"
    ts.process_claimed_runs(claimed, now=5, invoke=_passing_invoke)
    stored = db.get_trace_evaluation(run["uuid"])
    assert stored["status"] == "failed"
    assert stored["error"] == "corrupt_snapshot"


def test_vanished_trace_after_liveness_check_skips():
    org = _org()
    _, _, _, _, run = _setup_pending(org)
    _isolate([run["uuid"]], at=0)
    claimed = db.claim_trace_evaluations(now=1, lease_seconds=60, batch_size=1)
    with patch("db.get_trace", return_value=None), patch(
        "db.trace_scoring_skip_reason", return_value=None
    ):
        ts.process_claimed_runs(claimed, now=5, invoke=_passing_invoke)
    stored = db.get_trace_evaluation(run["uuid"])
    assert stored["status"] == "skipped"
    assert stored["error"] == "trace_deleted"


def test_invoke_exception_truncates_error():
    org = _org()
    _, _, _, _, run = _setup_pending(org)
    _isolate([run["uuid"]], at=0)
    claimed = db.claim_trace_evaluations(now=1, lease_seconds=60, batch_size=1)

    def boom(config, dataset, **kwargs):
        raise RuntimeError("e" * (ts.ERROR_MAX_CHARS + 20))

    ts.process_claimed_runs(claimed, now=5, invoke=boom, max_attempts=2)
    stored = db.get_trace_evaluation(run["uuid"])
    assert stored["status"] == "pending"
    assert stored["error"].endswith("...")
    assert len(stored["error"]) == ts.ERROR_MAX_CHARS


def test_settle_terminal_rejects_unknown_status():
    with pytest.raises(ValueError, match="failed or skipped"):
        db.settle_trace_evaluation_terminal(
            str(uuid.uuid4()), "completed", error=None, now=1
        )


def test_settle_unknown_run_is_noop():
    assert db.settle_trace_evaluation_completed(str(uuid.uuid4()), [], now=1) == "noop"
    assert (
        db.settle_trace_evaluation_terminal(
            str(uuid.uuid4()), "failed", error="x", now=1
        )
        is False
    )
    assert (
        db.defer_trace_evaluation(
            str(uuid.uuid4()), available_at=10, now=1, error="x"
        )
        is False
    )


def test_deleted_evaluators_are_omitted_unless_include_deleted():
    org = _org()
    ev, _ = _eligible_evaluator(org, name="SoonGone")
    assert ev in db.get_evaluators_by_uuids([ev])
    assert db.delete_evaluator(ev) is True
    assert db.get_evaluators_by_uuids([ev]) == {}
    assert ev in db.get_evaluators_by_uuids([ev], include_deleted=True)


def test_invoke_eval_only_cli_nonzero_reads_stderr():
    class FakeProc(_FakePopen):
        def __init__(self, cmd, kwargs):
            self.pid = 7
            self.returncode = 1
            kwargs["stderr"].write("last error line\n")
            kwargs["stderr"].flush()

        def wait(self, timeout=None):
            return 1

    with patch(
        "trace_scoring.subprocess.Popen",
        lambda cmd, **kwargs: FakeProc(cmd, kwargs),
    ), patch("trace_scoring.get_calibrate_agent_cli", return_value="calibrate-agent"):
        result = ts.invoke_eval_only_cli({"evaluators": []}, [], timeout_seconds=5)
    assert result.returncode == 1
    assert result.timed_out is False
    assert "last error line" in result.error


def test_invoke_eval_only_cli_timeout_second_wait_still_times_out():
    killed = []

    class FakeProc(_FakePopen):
        def __init__(self, cmd):
            self.cmd = cmd
            self.pid = 11
            self.returncode = None

        def wait(self, timeout=None):
            raise subprocess.TimeoutExpired(self.cmd, timeout)

    with patch(
        "trace_scoring.subprocess.Popen", lambda cmd, **kwargs: FakeProc(cmd)
    ), patch(
        "trace_scoring.get_calibrate_agent_cli", return_value="calibrate-agent"
    ), patch(
        "trace_scoring.kill_process_group", lambda pid, job: killed.append((pid, job))
    ):
        result = ts.invoke_eval_only_cli({"evaluators": []}, [], timeout_seconds=1)
    assert result.timed_out is True
    assert result.returncode == -9
    assert killed == [(11, "trace-scoring")]


def test_invoke_eval_only_cli_stderr_read_failure_is_empty():
    class FakeProc(_FakePopen):
        def __init__(self):
            self.pid = 3
            self.returncode = 2

        def wait(self, timeout=None):
            return 2

    real_read = Path.read_text

    def boom(self, *args, **kwargs):
        if self.name == "stderr.log":
            raise OSError("gone")
        return real_read(self, *args, **kwargs)

    with patch(
        "trace_scoring.subprocess.Popen", lambda cmd, **kwargs: FakeProc()
    ), patch(
        "trace_scoring.get_calibrate_agent_cli", return_value="calibrate-agent"
    ), patch.object(Path, "read_text", boom):
        result = ts.invoke_eval_only_cli({"evaluators": []}, [], timeout_seconds=5)
    assert result.returncode == 2
    assert "exited 2" in result.error


def test_invoke_timeout_falls_back_to_process_kill_when_group_kill_fails():
    group_calls = []
    killed = []

    class FakeProc(_FakePopen):
        def __init__(self, cmd):
            self.cmd = cmd
            self.pid = 44
            self.returncode = None

        def wait(self, timeout=None):
            if self.returncode is not None:
                return self.returncode
            raise subprocess.TimeoutExpired(self.cmd, timeout)

        def kill(self):
            killed.append(self.pid)
            self.returncode = -9

    with patch(
        "trace_scoring.subprocess.Popen", lambda cmd, **kwargs: FakeProc(cmd)
    ), patch(
        "trace_scoring.get_calibrate_agent_cli", return_value="calibrate-agent"
    ), patch(
        "trace_scoring.kill_process_group",
        lambda pid, job: group_calls.append((pid, job)) or False,
    ):
        result = ts.invoke_eval_only_cli({"evaluators": []}, [], timeout_seconds=1)
    assert result.timed_out is True
    assert result.returncode == -9
    assert group_calls == [(44, "trace-scoring")]
    assert killed == [44]


def test_invoke_timeout_skips_tempdir_cleanup_while_process_alive():
    rmtree_calls = []
    captured = {}

    class FakeProc(_FakePopen):
        def __init__(self, cmd):
            self.cmd = cmd
            self.pid = 55
            self.returncode = None

        def wait(self, timeout=None):
            raise subprocess.TimeoutExpired(self.cmd, timeout)

        def kill(self):
            return None

        def poll(self):
            return None

    def fake_popen(cmd, **kwargs):
        captured["cwd"] = kwargs.get("cwd")
        return FakeProc(cmd)

    def tracking_rmtree(path, *args, **kwargs):
        rmtree_calls.append(str(path))

    with patch("trace_scoring.subprocess.Popen", fake_popen), patch(
        "trace_scoring.get_calibrate_agent_cli", return_value="calibrate-agent"
    ), patch("trace_scoring.kill_process_group", return_value=False), patch(
        "trace_scoring.shutil.rmtree", tracking_rmtree
    ):
        result = ts.invoke_eval_only_cli({"evaluators": []}, [], timeout_seconds=1)
    assert result.timed_out is True
    assert result.returncode == -9
    assert rmtree_calls == []
    leftover = captured.get("cwd")
    if leftover:
        shutil.rmtree(leftover, ignore_errors=True)


def test_reap_cli_process_noops_when_already_exited():
    class FakeProc(_FakePopen):
        def __init__(self):
            self.pid = 1
            self.returncode = 0

        def wait(self, timeout=None):
            raise AssertionError("already exited")

        def kill(self):
            raise AssertionError("already exited")

    with patch("trace_scoring.kill_process_group") as killer:
        ts._reap_cli_process(FakeProc())
    killer.assert_not_called()


def test_reap_cli_process_swallows_kill_oserror():
    class FakeProc(_FakePopen):
        def __init__(self):
            self.pid = 8
            self.returncode = None

        def wait(self, timeout=None):
            raise subprocess.TimeoutExpired("cmd", timeout)

        def kill(self):
            raise OSError("esrch")

    with patch("trace_scoring.kill_process_group", return_value=True):
        ts._reap_cli_process(FakeProc())
