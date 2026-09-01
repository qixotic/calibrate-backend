"""Engine tests: claim, hydrate, invoke, settle.

Schema constraints (CHECKs, NOT NULLs, active-run uniqueness) are pinned in
`test_db_trace_eval.py` and the ingest-time snapshot contract in
`test_db_traces.py`; these tests exercise behaviour on top of both. Pure
helpers with no DB are unit-tested in `test_trace_scoring.py`.
"""

from __future__ import annotations

import json
import random
import subprocess
import time
import uuid
from dataclasses import asdict
from pathlib import Path

import db
import pytest
import trace_scoring as ts

RunStatus = ts.TraceEvalRunStatus


@pytest.fixture(autouse=True)
def _isolate_runs():
    """The claim scans every open run in the file, so leftovers from another
    test would be claimed by this one."""
    with db.get_db_connection() as conn:
        conn.execute("DELETE FROM trace_eval_runs")
        conn.execute("DELETE FROM trace_eval_scores")
        conn.commit()
    yield


def _org() -> str:
    return str(uuid.uuid4())


def _agent(org: str, *, interaction_type: str = "conversation") -> dict:
    agent_uuid = str(uuid.uuid4())
    with db.get_db_connection() as conn:
        conn.execute(
            "INSERT INTO agents (uuid, org_uuid, name, config, interaction_type, "
            "auto_score_traces) VALUES (?, ?, ?, ?, ?, 1)",
            (agent_uuid, org, f"agent-{agent_uuid[:8]}", "{}", interaction_type),
        )
        conn.commit()
    return db.get_agent(agent_uuid)


def _evaluator(
    org: str,
    *,
    name: str | None = None,
    evaluator_type: str = "llm",
    output_type: str = "binary",
    output_config: dict | None = None,
) -> tuple[str, str]:
    """Returns `(evaluator_uuid, live_version_uuid)`."""
    ev = db.create_evaluator(
        name=name or f"eval-{uuid.uuid4().hex[:6]}",
        evaluator_type=evaluator_type,
        org_uuid=org,
        owner_user_id=str(uuid.uuid4()),
        output_type=output_type,
    )
    version = db.create_evaluator_version(
        ev, "openai/gpt-4.1", "Judge it.", output_config=output_config
    )
    db.set_evaluator_live_version(ev, version["uuid"])
    return ev, version["uuid"]


def _rating_evaluator(org: str, *, name: str | None = None) -> tuple[str, str]:
    return _evaluator(
        org,
        name=name,
        output_type="rating",
        output_config={"scale": [{"value": 1, "name": "Bad"}, {"value": 5, "name": "Good"}]},
    )


def _trace(org: str, agent: dict, **overrides) -> dict:
    payload = {
        "input": [{"role": "user", "content": "hi"}],
        "output": {"response": "hello", "tool_calls": None},
    }
    payload.update(overrides)
    return db.create_trace(org_uuid=org, agent_id=agent["uuid"], **payload)


def _run(
    org: str,
    agent: dict,
    trace: dict,
    pins: list[tuple[str, str]],
    *,
    evaluation_type: str = "response",
    available_at: int = 0,
    attempts: int = 0,
    status: RunStatus = RunStatus.PENDING,
    scoring_plan: str | None = "",
) -> str:
    """Insert one open run carrying a real snapshot. Returns its uuid."""
    if scoring_plan == "":
        scoring_plan = json.dumps(
            asdict(
                ts.ScoringPlan(
                    evaluation_type=evaluation_type,
                    evaluators=[
                        ts.ScoringPlanPin(evaluator_uuid=e, evaluator_version_id=v)
                        for e, v in pins
                    ],
                )
            )
        )
    run_uuid = str(uuid.uuid4())
    with db.get_db_connection() as conn:
        conn.execute(
            "INSERT INTO trace_eval_runs (uuid, trace_uuid, org_uuid, agent_id, status, "
            "scoring_plan, available_at, attempts, created_at, updated_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, 1, 1)",
            (
                run_uuid,
                trace["uuid"],
                org,
                agent["uuid"],
                status.value,
                scoring_plan,
                available_at,
                attempts,
            ),
        )
        conn.commit()
    return run_uuid


def _judged(run_uuid: str, verdicts: dict[str, dict]) -> dict:
    """One CLI result entry: judge_results keyed by runtime evaluator name."""
    return {
        "test_case_id": run_uuid,
        "metrics": {"judge_results": verdicts},
    }


def _invoker(results, *, error: str = "", returncode: int = 0, timed_out: bool = False):
    """A stand-in for the CLI. `results` may be a callable taking the dataset."""
    captured: dict = {}

    def invoke(config, dataset, **kwargs):
        captured["config"] = config
        captured["dataset"] = dataset
        captured["kwargs"] = kwargs
        payload = results(dataset) if callable(results) else results
        return ts.EvalOnlyCliResult(
            returncode=returncode, timed_out=timed_out, results=payload, error=error
        )

    invoke.captured = captured
    return invoke


def _status(run_uuid: str) -> str:
    return db.get_trace_eval_run(run_uuid)["status"]


# --- claim -----------------------------------------------------------------


def test_claim_takes_oldest_first_and_stamps_the_lease():
    org = _org()
    agent = _agent(org)
    ev = _evaluator(org)
    newer = _run(org, agent, _trace(org, agent), [ev], available_at=200)
    older = _run(org, agent, _trace(org, agent), [ev], available_at=100)

    claimed = db.claim_trace_eval_runs(now=1000, lease_seconds=600, batch_size=1)

    assert [row["uuid"] for row in claimed] == [older]
    assert claimed[0]["attempts"] == 1
    row = db.get_trace_eval_run(older)
    assert row["status"] == RunStatus.PROCESSING.value
    assert row["available_at"] == 1600
    assert _status(newer) == RunStatus.PENDING.value


def test_claim_ignores_runs_not_yet_available():
    org = _org()
    agent = _agent(org)
    run = _run(org, agent, _trace(org, agent), [_evaluator(org)], available_at=5000)

    assert db.claim_trace_eval_runs(now=1000, lease_seconds=600, batch_size=10) == []
    assert _status(run) == RunStatus.PENDING.value


def test_expired_lease_is_reclaimed_and_counts_another_attempt():
    org = _org()
    agent = _agent(org)
    run = _run(
        org,
        agent,
        _trace(org, agent),
        [_evaluator(org)],
        available_at=100,
        attempts=1,
        status=RunStatus.PROCESSING,
    )

    claimed = db.claim_trace_eval_runs(now=1000, lease_seconds=600, batch_size=10)

    assert [row["uuid"] for row in claimed] == [run]
    assert claimed[0]["attempts"] == 2


def test_two_claimers_never_receive_the_same_run():
    org = _org()
    agent = _agent(org)
    ev = _evaluator(org)
    runs = {_run(org, agent, _trace(org, agent), [ev]) for _ in range(4)}

    first = db.claim_trace_eval_runs(now=1000, lease_seconds=600, batch_size=2)
    second = db.claim_trace_eval_runs(now=1000, lease_seconds=600, batch_size=2)

    got = [row["uuid"] for row in first] + [row["uuid"] for row in second]
    assert sorted(got) == sorted(runs)
    assert len(set(got)) == 4


def test_claim_with_no_capacity_is_a_noop():
    org = _org()
    agent = _agent(org)
    run = _run(org, agent, _trace(org, agent), [_evaluator(org)])

    assert db.claim_trace_eval_runs(now=1000, lease_seconds=600, batch_size=0) == []
    assert _status(run) == RunStatus.PENDING.value


def test_boot_check_rejects_a_sqlite_without_returning(monkeypatch):
    monkeypatch.setattr(db.sqlite3, "sqlite_version_info", (3, 34, 0))
    monkeypatch.setattr(db.sqlite3, "sqlite_version", "3.34.0")
    with pytest.raises(RuntimeError, match="3.35"):
        db.assert_sqlite_returning_support()


# --- snapshot round-trip and hydration --------------------------------------


def test_ingested_plan_survives_the_round_trip_to_the_claim():
    """The writer's format and the engine's parser must not drift apart. A
    parser reading a stale key still passes against a hand-built fixture, so
    this goes through a really-ingested row."""
    org = _org()
    agent = _agent(org, interaction_type="general")
    evaluator_uuid, version_uuid = _evaluator(org, evaluator_type="llm-general")
    db.add_evaluator_to_agent(agent["uuid"], evaluator_uuid)
    db.create_trace_with_eval_run(
        org_uuid=org,
        agent=agent,
        input="Summarize the schedule.",
        output={"response": "Done."},
    )

    claimed = db.claim_trace_eval_runs(
        now=int(time.time()) + 5, lease_seconds=600, batch_size=10
    )

    assert len(claimed) == 1
    assert ts.parse_scoring_plan(claimed[0]["scoring_plan"]) == ts.ScoringPlan(
        evaluation_type="general",
        evaluators=[
            ts.ScoringPlanPin(
                evaluator_uuid=evaluator_uuid, evaluator_version_id=version_uuid
            )
        ],
    )


def test_hydration_reads_the_pinned_version_not_the_live_one():
    org = _org()
    evaluator_uuid, pinned = _evaluator(org)
    newer = db.create_evaluator_version(evaluator_uuid, "openai/gpt-5", "Judge harder.")
    db.set_evaluator_live_version(evaluator_uuid, newer["uuid"])

    hydrated = ts.hydrate_pinned_evaluators(
        [ts.ScoringPlanPin(evaluator_uuid=evaluator_uuid, evaluator_version_id=pinned)]
    )

    assert [ev["evaluator_version_id"] for ev in hydrated] == [pinned]
    assert hydrated[0]["system_prompt"] == "Judge it."
    assert hydrated[0]["judge_model"] == "openai/gpt-4.1"


def test_hydration_survives_an_evaluator_deleted_after_it_was_pinned():
    org = _org()
    evaluator_uuid, pinned = _evaluator(org, name="Gone but pinned")
    db.delete_evaluator(evaluator_uuid)

    hydrated = ts.hydrate_pinned_evaluators(
        [ts.ScoringPlanPin(evaluator_uuid=evaluator_uuid, evaluator_version_id=pinned)]
    )

    assert hydrated is not None
    assert hydrated[0]["name"] == "Gone but pinned"


@pytest.mark.parametrize("break_it", ["missing_version", "wrong_evaluator"])
def test_a_pin_that_no_longer_resolves_fails_the_run(break_it):
    org = _org()
    agent = _agent(org)
    trace = _trace(org, agent)
    evaluator_uuid, version_uuid = _evaluator(org)
    other_uuid, _ = _evaluator(org)
    pins = [
        (evaluator_uuid, str(uuid.uuid4()))
        if break_it == "missing_version"
        else (other_uuid, version_uuid)
    ]
    run = _run(org, agent, trace, pins)

    invoke = _invoker([])
    ts.claim_and_score_batch(now=1000, invoke=invoke)

    row = db.get_trace_eval_run(run)
    assert row["status"] == RunStatus.FAILED.value
    assert row["error"] == ts.CORRUPT_SNAPSHOT_ERROR
    assert "dataset" not in invoke.captured


def test_an_unparseable_snapshot_fails_the_run_without_invoking():
    org = _org()
    agent = _agent(org)
    run = _run(org, agent, _trace(org, agent), [], scoring_plan="{not json")

    invoke = _invoker([])
    ts.claim_and_score_batch(now=1000, invoke=invoke)

    row = db.get_trace_eval_run(run)
    assert row["status"] == RunStatus.FAILED.value
    assert row["error"] == ts.CORRUPT_SNAPSHOT_ERROR
    assert "dataset" not in invoke.captured


# --- settlement -------------------------------------------------------------


def test_binary_and_rating_verdicts_settle_in_their_own_types():
    org = _org()
    agent = _agent(org)
    trace = _trace(org, agent)
    binary_uuid, binary_version = _evaluator(org, name="Correctness")
    rating_uuid, rating_version = _rating_evaluator(org, name="Helpfulness")
    run = _run(org, agent, trace, [(binary_uuid, binary_version), (rating_uuid, rating_version)])

    ts.claim_and_score_batch(
        now=1000,
        invoke=_invoker(
            [
                _judged(
                    run,
                    {
                        "Correctness": {"match": True, "reasoning": "spot on"},
                        "Helpfulness": {"score": 4, "reasoning": "useful"},
                    },
                )
            ]
        ),
    )

    assert _status(run) == RunStatus.COMPLETED.value
    scores = {s["evaluator_uuid"]: s for s in db.get_trace_eval_scores(run)}
    assert scores[binary_uuid]["value"] == 1
    assert scores[binary_uuid]["output_type"] == "binary"
    assert scores[binary_uuid]["reasoning"] == "spot on"
    assert scores[binary_uuid]["evaluator_version_id"] == binary_version
    assert scores[rating_uuid]["value"] == 4
    assert scores[rating_uuid]["output_type"] == "rating"
    assert scores[rating_uuid]["trace_uuid"] == trace["uuid"]


def test_a_failed_binary_verdict_stores_zero_not_a_missing_row():
    org = _org()
    agent = _agent(org)
    evaluator_uuid, version_uuid = _evaluator(org, name="Correctness")
    run = _run(org, agent, _trace(org, agent), [(evaluator_uuid, version_uuid)])

    ts.claim_and_score_batch(
        now=1000,
        invoke=_invoker([_judged(run, {"Correctness": {"match": False, "reasoning": "no"}})]),
    )

    assert _status(run) == RunStatus.COMPLETED.value
    assert [s["value"] for s in db.get_trace_eval_scores(run)] == [0]


def test_a_general_run_sends_input_and_no_history():
    org = _org()
    agent = _agent(org, interaction_type="general")
    trace = _trace(org, agent, input="Summarize this.", output={"response": "Summary."})
    evaluator_uuid, version_uuid = _evaluator(
        org, name="Quality", evaluator_type="llm-general"
    )
    run = _run(
        org,
        agent,
        trace,
        [(evaluator_uuid, version_uuid)],
        evaluation_type="general",
    )

    invoke = _invoker(lambda ds: [_judged(run, {"Quality": {"match": True}})])
    ts.claim_and_score_batch(now=1000, invoke=invoke)

    item = invoke.captured["dataset"][0]
    assert item["test_case"]["id"] == run
    assert item["test_case"]["input"] == "Summarize this."
    assert "history" not in item["test_case"]
    assert item["test_case"]["evaluation"]["type"] == "general"
    assert item["output"] == {"response": "Summary."}
    assert _status(run) == RunStatus.COMPLETED.value


def test_a_conversation_run_sends_history_and_tool_calls():
    org = _org()
    agent = _agent(org)
    calls = [{"tool": "get_schedule", "arguments": {"weeks": 14}}]
    trace = _trace(
        org,
        agent,
        input=[{"role": "user", "content": "when?"}],
        output={"response": "", "tool_calls": calls},
    )
    evaluator_uuid, version_uuid = _evaluator(org, name="Correctness")
    run = _run(org, agent, trace, [(evaluator_uuid, version_uuid)])

    invoke = _invoker([_judged(run, {"Correctness": {"match": False}})])
    ts.claim_and_score_batch(now=1000, invoke=invoke)

    item = invoke.captured["dataset"][0]
    assert item["test_case"]["history"] == [{"role": "user", "content": "when?"}]
    assert "input" not in item["test_case"]
    assert item["output"] == {"response": "", "tool_calls": calls}


def test_both_modes_share_one_invocation_and_one_evaluator_definition():
    org = _org()
    conversation = _agent(org)
    general = _agent(org, interaction_type="general")
    shared_uuid, shared_version = _evaluator(org, name="Correctness")
    general_uuid, general_version = _evaluator(
        org, name="Quality", evaluator_type="llm-general"
    )
    first = _run(org, conversation, _trace(org, conversation), [(shared_uuid, shared_version)])
    second = _run(org, conversation, _trace(org, conversation), [(shared_uuid, shared_version)])
    third = _run(
        org,
        general,
        _trace(org, general, input="x", output={"response": "y"}),
        [(general_uuid, general_version)],
        evaluation_type="general",
    )

    invoke = _invoker(
        lambda ds: [
            _judged(first, {"Correctness": {"match": True}}),
            _judged(second, {"Correctness": {"match": True}}),
            _judged(third, {"Quality": {"match": True}}),
        ]
    )
    ts.claim_and_score_batch(now=1000, invoke=invoke)

    assert len(invoke.captured["dataset"]) == 3
    definitions = invoke.captured["config"]["evaluators"]
    assert sorted(ev["id"] for ev in definitions) == sorted([shared_uuid, general_uuid])
    assert all(
        _status(run) == RunStatus.COMPLETED.value for run in (first, second, third)
    )


def test_same_named_evaluators_from_two_orgs_stay_distinct_in_one_batch():
    """A display name is unique only within an org, and a claim batch spans
    orgs, so calibrate — which keys its output by name — needs the suffix."""
    first_org, second_org = _org(), _org()
    first_agent, second_agent = _agent(first_org), _agent(second_org)
    first_uuid, first_version = _evaluator(first_org, name="Correctness")
    second_uuid, second_version = _evaluator(second_org, name="Correctness")
    first_run = _run(
        first_org,
        first_agent,
        _trace(first_org, first_agent),
        [(first_uuid, first_version)],
        available_at=1,
    )
    second_run = _run(
        second_org,
        second_agent,
        _trace(second_org, second_agent),
        [(second_uuid, second_version)],
        available_at=2,
    )

    def results(dataset):
        by_run = {
            item["test_case"]["id"]: item["test_case"]["evaluation"]["criteria"][0]["name"]
            for item in dataset
        }
        return [
            _judged(first_run, {by_run[first_run]: {"match": True}}),
            _judged(second_run, {by_run[second_run]: {"match": False}}),
        ]

    invoke = _invoker(results)
    ts.claim_and_score_batch(now=1000, invoke=invoke)

    names = [ev["name"] for ev in invoke.captured["config"]["evaluators"]]
    assert len(set(names)) == 2
    assert [s["value"] for s in db.get_trace_eval_scores(first_run)] == [1]
    assert [s["value"] for s in db.get_trace_eval_scores(second_run)] == [0]


def test_results_map_by_id_so_a_reordered_file_still_lands():
    org = _org()
    agent = _agent(org)
    evaluator_uuid, version_uuid = _evaluator(org, name="Correctness")
    first = _run(org, agent, _trace(org, agent), [(evaluator_uuid, version_uuid)], available_at=1)
    second = _run(org, agent, _trace(org, agent), [(evaluator_uuid, version_uuid)], available_at=2)

    ts.claim_and_score_batch(
        now=1000,
        invoke=_invoker(
            [
                _judged(second, {"Correctness": {"match": False}}),
                _judged(first, {"Correctness": {"match": True}}),
            ]
        ),
    )

    assert [s["value"] for s in db.get_trace_eval_scores(first)] == [1]
    assert [s["value"] for s in db.get_trace_eval_scores(second)] == [0]


def test_a_result_covering_only_part_of_the_snapshot_is_not_settled():
    org = _org()
    agent = _agent(org)
    first_uuid, first_version = _evaluator(org, name="Correctness")
    second_uuid, second_version = _evaluator(org, name="Safety")
    run = _run(
        org,
        agent,
        _trace(org, agent),
        [(first_uuid, first_version), (second_uuid, second_version)],
    )

    ts.claim_and_score_batch(
        now=1000, invoke=_invoker([_judged(run, {"Correctness": {"match": True}})])
    )

    assert _status(run) == RunStatus.PENDING.value
    assert db.get_trace_eval_scores(run) == []


def test_a_partial_batch_settles_the_finished_runs_and_retries_the_rest():
    org = _org()
    agent = _agent(org)
    evaluator_uuid, version_uuid = _evaluator(org, name="Correctness")
    done = _run(org, agent, _trace(org, agent), [(evaluator_uuid, version_uuid)], available_at=1)
    cut = _run(org, agent, _trace(org, agent), [(evaluator_uuid, version_uuid)], available_at=2)

    ts.claim_and_score_batch(
        now=1000,
        invoke=_invoker(
            [_judged(done, {"Correctness": {"match": True}})],
            timed_out=True,
            returncode=-9,
            error="timed out",
        ),
        rng=random.Random(7),
    )

    assert _status(done) == RunStatus.COMPLETED.value
    unfinished = db.get_trace_eval_run(cut)
    assert unfinished["status"] == RunStatus.PENDING.value
    assert unfinished["error"] == "timed out"
    assert unfinished["available_at"] > 1000


def test_deferral_and_completion_are_stamped_after_the_call_not_before(monkeypatch):
    """A long invocation must not write a backoff instant that has already
    passed, so settlement re-reads the clock."""
    org = _org()
    agent = _agent(org)
    evaluator_uuid, version_uuid = _evaluator(org, name="Correctness")
    done = _run(org, agent, _trace(org, agent), [(evaluator_uuid, version_uuid)], available_at=1)
    cut = _run(org, agent, _trace(org, agent), [(evaluator_uuid, version_uuid)], available_at=2)
    claim_time = 1000
    after_invoke = claim_time + 25 * 60

    def slow(config, dataset, **kwargs):
        monkeypatch.setattr(ts.time, "time", lambda: after_invoke)
        return ts.EvalOnlyCliResult(
            returncode=1,
            timed_out=False,
            results=[_judged(done, {"Correctness": {"match": True}})],
            error="boom",
        )

    ts.claim_and_score_batch(now=claim_time, invoke=slow, rng=random.Random(7))

    assert db.get_trace_eval_run(done)["completed_at"] == after_invoke
    assert db.get_trace_eval_run(cut)["available_at"] > after_invoke


def test_an_invocation_that_raises_defers_every_run_in_the_batch():
    org = _org()
    agent = _agent(org)
    evaluator_uuid, version_uuid = _evaluator(org, name="Correctness")
    runs = [
        _run(org, agent, _trace(org, agent), [(evaluator_uuid, version_uuid)])
        for _ in range(2)
    ]

    def explode(config, dataset, **kwargs):
        raise RuntimeError("cli vanished")

    ts.claim_and_score_batch(now=1000, invoke=explode, rng=random.Random(3))

    for run in runs:
        row = db.get_trace_eval_run(run)
        assert row["status"] == RunStatus.PENDING.value
        assert row["error"] == "cli vanished"
        assert row["available_at"] > 1000


def test_a_settle_failure_defers_that_run_and_spares_the_rest_of_the_batch(
    monkeypatch,
):
    """A busy write inside one run's settle transaction must not raise through
    the batch: the failing run defers for a retry and the runs after it still
    settle, instead of the whole tail stranding in `processing` until the
    lease expires."""
    org = _org()
    agent = _agent(org)
    evaluator_uuid, version_uuid = _evaluator(org, name="Correctness")
    run_a = _run(org, agent, _trace(org, agent), [(evaluator_uuid, version_uuid)])
    run_b = _run(org, agent, _trace(org, agent), [(evaluator_uuid, version_uuid)])

    real_settle = db.settle_trace_eval_run_completed

    def flaky_settle(run_uuid, scores, *, now):
        if run_uuid == run_a:
            raise RuntimeError("database table is locked")
        return real_settle(run_uuid, scores, now=now)

    monkeypatch.setattr(db, "settle_trace_eval_run_completed", flaky_settle)
    ts.claim_and_score_batch(
        now=1000,
        invoke=_invoker(
            [
                _judged(run_a, {"Correctness": {"match": True, "reasoning": "ok"}}),
                _judged(run_b, {"Correctness": {"match": True, "reasoning": "ok"}}),
            ]
        ),
        rng=random.Random(5),
    )

    deferred = db.get_trace_eval_run(run_a)
    assert deferred["status"] == RunStatus.PENDING.value
    assert deferred["error"] == "database table is locked"
    assert deferred["available_at"] > 1000
    assert db.get_trace_eval_scores(run_a) == []
    assert _status(run_b) == RunStatus.COMPLETED.value
    assert [s["value"] for s in db.get_trace_eval_scores(run_b)] == [1]


def test_retries_stop_at_the_ceiling_and_the_run_is_buried():
    org = _org()
    agent = _agent(org)
    evaluator_uuid, version_uuid = _evaluator(org, name="Correctness")
    run = _run(
        org,
        agent,
        _trace(org, agent),
        [(evaluator_uuid, version_uuid)],
        attempts=ts.MAX_ATTEMPTS - 1,
    )

    ts.claim_and_score_batch(
        now=1000, invoke=_invoker([], error="judge exploded"), max_attempts=ts.MAX_ATTEMPTS
    )

    row = db.get_trace_eval_run(run)
    assert row["status"] == RunStatus.FAILED.value
    assert row["error"] == "judge exploded"
    assert row["completed_at"] is not None


def test_an_unreadable_verdict_defers_rather_than_hitting_the_value_constraint():
    """A null value would violate `trace_eval_scores.value NOT NULL`. Rejecting
    it at the mapping layer keeps that an ordinary retry instead of an
    IntegrityError raised inside the settle transaction."""
    org = _org()
    agent = _agent(org)
    evaluator_uuid, version_uuid = _rating_evaluator(org, name="Helpfulness")
    run = _run(org, agent, _trace(org, agent), [(evaluator_uuid, version_uuid)])

    ts.claim_and_score_batch(
        now=1000,
        invoke=_invoker([_judged(run, {"Helpfulness": {"score": None, "reasoning": "?"}})]),
        rng=random.Random(1),
    )

    assert _status(run) == RunStatus.PENDING.value
    assert db.get_trace_eval_scores(run) == []


# --- deletion ---------------------------------------------------------------


def test_a_trace_deleted_before_the_claim_is_skipped_without_a_judge_call():
    org = _org()
    agent = _agent(org)
    trace = _trace(org, agent)
    run = _run(org, agent, trace, [_evaluator(org)])
    db.soft_delete_traces(org, trace_ids=[trace["uuid"]])

    invoke = _invoker([])
    ts.claim_and_score_batch(now=1000, invoke=invoke)

    row = db.get_trace_eval_run(run)
    assert row["status"] == RunStatus.SKIPPED.value
    assert row["error"] == ts.TraceEvalSettleSkipReason.TRACE_DELETED.value
    assert "dataset" not in invoke.captured


def test_a_deleted_agent_is_skipped_with_its_own_reason():
    org = _org()
    agent = _agent(org)
    run = _run(org, agent, _trace(org, agent), [_evaluator(org)])
    db.delete_agent(agent["uuid"])

    ts.claim_and_score_batch(now=1000, invoke=_invoker([]))

    row = db.get_trace_eval_run(run)
    assert row["status"] == RunStatus.SKIPPED.value
    assert row["error"] == ts.TraceEvalSettleSkipReason.AGENT_DELETED.value


def test_a_trace_deleted_during_the_call_settles_skipped_and_stores_no_scores():
    org = _org()
    agent = _agent(org)
    trace = _trace(org, agent)
    evaluator_uuid, version_uuid = _evaluator(org, name="Correctness")
    run = _run(org, agent, trace, [(evaluator_uuid, version_uuid)])

    def delete_then_return(config, dataset, **kwargs):
        db.soft_delete_traces(org, trace_ids=[trace["uuid"]])
        return ts.EvalOnlyCliResult(
            returncode=0,
            timed_out=False,
            results=[_judged(run, {"Correctness": {"match": True}})],
        )

    ts.claim_and_score_batch(now=1000, invoke=delete_then_return)

    row = db.get_trace_eval_run(run)
    assert row["status"] == RunStatus.SKIPPED.value
    assert row["error"] == ts.TraceEvalSettleSkipReason.TRACE_DELETED.value
    assert db.get_trace_eval_scores(run) == []


def test_a_trace_deleted_during_a_failed_call_skips_instead_of_deferring():
    org = _org()
    agent = _agent(org)
    trace = _trace(org, agent)
    run = _run(org, agent, trace, [_evaluator(org)])

    def delete_then_fail(config, dataset, **kwargs):
        db.soft_delete_traces(org, trace_ids=[trace["uuid"]])
        return ts.EvalOnlyCliResult(
            returncode=1, timed_out=False, results=[], error="boom"
        )

    ts.claim_and_score_batch(now=1000, invoke=delete_then_fail, rng=random.Random(5))

    row = db.get_trace_eval_run(run)
    assert row["status"] == RunStatus.SKIPPED.value
    assert row["error"] == ts.TraceEvalSettleSkipReason.TRACE_DELETED.value


def test_a_trace_deleted_at_the_ceiling_skips_rather_than_failing():
    org = _org()
    agent = _agent(org)
    trace = _trace(org, agent)
    run = _run(org, agent, trace, [_evaluator(org)], attempts=ts.MAX_ATTEMPTS - 1)

    def delete_then_fail(config, dataset, **kwargs):
        db.soft_delete_traces(org, trace_ids=[trace["uuid"]])
        return ts.EvalOnlyCliResult(
            returncode=1, timed_out=False, results=[], error="boom"
        )

    ts.claim_and_score_batch(now=1000, invoke=delete_then_fail)

    row = db.get_trace_eval_run(run)
    assert row["status"] == RunStatus.SKIPPED.value
    assert row["error"] == ts.TraceEvalSettleSkipReason.TRACE_DELETED.value


# --- settlement guards ------------------------------------------------------


def test_only_the_first_settler_of_a_run_writes():
    org = _org()
    agent = _agent(org)
    trace = _trace(org, agent)
    evaluator_uuid, version_uuid = _evaluator(org)
    run = _run(org, agent, trace, [(evaluator_uuid, version_uuid)], status=RunStatus.PROCESSING)
    scores = [
        {
            "evaluator_uuid": evaluator_uuid,
            "evaluator_version_id": version_uuid,
            "value": 1,
            "output_type": "binary",
            "reasoning": "first",
        }
    ]

    assert db.settle_trace_eval_run_completed(run, scores, now=10) == "completed"
    late = [{**scores[0], "value": 0, "reasoning": "late"}]
    assert db.settle_trace_eval_run_completed(run, late, now=20) == "noop"

    stored = db.get_trace_eval_scores(run)
    assert [s["value"] for s in stored] == [1]
    assert stored[0]["reasoning"] == "first"


def test_a_retry_of_the_same_run_overwrites_its_own_score_rows():
    """Scores are keyed on the run, so a retry re-upserts rather than
    duplicating — and a rescore, being a different run, cannot collide."""
    org = _org()
    agent = _agent(org)
    trace = _trace(org, agent)
    evaluator_uuid, version_uuid = _evaluator(org)
    run = _run(org, agent, trace, [(evaluator_uuid, version_uuid)], status=RunStatus.PROCESSING)
    score = {
        "evaluator_uuid": evaluator_uuid,
        "evaluator_version_id": version_uuid,
        "value": 0,
        "output_type": "binary",
        "reasoning": "first pass",
    }

    db.settle_trace_eval_run_completed(run, [score], now=10)
    with db.get_db_connection() as conn:
        conn.execute(
            "UPDATE trace_eval_runs SET status = ? WHERE uuid = ?",
            (RunStatus.PROCESSING.value, run),
        )
        conn.commit()
    db.settle_trace_eval_run_completed(
        run, [{**score, "value": 1, "reasoning": "retry"}], now=20
    )

    stored = db.get_trace_eval_scores(run)
    assert len(stored) == 1
    assert stored[0]["value"] == 1
    assert stored[0]["reasoning"] == "retry"


def test_settling_a_run_nobody_claimed_is_a_noop():
    assert db.settle_trace_eval_run_completed(str(uuid.uuid4()), [], now=10) == "noop"
    assert not db.settle_trace_eval_run_terminal(
        str(uuid.uuid4()), status=RunStatus.FAILED, error="x", now=10
    )
    assert not db.defer_trace_eval_run(str(uuid.uuid4()), available_at=50, now=10)


def test_terminal_settlement_refuses_a_non_terminal_status():
    with pytest.raises(ValueError, match="failed or skipped"):
        db.settle_trace_eval_run_terminal(
            str(uuid.uuid4()), status=RunStatus.COMPLETED, error=None, now=10
        )


def test_settling_a_pending_run_is_refused():
    org = _org()
    agent = _agent(org)
    run = _run(org, agent, _trace(org, agent), [_evaluator(org)])

    assert db.settle_trace_eval_run_completed(run, [], now=10) == "noop"
    assert not db.defer_trace_eval_run(run, available_at=50, now=10)
    assert _status(run) == RunStatus.PENDING.value


# --- the CLI seam -----------------------------------------------------------


class _FakePopen:
    def __init__(self, *, returncode=0, hangs=False, stderr_text=""):
        self.pid = 4321
        self.returncode = None if hangs else returncode
        self._hangs = hangs
        self.killed = False
        self.waits = 0
        self.args = None
        self.kwargs = None
        self.stderr_text = stderr_text

    def wait(self, timeout=None):
        self.waits += 1
        if self._hangs and not self.killed:
            raise subprocess.TimeoutExpired("calibrate-agent", timeout or 0)
        return self.returncode

    def poll(self):
        return self.returncode

    def kill(self):
        self.killed = True
        self.returncode = -9


def _patch_popen(monkeypatch, proc, *, results=None, stderr=""):
    def fake_popen(cmd, **kwargs):
        proc.args = cmd
        proc.kwargs = kwargs
        output_dir = Path(cmd[cmd.index("-o") + 1])
        if results is not None:
            (output_dir / "results.json").write_text(json.dumps(results))
        if stderr:
            (output_dir / "stderr.log").write_text(stderr)
        return proc

    monkeypatch.setattr(ts.subprocess, "Popen", fake_popen)
    return proc


def test_the_cli_call_goes_through_the_seam_with_file_stdio_and_its_own_session(
    monkeypatch,
):
    proc = _patch_popen(monkeypatch, _FakePopen(), results=[{"test_case_id": "r-1"}])

    result = ts.invoke_eval_only_cli({"evaluators": []}, [{"test_case": {"id": "r-1"}}])

    assert result.returncode == 0 and not result.timed_out
    assert result.results == [{"test_case_id": "r-1"}]
    assert proc.args[1] == "llm" and "--eval-only" in proc.args
    assert proc.kwargs["start_new_session"] is True
    # Temp files, not pipes: a chatty run must not deadlock on a full buffer.
    assert proc.kwargs["stdout"] is not None and proc.kwargs["stderr"] is not None


def test_a_hung_cli_is_killed_and_its_partial_results_are_still_read(monkeypatch):
    proc = _patch_popen(
        monkeypatch, _FakePopen(hangs=True), results=[{"test_case_id": "r-1"}]
    )
    killed: list[int] = []
    monkeypatch.setattr(
        ts, "kill_process_group", lambda pid, job: (killed.append(pid), True)[1]
    )

    result = ts.invoke_eval_only_cli(
        {"evaluators": []}, [{"test_case": {"id": "r-1"}}], timeout_seconds=1
    )

    assert result.timed_out and result.returncode == -9
    assert result.results == [{"test_case_id": "r-1"}]
    assert killed == [proc.pid]
    assert "timed out" in result.error


def test_a_process_surviving_the_group_kill_is_killed_directly(monkeypatch):
    proc = _patch_popen(monkeypatch, _FakePopen(hangs=True))
    monkeypatch.setattr(ts, "kill_process_group", lambda pid, job: False)

    ts.invoke_eval_only_cli({"evaluators": []}, [], timeout_seconds=1)

    assert proc.killed


def test_a_nonzero_exit_reports_the_last_stderr_line(monkeypatch):
    _patch_popen(
        monkeypatch,
        _FakePopen(returncode=2),
        results=[],
        stderr="warming up\nprovider refused the key\n",
    )

    result = ts.invoke_eval_only_cli({"evaluators": []}, [])

    assert result.returncode == 2
    assert result.error == "provider refused the key"


def test_a_nonzero_exit_with_no_stderr_still_reports_the_code(monkeypatch):
    _patch_popen(monkeypatch, _FakePopen(returncode=3), results=[])

    assert "exited 3" in ts.invoke_eval_only_cli({"evaluators": []}, []).error


def test_the_temp_dir_is_kept_while_the_child_is_still_running(monkeypatch):
    """Deleting it out from under a live child would pull its output away."""
    proc = _FakePopen(hangs=True)
    _patch_popen(monkeypatch, proc)
    monkeypatch.setattr(ts, "kill_process_group", lambda pid, job: True)
    monkeypatch.setattr(proc, "kill", lambda: None)
    removed: list = []
    monkeypatch.setattr(ts.shutil, "rmtree", lambda p, **kw: removed.append(p))

    ts.invoke_eval_only_cli({"evaluators": []}, [], timeout_seconds=1)

    assert removed == []


def test_reaping_an_already_exited_process_does_nothing(monkeypatch):
    monkeypatch.setattr(
        ts,
        "kill_process_group",
        lambda pid, job: pytest.fail("should not kill an exited process"),
    )
    ts._reap_cli_process(_FakePopen(returncode=0))


def test_reaping_swallows_a_kill_that_races_the_exit(monkeypatch):
    proc = _FakePopen(hangs=True)

    def boom():
        raise OSError("already gone")

    monkeypatch.setattr(proc, "kill", boom)
    monkeypatch.setattr(ts, "kill_process_group", lambda pid, job: True)

    ts._reap_cli_process(proc)


def test_unreadable_stderr_does_not_mask_the_exit_code(monkeypatch):
    _patch_popen(monkeypatch, _FakePopen(returncode=4), results=[])
    original = Path.read_text

    def flaky(self, *args, **kwargs):
        if self.name == "stderr.log":
            raise OSError("gone")
        return original(self, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", flaky)

    assert "exited 4" in ts.invoke_eval_only_cli({"evaluators": []}, []).error


# --- driver -----------------------------------------------------------------


def test_an_empty_queue_scores_nothing():
    assert ts.claim_and_score_batch(now=1000, invoke=_invoker([])) == []
    ts.process_claimed_runs([], invoke=_invoker([]))


def test_a_process_that_dies_to_the_group_kill_is_not_killed_again(monkeypatch):
    proc = _FakePopen(hangs=True)
    _patch_popen(monkeypatch, proc)

    def group_kill(pid, job):
        proc.returncode = -15
        return True

    monkeypatch.setattr(ts, "kill_process_group", group_kill)
    monkeypatch.setattr(
        proc, "kill", lambda: pytest.fail("the group kill already reaped it")
    )

    assert ts.invoke_eval_only_cli({"evaluators": []}, [], timeout_seconds=1).timed_out


def test_a_trace_deleted_between_the_liveness_check_and_the_read_is_skipped(monkeypatch):
    """Two reads on two connections, so the row can vanish in between."""
    org = _org()
    agent = _agent(org)
    run = _run(org, agent, _trace(org, agent), [_evaluator(org)])
    monkeypatch.setattr(db, "get_trace", lambda *a, **kw: None)

    invoke = _invoker([])
    ts.claim_and_score_batch(now=1000, invoke=invoke)

    row = db.get_trace_eval_run(run)
    assert row["status"] == RunStatus.SKIPPED.value
    assert row["error"] == ts.TraceEvalSettleSkipReason.TRACE_DELETED.value
    assert "dataset" not in invoke.captured


def test_a_run_whose_preparation_raises_is_deferred_not_left_claimed(monkeypatch):
    """Left in `processing` it would be invisible to the attempt ceiling and
    reclaimed forever."""
    org = _org()
    agent = _agent(org)
    healthy_uuid, healthy_version = _evaluator(org, name="Correctness")
    broken = _run(org, agent, _trace(org, agent), [_evaluator(org)], available_at=1)
    healthy = _run(
        org, agent, _trace(org, agent), [(healthy_uuid, healthy_version)], available_at=2
    )
    real_get_trace = db.get_trace
    calls: list[str] = []

    def flaky(org_uuid, trace_uuid):
        calls.append(trace_uuid)
        if len(calls) == 1:
            raise ValueError("unreadable trace payload")
        return real_get_trace(org_uuid, trace_uuid)

    monkeypatch.setattr(db, "get_trace", flaky)

    ts.claim_and_score_batch(
        now=1000,
        invoke=_invoker(lambda ds: [_judged(healthy, {"Correctness": {"match": True}})]),
        rng=random.Random(2),
    )

    deferred = db.get_trace_eval_run(broken)
    assert deferred["status"] == RunStatus.PENDING.value
    assert deferred["error"] == "unreadable trace payload"
    assert deferred["available_at"] > 1000
    assert _status(healthy) == RunStatus.COMPLETED.value
