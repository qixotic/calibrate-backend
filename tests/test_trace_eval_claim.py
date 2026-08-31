"""Storage tests: claiming an open run, and the guards around settling it.

Schema constraints (CHECKs, NOT NULLs, active-run uniqueness) are pinned in
`test_db_trace_eval.py` and the ingest-time snapshot contract in
`test_db_traces.py`; these tests exercise the transactions on top of both.
"""

from __future__ import annotations

import json
import uuid
from dataclasses import asdict

import pytest

import db
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


def _status(run_uuid: str) -> str:
    return db.get_trace_eval_run(run_uuid)["status"]


# --- claim --------------------------------------------------------------

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


# --- settlement guards ---------------------------------------------------

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
