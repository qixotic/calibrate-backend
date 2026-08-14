"""Unit tests for the trace-eval store (src/traces/eval_store.py)."""

from __future__ import annotations

import uuid

from traces import eval_store, store

_AGENT = "11111111-1111-4111-8111-111111111111"


def _org() -> str:
    return str(uuid.uuid4())


def _seed(org: str, count: int, agent_id: str = _AGENT):
    """Create `count` live traces and return their uuids, oldest first."""
    uuids = []
    for i in range(count):
        row, _ = store.create_trace(
            org_uuid=org,
            agent_id=agent_id,
            message_id=f"m-{uuid.uuid4().hex[:10]}",
            conversation_id="conv-1",
            input=[{"role": "user", "content": f"q{i}"}],
            output={"response": f"a{i}"},
        )
        uuids.append(row["uuid"])
    return uuids


def _run(org: str, agent_id: str = _AGENT, status: str = "in_progress"):
    return eval_store.create_eval_run(
        org,
        agent_id,
        trigger=eval_store.TRIGGER_AUTO,
        inferred_type="response",
        status=status,
    )


class TestPendingSelection:
    def test_new_traces_are_pending(self):
        org = _org()
        seeded = _seed(org, 3)
        pending = eval_store.list_pending_traces(org, _AGENT, limit=10)
        assert {p["uuid"] for p in pending} == set(seeded)

    def test_pending_is_oldest_first_so_a_backlog_cannot_starve(self):
        org = _org()
        seeded = _seed(org, 5)
        pending = eval_store.list_pending_traces(org, _AGENT, limit=3)
        assert [p["uuid"] for p in pending] == seeded[:3]

    def test_pending_is_scoped_to_one_agent(self):
        org = _org()
        mine = _seed(org, 2, agent_id=_AGENT)
        other = "22222222-2222-4222-8222-222222222222"
        _seed(org, 2, agent_id=other)

        pending = eval_store.list_pending_traces(org, _AGENT, limit=10)
        assert {p["uuid"] for p in pending} == set(mine)

    def test_another_workspace_cannot_see_them(self):
        org = _org()
        _seed(org, 2)
        assert eval_store.list_pending_traces(_org(), _AGENT, limit=10) == []

    def test_agents_with_pending_traces_reports_the_pair(self):
        org = _org()
        _seed(org, 1)
        assert (org, _AGENT) in eval_store.agents_with_pending_traces(limit=1000)


class TestClaiming:
    def test_claiming_removes_traces_from_the_pending_set(self):
        org = _org()
        seeded = _seed(org, 3)
        run = _run(org)

        won = eval_store.claim_traces(org, run["uuid"], seeded)

        assert set(won) == set(seeded)
        assert eval_store.list_pending_traces(org, _AGENT, limit=10) == []

    def test_a_second_run_cannot_win_an_already_claimed_trace(self):
        """The concurrency guard: two schedulers racing the same backlog."""
        org = _org()
        seeded = _seed(org, 2)
        first, second = _run(org), _run(org)

        won_first = eval_store.claim_traces(org, first["uuid"], seeded)
        won_second = eval_store.claim_traces(org, second["uuid"], seeded)

        assert set(won_first) == set(seeded)
        assert won_second == []

    def test_claiming_is_scoped_to_the_workspace(self):
        org = _org()
        seeded = _seed(org, 1)
        run = _run(org)
        assert eval_store.claim_traces(_org(), run["uuid"], seeded) == []

    def test_claiming_nothing_is_a_no_op(self):
        assert eval_store.claim_traces(_org(), "r", []) == []


class TestCompletionAndRelease:
    def test_marking_evaluated_keeps_traces_out_of_the_pending_set(self):
        org = _org()
        seeded = _seed(org, 2)
        run = _run(org)
        eval_store.claim_traces(org, run["uuid"], seeded)

        assert eval_store.mark_traces_evaluated(run["uuid"], seeded) == 2

        eval_store.release_claims(run["uuid"])
        assert eval_store.list_pending_traces(org, _AGENT, limit=10) == []

    def test_releasing_returns_unjudged_traces_to_the_queue(self):
        org = _org()
        seeded = _seed(org, 3)
        run = _run(org)
        eval_store.claim_traces(org, run["uuid"], seeded)

        assert eval_store.release_claims(run["uuid"]) == 3
        assert len(eval_store.list_pending_traces(org, _AGENT, limit=10)) == 3

    def test_releasing_a_partly_finished_run_keeps_the_finished_traces(self):
        """A crashed run must not re-judge what already succeeded."""
        org = _org()
        seeded = _seed(org, 3)
        run = _run(org)
        eval_store.claim_traces(org, run["uuid"], seeded)
        eval_store.mark_traces_evaluated(run["uuid"], seeded[:2])

        assert eval_store.release_claims(run["uuid"]) == 1
        pending = eval_store.list_pending_traces(org, _AGENT, limit=10)
        assert [p["uuid"] for p in pending] == [seeded[2]]


class TestRuns:
    def test_create_read_and_patch(self):
        org = _org()
        run = _run(org, status="queued")

        assert eval_store.get_eval_run(org, run["uuid"])["status"] == "queued"

        patched = eval_store.update_eval_run(
            run["uuid"], status="done", trace_count=7, finished_at="now"
        )
        assert patched["status"] == "done"
        assert patched["trace_count"] == 7
        assert patched["finished_at"].endswith("Z")

    def test_another_workspace_cannot_read_a_run(self):
        run = _run(_org())
        assert eval_store.get_eval_run(_org(), run["uuid"]) is None

    def test_runs_list_is_newest_first_and_agent_scoped(self):
        org = _org()
        first = _run(org)
        second = _run(org)
        _run(org, agent_id="33333333-3333-4333-8333-333333333333")

        rows, total = eval_store.list_eval_runs(org, _AGENT, limit=10, offset=0)
        assert total == 2
        assert [r["uuid"] for r in rows] == [second["uuid"], first["uuid"]]

    def test_unfinished_runs_are_the_restart_recovery_set(self):
        org = _org()
        live = _run(org, status="in_progress")
        done = _run(org, status="done")

        uuids = {r["uuid"] for r in eval_store.unfinished_runs()}
        assert live["uuid"] in uuids
        assert done["uuid"] not in uuids


class TestResults:
    def _record(self, org, run_uuid, trace_uuid, **overrides):
        row = {
            "trace_uuid": trace_uuid,
            "evaluator_uuid": "ev-1",
            "evaluator_name": "Helpfulness",
            "output_type": "binary",
            "passed": True,
            "reasoning": "clear answer",
        }
        row.update(overrides)
        return eval_store.record_results(org, run_uuid, [row])

    def test_results_round_trip_with_the_snapshotted_evaluator_name(self):
        org = _org()
        trace = _seed(org, 1)[0]
        run = _run(org)
        self._record(org, run["uuid"], trace)

        results = eval_store.results_for_trace(org, trace)
        assert len(results) == 1
        assert results[0]["evaluator_name"] == "Helpfulness"
        assert results[0]["passed"] is True
        assert results[0]["reasoning"] == "clear answer"

    def test_rating_results_carry_a_score_and_its_scale(self):
        org = _org()
        trace = _seed(org, 1)[0]
        run = _run(org)
        self._record(
            org,
            run["uuid"],
            trace,
            output_type="rating",
            passed=None,
            score=4.0,
            scale_min=1.0,
            scale_max=5.0,
        )

        result = eval_store.results_for_trace(org, trace)[0]
        assert result["score"] == 4.0
        assert (result["scale_min"], result["scale_max"]) == (1.0, 5.0)
        assert result["passed"] is None

    def test_another_workspace_cannot_read_results(self):
        org = _org()
        trace = _seed(org, 1)[0]
        run = _run(org)
        self._record(org, run["uuid"], trace)
        assert eval_store.results_for_trace(_org(), trace) == []

    def test_summaries_batch_every_trace_on_the_page(self):
        org = _org()
        traces = _seed(org, 3)
        run = _run(org)
        self._record(org, run["uuid"], traces[0], passed=True)
        self._record(org, run["uuid"], traces[1], evaluator_uuid="ev-2", passed=False)

        summaries = eval_store.eval_summaries_for_traces(org, traces)

        assert summaries[traces[0]] == {"total": 1, "passed": 1}
        assert summaries[traces[1]] == {"total": 1, "passed": 0}
        # An unjudged trace is absent rather than zero-filled, so the badge can
        # tell "not evaluated" from "evaluated and failed".
        assert traces[2] not in summaries

    def test_summaries_of_nothing_is_empty(self):
        assert eval_store.eval_summaries_for_traces(_org(), []) == {}
