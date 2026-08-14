"""Unit tests for the background trace-evaluation scheduler.

The real `eval_store` runs against the test traces database; only the agent
lookups in pense.db and `launch_trace_eval` are stubbed, so no test ever
spawns judging work.
"""

from __future__ import annotations

import asyncio
import uuid
from typing import Any, Dict, List

import pytest

from traces import eval_scheduler, eval_store, inference, store


def _uuid() -> str:
    return str(uuid.uuid4())


def _evaluator(evaluator_type: str) -> Dict[str, Any]:
    return {
        "uuid": _uuid(),
        "name": f"ev-{evaluator_type}",
        "evaluator_type": evaluator_type,
        "output_type": "binary",
    }


class _Scenario:
    """One isolated (org, agent) pair with stubbed pense.db lookups."""

    def __init__(self, monkeypatch: pytest.MonkeyPatch):
        self.org_uuid = _uuid()
        self.agent_id = _uuid()
        self.agent: Dict[str, Any] = {
            "uuid": self.agent_id,
            "org_uuid": self.org_uuid,
            "name": "traced-agent",
            "type": "connection",
            "config": {"auto_eval_enabled": True},
        }
        self.evaluators: List[Dict[str, Any]] = []
        self.calls: List[Dict[str, Any]] = []

        monkeypatch.setattr(
            eval_scheduler.db,
            "get_agent",
            lambda agent_id: self.agent if agent_id == self.agent_id else None,
        )
        monkeypatch.setattr(
            eval_scheduler.db,
            "get_evaluators_for_agent",
            lambda agent_id: (
                list(self.evaluators) if agent_id == self.agent_id else []
            ),
        )
        monkeypatch.setattr(eval_scheduler, "launch_trace_eval", self._fake_launch)
        # The traces DB is shared across the session, so headroom is measured
        # from whatever other tests left behind rather than from zero.
        self.baseline_live = len(eval_store.unfinished_runs())
        monkeypatch.setenv(
            "TRACE_EVAL_MAX_CONCURRENT_RUNS", str(self.baseline_live + 10)
        )

    def _fake_launch(self, **kwargs: Any):
        self.calls.append(kwargs)
        return _uuid(), "queued"

    def seed_trace(self, *, input: Any, output: Any) -> str:
        row, created = store.create_trace(
            org_uuid=self.org_uuid,
            agent_id=self.agent_id,
            message_id=f"m-{uuid.uuid4().hex}",
            conversation_id="conv-1",
            input=input,
            output=output,
        )
        assert created
        return row["uuid"]

    def seed_response_trace(self) -> str:
        return self.seed_trace(
            input=[{"role": "user", "content": "hi"}],
            output={"response": "hello there"},
        )

    def seed_conversation_trace(self) -> str:
        return self.seed_trace(
            input=[
                {"role": "system", "content": "You are helpful."},
                {"role": "user", "content": "hi"},
            ],
            output={"response": "hello there"},
        )

    def seed_tool_call_trace(self) -> str:
        return self.seed_trace(
            input=[{"role": "user", "content": "book it"}],
            output={"tool_calls": [{"tool": "book", "arguments": {"id": 1}}]},
        )

    def pending_uuids(self) -> List[str]:
        return [
            t["uuid"]
            for t in eval_store.list_pending_traces(
                self.org_uuid, self.agent_id, limit=100
            )
        ]


@pytest.fixture
def scenario(monkeypatch: pytest.MonkeyPatch) -> _Scenario:
    return _Scenario(monkeypatch)


def test_tick_skips_agent_with_auto_eval_disabled(scenario):
    scenario.agent["config"] = {"auto_eval_enabled": False}
    scenario.evaluators = [_evaluator("llm")]
    trace_uuid = scenario.seed_response_trace()

    assert eval_scheduler.run_one_tick() == 0
    assert scenario.calls == []
    assert trace_uuid in scenario.pending_uuids()


def test_tick_skips_agent_missing_the_auto_eval_flag(scenario):
    scenario.agent["config"] = {}
    scenario.evaluators = [_evaluator("llm")]
    scenario.seed_response_trace()

    assert eval_scheduler.run_one_tick() == 0
    assert scenario.calls == []


def test_tick_skips_agent_without_linked_evaluators(scenario):
    scenario.evaluators = []
    trace_uuid = scenario.seed_response_trace()

    assert eval_scheduler.run_one_tick() == 0
    assert scenario.calls == []
    assert trace_uuid in scenario.pending_uuids()


def test_tick_launches_one_run_per_inferred_type(scenario):
    conversation_evaluator = _evaluator("conversation")
    tool_call_evaluator = _evaluator("llm-general")
    scenario.evaluators = [conversation_evaluator, tool_call_evaluator]
    conversation_uuid = scenario.seed_conversation_trace()
    tool_call_uuid = scenario.seed_tool_call_trace()

    assert eval_scheduler.run_one_tick() == 2

    by_type = {call["inferred_type"]: call for call in scenario.calls}
    assert set(by_type) == {inference.TYPE_CONVERSATION, inference.TYPE_TOOL_CALL}

    conversation_call = by_type[inference.TYPE_CONVERSATION]
    assert [t["uuid"] for t in conversation_call["traces"]] == [conversation_uuid]
    assert conversation_call["evaluators"] == [conversation_evaluator]

    tool_call_call = by_type[inference.TYPE_TOOL_CALL]
    assert [t["uuid"] for t in tool_call_call["traces"]] == [tool_call_uuid]
    assert tool_call_call["evaluators"] == [tool_call_evaluator]

    for call in scenario.calls:
        assert call["org_uuid"] == scenario.org_uuid
        assert call["agent"] is scenario.agent
        assert call["trigger"] == eval_store.TRIGGER_AUTO


def test_tick_batches_no_more_than_the_configured_size(scenario, monkeypatch):
    monkeypatch.setenv("TRACE_EVAL_BATCH_SIZE", "1")
    scenario.evaluators = [_evaluator("llm")]
    scenario.seed_response_trace()
    scenario.seed_response_trace()

    assert eval_scheduler.run_one_tick() == 1
    assert len(scenario.calls[0]["traces"]) == 1


def test_tick_stops_at_the_concurrency_ceiling(scenario, monkeypatch):
    monkeypatch.setenv(
        "TRACE_EVAL_MAX_CONCURRENT_RUNS", str(scenario.baseline_live + 1)
    )
    scenario.evaluators = [_evaluator("conversation"), _evaluator("llm-general")]
    scenario.seed_conversation_trace()
    scenario.seed_tool_call_trace()

    assert eval_scheduler.run_one_tick() == 1
    assert len(scenario.calls) == 1


def test_tick_launches_nothing_when_already_at_the_ceiling(scenario, monkeypatch):
    monkeypatch.setenv("TRACE_EVAL_MAX_CONCURRENT_RUNS", "0")
    scenario.evaluators = [_evaluator("llm")]
    scenario.seed_response_trace()

    assert eval_scheduler.run_one_tick() == 0
    assert scenario.calls == []


def test_tick_survives_a_failing_agent(scenario, monkeypatch):
    scenario.evaluators = [_evaluator("llm")]
    trace_uuid = scenario.seed_response_trace()

    def boom(**kwargs):
        raise RuntimeError("launch exploded")

    monkeypatch.setattr(eval_scheduler, "launch_trace_eval", boom)

    assert eval_scheduler.run_one_tick() == 0
    assert trace_uuid in scenario.pending_uuids()


def test_recover_orphaned_runs_returns_traces_to_the_pending_set(scenario):
    trace_uuid = scenario.seed_response_trace()
    run = eval_store.create_eval_run(
        scenario.org_uuid,
        scenario.agent_id,
        trigger=eval_store.TRIGGER_AUTO,
        inferred_type=inference.TYPE_RESPONSE,
        status="in_progress",
        trace_count=1,
    )
    assert eval_store.claim_traces(scenario.org_uuid, run["uuid"], [trace_uuid]) == [
        trace_uuid
    ]
    assert trace_uuid not in scenario.pending_uuids()

    assert eval_scheduler.recover_orphaned_runs() >= 1

    recovered = eval_store.get_eval_run(scenario.org_uuid, run["uuid"])
    assert recovered["status"] == "failed"
    assert recovered["error"] == eval_scheduler.ORPHANED_RUN_ERROR
    assert recovered["finished_at"] is not None
    assert trace_uuid in scenario.pending_uuids()


def test_recover_orphaned_runs_leaves_judged_traces_alone(scenario):
    trace_uuid = scenario.seed_response_trace()
    run = eval_store.create_eval_run(
        scenario.org_uuid,
        scenario.agent_id,
        trigger=eval_store.TRIGGER_AUTO,
        inferred_type=inference.TYPE_RESPONSE,
        status="queued",
        trace_count=1,
    )
    eval_store.claim_traces(scenario.org_uuid, run["uuid"], [trace_uuid])
    eval_store.mark_traces_evaluated(run["uuid"], [trace_uuid])

    eval_scheduler.recover_orphaned_runs()

    assert trace_uuid not in scenario.pending_uuids()


def test_poll_loop_keeps_running_after_a_failing_tick(monkeypatch):
    monkeypatch.setenv("TRACE_EVAL_POLL_SECONDS", "7")
    ticks: List[int] = []

    def flaky_tick() -> int:
        ticks.append(len(ticks))
        if len(ticks) == 1:
            raise RuntimeError("tick exploded")
        return 1

    sleeps: List[float] = []

    async def fake_sleep(seconds: float) -> None:
        sleeps.append(seconds)
        if len(sleeps) >= 3:
            raise asyncio.CancelledError

    monkeypatch.setattr(eval_scheduler, "run_one_tick", flaky_tick)
    monkeypatch.setattr(eval_scheduler.asyncio, "sleep", fake_sleep)

    with pytest.raises(asyncio.CancelledError):
        asyncio.run(eval_scheduler.poll_loop())

    assert len(ticks) == 3
    assert sleeps == [7, 7, 7]


def test_unjudgeable_traces_are_retired_so_they_cannot_starve_the_queue(scenario):
    """Selection is oldest-first, so traces nothing can judge must leave the
    window. Left pending they would occupy it every tick and starve newer work,
    silently stopping automatic judging for the agent."""
    # Only a conversation evaluator is linked, but the traces are single-turn,
    # so the ladder infers `response` and finds nothing able to judge them.
    scenario.evaluators = [_evaluator("conversation")]
    for _ in range(3):
        scenario.seed_response_trace()

    assert len(scenario.pending_uuids()) == 3

    eval_scheduler.run_one_tick()

    assert scenario.pending_uuids() == []
    assert scenario.calls == []

    runs, _ = eval_store.list_eval_runs(
        scenario.org_uuid, scenario.agent_id, limit=10, offset=0
    )
    bookkeeping = [
        r for r in runs if r["inferred_type"] == eval_scheduler.UNJUDGEABLE_RUN_TYPE
    ]
    assert len(bookkeeping) == 1
    assert bookkeeping[0]["skipped_count"] == 3
    assert bookkeeping[0]["status"] == "done"


def test_retiring_does_not_touch_traces_that_can_be_judged(scenario):
    """A mixed batch still judges what it can."""
    scenario.evaluators = [_evaluator("llm")]
    judgeable = scenario.seed_response_trace()
    unjudgeable = scenario.seed_tool_call_trace()  # needs an llm-general evaluator

    eval_scheduler.run_one_tick()

    assert len(scenario.calls) == 1
    assert [t["uuid"] for t in scenario.calls[0]["traces"]] == [judgeable]
    # Only the unjudgeable one is retired. The judgeable trace is still pending
    # here solely because the stubbed launcher never claims it; the real
    # `launch_trace_eval` does that.
    assert unjudgeable not in scenario.pending_uuids()
    assert judgeable in scenario.pending_uuids()

