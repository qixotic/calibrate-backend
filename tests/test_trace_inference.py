"""The D4 inference ladder: how each trace gets judged."""

from __future__ import annotations

import pytest

from traces.inference import (
    SKIP_NO_EVALUATORS,
    SKIP_NOTHING_TO_JUDGE,
    TYPE_CONVERSATION,
    TYPE_RESPONSE,
    TYPE_TOOL_CALL,
    evaluators_for_type,
    infer_trace_type,
    group_evaluators_by_type,
    plan_batches,
)


def _ev(uuid: str, evaluator_type: str):
    return {"uuid": uuid, "name": f"e-{uuid}", "evaluator_type": evaluator_type}


def _trace(uuid="t1", turns=1, response="hi", tool_calls=None):
    output = {}
    if response is not None:
        output["response"] = response
    if tool_calls:
        output["tool_calls"] = tool_calls
    return {
        "uuid": uuid,
        "input": [{"role": "user", "content": f"m{i}"} for i in range(turns)],
        "output": output,
    }


CONV = [_ev("c1", "conversation")]
LLM = [_ev("l1", "llm")]
GENERAL = [_ev("g1", "llm-general")]


class TestLadder:
    def test_multi_turn_with_a_conversation_evaluator_is_a_conversation(self):
        by_type = group_evaluators_by_type(CONV + LLM)
        assert infer_trace_type(_trace(turns=3), by_type) == TYPE_CONVERSATION

    def test_multi_turn_without_one_falls_back_to_the_reply(self):
        """The documented fallback: judge the response rather than skip."""
        by_type = group_evaluators_by_type(LLM)
        assert infer_trace_type(_trace(turns=3), by_type) == TYPE_RESPONSE

    def test_single_turn_is_a_response_even_when_conversation_evaluators_exist(self):
        by_type = group_evaluators_by_type(CONV + LLM)
        assert infer_trace_type(_trace(turns=1), by_type) == TYPE_RESPONSE

    def test_tool_calls_without_reply_text_are_a_tool_call(self):
        by_type = group_evaluators_by_type(GENERAL)
        trace = _trace(response=None, tool_calls=[{"tool": "get_schedule"}])
        assert infer_trace_type(trace, by_type) == TYPE_TOOL_CALL

    def test_a_reply_alongside_tool_calls_is_still_a_response(self):
        """tool_call is the last rung, reached only when there is no prose."""
        by_type = group_evaluators_by_type(LLM + GENERAL)
        trace = _trace(response="here you go", tool_calls=[{"tool": "x"}])
        assert infer_trace_type(trace, by_type) == TYPE_RESPONSE

    @pytest.mark.parametrize("blank", ["", "   ", None])
    def test_blank_reply_and_no_tool_calls_is_unjudgeable(self, blank):
        by_type = group_evaluators_by_type(LLM)
        assert infer_trace_type(_trace(response=blank), by_type) is None


class TestBatching:
    def test_traces_group_by_inferred_type(self):
        traces = [
            _trace("conv", turns=4),
            _trace("resp", turns=1),
            _trace("tool", response=None, tool_calls=[{"tool": "x"}]),
        ]
        batches, skipped = plan_batches(traces, CONV + LLM + GENERAL)

        assert [t["uuid"] for t in batches[TYPE_CONVERSATION]] == ["conv"]
        assert [t["uuid"] for t in batches[TYPE_RESPONSE]] == ["resp"]
        assert [t["uuid"] for t in batches[TYPE_TOOL_CALL]] == ["tool"]
        assert skipped == []

    def test_a_type_with_no_matching_evaluator_is_skipped_with_a_reason(self):
        """Silently dropping these would look like the judge lost the trace."""
        traces = [_trace("tool", response=None, tool_calls=[{"tool": "x"}])]
        batches, skipped = plan_batches(traces, LLM)

        assert batches == {}
        assert skipped == [
            {
                "trace_uuid": "tool",
                "reason": SKIP_NO_EVALUATORS,
                "inferred_type": TYPE_TOOL_CALL,
            }
        ]

    def test_unjudgeable_traces_are_reported_separately(self):
        batches, skipped = plan_batches([_trace("empty", response="")], LLM)
        assert batches == {}
        assert skipped == [
            {"trace_uuid": "empty", "reason": SKIP_NOTHING_TO_JUDGE}
        ]

    def test_an_agent_with_no_evaluators_judges_nothing(self):
        batches, skipped = plan_batches([_trace("a"), _trace("b")], [])
        assert batches == {}
        assert {s["reason"] for s in skipped} == {SKIP_NO_EVALUATORS}


class TestEvaluatorSelection:
    def test_each_type_pulls_only_its_own_evaluators(self):
        linked = CONV + LLM + GENERAL
        assert evaluators_for_type(TYPE_CONVERSATION, linked) == CONV
        assert evaluators_for_type(TYPE_RESPONSE, linked) == LLM
        assert evaluators_for_type(TYPE_TOOL_CALL, linked) == GENERAL

    def test_evaluators_default_to_llm_when_untyped(self):
        assert group_evaluators_by_type([{"uuid": "x", "name": "x"}]) == {
            "llm": [{"uuid": "x", "name": "x"}]
        }
