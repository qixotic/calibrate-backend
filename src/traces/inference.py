"""Decide how each trace should be judged, and with which evaluators.

A trace records what an agent actually did, so unlike a test it carries no
expected answer to diff against. Every mode here therefore judges the stored
output as-is; nothing re-runs the agent.

The ladder is ordered, first match wins:

1. `conversation` when the trace has more than one turn of history AND the
   agent has a conversation evaluator to judge it with.
2. `response` when the reply has text. This also catches the multi-turn case
   from step 1 when no conversation evaluator is linked, so a rich trace still
   gets judged on its reply rather than skipped.
3. `tool_call` when nothing else matched, which in practice means the turn
   produced only tool calls and no prose.
"""

from typing import Any, Dict, List, Optional, Tuple

TYPE_CONVERSATION = "conversation"
TYPE_RESPONSE = "response"
TYPE_TOOL_CALL = "tool_call"

# Which `evaluator_type` judges each inferred trace type. `tool_call` uses
# `llm-general` because serialized tool calls are a standalone input/output
# pair, not a conversation.
EVALUATOR_TYPE_FOR = {
    TYPE_CONVERSATION: "conversation",
    TYPE_RESPONSE: "llm",
    TYPE_TOOL_CALL: "llm-general",
}

SKIP_NO_EVALUATORS = "no_evaluator_for_type"
SKIP_NOTHING_TO_JUDGE = "nothing_to_judge"


def group_evaluators_by_type(
    evaluators: List[Dict[str, Any]],
) -> Dict[str, List[Dict[str, Any]]]:
    """Bucket an agent's linked evaluators by their `evaluator_type`."""
    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for ev in evaluators:
        grouped.setdefault(ev.get("evaluator_type") or "llm", []).append(ev)
    return grouped


def _has_response_text(output: Dict[str, Any]) -> bool:
    response = (output or {}).get("response")
    return isinstance(response, str) and bool(response.strip())


def _has_tool_calls(output: Dict[str, Any]) -> bool:
    return bool((output or {}).get("tool_calls"))


def infer_trace_type(
    trace: Dict[str, Any], evaluators_by_type: Dict[str, List[Dict[str, Any]]]
) -> Optional[str]:
    """Apply the ladder. Returns None when the trace carries nothing judgeable."""
    turns = trace.get("input") or []
    output = trace.get("output") or {}

    if len(turns) > 1 and evaluators_by_type.get("conversation"):
        return TYPE_CONVERSATION
    if _has_response_text(output):
        return TYPE_RESPONSE
    if _has_tool_calls(output):
        return TYPE_TOOL_CALL
    return None


def plan_batches(
    traces: List[Dict[str, Any]], evaluators: List[Dict[str, Any]]
) -> Tuple[Dict[str, List[Dict[str, Any]]], List[Dict[str, str]]]:
    """Split traces into one judgeable batch per inferred type.

    Returns `(batches, skipped)`. A batch only survives when the agent actually
    has an evaluator of the matching type, so a trace inferred as `tool_call`
    against an agent with no `llm-general` evaluator is reported as skipped
    rather than silently dropped.
    """
    by_type = group_evaluators_by_type(evaluators)
    batches: Dict[str, List[Dict[str, Any]]] = {}
    skipped: List[Dict[str, str]] = []

    for trace in traces:
        inferred = infer_trace_type(trace, by_type)
        if inferred is None:
            skipped.append(
                {"trace_uuid": trace["uuid"], "reason": SKIP_NOTHING_TO_JUDGE}
            )
            continue
        if not by_type.get(EVALUATOR_TYPE_FOR[inferred]):
            skipped.append(
                {
                    "trace_uuid": trace["uuid"],
                    "reason": SKIP_NO_EVALUATORS,
                    "inferred_type": inferred,
                }
            )
            continue
        batches.setdefault(inferred, []).append(trace)

    return batches, skipped


def evaluators_for_type(
    inferred_type: str, evaluators: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    return group_evaluators_by_type(evaluators).get(
        EVALUATOR_TYPE_FOR[inferred_type], []
    )
