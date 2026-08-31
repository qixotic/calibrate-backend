"""Unit tests for trace-scoring eligibility, plan resolution, and the pure
engine helpers.

Anything touching the DB or the claim/settle transactions lives in
`test_trace_eval_claim.py`.
"""

from __future__ import annotations

import json
import random
import uuid
from dataclasses import asdict

import pytest

import db
import trace_scoring as ts


def test_trace_eval_run_status_is_the_closed_set():
    assert tuple(s.value for s in ts.TraceEvalRunStatus) == (
        "pending",
        "processing",
        "completed",
        "failed",
        "skipped",
    )
    assert ts.OPEN_TRACE_EVAL_RUN_STATUSES == (
        ts.TraceEvalRunStatus.PENDING,
        ts.TraceEvalRunStatus.PROCESSING,
    )


def _ev(evaluator_type, *, live=None, name="ev", ev_uuid=None):
    return {
        "uuid": ev_uuid or str(uuid.uuid4()),
        "name": name,
        "evaluator_type": evaluator_type,
        "live_version_id": live,
    }


def _version(version_uuid, *, variables=None):
    return {"uuid": version_uuid, "variables": variables}


def test_conversation_agent_maps_to_response_llm():
    live = str(uuid.uuid4())
    ev = _ev("llm", live=live, name="clean")
    result = ts.resolve_trace_scoring("conversation", [(ev, _version(live))])
    assert result.evaluation_type == "response"
    assert result.evaluator_type == "llm"
    assert [p.pin.evaluator_uuid for p in result.eligible] == [ev["uuid"]]
    assert result.eligible[0].pin.evaluator_version_id == live
    assert result.ineligible == []
    assert result.as_plan() == ts.ScoringPlan(
        evaluation_type="response",
        evaluators=[
            ts.ScoringPlanPin(
                evaluator_uuid=ev["uuid"], evaluator_version_id=live
            )
        ],
    )


def test_general_agent_maps_to_general_llm_general():
    live = str(uuid.uuid4())
    ev = _ev("llm-general", live=live, name="gen")
    result = ts.resolve_trace_scoring("general", [(ev, _version(live))])
    assert result.evaluation_type == "general"
    assert result.evaluator_type == "llm-general"
    assert result.eligible[0].pin.evaluator_uuid == ev["uuid"]
    assert result.as_plan().evaluation_type == "general"


def test_mixed_evaluator_types_are_filtered_before_validation():
    """Wrong-type evaluators are ineligible even if they also declare variables
    or lack a live version — type is checked first so a mixed set never raises."""
    live = str(uuid.uuid4())
    vars_id = str(uuid.uuid4())
    clean = _ev("llm", live=live, name="clean")
    wrong_with_vars = _ev("llm-general", live=vars_id, name="general-vars")
    wrong_no_live = _ev("stt", live=None, name="stt")
    conversation = _ev("conversation", live=live, name="sim")
    result = ts.resolve_trace_scoring(
        "conversation",
        [
            (clean, _version(live)),
            (wrong_with_vars, _version(vars_id, variables=[{"name": "criteria"}])),
            (wrong_no_live, None),
            (conversation, _version(live)),
        ],
    )
    assert [p.name for p in result.eligible] == ["clean"]
    by_name = {i.name: i.reason for i in result.ineligible}
    assert by_name == {
        "general-vars": ts.IneligibleReason.WRONG_TYPE,
        "stt": ts.IneligibleReason.WRONG_TYPE,
        "sim": ts.IneligibleReason.WRONG_TYPE,
    }


def test_no_live_version_disqualifies():
    ev_none = _ev("llm", live=None, name="none")
    ev_missing = _ev("llm", live=str(uuid.uuid4()), name="missing")
    result = ts.resolve_trace_scoring(
        "conversation", [(ev_none, None), (ev_missing, None)]
    )
    assert result.eligible == []
    assert {i.name: i.reason for i in result.ineligible} == {
        "none": ts.IneligibleReason.NO_LIVE_VERSION,
        "missing": ts.IneligibleReason.NO_LIVE_VERSION,
    }
    assert result.as_plan() == ts.ScoringPlanSkip(skip="no_usable_evaluators")


def test_declares_variables_disqualifies():
    live = str(uuid.uuid4())
    ev = _ev("llm", live=live, name="criteria")
    result = ts.resolve_trace_scoring(
        "conversation",
        [(ev, _version(live, variables=[{"name": "criteria"}]))],
    )
    assert result.eligible == []
    assert result.ineligible[0].reason == ts.IneligibleReason.DECLARES_VARIABLES


def test_empty_variables_list_is_eligible():
    live = str(uuid.uuid4())
    ev = _ev("llm", live=live)
    result = ts.resolve_trace_scoring(
        "conversation", [(ev, _version(live, variables=[]))]
    )
    assert len(result.eligible) == 1


def test_unsupported_interaction_type_skips_and_marks_wrong_type():
    live = str(uuid.uuid4())
    ev = _ev("llm", live=live, name="ok")
    result = ts.resolve_trace_scoring("voice", [(ev, _version(live))])
    assert result.evaluation_type is None
    assert result.eligible == []
    assert result.ineligible[0].reason == ts.IneligibleReason.WRONG_TYPE
    assert result.as_plan() == ts.ScoringPlanSkip(skip="unsupported_interaction_type")


def test_empty_linked_set_is_not_usable():
    result = ts.resolve_trace_scoring("conversation", [])
    assert result.eligible == []
    assert result.ineligible == []
    assert result.as_plan() == ts.ScoringPlanSkip(skip="no_usable_evaluators")


def test_resolve_live_evaluators_pairs_version_or_none():
    org = str(uuid.uuid4())
    agent_uuid = str(uuid.uuid4())
    with db.get_db_connection() as conn:
        conn.execute(
            "INSERT INTO agents (uuid, org_uuid, name, config, interaction_type) "
            "VALUES (?, ?, ?, ?, ?)",
            (agent_uuid, org, "score-me", "{}", "general"),
        )
        conn.commit()
    live_ev = db.create_evaluator(
        name=f"gen-{uuid.uuid4().hex[:6]}",
        evaluator_type="llm-general",
        org_uuid=org,
    )
    version = db.create_evaluator_version(live_ev, "openai/gpt-4.1", "Judge it.")
    db.set_evaluator_live_version(live_ev, version["uuid"])
    db.add_evaluator_to_agent(agent_uuid, live_ev)

    bare_ev = db.create_evaluator(
        name=f"bare-{uuid.uuid4().hex[:6]}",
        evaluator_type="llm-general",
        org_uuid=org,
    )
    db.add_evaluator_to_agent(agent_uuid, bare_ev)

    pairs = db.resolve_live_evaluators(agent_uuid)
    by_uuid = {ev["uuid"]: version_row for ev, version_row in pairs}
    assert by_uuid[live_ev]["uuid"] == version["uuid"]
    assert by_uuid[bare_ev] is None

    agent = db.get_agent(agent_uuid)
    result = ts.resolve_trace_scoring(agent["interaction_type"], pairs)
    assert result.evaluation_type == "general"
    assert [p.pin.evaluator_uuid for p in result.eligible] == [live_ev]
    assert result.as_plan() == ts.ScoringPlan(
        evaluation_type="general",
        evaluators=[
            ts.ScoringPlanPin(
                evaluator_uuid=live_ev, evaluator_version_id=version["uuid"]
            )
        ],
    )
    assert result.ineligible[0].evaluator_uuid == bare_ev
    assert result.ineligible[0].reason == ts.IneligibleReason.NO_LIVE_VERSION


def _plan(evaluation_type="response", pins=(("ev-1", "ver-1"),)) -> ts.ScoringPlan:
    return ts.ScoringPlan(
        evaluation_type=evaluation_type,
        evaluators=[
            ts.ScoringPlanPin(evaluator_uuid=e, evaluator_version_id=v) for e, v in pins
        ],
    )


def test_a_serialized_plan_parses_back_to_the_same_dataclass():
    plan = _plan(pins=(("ev-1", "ver-1"), ("ev-2", "ver-2")))
    assert ts.parse_scoring_plan(json.dumps(asdict(plan))) == plan


@pytest.mark.parametrize(
    "raw",
    [
        None,
        "",
        "{not json",
        123,
        json.dumps([]),
        json.dumps({"evaluation_type": "conversation", "evaluators": [{"a": 1}]}),
        json.dumps({"evaluators": [{"evaluator_uuid": "e", "evaluator_version_id": "v"}]}),
        json.dumps({"evaluation_type": "response"}),
        json.dumps({"evaluation_type": "response", "evaluators": []}),
        json.dumps({"evaluation_type": "response", "evaluators": "nope"}),
        json.dumps({"evaluation_type": "response", "evaluators": ["ev-1"]}),
        json.dumps({"evaluation_type": "response", "evaluators": [{"evaluator_uuid": "e"}]}),
        json.dumps(
            {"evaluation_type": "response", "evaluators": [{"evaluator_version_id": "v"}]}
        ),
        json.dumps(
            {
                "evaluation_type": "response",
                "evaluators": [
                    {"evaluator_uuid": "e", "evaluator_version_id": "v1"},
                    {"evaluator_uuid": "e", "evaluator_version_id": "v2"},
                ],
            }
        ),
    ],
)
def test_an_unusable_snapshot_never_parses_into_a_partial_plan(raw):
    assert ts.parse_scoring_plan(raw) is None


def test_results_are_indexed_by_id_and_a_repeated_id_drops_both():
    entries = [
        {"test_case_id": "run-1", "metrics": {}},
        {"test_case": {"id": "run-2"}, "metrics": {}},
        {"test_case_id": "run-3", "metrics": {}},
        {"test_case_id": "run-3", "metrics": {}},
    ]
    indexed = ts.index_cli_results(entries)
    assert sorted(indexed) == ["run-1", "run-2"]


def test_entries_with_no_id_and_a_non_list_file_are_ignored():
    assert ts.index_cli_results("not a list") == {}
    assert ts.index_cli_results([{"metrics": {}}, "junk", None]) == {}


def _hydrated(uuid_: str, output_type: str) -> dict:
    return {
        "uuid": uuid_,
        "output_type": output_type,
        "evaluator_version_id": f"{uuid_}-version",
    }


def test_a_verdict_maps_by_evaluator_id_ahead_of_its_runtime_name():
    entry = {
        "metrics": {
            "judge_results": {
                "renamed since the run": {
                    "evaluator_id": "ev-1",
                    "match": True,
                    "reasoning": "ok",
                }
            }
        }
    }
    scores = ts.map_item_scores(
        entry,
        pins=[ts.ScoringPlanPin(evaluator_uuid="ev-1", evaluator_version_id="ev-1-version")],
        name_to_uuid={},
        hydrated_by_uuid={"ev-1": _hydrated("ev-1", "binary")},
    )
    assert scores == [
        {
            "evaluator_uuid": "ev-1",
            "evaluator_version_id": "ev-1-version",
            "value": 1,
            "output_type": "binary",
            "reasoning": "ok",
        }
    ]


def test_a_verdict_falls_back_to_the_runtime_name_when_the_runner_sends_no_id():
    entry = {"metrics": {"judge_results": {"Correctness-ab12": {"score": 3}}}}
    scores = ts.map_item_scores(
        entry,
        pins=[ts.ScoringPlanPin(evaluator_uuid="ev-1", evaluator_version_id="ev-1-version")],
        name_to_uuid={"Correctness-ab12": "ev-1"},
        hydrated_by_uuid={"ev-1": _hydrated("ev-1", "rating")},
    )
    assert scores == [
        {
            "evaluator_uuid": "ev-1",
            "evaluator_version_id": "ev-1-version",
            "value": 3.0,
            "output_type": "rating",
            "reasoning": None,
        }
    ]


@pytest.mark.parametrize(
    "judge_results",
    [
        {"Unknown": {"match": True}},
        {"A": {"match": True}, "B": {"evaluator_id": "ev-1", "match": False}},
        {"A": "not a dict"},
        {"A": {"reasoning": "no verdict at all"}},
        {"A": {"match": "maybe"}},
        {},
    ],
)
def test_a_result_that_cannot_be_read_cleanly_leaves_the_run_unsettled(judge_results):
    entry = {"metrics": {"judge_results": judge_results}}
    assert (
        ts.map_item_scores(
            entry,
            pins=[
                ts.ScoringPlanPin(evaluator_uuid="ev-1", evaluator_version_id="ev-1-version")
            ],
            name_to_uuid={"A": "ev-1", "B": "ev-1"},
            hydrated_by_uuid={"ev-1": _hydrated("ev-1", "binary")},
        )
        is None
    )


@pytest.mark.parametrize("entry", [{}, {"metrics": "nope"}, {"metrics": {"judge_results": []}}])
def test_a_result_with_no_judge_block_leaves_the_run_unsettled(entry):
    assert (
        ts.map_item_scores(
            entry,
            pins=[
                ts.ScoringPlanPin(evaluator_uuid="ev-1", evaluator_version_id="ev-1-version")
            ],
            name_to_uuid={},
            hydrated_by_uuid={"ev-1": _hydrated("ev-1", "binary")},
        )
        is None
    )


def test_a_result_missing_one_of_two_pinned_evaluators_is_unsettleable():
    entry = {"metrics": {"judge_results": {"A": {"evaluator_id": "ev-1", "match": True}}}}
    assert (
        ts.map_item_scores(
            entry,
            pins=[
                ts.ScoringPlanPin(evaluator_uuid="ev-1", evaluator_version_id="ev-1-version"),
                ts.ScoringPlanPin(evaluator_uuid="ev-2", evaluator_version_id="ev-2-version"),
            ],
            name_to_uuid={},
            hydrated_by_uuid={
                "ev-1": _hydrated("ev-1", "binary"),
                "ev-2": _hydrated("ev-2", "binary"),
            },
        )
        is None
    )


@pytest.mark.parametrize(
    "judgement,output_type",
    [
        ({"score": None}, "rating"),
        ({"reasoning": "no score key"}, "rating"),
        ({"score": "high"}, "rating"),
        ({"score": True}, "rating"),
    ],
)
def test_a_rating_without_a_usable_number_is_rejected_before_the_insert(
    judgement, output_type
):
    """`trace_eval_scores.value` is NOT NULL, so a verdict the schema would
    reject has to fail here — inside the settle transaction it would be an
    IntegrityError, not a retryable run."""
    assert ts._typed_score(judgement, output_type) is None


def test_a_binary_verdict_normalizes_truthy_and_falsy_forms():
    assert ts._typed_score({"match": 1}, "binary")["value"] == 1
    assert ts._typed_score({"match": 0}, "binary")["value"] == 0


def test_non_string_reasoning_is_coerced_rather_than_dropped():
    assert ts._typed_score({"match": True, "reasoning": 7}, "binary")["reasoning"] == "7"


def test_backoff_grows_with_attempts_and_never_lands_on_one_instant():
    """Without jitter a whole-batch failure defers every run to the same
    moment, and the next claim reassembles the identical failing batch."""
    delays = {ts.backoff_available_at(1, 0, random.Random(seed)) for seed in range(20)}
    assert len(delays) > 1
    assert min(delays) >= ts._BACKOFF_BASE_SECONDS
    assert ts.backoff_available_at(6, 0, random.Random(1)) >= ts._BACKOFF_CAP_SECONDS


def test_a_long_error_is_truncated_and_an_empty_one_still_says_something():
    assert ts._truncate_error("") == "scoring failed"
    truncated = ts._truncate_error("x" * (ts.ERROR_MAX_CHARS + 500))
    assert len(truncated) == ts.ERROR_MAX_CHARS
    assert truncated.endswith("...")


def test_a_results_file_that_is_missing_or_mid_rewrite_reads_as_nothing_finished(tmp_path):
    assert ts.parse_results_json(tmp_path / "missing.json") == []
    partial = tmp_path / "results.json"
    partial.write_text('[{"test_case_id": "run-1"')
    assert ts.parse_results_json(partial) == []
    partial.write_text('{"not": "a list"}')
    assert ts.parse_results_json(partial) == []


def test_trace_evaluator_passed_matches_cli_rule():
    assert ts.trace_evaluator_passed("binary", 1, None) is True
    assert ts.trace_evaluator_passed("binary", 0, None) is False
    assert ts.trace_evaluator_passed("binary", None, None) is False
    assert ts.trace_evaluator_passed("rating", 5, 5) is True
    assert ts.trace_evaluator_passed("rating", 5.0, 5) is True
    assert ts.trace_evaluator_passed("rating", 4, 5) is False
    assert ts.trace_evaluator_passed("rating", 0, 5) is False
    assert ts.trace_evaluator_passed("rating", 5, None) is False
    assert ts.trace_evaluator_passed("rating", None, 5) is False
    assert ts.trace_evaluator_passed("rating", "high", 5) is False


def test_scale_bounds_read_stored_text_and_tolerate_junk():
    assert ts.scale_bounds_from_output_config(
        '{"scale": [{"value": 1, "name": "Low"}, {"value": 5, "name": "High"}]}'
    ) == (1, 5)
    assert ts.scale_bounds_from_output_config(
        {"scale": [{"value": 2}, {"value": 4}]}
    ) == (2, 4)
    assert ts.scale_bounds_from_output_config("not json") == (None, None)
    assert ts.scale_bounds_from_output_config(None) == (None, None)
    assert ts.scale_bounds_from_output_config({"scale": []}) == (None, None)
