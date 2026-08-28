"""Unit tests for trace-scoring eligibility / plan resolution."""

from __future__ import annotations

import json
import random
import uuid

import db
import trace_scoring as ts


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
    result = ts.partition_trace_scoring_evaluators(
        "conversation", [ev], {live: _version(live)}
    )
    assert result.evaluation_type == "response"
    assert result.evaluator_type == "llm"
    assert [p.evaluator_uuid for p in result.eligible] == [ev["uuid"]]
    assert result.eligible[0].evaluator_version_id == live
    assert result.ineligible == []
    assert result.as_plan() == {
        "type": "response",
        "evaluators": [
            {"evaluator_uuid": ev["uuid"], "evaluator_version_id": live}
        ],
    }


def test_general_agent_maps_to_general_llm_general():
    live = str(uuid.uuid4())
    ev = _ev("llm-general", live=live, name="gen")
    result = ts.partition_trace_scoring_evaluators(
        "general", [ev], {live: _version(live)}
    )
    assert result.evaluation_type == "general"
    assert result.evaluator_type == "llm-general"
    assert result.eligible[0].evaluator_uuid == ev["uuid"]
    assert result.as_plan()["type"] == "general"


def test_mixed_evaluator_types_are_filtered_before_validation():
    """Wrong-type evaluators are ineligible even if they also declare variables
    or lack a live version — type is checked first so a mixed set never raises."""
    live = str(uuid.uuid4())
    missing = str(uuid.uuid4())
    vars_id = str(uuid.uuid4())
    clean = _ev("llm", live=live, name="clean")
    wrong_with_vars = _ev("llm-general", live=vars_id, name="general-vars")
    wrong_no_live = _ev("stt", live=None, name="stt")
    conversation = _ev("conversation", live=live, name="sim")
    result = ts.partition_trace_scoring_evaluators(
        "conversation",
        [clean, wrong_with_vars, wrong_no_live, conversation],
        {
            live: _version(live),
            vars_id: _version(vars_id, variables=[{"name": "criteria"}]),
            missing: _version(missing),
        },
    )
    assert [p.name for p in result.eligible] == ["clean"]
    by_name = {i.name: i.reason for i in result.ineligible}
    assert by_name == {
        "general-vars": ts.INELIGIBLE_REASON_WRONG_TYPE,
        "stt": ts.INELIGIBLE_REASON_WRONG_TYPE,
        "sim": ts.INELIGIBLE_REASON_WRONG_TYPE,
    }
    payload = result.ineligible_payload()
    assert {row["name"]: row["reason"] for row in payload} == by_name
    assert all("evaluator_uuid" in row for row in payload)


def test_no_live_version_disqualifies():
    ev_none = _ev("llm", live=None, name="none")
    ev_missing = _ev("llm", live=str(uuid.uuid4()), name="missing")
    result = ts.partition_trace_scoring_evaluators(
        "conversation", [ev_none, ev_missing], {}
    )
    assert result.eligible == []
    assert {i.name: i.reason for i in result.ineligible} == {
        "none": ts.INELIGIBLE_REASON_NO_LIVE_VERSION,
        "missing": ts.INELIGIBLE_REASON_NO_LIVE_VERSION,
    }
    assert result.as_plan() == {"skip": "no_usable_evaluators"}


def test_declares_variables_disqualifies():
    live = str(uuid.uuid4())
    ev = _ev("llm", live=live, name="criteria")
    result = ts.partition_trace_scoring_evaluators(
        "conversation",
        [ev],
        {live: _version(live, variables=[{"name": "criteria"}])},
    )
    assert result.eligible == []
    assert result.ineligible[0].reason == ts.INELIGIBLE_REASON_DECLARES_VARIABLES


def test_empty_variables_list_is_eligible():
    live = str(uuid.uuid4())
    ev = _ev("llm", live=live)
    result = ts.partition_trace_scoring_evaluators(
        "conversation", [ev], {live: _version(live, variables=[])}
    )
    assert len(result.eligible) == 1


def test_unsupported_interaction_type_skips_and_marks_wrong_type():
    live = str(uuid.uuid4())
    ev = _ev("llm", live=live, name="ok")
    result = ts.partition_trace_scoring_evaluators(
        "voice", [ev], {live: _version(live)}
    )
    assert result.evaluation_type is None
    assert result.eligible == []
    assert result.ineligible[0].reason == ts.INELIGIBLE_REASON_WRONG_TYPE
    assert result.as_plan() == {"skip": "unsupported_interaction_type"}


def test_empty_linked_set_is_not_usable():
    result = ts.partition_trace_scoring_evaluators("conversation", [], {})
    assert result.eligible == []
    assert result.ineligible == []
    assert result.as_plan() == {"skip": "no_usable_evaluators"}


def test_resolve_trace_scoring_loads_linked_evaluators():
    org = str(uuid.uuid4())
    agent_uuid = str(uuid.uuid4())
    with db.get_db_connection() as conn:
        conn.execute(
            "INSERT INTO agents (uuid, org_uuid, name, config, interaction_type) "
            "VALUES (?, ?, ?, ?, ?)",
            (agent_uuid, org, "score-me", "{}", "general"),
        )
        conn.commit()
    ev = db.create_evaluator(
        name=f"gen-{uuid.uuid4().hex[:6]}",
        evaluator_type="llm-general",
        org_uuid=org,
    )
    version = db.create_evaluator_version(ev, "openai/gpt-4.1", "Judge it.")
    db.set_evaluator_live_version(ev, version["uuid"])
    db.add_evaluator_to_agent(agent_uuid, ev)

    agent = db.get_agent(agent_uuid)
    result = ts.resolve_trace_scoring(agent)
    assert result.evaluation_type == "general"
    assert [p.evaluator_uuid for p in result.eligible] == [ev]
    assert ts.resolve_scoring_plan(agent)["type"] == "general"


def test_parse_criteria_snapshot_rejects_malformed():
    live = str(uuid.uuid4())
    ev = str(uuid.uuid4())
    good = {"type": "response", "evaluators": [{"evaluator_uuid": ev, "evaluator_version_id": live}]}
    assert ts.parse_criteria_snapshot(json.dumps(good)) == good
    assert ts.parse_criteria_snapshot(good) == good
    assert ts.parse_criteria_snapshot(None) is None
    assert ts.parse_criteria_snapshot("{") is None
    assert ts.parse_criteria_snapshot({"type": "tool_call", "evaluators": good["evaluators"]}) is None
    assert ts.parse_criteria_snapshot({"type": "response", "evaluators": []}) is None
    assert ts.parse_criteria_snapshot(
        {
            "type": "response",
            "evaluators": [
                {"evaluator_uuid": ev, "evaluator_version_id": live},
                {"evaluator_uuid": ev, "evaluator_version_id": live},
            ],
        }
    ) is None


def test_build_dataset_item_response_and_general_shapes():
    run_uuid = str(uuid.uuid4())
    refs = [{"name": "Correctness"}]
    response_item = ts.build_dataset_item(
        run_uuid,
        "response",
        {
            "input": [{"role": "user", "content": "hi"}],
            "output": {"response": "hello", "tool_calls": [{"tool": "x", "arguments": {}}]},
        },
        refs,
    )
    assert response_item["test_case"]["id"] == run_uuid
    assert response_item["test_case"]["history"][0]["content"] == "hi"
    assert "input" not in response_item["test_case"]
    assert response_item["output"]["tool_calls"][0]["tool"] == "x"

    general_item = ts.build_dataset_item(
        run_uuid,
        "general",
        {"input": "summarize", "output": {"response": "done", "tool_calls": []}},
        refs,
    )
    assert general_item["test_case"]["id"] == run_uuid
    assert general_item["test_case"]["input"] == "summarize"
    assert "history" not in general_item["test_case"]
    assert "tool_calls" not in general_item["output"]

    empty_response = ts.build_dataset_item(
        run_uuid, "response", {"input": "turns", "output": None}, refs
    )
    assert empty_response["test_case"]["history"] == []
    assert empty_response["output"]["tool_calls"] == []
    assert empty_response["output"]["response"] == ""

    general_coerced = ts.build_dataset_item(
        run_uuid, "general", {"input": 12, "output": {"response": None}}, refs
    )
    assert general_coerced["test_case"]["input"] == "12"
    assert general_coerced["output"]["response"] == ""


def test_index_cli_results_uses_embedded_id_and_flags_duplicates():
    run_a, run_b = str(uuid.uuid4()), str(uuid.uuid4())
    indexed = ts.index_cli_results(
        [
            {"test_case": {"id": run_a}, "metrics": {}},
            {"test_case_id": run_b, "test_case": {"id": "ignored"}, "metrics": {}},
            {"test_case_id": run_a, "metrics": {}},
        ]
    )
    assert indexed.get(run_a) is None
    assert indexed[run_b]["test_case_id"] == run_b


def test_map_item_scores_evaluator_id_first_name_fallback_and_rejects():
    ev_a, ev_b = str(uuid.uuid4()), str(uuid.uuid4())
    hydrated = {
        ev_a: {"output_type": "binary", "evaluator_version_id": "v-a"},
        ev_b: {"output_type": "rating", "evaluator_version_id": "v-b"},
    }
    name_to_uuid = {"Nice": ev_a, "Score": ev_b}
    pins = [ev_a, ev_b]
    ok = ts.map_item_scores(
        {
            "metrics": {
                "judge_results": {
                    "Nice": {"match": True, "reasoning": "yes", "evaluator_id": ev_a},
                    "other-name": {"score": 4, "evaluator_id": ev_b},
                }
            }
        },
        pin_uuids=pins,
        name_to_uuid=name_to_uuid,
        hydrated_by_uuid=hydrated,
    )
    assert ok is not None
    by_ev = {row["evaluator_uuid"]: row for row in ok}
    assert by_ev[ev_a] == {
        "evaluator_uuid": ev_a,
        "evaluator_version_id": "v-a",
        "match": 1,
        "score": None,
        "reasoning": "yes",
    }
    assert by_ev[ev_b]["match"] is None and by_ev[ev_b]["score"] == 4.0

    name_only = ts.map_item_scores(
        {
            "metrics": {
                "judge_results": {
                    "Nice": {"match": False},
                    "Score": {"score": 1},
                }
            }
        },
        pin_uuids=pins,
        name_to_uuid=name_to_uuid,
        hydrated_by_uuid=hydrated,
    )
    assert name_only is not None
    assert name_only[0]["match"] == 0

    assert (
        ts.map_item_scores(
            {
                "metrics": {
                    "judge_results": {
                        "Nice": {"match": True, "evaluator_id": ev_a},
                    }
                }
            },
            pin_uuids=pins,
            name_to_uuid=name_to_uuid,
            hydrated_by_uuid=hydrated,
        )
        is None
    )
    assert (
        ts.map_item_scores(
            {
                "metrics": {
                    "judge_results": {
                        "Nice": {"match": True, "evaluator_id": ev_a},
                        "Score": {"score": 1, "evaluator_id": ev_a},
                    }
                }
            },
            pin_uuids=pins,
            name_to_uuid=name_to_uuid,
            hydrated_by_uuid=hydrated,
        )
        is None
    )
    assert (
        ts.map_item_scores(
            {
                "metrics": {
                    "judge_results": {
                        "Nice": {"match": True, "evaluator_id": ev_a},
                        "Score": {"score": 1, "evaluator_id": str(uuid.uuid4())},
                    }
                }
            },
            pin_uuids=pins,
            name_to_uuid=name_to_uuid,
            hydrated_by_uuid=hydrated,
        )
        is None
    )


def test_backoff_available_at_includes_jitter():
    rng = random.Random(0)
    first = ts.backoff_available_at(1, 1000, rng)
    rng = random.Random(0)
    assert ts.backoff_available_at(1, 1000, rng) == first
    assert first > 1000
    later = ts.backoff_available_at(8, 1000, random.Random(1))
    assert later >= 1000 + ts._BACKOFF_CAP_SECONDS


def test_parse_criteria_snapshot_rejects_non_objects_and_incomplete_pins():
    ev = str(uuid.uuid4())
    assert ts.parse_criteria_snapshot([{"type": "response"}]) is None
    assert ts.parse_criteria_snapshot(
        {"type": "response", "evaluators": ["not-a-pin"]}
    ) is None
    assert ts.parse_criteria_snapshot(
        {"type": "response", "evaluators": [{"evaluator_uuid": ev}]}
    ) is None


def test_index_cli_results_ignores_non_list_and_entries_without_ids():
    assert ts.index_cli_results({"test_case_id": "x"}) == {}
    indexed = ts.index_cli_results(
        ["skip", {"metrics": {}}, {"test_case_id": "a", "metrics": {"ok": True}}]
    )
    assert "a" in indexed
    assert len(indexed) == 1


def test_map_item_scores_rejects_malformed_judgements():
    ev_a = str(uuid.uuid4())
    hydrated = {ev_a: {"output_type": "binary", "evaluator_version_id": "v-a"}}
    name_to_uuid = {"Nice": ev_a}
    pins = [ev_a]
    kwargs = dict(
        pin_uuids=pins, name_to_uuid=name_to_uuid, hydrated_by_uuid=hydrated
    )
    assert ts.map_item_scores({"metrics": "nope"}, **kwargs) is None
    assert ts.map_item_scores({"metrics": {"judge_results": {}}}, **kwargs) is None
    assert ts.map_item_scores(
        {"metrics": {"judge_results": {"Nice": "nope"}}}, **kwargs
    ) is None
    assert ts.map_item_scores(
        {"metrics": {"judge_results": {"Nice": {"match": True, "evaluator_id": ev_a}}}},
        pin_uuids=pins,
        name_to_uuid=name_to_uuid,
        hydrated_by_uuid={},
    ) is None
    assert ts.map_item_scores(
        {"metrics": {"judge_results": {"Nice": {"evaluator_id": ev_a}}}},
        **kwargs,
    ) is None
    assert ts.map_item_scores(
        {"metrics": {"judge_results": {"Nice": {"match": "yes", "evaluator_id": ev_a}}}},
        **kwargs,
    ) is None

    rating = {ev_a: {"output_type": "rating", "evaluator_version_id": "v-a"}}
    rating_kwargs = dict(
        pin_uuids=pins, name_to_uuid=name_to_uuid, hydrated_by_uuid=rating
    )
    assert ts.map_item_scores(
        {"metrics": {"judge_results": {"Nice": {"score": None, "evaluator_id": ev_a}}}},
        **rating_kwargs,
    ) is None
    assert ts.map_item_scores(
        {"metrics": {"judge_results": {"Nice": {"score": "bad", "evaluator_id": ev_a}}}},
        **rating_kwargs,
    ) is None
    coerced = ts.map_item_scores(
        {
            "metrics": {
                "judge_results": {
                    "Nice": {"score": 3, "reasoning": 12, "evaluator_id": ev_a}
                }
            }
        },
        **rating_kwargs,
    )
    assert coerced is not None
    assert coerced[0]["reasoning"] == "12"


def test_parse_results_json_best_effort(tmp_path):
    missing = tmp_path / "nope.json"
    assert ts.parse_results_json(missing) == []
    bad = tmp_path / "bad.json"
    bad.write_text("{", encoding="utf-8")
    assert ts.parse_results_json(bad) == []
    obj = tmp_path / "obj.json"
    obj.write_text("{}", encoding="utf-8")
    assert ts.parse_results_json(obj) == []
    ok = tmp_path / "ok.json"
    ok.write_text('[{"test_case_id": "r1"}]', encoding="utf-8")
    assert ts.parse_results_json(ok)[0]["test_case_id"] == "r1"


def test_truncate_error_caps_detail():
    huge = "x" * (ts.ERROR_MAX_CHARS + 50)
    truncated = ts._truncate_error(huge)
    assert truncated.endswith("...")
    assert len(truncated) == ts.ERROR_MAX_CHARS
    assert ts._truncate_error("   ") == "scoring failed"
