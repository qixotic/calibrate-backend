"""Unit tests for trace-scoring eligibility / plan resolution."""

from __future__ import annotations

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
