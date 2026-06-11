"""Unit tests for pure helpers in routers/agent_tests.py."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest


def test_read_agent_test_results_json_missing(tmp_path):
    from routers.agent_tests import _read_agent_test_results_json

    assert _read_agent_test_results_json(None) is None
    assert _read_agent_test_results_json(tmp_path / "missing") is None


def test_read_agent_test_results_json_found(tmp_path):
    from routers.agent_tests import _read_agent_test_results_json

    sub = tmp_path / "sub"
    sub.mkdir()
    (sub / "results.json").write_text(json.dumps([{"x": 1}]))
    assert _read_agent_test_results_json(tmp_path) == [{"x": 1}]


def test_read_agent_test_results_json_malformed(tmp_path):
    from routers.agent_tests import _read_agent_test_results_json

    (tmp_path / "results.json").write_text("{not json")
    assert _read_agent_test_results_json(tmp_path) is None


def test_read_agent_test_metrics_json_missing(tmp_path):
    from routers.agent_tests import _read_agent_test_metrics_json

    assert _read_agent_test_metrics_json(None) is None
    assert _read_agent_test_metrics_json(tmp_path / "missing") is None


def test_read_agent_test_metrics_json_found(tmp_path):
    from routers.agent_tests import _read_agent_test_metrics_json

    (tmp_path / "metrics.json").write_text(json.dumps({"a": 1}))
    assert _read_agent_test_metrics_json(tmp_path) == {"a": 1}


def test_parse_agent_test_results():
    from routers.agent_tests import _parse_agent_test_results

    data = [
        {
            "test_case_id": "t1",
            # cost is nested in output; latency_ms is top-level on the result object
            "output": {"response": "hi", "tool_calls": None, "cost": 0.000942},
            "metrics": {"passed": True, "reasoning": "ok"},
            "test_case": {"name": "T1", "id": "t1"},
            "latency_ms": 842,
        }
    ]
    out = _parse_agent_test_results(data)
    assert out[0]["passed"] is True
    assert out[0]["latency_ms"] == 842
    assert out[0]["cost"] == 0.000942

    # Eval-only / older rows without latency or cost: keys present, values None
    no_metrics = _parse_agent_test_results(
        [{"output": {}, "metrics": {"passed": False}, "test_case": {"name": "T2"}}]
    )
    assert no_metrics[0]["latency_ms"] is None
    assert no_metrics[0]["cost"] is None

    assert _parse_agent_test_results(None) == []
    assert _parse_agent_test_results("not-a-list") == []


def test_parse_agent_test_results_preserves_tool_call_output():
    """Tool-call entries from agent-connection runs may carry an `output`
    (the tool's execution result); it must survive parsing verbatim."""
    from routers.agent_tests import _parse_agent_test_results

    data = [
        {
            "test_case_id": "t1",
            "output": {
                "response": None,
                "tool_calls": [
                    {
                        "tool": "get_weather",
                        "arguments": {"city": "NYC"},
                        "output": {"temp": 72},
                    }
                ],
            },
            "metrics": {"passed": True},
            "test_case": {"name": "T1", "id": "t1"},
        }
    ]
    out = _parse_agent_test_results(data)
    assert out[0]["output"]["tool_calls"][0]["output"] == {"temp": 72}


def test_test_case_result_accepts_fractional_latency():
    """External agent-connection tests self-report `metrics.latency_ms` verbatim,
    which may be a float (e.g. 1955.7) — the model must not reject it as a
    non-integer. Regression for a ValidationError when latency_ms was Optional[int]."""
    from routers.agent_tests import TestCaseResult

    r = TestCaseResult.model_validate({"name": "T1", "latency_ms": 1955.7, "cost": 0.0021})
    assert r.latency_ms == 1955.7
    # Integers still validate fine.
    assert TestCaseResult.model_validate({"latency_ms": 842}).latency_ms == 842


def test_perf_aggregate_means_accept_floats():
    """Aggregate blocks are Dict[str, Any], so a fractional `mean` (e.g.
    total_tokens averaged over runs) must validate, not be coerced/rejected."""
    from routers.agent_tests import TestRunStatusResponse, ModelResult

    resp = TestRunStatusResponse(
        task_id="t",
        status="done",
        total_tokens={"mean": 4378.5, "min": 4369, "max": 4387, "count": 2},
        latency_ms={"mean": 1955.7, "min": 1851.0, "max": 2060.4, "count": 2},
    )
    assert resp.total_tokens["mean"] == 4378.5
    assert resp.latency_ms["mean"] == 1955.7

    mr = ModelResult(
        model="m", message="ok",
        total_tokens={"mean": 4378.5, "min": 4369, "max": 4387, "count": 2},
    )
    assert mr.total_tokens["mean"] == 4378.5


def test_tool_call_output_model_surfaces_output():
    """The `output` field must be declared on ToolCallOutput, otherwise the
    response_model drops it on serialization."""
    from routers.agent_tests import ToolCallOutput, TestOutput

    tc = ToolCallOutput.model_validate(
        {"tool": "get_weather", "arguments": {"city": "NYC"}, "output": {"temp": 72}}
    )
    assert tc.output == {"temp": 72}

    # Optional: absent output serializes as None, not dropped.
    out = TestOutput.model_validate(
        {"tool_calls": [{"tool": "noop", "arguments": {}}]}
    )
    assert out.tool_calls[0].output is None


def test_merge_test_results_by_test_names():
    from routers.agent_tests import _merge_test_results_by_test_names

    completed = [{"name": "t1", "passed": True}]
    merged = _merge_test_results_by_test_names(["t1", "t2"], completed)
    assert merged[0]["passed"] is True
    assert merged[1]["name"] == "t2"
    assert merged[1]["passed"] is None

    # No test_names
    assert _merge_test_results_by_test_names([], completed) == []


def test_benchmark_queued_model_results():
    from routers.agent_tests import _benchmark_queued_model_results

    out = _benchmark_queued_model_results(["m1", "m2"], ["t1"])
    assert len(out) == 2
    assert out[0]["model"] == "m1"
    assert out[0]["success"] is None


def test_enrich_test_results_with_evaluators_none():
    from routers.agent_tests import _enrich_test_results_with_evaluators

    # No-op for None / empty
    _enrich_test_results_with_evaluators(None, {})
    _enrich_test_results_with_evaluators([], {})


def test_enrich_test_results_with_evaluators_dict_judge():
    """judge_results is the raw dict shape calibrate emits — converted to
    a minimal list (per-row data only; name/description live on the
    top-level evaluators[] block)."""
    from routers.agent_tests import _enrich_test_results_with_evaluators

    test_results = [
        {
            "test_case_id": "t1",
            "judge_results": {
                "Safety": {
                    "evaluator_id": "ev-1",
                    "reasoning": "ok",
                    "match": True,
                }
            },
        }
    ]
    snapshot = {
        "t1": [{"uuid": "ev-1", "name": "Safety", "variable_values": {"x": 1}}]
    }
    with patch(
        "db.get_evaluator",
        return_value={"uuid": "ev-1", "name": "Safety NEW", "description": "d"},
    ):
        _enrich_test_results_with_evaluators(test_results, snapshot)
    entry = test_results[0]["judge_results"][0]
    assert entry["evaluator_uuid"] == "ev-1"
    assert entry["match"] is True
    assert entry["reasoning"] == "ok"
    # Evaluator-level fields are promoted to the top-level evaluators[]
    # block; they MUST NOT be duplicated on each row.
    for k in ("name", "description", "scale_min", "scale_max"):
        assert k not in entry, f"{k} should not be on judge_results row"


def test_enrich_test_results_with_evaluators_list_judge():
    """Idempotent when judge_results is already a structured list. The list
    path also strips evaluator-level fields if a legacy row carries them."""
    from routers.agent_tests import _enrich_test_results_with_evaluators

    test_results = [
        {
            "test_case_id": "t1",
            "judge_results": [
                {
                    "evaluator_uuid": "ev-1",
                    "name": "Stale",
                    "description": "old",
                    "scale_min": 1,
                    "scale_max": 5,
                    "match": True,
                },
            ],
        }
    ]
    with patch(
        "db.get_evaluator",
        return_value={"uuid": "ev-1", "name": "Refreshed", "description": "d"},
    ):
        _enrich_test_results_with_evaluators(test_results, None)
    entry = test_results[0]["judge_results"][0]
    assert entry["evaluator_uuid"] == "ev-1"
    for k in ("name", "description", "scale_min", "scale_max"):
        assert k not in entry


def test_enrich_test_results_with_evaluators_value_name_binary():
    """Binary judge_results pick up `value_name` from the snapshot's rubric."""
    from routers.agent_tests import _enrich_test_results_with_evaluators

    test_results = [
        {
            "test_case_id": "t1",
            "judge_results": {
                "Safety": {
                    "evaluator_id": "ev-1",
                    "reasoning": "ok",
                    "match": True,
                }
            },
        }
    ]
    snapshot = {
        "t1": [
            {
                "uuid": "ev-1",
                "name": "Safety",
                "output_type": "binary",
                "output_config": {
                    "scale": [
                        {"value": True, "name": "Safe"},
                        {"value": False, "name": "Unsafe"},
                    ]
                },
            }
        ]
    }
    with patch(
        "db.get_evaluator",
        return_value={"uuid": "ev-1", "name": "Safety", "description": "d"},
    ):
        _enrich_test_results_with_evaluators(test_results, snapshot)
    entry = test_results[0]["judge_results"][0]
    assert entry["value_name"] == "Safe"


def test_enrich_test_results_with_evaluators_value_name_rating():
    """Rating judge_results resolve `value_name` via the numeric scale entry."""
    from routers.agent_tests import _enrich_test_results_with_evaluators

    test_results = [
        {
            "test_case_id": "t1",
            "judge_results": {
                "Helpfulness": {
                    "evaluator_id": "ev-2",
                    "reasoning": "great",
                    "score": 4,
                }
            },
        }
    ]
    snapshot = {
        "t1": [
            {
                "uuid": "ev-2",
                "name": "Helpfulness",
                "output_type": "rating",
                "scale_min": 1,
                "scale_max": 5,
                "output_config": {
                    "scale": [
                        {"value": 1, "name": "Terrible"},
                        {"value": 4, "name": "Good"},
                        {"value": 5, "name": "Excellent"},
                    ]
                },
            }
        ]
    }
    with patch(
        "db.get_evaluator",
        return_value={"uuid": "ev-2", "name": "Helpfulness", "description": "d"},
    ):
        _enrich_test_results_with_evaluators(test_results, snapshot)
    entry = test_results[0]["judge_results"][0]
    assert entry["value_name"] == "Good"


def test_enrich_test_results_with_evaluators_value_name_list_path():
    """List-shape judge_results (idempotent re-enrichment) also resolves
    `value_name` from the snapshot — matches the dict-path behavior so the
    field doesn't disappear on re-read."""
    from routers.agent_tests import _enrich_test_results_with_evaluators

    test_results = [
        {
            "test_case_id": "t1",
            "judge_results": [
                {"evaluator_uuid": "ev-1", "name": "Safety", "match": False},
            ],
        }
    ]
    snapshot = {
        "t1": [
            {
                "uuid": "ev-1",
                "name": "Safety",
                "output_type": "binary",
                "output_config": {
                    "scale": [
                        {"value": True, "name": "Safe"},
                        {"value": False, "name": "Unsafe"},
                    ]
                },
            }
        ]
    }
    with patch(
        "db.get_evaluator",
        return_value={"uuid": "ev-1", "name": "Safety", "description": "d"},
    ):
        _enrich_test_results_with_evaluators(test_results, snapshot)
    assert test_results[0]["judge_results"][0]["value_name"] == "Unsafe"


def test_build_evaluators_block_for_test_run_dedupes_and_enriches():
    """Block dedupes evaluators across test cases, pulls
    name/description from the live DB, and falls back to the snapshot's
    name when the DB lookup misses (or returns None)."""
    from routers.agent_tests import _build_evaluators_block_for_test_run

    snapshot = {
        "t1": [
            {
                "uuid": "ev-1",
                "name": "Safety",
                "output_type": "binary",
                "output_config": {
                    "scale": [
                        {"value": True, "name": "Safe"},
                        {"value": False, "name": "Unsafe"},
                    ]
                },
            },
            {
                "uuid": "ev-2",
                "name": "Helpfulness",
                "output_type": "rating",
                "scale_min": 1,
                "scale_max": 5,
                "version_number": 3,
                "output_config": {
                    "scale": [{"value": 5, "name": "Great"}]
                },
            },
        ],
        # Same evaluator appears again under another test — dedup expected.
        "t2": [
            {
                "uuid": "ev-1",
                "name": "Safety",
                "output_type": "binary",
                "output_config": {
                    "scale": [
                        {"value": True, "name": "Safe"},
                        {"value": False, "name": "Unsafe"},
                    ]
                },
            },
        ],
    }
    fake_db = {
        "ev-1": {"uuid": "ev-1", "name": "Safety LIVE", "description": "d1"},
        "ev-2": {"uuid": "ev-2", "name": "Helpfulness LIVE", "description": "d2"},
    }
    with patch("db.get_evaluator", side_effect=lambda u: fake_db.get(u)):
        block = _build_evaluators_block_for_test_run(snapshot)
    assert len(block) == 2
    by_uuid = {e["uuid"]: e for e in block}
    assert by_uuid["ev-1"]["name"] == "Safety LIVE"
    assert by_uuid["ev-1"]["output_type"] == "binary"
    assert by_uuid["ev-1"]["output_config"]["scale"][0]["name"] == "Safe"
    assert by_uuid["ev-2"]["scale_min"] == 1
    assert by_uuid["ev-2"]["scale_max"] == 5
    # version_number surfaces from the snapshot; absent ⇒ None.
    assert by_uuid["ev-2"]["version_number"] == 3
    assert by_uuid["ev-1"]["version_number"] is None


def test_build_evaluators_block_for_test_run_default_output_config():
    """Binary evaluators without a snapshotted output_config still get a
    Correct/Wrong scale via the shared default."""
    from routers.agent_tests import _build_evaluators_block_for_test_run

    snapshot = {
        "t1": [{"uuid": "ev-1", "name": "Anything", "output_type": "binary"}]
    }
    with patch("db.get_evaluator", return_value={"uuid": "ev-1", "name": "x", "description": None}):
        block = _build_evaluators_block_for_test_run(snapshot)
    assert block[0]["output_config"]["scale"] == [
        {"value": True, "name": "Correct"},
        {"value": False, "name": "Wrong"},
    ]


def test_build_evaluators_block_for_test_run_legacy_row_fallback():
    """Legacy run with no snapshot still emits a block entry for the
    evaluator referenced by judge_results so the FE doesn't see an unknown
    evaluator_uuid."""
    from routers.agent_tests import _build_evaluators_block_for_test_run

    test_results = [
        {
            "test_case_id": "t1",
            "judge_results": {
                "Safety": {"evaluator_id": "ev-legacy", "match": True},
            },
        }
    ]
    with patch(
        "db.get_evaluator",
        return_value={
            "uuid": "ev-legacy",
            "name": "Legacy",
            "description": "d",
            "output_type": "binary",
        },
    ):
        block = _build_evaluators_block_for_test_run(
            None, test_results=test_results
        )
    assert len(block) == 1
    assert block[0]["uuid"] == "ev-legacy"
    assert block[0]["name"] == "Legacy"
    assert block[0]["output_type"] == "binary"
    # Default binary scale was injected.
    assert block[0]["output_config"]["scale"][0]["name"] == "Correct"


def test_enrich_test_results_with_evaluators_dict_output_type_live_fallback():
    """Dict-path enrichment must fall back to the LIVE evaluator's
    output_type when the snapshot lacks it — otherwise value_name comes
    out null on legacy runs whose snapshot didn't capture output_type."""
    from routers.agent_tests import _enrich_test_results_with_evaluators

    test_results = [
        {
            "test_case_id": "t1",
            "judge_results": {
                "Safety": {
                    "evaluator_id": "ev-1",
                    "match": True,
                }
            },
        }
    ]
    # Snapshot has the evaluator but no output_type — simulates a legacy
    # capture that pre-dates the field.
    snapshot = {"t1": [{"uuid": "ev-1"}]}
    with patch(
        "db.get_evaluator",
        return_value={
            "uuid": "ev-1",
            "name": "Safety",
            "description": "d",
            "output_type": "binary",
        },
    ):
        _enrich_test_results_with_evaluators(test_results, snapshot)
    # Live evaluator's output_type kicks in and the binary fallback
    # resolves the label.
    assert test_results[0]["judge_results"][0]["value_name"] == "Correct"


def test_enrich_test_results_with_evaluators_value_name_legacy_fallback():
    """Legacy snapshot without `output_config` falls back to Correct/Wrong
    for binary so old runs still surface a label."""
    from routers.agent_tests import _enrich_test_results_with_evaluators

    test_results = [
        {
            "test_case_id": "t1",
            "judge_results": {
                "Safety": {
                    "evaluator_id": "ev-1",
                    "match": True,
                }
            },
        }
    ]
    snapshot = {
        "t1": [{"uuid": "ev-1", "name": "Safety", "output_type": "binary"}]
    }
    with patch(
        "db.get_evaluator",
        return_value={"uuid": "ev-1", "name": "Safety", "description": "d"},
    ):
        _enrich_test_results_with_evaluators(test_results, snapshot)
    assert test_results[0]["judge_results"][0]["value_name"] == "Correct"


def test_enrich_model_results_with_evaluators():
    from routers.agent_tests import _enrich_model_results_with_evaluators

    _enrich_model_results_with_evaluators(None, {})
    _enrich_model_results_with_evaluators([], {})
    # Happy path: nested test_results
    mr = [
        {
            "test_results": [
                {
                    "test_case_id": "t1",
                    "judge_results": {
                        "Safety": {
                            "evaluator_id": "ev-1",
                            "match": True,
                        }
                    },
                }
            ]
        }
    ]
    with patch(
        "db.get_evaluator",
        return_value={"uuid": "ev-1", "name": "Safety", "description": "d"},
    ):
        _enrich_model_results_with_evaluators(mr, {})
    assert mr[0]["test_results"][0]["judge_results"][0]["evaluator_uuid"] == "ev-1"


def test_build_evaluator_summary():
    from routers.agent_tests import _build_evaluator_summary

    assert _build_evaluator_summary(None) is None
    assert _build_evaluator_summary({"criteria": "not-a-dict"}) is None

    out = _build_evaluator_summary(
        {
            "criteria": {
                "Safety": {
                    "type": "binary",
                    "passed": 4,
                    "total": 5,
                    "evaluator_id": "ev-1",
                },
                "Quality": {
                    "type": "rating",
                    "mean": 3.5,
                    "evaluator_id": "ev-2",
                },
                "Skipped": {"type": "other"},
                "AlsoSkipped": "not-a-dict",
            }
        }
    )
    assert any(e["type"] == "binary" for e in out)
    assert any(e["type"] == "rating" for e in out)


def test_update_agent_test_intermediate_results_stores_perf_aggregates(tmp_path):
    """The aggregated latency_ms/cost/total_tokens blocks from metrics.json must
    land in the unit-test job results, and the per-case values must ride on each
    row (latency top-level, cost nested in output)."""
    from routers.agent_tests import _update_agent_test_intermediate_results
    from db import create_agent_test_job, get_agent_test_job

    job_id = create_agent_test_job(agent_id="agent-x", job_type="llm-unit-test")

    (tmp_path / "results.json").write_text(
        json.dumps(
            [
                {
                    "output": {"response": "hi", "cost": 0.000942},
                    "metrics": {"passed": True},
                    "test_case": {"name": "T1"},
                    "latency_ms": 842,
                }
            ]
        )
    )
    (tmp_path / "metrics.json").write_text(
        json.dumps(
            {
                "total": 1,
                "passed": 1,
                "latency_ms": {"mean": 842, "min": 842, "max": 842, "count": 1},
                "cost": {"mean": 0.000942, "min": 0.000942, "max": 0.000942, "count": 1},
                # Fractional mean: per-run tokens are ints but the aggregate mean
                # can be a float — it must round-trip, not be coerced to int.
                "total_tokens": {"mean": 4378.0, "min": 4369, "max": 4387, "count": 2},
            }
        )
    )

    _update_agent_test_intermediate_results(job_id, tmp_path, ["T1"])

    results = get_agent_test_job(job_id)["results"]
    assert results["latency_ms"] == {"mean": 842, "min": 842, "max": 842, "count": 1}
    assert results["cost"]["mean"] == 0.000942
    assert results["total_tokens"] == {"mean": 4378.0, "min": 4369, "max": 4387, "count": 2}
    assert results["test_results"][0]["latency_ms"] == 842
    assert results["test_results"][0]["cost"] == 0.000942


def test_update_benchmark_intermediate_results_stores_per_model_perf_aggregates(tmp_path):
    """Each completed model's aggregated latency_ms/cost/total_tokens ride on its
    model_results entry."""
    from routers.agent_tests import _update_benchmark_intermediate_results
    from db import create_agent_test_job, get_agent_test_job

    job_id = create_agent_test_job(agent_id="agent-y", job_type="llm-benchmark")

    model_dir = tmp_path / "gpt-4o"
    model_dir.mkdir()
    (model_dir / "results.json").write_text(
        json.dumps(
            [
                {
                    "output": {"response": "hi", "cost": 0.0021},
                    "metrics": {"passed": True},
                    "test_case": {"name": "T1"},
                    "latency_ms": 500,
                }
            ]
        )
    )
    (model_dir / "metrics.json").write_text(
        json.dumps(
            {
                "total": 1,
                "passed": 1,
                "latency_ms": {"mean": 500, "min": 500, "max": 500, "count": 1},
                "cost": {"mean": 0.0021, "min": 0.0021, "max": 0.0021, "count": 1},
                "total_tokens": {"mean": 4378, "min": 4369, "max": 4387, "count": 2},
            }
        )
    )

    _update_benchmark_intermediate_results(job_id, tmp_path, ["gpt-4o"], ["T1"])

    model_results = get_agent_test_job(job_id)["results"]["model_results"]
    completed = next(m for m in model_results if m["model"] == "gpt-4o")
    assert completed["latency_ms"] == {"mean": 500, "min": 500, "max": 500, "count": 1}
    assert completed["cost"]["mean"] == 0.0021
    assert completed["total_tokens"] == {"mean": 4378, "min": 4369, "max": 4387, "count": 2}
    assert completed["test_results"][0]["latency_ms"] == 500
    assert completed["test_results"][0]["cost"] == 0.0021


def test_calibrate_config_from_agent_test_job_stored():
    """If stored calibrate_config is on the job, it's reused."""
    from routers.agent_tests import _calibrate_config_from_agent_test_job

    with patch(
        "routers.agent_tests.get_agent_test_job",
        return_value={"details": {"calibrate_config": {"a": 1}}},
    ):
        out = _calibrate_config_from_agent_test_job("j", None, None)
    assert out == {"a": 1}


def test_pending_test_case_result_placeholder():
    from routers.agent_tests import _pending_test_case_result_placeholder

    out = _pending_test_case_result_placeholder("t1")
    assert out["name"] == "t1"
    assert out["passed"] is None
    assert out["latency_ms"] is None
    assert out["cost"] is None


def test_get_evaluator_cached_for_enrichment():
    from routers.agent_tests import _get_evaluator_cached_for_enrichment

    cache = {}
    with patch("db.get_evaluator", return_value={"uuid": "e", "name": "n"}):
        ev = _get_evaluator_cached_for_enrichment("e", cache)
    assert ev["name"] == "n"
    # second call doesn't refetch
    ev2 = _get_evaluator_cached_for_enrichment("e", cache)
    assert ev2 is ev
