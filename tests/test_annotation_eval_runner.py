"""Unit tests for the annotation evaluator-run worker.

Covers the pure helpers (`_resolve_evaluator_dicts`, `build_dataset_for_task_type`,
`calibrate_command_for_task_type`, parsers, error extractor) and the
high-level dispatch (`start_annotation_eval_job`, `resume_annotation_eval_job`).
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

import annotation_eval_runner as runner


# ---------------------------------------------------------------------------
# trivial helpers
# ---------------------------------------------------------------------------


def test_utcnow_str_format():
    s = runner._utcnow_str()
    assert len(s) == len("2024-01-01 00:00:00")
    assert s[4] == "-" and s[7] == "-" and s[10] == " "


def test_output_dir_snapshot_missing_and_present(tmp_path):
    assert runner._output_dir_snapshot(tmp_path / "missing") == (0, 0)
    (tmp_path / "a.txt").write_text("hi")
    sub = tmp_path / "sub"
    sub.mkdir()
    (sub / "b.txt").write_text("world")
    count, size = runner._output_dir_snapshot(tmp_path)
    assert count == 2 and size > 0


# ---------------------------------------------------------------------------
# _resolve_evaluator_dicts
# ---------------------------------------------------------------------------


def _ev_record(**overrides):
    base = {
        "uuid": "ev-1",
        "name": "Safety",
        "live_version_id": "ver-1",
        "output_type": "binary",
        "kind": "single",
        "data_type": "text",
    }
    base.update(overrides)
    return base


def _version_record(**overrides):
    base = {
        "uuid": "ver-1",
        "evaluator_id": "ev-1",
        "judge_model": "gpt",
        "system_prompt": "p",
        "output_config": {},
        "variables": [],
    }
    base.update(overrides)
    return base


def test_resolve_evaluator_dicts_missing_id():
    with pytest.raises(runner.EvaluatorResolutionError):
        runner._resolve_evaluator_dicts([{"evaluator_id": None}], {"ev-1"})


def test_resolve_evaluator_dicts_not_linked():
    with pytest.raises(runner.EvaluatorResolutionError):
        runner._resolve_evaluator_dicts(
            [{"evaluator_id": "ev-1"}], set()  # not linked
        )


def test_resolve_evaluator_dicts_evaluator_missing():
    with patch("annotation_eval_runner.get_evaluator", return_value=None):
        with pytest.raises(runner.EvaluatorResolutionError):
            runner._resolve_evaluator_dicts(
                [{"evaluator_id": "ev-1"}], {"ev-1"}
            )


def test_resolve_evaluator_dicts_no_live_version():
    with patch(
        "annotation_eval_runner.get_evaluator",
        return_value=_ev_record(live_version_id=None),
    ):
        with pytest.raises(runner.EvaluatorResolutionError):
            runner._resolve_evaluator_dicts(
                [{"evaluator_id": "ev-1"}], {"ev-1"}
            )


def test_resolve_evaluator_dicts_version_mismatch():
    with patch(
        "annotation_eval_runner.get_evaluator", return_value=_ev_record()
    ), patch(
        "annotation_eval_runner.get_evaluator_version",
        return_value=_version_record(evaluator_id="other"),
    ):
        with pytest.raises(runner.EvaluatorResolutionError):
            runner._resolve_evaluator_dicts(
                [{"evaluator_id": "ev-1"}], {"ev-1"}
            )


def test_resolve_evaluator_dicts_missing_identity_field():
    with patch(
        "annotation_eval_runner.get_evaluator",
        return_value=_ev_record(output_type=None),
    ), patch(
        "annotation_eval_runner.get_evaluator_version", return_value=_version_record()
    ):
        with pytest.raises(runner.EvaluatorResolutionError):
            runner._resolve_evaluator_dicts(
                [{"evaluator_id": "ev-1"}], {"ev-1"}
            )


def test_resolve_evaluator_dicts_happy_path():
    with patch(
        "annotation_eval_runner.get_evaluator", return_value=_ev_record()
    ), patch(
        "annotation_eval_runner.get_evaluator_version", return_value=_version_record()
    ):
        out = runner._resolve_evaluator_dicts(
            [{"evaluator_id": "ev-1"}], {"ev-1"}
        )
    assert out[0]["uuid"] == "ev-1"
    assert out[0]["_evaluator_version_id"] == "ver-1"


def test_dedupe_evaluator_names():
    evs = [
        {"uuid": "aaaaaaaa11", "name": "Safety"},
        {"uuid": "bbbbbbbb22", "name": "Safety"},
        {"uuid": "cccccccc33", "name": "Other"},
    ]
    runner._dedupe_evaluator_names(evs)
    assert evs[0]["name"] == "Safety"
    assert evs[1]["name"].startswith("Safety-")
    assert evs[2]["name"] == "Other"


# ---------------------------------------------------------------------------
# Dataset builders
# ---------------------------------------------------------------------------


def test_build_stt_dataset():
    items = [
        {
            "uuid": "i1",
            "payload": {
                "predicted_transcript": "pred",
                "reference_transcript": "ref",
            },
        }
    ]
    out = runner._build_stt_dataset(items)
    assert out == [{"id": "i1", "gt": "ref", "pred": "pred"}]


def test_build_stt_dataset_bad_payload():
    with pytest.raises(runner.DatasetBuildError):
        runner._build_stt_dataset([{"uuid": "i1", "payload": "not-a-dict"}])
    with pytest.raises(runner.DatasetBuildError):
        runner._build_stt_dataset([{"uuid": "i1", "payload": {}}])


def test_build_llm_dataset_no_evaluators():
    with pytest.raises(runner.DatasetBuildError):
        runner._build_llm_dataset([], [])


def test_build_llm_dataset_bad_payload():
    evs = [{"uuid": "e1", "name": "judge"}]
    with pytest.raises(runner.DatasetBuildError):
        runner._build_llm_dataset(
            [{"uuid": "i1", "payload": {"chat_history": "not-a-list"}}],
            evs,
        )
    with pytest.raises(runner.DatasetBuildError):
        runner._build_llm_dataset(
            [
                {
                    "uuid": "i1",
                    "payload": {
                        "chat_history": [],
                        "agent_response": "r",
                        "tool_calls": "not-a-list",
                    },
                }
            ],
            evs,
        )
    with pytest.raises(runner.DatasetBuildError):
        runner._build_llm_dataset(
            [
                {
                    "uuid": "i1",
                    "payload": {
                        "chat_history": [],
                        "agent_response": "r",
                        "evaluator_variables": "not-dict",
                    },
                }
            ],
            evs,
        )


def test_build_llm_dataset_happy():
    evs = [{"uuid": "e1", "name": "judge"}]
    out = runner._build_llm_dataset(
        [
            {
                "uuid": "i1",
                "payload": {
                    "chat_history": [{"role": "user", "content": "hi"}],
                    "agent_response": "hi back",
                    "evaluator_variables": {"e1": {"x": 1}},
                },
            }
        ],
        evs,
    )
    assert out[0]["test_case"]["id"] == "i1"
    assert out[0]["test_case"]["evaluation"]["criteria"][0]["arguments"] == {"x": 1}


def test_build_llm_general_dataset_no_evaluators():
    with pytest.raises(runner.DatasetBuildError):
        runner._build_llm_general_dataset([], [])


def test_build_llm_general_dataset_bad_payload():
    evs = [{"uuid": "e1", "name": "judge"}]
    # Missing `input`.
    with pytest.raises(runner.DatasetBuildError):
        runner._build_llm_general_dataset(
            [{"uuid": "i1", "payload": {"output": "o"}}], evs
        )
    # Missing `output`.
    with pytest.raises(runner.DatasetBuildError):
        runner._build_llm_general_dataset(
            [{"uuid": "i1", "payload": {"input": "i"}}], evs
        )
    # Bad evaluator_variables.
    with pytest.raises(runner.DatasetBuildError):
        runner._build_llm_general_dataset(
            [
                {
                    "uuid": "i1",
                    "payload": {
                        "input": "i",
                        "output": "o",
                        "evaluator_variables": "not-dict",
                    },
                }
            ],
            evs,
        )


def test_build_llm_general_dataset_happy():
    evs = [{"uuid": "e1", "name": "judge"}]
    out = runner._build_llm_general_dataset(
        [
            {
                "uuid": "i1",
                "payload": {"input": "summarize this", "output": "a summary"},
            }
        ],
        evs,
    )
    # Flat `calibrate general` shape: {id, input, output}. No vars → no arguments.
    assert out == [
        {"id": "i1", "input": "summarize this", "output": "a summary"}
    ]


def test_build_llm_general_dataset_with_arguments():
    """Per-item `evaluator_variables` (same contract as the llm task type) are
    keyed by evaluator NAME in the per-row `arguments` map — each evaluator gets
    its own box, exactly like the llm path (no shared bag, no collision)."""
    evs = [{"uuid": "e1", "name": "judge1"}, {"uuid": "e2", "name": "judge2"}]
    out = runner._build_llm_general_dataset(
        [
            {
                "uuid": "i1",
                "payload": {
                    "input": "in",
                    "output": "out",
                    "evaluator_variables": {
                        # same {{var}} name for both — must NOT collide
                        "e1": {"criteria": "be concise"},
                        "e2": {"criteria": "be accurate"},
                        # not in this run → ignored
                        "e9": {"ignored": "x"},
                    },
                },
            }
        ],
        evs,
    )
    assert out[0]["id"] == "i1"
    assert out[0]["arguments"] == {
        "judge1": {"criteria": "be concise"},
        "judge2": {"criteria": "be accurate"},
    }


def test_build_dataset_dispatch_llm_general():
    out = runner.build_dataset_for_task_type(
        "llm-general",
        [{"uuid": "i1", "payload": {"input": "i", "output": "o"}}],
        [{"uuid": "e1", "name": "judge"}],
    )
    assert out == [{"id": "i1", "input": "i", "output": "o"}]


def test_build_simulation_dataset():
    out = runner._build_simulation_dataset(
        [{"uuid": "i1", "payload": {"transcript": [{"role": "user", "content": "x"}]}}]
    )
    assert out[0]["name"] == "i1"


def test_build_simulation_dataset_bad_payload():
    with pytest.raises(runner.DatasetBuildError):
        runner._build_simulation_dataset(
            [{"uuid": "i1", "payload": {"transcript": []}}]
        )
    with pytest.raises(runner.DatasetBuildError):
        runner._build_simulation_dataset(
            [{"uuid": "i1", "payload": {}}]
        )


def test_build_dataset_dispatch_unknown_task_type():
    with pytest.raises(runner.DatasetBuildError):
        runner.build_dataset_for_task_type("unknown", [], [])


def test_build_dataset_dispatch_conversation():
    # The `conversation` task type routes to the simulation-shaped dataset builder.
    out = runner.build_dataset_for_task_type(
        "conversation",
        [{"uuid": "i1", "payload": {"transcript": [{"role": "user", "content": "x"}]}}],
        [],
    )
    assert out[0]["name"] == "i1"
    assert out[0]["conversation_history"] == [{"role": "user", "content": "x"}]


# ---------------------------------------------------------------------------
# calibrate_command_for_task_type
# ---------------------------------------------------------------------------


def test_calibrate_command_for_task_type():
    p = Path("/tmp")
    out_stt = runner.calibrate_command_for_task_type("stt", p, p, p)
    assert out_stt[:3] == ["calibrate", "stt", "--eval-only"]
    out_llm = runner.calibrate_command_for_task_type("llm", p, p, p)
    assert out_llm[:2] == ["calibrate", "llm"]
    # llm-general uses the dedicated `calibrate general` command (no --eval-only).
    out_llm_general = runner.calibrate_command_for_task_type("llm-general", p, p, p)
    assert out_llm_general[:2] == ["calibrate", "general"]
    assert "--eval-only" not in out_llm_general
    out_sim = runner.calibrate_command_for_task_type("conversation", p, p, p)
    assert out_sim[:2] == ["calibrate", "simulations"]
    with pytest.raises(runner.DatasetBuildError):
        runner.calibrate_command_for_task_type("unknown", p, p, p)


# ---------------------------------------------------------------------------
# Output reading
# ---------------------------------------------------------------------------


def test_read_results_csv(tmp_path):
    p = tmp_path / "results.csv"
    p.write_text("id,gt,pred\n1,hi,hi\n2,bye,bye\n")
    rows = runner._read_results_csv(tmp_path)
    assert len(rows) == 2

    # Missing file
    assert runner._read_results_csv(tmp_path / "missing") is None

    # Malformed (force open() to raise via mock)
    bad = tmp_path / "bad"
    bad.mkdir()
    (bad / "results.csv").write_text("col1,col2\n")
    with patch("builtins.open", side_effect=RuntimeError("boom")):
        assert runner._read_results_csv(bad) is None


def test_read_metrics_json(tmp_path):
    (tmp_path / "metrics.json").write_text(json.dumps({"a": 1}))
    assert runner._read_metrics_json(tmp_path) == {"a": 1}
    assert runner._read_metrics_json(tmp_path / "missing") is None
    bad = tmp_path / "bad"
    bad.mkdir()
    (bad / "metrics.json").write_text("{not json")
    assert runner._read_metrics_json(bad) is None


def test_read_config_evaluators_map(tmp_path):
    assert runner._read_config_evaluators_map(tmp_path) == {}
    (tmp_path / "config.json").write_text(
        json.dumps({"evaluators_map": {"u1": "Safety", "u2": "Faithfulness"}})
    )
    # Inverts to name → uuid
    assert runner._read_config_evaluators_map(tmp_path) == {
        "Safety": "u1",
        "Faithfulness": "u2",
    }
    bad = tmp_path / "bad"
    bad.mkdir()
    (bad / "config.json").write_text("not json")
    assert runner._read_config_evaluators_map(bad) == {}


def test_read_simulation_dataset_map(tmp_path):
    assert runner._read_simulation_dataset_map(tmp_path) == {}
    (tmp_path / "dataset_map.json").write_text(
        json.dumps({"row_1": {"index": 0}, "row_2": {"index": 1}, "row_bad": "x"})
    )
    out = runner._read_simulation_dataset_map(tmp_path)
    assert out == {"row_1": 0, "row_2": 1}
    bad = tmp_path / "bad"
    bad.mkdir()
    (bad / "dataset_map.json").write_text("not json")
    assert runner._read_simulation_dataset_map(bad) == {}


# ---------------------------------------------------------------------------
# Results parsers
# ---------------------------------------------------------------------------


def _ev_resolved(uuid="ev-1", name="Safety", output_type="binary", version="ver-1"):
    return {
        "uuid": uuid,
        "name": name,
        "output_type": output_type,
        "_evaluator_version_id": version,
    }


def test_parse_results_stt(tmp_path):
    (tmp_path / "config.json").write_text(
        json.dumps({"evaluators_map": {"ev-1": "Safety"}})
    )
    (tmp_path / "results.csv").write_text(
        "id,gt,pred,Safety,Safety_reasoning\n"
        "i1,a,a,1,ok\n"
        "i2,b,b,0,not ok\n"
    )
    runs = runner._parse_results_stt(tmp_path, [_ev_resolved()], "job-1")
    assert len(runs) == 2
    assert runs[0]["evaluator_id"] == "ev-1"
    assert runs[0]["value"]["value"] is True


def test_parse_results_stt_no_config_fallback(tmp_path):
    """Without config.json/evaluators_map, falls back to display name."""
    (tmp_path / "results.csv").write_text(
        "id,gt,pred,Safety,Safety_reasoning\ni1,a,a,1,ok\n"
    )
    runs = runner._parse_results_stt(tmp_path, [_ev_resolved()], "job-1")
    assert runs and runs[0]["value"]["value"] is True


def test_parse_results_stt_empty_value(tmp_path):
    """Empty score → run marked failed."""
    (tmp_path / "results.csv").write_text(
        "id,gt,pred,Safety,Safety_reasoning\ni1,a,a,,\n"
    )
    runs = runner._parse_results_stt(tmp_path, [_ev_resolved()], "job-1")
    assert runs[0]["status"] == "failed"


def test_parse_results_stt_no_id_row(tmp_path):
    (tmp_path / "results.csv").write_text(
        "id,gt,pred,Safety,Safety_reasoning\n,a,a,1,ok\n"
    )
    runs = runner._parse_results_stt(tmp_path, [_ev_resolved()], "job-1")
    assert runs == []


def test_parse_results_stt_unknown_uuid_in_map(tmp_path):
    """evaluators_map references a UUID we don't have a snapshot for → skip."""
    (tmp_path / "config.json").write_text(
        json.dumps({"evaluators_map": {"unknown": "Other"}})
    )
    (tmp_path / "results.csv").write_text(
        "id,gt,pred,Other\ni1,a,a,1\n"
    )
    runs = runner._parse_results_stt(tmp_path, [_ev_resolved()], "job-1")
    assert runs == []


def test_parse_results_llm(tmp_path):
    (tmp_path / "config.json").write_text(
        json.dumps({"evaluators_map": {"ev-1": "Safety"}})
    )
    (tmp_path / "results.json").write_text(
        json.dumps(
            [
                {
                    "test_case": {"id": "i1"},
                    "metrics": {
                        "judge_results": {
                            "Safety": {"match": True, "reasoning": "ok"}
                        }
                    },
                }
            ]
        )
    )
    runs = runner._parse_results_llm(tmp_path, [_ev_resolved()], "job-1")
    assert len(runs) == 1
    assert runs[0]["value"]["value"] is True


def test_parse_results_llm_score_field(tmp_path):
    """Rating evaluator uses `score` not `match`."""
    (tmp_path / "results.json").write_text(
        json.dumps(
            [
                {
                    "test_case": {"id": "i1"},
                    "metrics": {
                        "judge_results": {
                            "Safety": {"score": 3.5}
                        }
                    },
                }
            ]
        )
    )
    runs = runner._parse_results_llm(
        tmp_path, [_ev_resolved(output_type="rating")], "job-1"
    )
    assert runs[0]["value"]["value"] == 3.5


def test_parse_results_llm_missing_file(tmp_path):
    assert runner._parse_results_llm(tmp_path, [_ev_resolved()], "job-1") == []


def test_parse_results_llm_malformed(tmp_path):
    (tmp_path / "results.json").write_text("not json")
    assert runner._parse_results_llm(tmp_path, [_ev_resolved()], "job-1") == []


def test_parse_results_llm_missing_id_or_judge_results(tmp_path):
    (tmp_path / "results.json").write_text(
        json.dumps(
            [
                "not-dict",
                {"metrics": {}},  # missing id
                {"test_case": {"id": "i1"}},  # missing metrics.judge_results
            ]
        )
    )
    assert runner._parse_results_llm(tmp_path, [_ev_resolved()], "job-1") == []


def test_parse_results_simulation(tmp_path):
    (tmp_path / "config.json").write_text(
        json.dumps({"evaluators_map": {"ev-1": "Safety"}})
    )
    (tmp_path / "dataset_map.json").write_text(
        json.dumps({"row_1": {"index": 0, "name": "fallback-uuid"}})
    )
    sim_dir = tmp_path / "row_1"
    sim_dir.mkdir()
    (sim_dir / "evaluation_results.csv").write_text(
        "name,type,value,reasoning,evaluator_id\nSafety,binary,1,ok,ev-1\n"
    )
    items = [{"uuid": "real-uuid"}]
    runs = runner._parse_results_simulation(
        tmp_path, [_ev_resolved()], "job-1", items=items
    )
    assert runs[0]["item_id"] == "real-uuid"


def test_parse_results_simulation_no_output_dir(tmp_path):
    """Missing output_dir → empty list."""
    runs = runner._parse_results_simulation(
        tmp_path / "missing", [_ev_resolved()], "job-1"
    )
    assert runs == []


def test_parse_results_simulation_fallback_name(tmp_path):
    """When items not supplied, fall back to dataset_map's name field."""
    (tmp_path / "dataset_map.json").write_text(
        json.dumps({"row_1": {"index": 0, "name": "from-name"}})
    )
    sim_dir = tmp_path / "row_1"
    sim_dir.mkdir()
    (sim_dir / "evaluation_results.csv").write_text(
        "name,type,value,reasoning,evaluator_id\nSafety,binary,1,ok,ev-1\n"
    )
    runs = runner._parse_results_simulation(
        tmp_path, [_ev_resolved()], "job-1", items=None
    )
    assert runs[0]["item_id"] == "from-name"


def test_parse_results_simulation_unmapped_row(tmp_path):
    """A sim_dir with no dataset_map entry is skipped."""
    sim_dir = tmp_path / "row_99"
    sim_dir.mkdir()
    (sim_dir / "evaluation_results.csv").write_text("name,value\nSafety,1\n")
    runs = runner._parse_results_simulation(tmp_path, [_ev_resolved()], "job-1")
    assert runs == []


def test_parse_results_for_task_type_dispatch(tmp_path):
    # Unknown task type → []
    assert (
        runner.parse_results_for_task_type(
            "unknown", tmp_path, [_ev_resolved()], "job-1"
        )
        == []
    )


def test_parse_results_general(tmp_path):
    """`calibrate general` writes the same results.csv shape as STT, with
    input/output as the built-in columns; the column-map parser is reused."""
    (tmp_path / "config.json").write_text(
        json.dumps({"evaluators_map": {"ev-1": "Safety"}})
    )
    (tmp_path / "results.csv").write_text(
        "id,input,output,Safety,Safety_reasoning\n"
        "i1,in1,out1,1,ok\n"
        "i2,in2,out2,0,not ok\n"
    )
    runs = runner._parse_results_general(tmp_path, [_ev_resolved()], "job-1")
    assert len(runs) == 2
    assert runs[0]["evaluator_id"] == "ev-1"
    assert runs[0]["value"]["value"] is True
    assert runs[1]["value"]["value"] is False


def test_parse_results_for_task_type_dispatch_llm_general(tmp_path):
    # llm-general routes to the `calibrate general` CSV parser.
    (tmp_path / "config.json").write_text(
        json.dumps({"evaluators_map": {"ev-1": "Safety"}})
    )
    (tmp_path / "results.csv").write_text(
        "id,input,output,Safety,Safety_reasoning\ni1,in1,out1,1,ok\n"
    )
    runs = runner.parse_results_for_task_type(
        "llm-general", tmp_path, [_ev_resolved()], "job-1"
    )
    assert len(runs) == 1
    assert runs[0]["value"]["value"] is True


def test_parse_results_for_task_type_dispatch_conversation(tmp_path):
    # The `conversation` task type routes to the simulation results parser.
    (tmp_path / "dataset_map.json").write_text(
        json.dumps({"row_1": {"index": 0, "name": "from-name"}})
    )
    sim_dir = tmp_path / "row_1"
    sim_dir.mkdir()
    (sim_dir / "evaluation_results.csv").write_text(
        "name,type,value,reasoning,evaluator_id\nSafety,binary,1,ok,ev-1\n"
    )
    runs = runner.parse_results_for_task_type(
        "conversation", tmp_path, [_ev_resolved()], "job-1", items=None
    )
    assert runs[0]["item_id"] == "from-name"


# ---------------------------------------------------------------------------
# _extract_calibrate_error
# ---------------------------------------------------------------------------


def test_extract_calibrate_error_structured():
    stdout = '{"status":"error","error":"bad config"}'
    assert runner._extract_calibrate_error(stdout, "") == "bad config"


def test_extract_calibrate_error_stderr_fallback():
    assert runner._extract_calibrate_error("", "boom\nspecific failure") == (
        "specific failure"
    )


def test_extract_calibrate_error_stdout_fallback():
    assert runner._extract_calibrate_error("blah\nfinal", "") == "final"


def test_extract_calibrate_error_empty():
    assert runner._extract_calibrate_error("", "") == (
        "calibrate exited non-zero with no diagnostic output"
    )


# ---------------------------------------------------------------------------
# _persist_pgid
# ---------------------------------------------------------------------------


def test_persist_pgid_success():
    with patch("annotation_eval_runner.os.getpgid", return_value=999), patch(
        "annotation_eval_runner.update_job"
    ) as uj:
        runner._persist_pgid("j-1", 1234)
        uj.assert_called_once()


def test_persist_pgid_fallback_to_pid():
    with patch(
        "annotation_eval_runner.os.getpgid", side_effect=ProcessLookupError()
    ), patch("annotation_eval_runner.update_job") as uj:
        runner._persist_pgid("j-1", 1234)
        # Falls back to pid as pgid
        assert uj.call_args.kwargs["details"]["pgid"] == 1234


# ---------------------------------------------------------------------------
# _try_upload_partial_outputs
# ---------------------------------------------------------------------------


def test_try_upload_partial_outputs_missing(tmp_path):
    assert runner._try_upload_partial_outputs(None, "t", "j") is None
    assert (
        runner._try_upload_partial_outputs(tmp_path / "missing", "t", "j") is None
    )


def test_try_upload_partial_outputs_success(tmp_path):
    (tmp_path / "a.log").write_text("x")
    with patch(
        "annotation_eval_runner.get_s3_output_config", return_value="bucket"
    ), patch("annotation_eval_runner.get_s3_client", return_value=MagicMock()), patch(
        "annotation_eval_runner.upload_file_to_s3"
    ):
        s3_prefix = runner._try_upload_partial_outputs(tmp_path, "t", "j")
    assert s3_prefix and "annotation-tasks/t/evaluator-runs/j/outputs" in s3_prefix


def test_try_upload_partial_outputs_swallows_exception(tmp_path):
    with patch(
        "annotation_eval_runner.get_s3_output_config",
        side_effect=RuntimeError("nope"),
    ):
        assert runner._try_upload_partial_outputs(tmp_path, "t", "j") is None


# ---------------------------------------------------------------------------
# start_annotation_eval_job + queue starter
# ---------------------------------------------------------------------------


def test_start_annotation_eval_job_spawns_thread():
    with patch("annotation_eval_runner.threading.Thread") as thread_mock:
        runner.start_annotation_eval_job(
            "j-1", "t-1", "u-1", [_ev_resolved()], item_ids=None
        )
        thread_mock.return_value.start.assert_called_once()


def test_queue_starter_missing_fields():
    with pytest.raises(RuntimeError):
        runner._start_annotation_eval_job_from_queue(
            {"uuid": "j-1"}  # no details
        )


def test_queue_starter_no_evaluators():
    with pytest.raises(RuntimeError):
        runner._start_annotation_eval_job_from_queue(
            {"uuid": "j-1", "user_id": "u-1", "details": {"task_id": "t-1"}}
        )


def test_queue_starter_happy_path():
    with patch(
        "db.get_evaluators_for_annotation_task",
        return_value=[{"uuid": "ev-1"}],
    ), patch(
        "annotation_eval_runner._resolve_evaluator_dicts",
        return_value=[_ev_resolved()],
    ), patch(
        "annotation_eval_runner.start_annotation_eval_job"
    ) as sj:
        runner._start_annotation_eval_job_from_queue(
            {
                "uuid": "j-1",
                "user_id": "u-1",
                "details": {
                    "task_id": "t-1",
                    "evaluators": [
                        {"evaluator_id": "ev-1", "evaluator_version_id": "ver-1"}
                    ],
                    "item_ids": None,
                },
            }
        )
        sj.assert_called_once()


def test_resume_annotation_eval_job_clears_runs():
    with patch(
        "db.clear_evaluator_runs_for_job", return_value=3
    ), patch(
        "annotation_eval_runner._start_annotation_eval_job_from_queue"
    ) as sj:
        runner.resume_annotation_eval_job(
            {
                "uuid": "j-1",
                "user_id": "u-1",
                "details": {
                    "task_id": "t-1",
                    "evaluators": [
                        {"evaluator_id": "ev-1", "evaluator_version_id": "ver-1"}
                    ],
                },
            }
        )
        sj.assert_called_once()
