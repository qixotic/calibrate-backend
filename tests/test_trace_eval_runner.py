"""Unit tests for the trace-eval runner (src/traces/eval_runner.py).

The calibrate CLI is never spawned: `_run_calibrate_eval_only` is replaced with
a stub that writes the output files each shared parser expects.
"""

from __future__ import annotations

import json
import uuid
from unittest.mock import MagicMock, patch

import pytest

from traces import eval_runner, eval_store, store
from traces.inference import TYPE_CONVERSATION, TYPE_RESPONSE, TYPE_TOOL_CALL

_AGENT = "44444444-4444-4444-8444-444444444444"


def _org() -> str:
    return str(uuid.uuid4())


def _evaluator(**overrides):
    """An already-hydrated evaluator, as `refresh_evaluators_to_live` returns."""
    base = {
        "uuid": "ev-1",
        "name": "Helpfulness",
        "judge_model": "gpt",
        "system_prompt": "judge it",
        "output_type": "binary",
        "output_config": None,
        "variables": [],
        "variable_values": {},
        "kind": "single",
        "data_type": "text",
        "evaluator_type": "llm",
        "evaluator_version_id": "ver-1",
    }
    base.update(overrides)
    return base


def _resolved(*evaluators):
    """Resolve without a live-version lookup — there is no pense.db row here."""
    with patch("db.get_evaluator", return_value=None):
        return eval_runner.resolve_evaluators(list(evaluators) or [_evaluator()])


def _trace(trace_uuid, turns, output):
    return {"uuid": trace_uuid, "input": turns, "output": output}


def _seed(org, count, agent_id=_AGENT):
    """Create `count` live traces and return them oldest-first."""
    for i in range(count):
        store.create_trace(
            org_uuid=org,
            agent_id=agent_id,
            message_id=f"m-{uuid.uuid4().hex[:10]}",
            conversation_id="conv-1",
            input=[{"role": "user", "content": f"q{i}"}],
            output={"response": f"a{i}"},
        )
    return eval_store.list_pending_traces(org, agent_id, limit=100)


class _ImmediateThread:
    """Runs the worker inline so a test sees the finished run."""

    def __init__(self, target=None, args=(), daemon=None):
        self._target = target
        self._args = args

    def start(self):
        self._target(*self._args)


# ---------------------------------------------------------------------------
# Dataset shapes
# ---------------------------------------------------------------------------


class TestDatasetShapes:
    def test_response_rows_carry_the_trace_verbatim(self):
        calls = [{"tool": "book", "arguments": {"day": "mon"}}]
        traces = [
            _trace(
                "t-1",
                [{"role": "user", "content": "hi"}],
                {"response": "hello", "tool_calls": calls},
            )
        ]

        _, dataset = eval_runner.build_trace_dataset(
            TYPE_RESPONSE, traces, _resolved()
        )

        assert dataset == [
            {
                "test_case": {
                    "id": "t-1",
                    "history": [{"role": "user", "content": "hi"}],
                    "evaluation": {
                        "type": "response",
                        "criteria": [{"name": "Helpfulness"}],
                    },
                },
                "output": {"response": "hello", "tool_calls": calls},
            }
        ]

    def test_response_rows_default_tool_calls_to_empty(self):
        traces = [_trace("t-1", [{"role": "user", "content": "hi"}], {"response": "yo"})]
        _, dataset = eval_runner.build_trace_dataset(
            TYPE_RESPONSE, traces, _resolved()
        )
        assert dataset[0]["output"]["tool_calls"] == []

    def test_conversation_rows_append_the_agent_reply(self):
        turns = [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hey"},
            {"role": "user", "content": "help"},
        ]
        traces = [_trace("t-1", turns, {"response": "sure"})]

        _, dataset = eval_runner.build_trace_dataset(
            TYPE_CONVERSATION, traces, _resolved()
        )

        assert dataset == [
            {
                "name": "t-1",
                "conversation_history": turns + [{"role": "assistant", "content": "sure"}],
            }
        ]

    def test_conversation_rows_keep_the_given_trace_order(self):
        """Results map back by position, so order is load-bearing."""
        traces = [
            _trace(f"t-{i}", [{"role": "user", "content": f"q{i}"}], {"response": "r"})
            for i in range(4)
        ]

        items, dataset = eval_runner.build_trace_dataset(
            TYPE_CONVERSATION, traces, _resolved()
        )

        assert [r["name"] for r in dataset] == ["t-0", "t-1", "t-2", "t-3"]
        assert [i["uuid"] for i in items] == ["t-0", "t-1", "t-2", "t-3"]

    def test_tool_call_rows_pair_the_last_user_turn_with_the_calls(self):
        calls = [{"tool": "book", "arguments": {"day": "mon"}}]
        turns = [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hey"},
            {"role": "user", "content": "book it"},
        ]
        traces = [_trace("t-1", turns, {"response": "", "tool_calls": calls})]

        _, dataset = eval_runner.build_trace_dataset(
            TYPE_TOOL_CALL, traces, _resolved()
        )

        assert dataset[0]["id"] == "t-1"
        assert dataset[0]["input"] == "book it"
        assert json.loads(dataset[0]["output"]) == calls

    def test_tool_call_rows_serialize_non_string_user_content(self):
        turns = [{"role": "user", "content": [{"type": "text", "text": "hi"}]}]
        traces = [_trace("t-1", turns, {"tool_calls": [{"tool": "t", "arguments": {}}]})]

        _, dataset = eval_runner.build_trace_dataset(
            TYPE_TOOL_CALL, traces, _resolved()
        )

        assert json.loads(dataset[0]["input"]) == [{"type": "text", "text": "hi"}]

    def test_an_unjudgeable_type_is_rejected(self):
        with pytest.raises(ValueError):
            eval_runner.trace_to_item(_trace("t-1", [], {}), "nonsense")


# ---------------------------------------------------------------------------
# Evaluator resolution
# ---------------------------------------------------------------------------


class TestEvaluatorResolution:
    def test_the_live_version_is_pinned_at_run_time(self):
        live_evaluator = {
            "uuid": "ev-9",
            "name": "Live name",
            "live_version_id": "ver-9",
            "output_type": "rating",
            "kind": "single",
            "data_type": "text",
            "evaluator_type": "llm",
        }
        live_version = {
            "uuid": "ver-9",
            "evaluator_id": "ev-9",
            "judge_model": "gpt",
            "system_prompt": "live prompt",
            "output_config": {"scale": [{"value": 1}, {"value": 5}]},
            "variables": [],
        }
        with patch("db.get_evaluator", return_value=live_evaluator), patch(
            "db.get_evaluator_version", return_value=live_version
        ):
            resolved = eval_runner.resolve_evaluators([{"uuid": "ev-9", "name": "stale"}])

        assert resolved[0]["name"] == "Live name"
        assert resolved[0]["system_prompt"] == "live prompt"
        assert resolved[0]["_evaluator_version_id"] == "ver-9"
        assert (resolved[0]["_scale_min"], resolved[0]["_scale_max"]) == (1, 5)

    def test_snapshot_carries_what_a_result_needs_to_render(self):
        resolved = _resolved(
            _evaluator(
                output_type="rating",
                output_config={"scale": [{"value": 1}, {"value": 5}]},
            )
        )

        assert eval_runner.evaluator_snapshot(resolved) == [
            {
                "uuid": "ev-1",
                "name": "Helpfulness",
                "evaluator_version_id": "ver-1",
                "output_type": "rating",
                "scale_min": 1,
                "scale_max": 5,
            }
        ]

    def test_colliding_names_are_deduped_before_the_cli_sees_them(self):
        resolved = _resolved(_evaluator(), _evaluator(uuid="ev-2abcdef0"))
        assert len({ev["name"] for ev in resolved}) == 2


# ---------------------------------------------------------------------------
# Result mapping
# ---------------------------------------------------------------------------


class TestResultMapping:
    def test_binary_and_rating_land_in_their_own_columns(self):
        binary, rating = _resolved(
            _evaluator(),
            _evaluator(
                uuid="ev-2",
                name="Quality",
                output_type="rating",
                output_config={"scale": [{"value": 1}, {"value": 5}]},
            ),
        )
        parsed = [
            {
                "item_id": "t-1",
                "evaluator_id": "ev-1",
                "evaluator_version_id": "ver-1",
                "value": {"value": "true", "reasoning": "clear"},
            },
            {
                "item_id": "t-1",
                "evaluator_id": "ev-2",
                "evaluator_version_id": "ver-1",
                "value": {"value": "4"},
            },
        ]

        results = eval_runner.to_trace_results(parsed, [binary, rating])

        assert results[0]["passed"] is True
        assert results[0]["score"] is None
        assert results[0]["reasoning"] == "clear"
        assert results[1]["passed"] is None
        assert results[1]["score"] == 4.0
        assert (results[1]["scale_min"], results[1]["scale_max"]) == (1, 5)

    def test_rows_without_a_verdict_are_dropped(self):
        parsed = [{"item_id": "t-1", "evaluator_id": "ev-1", "value": None}]
        assert eval_runner.to_trace_results(parsed, _resolved()) == []

    def test_rows_for_an_unknown_evaluator_are_dropped(self):
        parsed = [
            {"item_id": "t-1", "evaluator_id": "gone", "value": {"value": True}}
        ]
        assert eval_runner.to_trace_results(parsed, _resolved()) == []


# ---------------------------------------------------------------------------
# Launch: claim, then judge only what was won
# ---------------------------------------------------------------------------


class TestLaunch:
    def _launch(self, org, traces, inferred_type=TYPE_RESPONSE):
        """Launch with the worker thread stubbed out, so nothing is judged."""
        thread = MagicMock()
        with patch("db.get_evaluator", return_value=None), patch(
            "traces.eval_runner.threading.Thread", thread
        ):
            run_uuid, status = eval_runner.launch_trace_eval(
                org,
                {"uuid": _AGENT},
                inferred_type,
                traces,
                [_evaluator()],
                eval_store.TRIGGER_MANUAL,
            )
        return run_uuid, status, thread

    def test_only_traces_won_by_the_claim_are_judged(self):
        org = _org()
        traces = _seed(org, 3)
        other = eval_store.create_eval_run(
            org,
            _AGENT,
            trigger=eval_store.TRIGGER_AUTO,
            inferred_type=TYPE_RESPONSE,
            status="in_progress",
        )
        eval_store.claim_traces(org, other["uuid"], [traces[0]["uuid"]])

        run_uuid, status, thread_mock = self._launch(org, traces)

        claimed = thread_mock.call_args.kwargs["args"][3]
        assert [t["uuid"] for t in claimed] == [t["uuid"] for t in traces[1:]]
        assert status == "in_progress"
        assert eval_store.get_eval_run(org, run_uuid)["trace_count"] == 2

    def test_winning_nothing_finishes_the_run_without_a_worker(self):
        org = _org()
        traces = _seed(org, 2)
        other = eval_store.create_eval_run(
            org,
            _AGENT,
            trigger=eval_store.TRIGGER_AUTO,
            inferred_type=TYPE_RESPONSE,
            status="in_progress",
        )
        eval_store.claim_traces(org, other["uuid"], [t["uuid"] for t in traces])

        run_uuid, status, thread_mock = self._launch(org, traces)

        thread_mock.assert_not_called()
        assert status == "done"
        run = eval_store.get_eval_run(org, run_uuid)
        assert run["trace_count"] == 0
        assert run["finished_at"] is not None

    def test_the_run_row_snapshots_the_evaluators(self):
        org = _org()
        traces = _seed(org, 1)
        run_uuid, _, _ = self._launch(org, traces)
        run = eval_store.get_eval_run(org, run_uuid)
        assert run["evaluator_snapshot"][0]["uuid"] == "ev-1"
        assert run["trigger"] == eval_store.TRIGGER_MANUAL


# ---------------------------------------------------------------------------
# The worker, end to end against stubbed CLI output
# ---------------------------------------------------------------------------


def _llm_cli(evaluator_uuid, name, verdicts):
    """Stub `calibrate llm --eval-only`: `verdicts` is {trace_uuid: judgement}."""

    def _run(cmd, cwd, log_dir, **kwargs):
        (log_dir / "config.json").write_text(
            json.dumps({"evaluators_map": {evaluator_uuid: name}})
        )
        (log_dir / "results.json").write_text(
            json.dumps(
                [
                    {
                        "test_case": {"id": trace_uuid},
                        "metrics": {"judge_results": {name: judgement}},
                    }
                    for trace_uuid, judgement in verdicts.items()
                ]
            )
        )
        return 0, "", ""

    return _run


def _simulation_cli(name, rows_by_index):
    """Stub `calibrate simulations --eval-only`.

    Calibrate names its output directories `row_<i>` independently of the row's
    dataset position, so the map here deliberately crosses them: `row_1` holds
    index 1 and `row_2` index 0.
    """

    def _run(cmd, cwd, log_dir, **kwargs):
        dataset_map = {
            "row_1": {"index": 1},
            "row_2": {"index": 0},
        }
        (log_dir / "dataset_map.json").write_text(json.dumps(dataset_map))
        for row_id, entry in dataset_map.items():
            sim_dir = log_dir / row_id
            sim_dir.mkdir()
            value, reasoning = rows_by_index[entry["index"]]
            (sim_dir / "evaluation_results.csv").write_text(
                "name,type,value,reasoning\n" f"{name},binary,{value},{reasoning}\n"
            )
        return 0, "", ""

    return _run


class TestWorker:
    def _run_worker(self, org, traces, inferred_type, cli, evaluators=None):
        evaluators = evaluators or [_evaluator()]
        with patch("db.get_evaluator", return_value=None), patch(
            "traces.eval_runner.threading.Thread", _ImmediateThread
        ), patch("traces.eval_runner._run_calibrate_eval_only", cli):
            return eval_runner.launch_trace_eval(
                org,
                {"uuid": _AGENT},
                inferred_type,
                traces,
                evaluators,
                eval_store.TRIGGER_AUTO,
            )

    def test_a_response_verdict_lands_on_its_own_trace(self):
        org = _org()
        traces = _seed(org, 2)
        cli = _llm_cli(
            "ev-1",
            "Helpfulness",
            {
                traces[0]["uuid"]: {"match": True, "reasoning": "good"},
                traces[1]["uuid"]: {"match": False, "reasoning": "bad"},
            },
        )

        run_uuid, _ = self._run_worker(org, traces, TYPE_RESPONSE, cli)

        first = eval_store.results_for_trace(org, traces[0]["uuid"])
        second = eval_store.results_for_trace(org, traces[1]["uuid"])
        assert (first[0]["passed"], first[0]["reasoning"]) == (True, "good")
        assert (second[0]["passed"], second[0]["reasoning"]) == (False, "bad")
        assert first[0]["evaluator_name"] == "Helpfulness"
        assert first[0]["evaluator_version_id"] == "ver-1"

    def test_a_finished_run_takes_its_traces_out_of_the_queue(self):
        org = _org()
        traces = _seed(org, 2)
        cli = _llm_cli(
            "ev-1",
            "Helpfulness",
            {t["uuid"]: {"match": True} for t in traces},
        )

        run_uuid, _ = self._run_worker(org, traces, TYPE_RESPONSE, cli)

        run = eval_store.get_eval_run(org, run_uuid)
        assert run["status"] == "done"
        assert run["finished_at"] is not None
        assert eval_store.list_pending_traces(org, _AGENT, limit=10) == []

    def test_a_rating_verdict_keeps_its_scale(self):
        org = _org()
        traces = _seed(org, 1)
        evaluator = _evaluator(
            output_type="rating",
            output_config={"scale": [{"value": 1}, {"value": 5}]},
        )
        cli = _llm_cli(
            "ev-1", "Helpfulness", {traces[0]["uuid"]: {"score": 4, "reasoning": "ok"}}
        )

        self._run_worker(org, traces, TYPE_RESPONSE, cli, evaluators=[evaluator])

        result = eval_store.results_for_trace(org, traces[0]["uuid"])[0]
        assert result["score"] == 4.0
        assert result["passed"] is None
        assert (result["scale_min"], result["scale_max"]) == (1.0, 5.0)

    def test_conversation_verdicts_map_back_by_dataset_position(self):
        org = _org()
        traces = _seed(org, 2)
        cli = _simulation_cli(
            "Helpfulness", {0: ("True", "first row"), 1: ("False", "second row")}
        )

        self._run_worker(org, traces, TYPE_CONVERSATION, cli)

        first = eval_store.results_for_trace(org, traces[0]["uuid"])[0]
        second = eval_store.results_for_trace(org, traces[1]["uuid"])[0]
        assert (first["passed"], first["reasoning"]) == (True, "first row")
        assert (second["passed"], second["reasoning"]) == (False, "second row")

    def test_a_trace_the_judge_skipped_stays_unevaluated(self):
        org = _org()
        traces = _seed(org, 2)
        cli = _llm_cli("ev-1", "Helpfulness", {traces[0]["uuid"]: {"match": True}})

        self._run_worker(org, traces, TYPE_RESPONSE, cli)

        assert eval_store.results_for_trace(org, traces[1]["uuid"]) == []
        assert eval_store.eval_summaries_for_traces(
            org, [t["uuid"] for t in traces]
        ) == {traces[0]["uuid"]: {"total": 1, "passed": 1}}


class TestFailure:
    def _failing_cli(self, stderr="boom"):
        def _run(cmd, cwd, log_dir, **kwargs):
            return 1, "", stderr

        return _run

    def test_a_failed_run_hands_its_traces_back_to_the_queue(self):
        org = _org()
        traces = _seed(org, 2)

        with patch("db.get_evaluator", return_value=None), patch(
            "traces.eval_runner.threading.Thread", _ImmediateThread
        ), patch(
            "traces.eval_runner._run_calibrate_eval_only", self._failing_cli()
        ), patch(
            "traces.eval_runner.capture_exception_to_sentry"
        ) as sentry:
            run_uuid, _ = eval_runner.launch_trace_eval(
                org,
                {"uuid": _AGENT},
                TYPE_RESPONSE,
                traces,
                [_evaluator()],
                eval_store.TRIGGER_AUTO,
            )

        run = eval_store.get_eval_run(org, run_uuid)
        assert run["status"] == "failed"
        assert "boom" in run["error"]
        assert run["finished_at"] is not None
        sentry.assert_called_once()
        pending = eval_store.list_pending_traces(org, _AGENT, limit=10)
        assert {p["uuid"] for p in pending} == {t["uuid"] for t in traces}

    def test_claims_are_released_even_if_the_status_write_fails(self):
        org = _org()
        traces = _seed(org, 1)
        run = eval_store.create_eval_run(
            org,
            _AGENT,
            trigger=eval_store.TRIGGER_AUTO,
            inferred_type=TYPE_RESPONSE,
            status="in_progress",
        )
        eval_store.claim_traces(org, run["uuid"], [traces[0]["uuid"]])

        with patch(
            "traces.eval_store.update_eval_run", side_effect=RuntimeError("db down")
        ), pytest.raises(RuntimeError):
            eval_runner._fail_run(run["uuid"], "boom")

        assert len(eval_store.list_pending_traces(org, _AGENT, limit=10)) == 1
