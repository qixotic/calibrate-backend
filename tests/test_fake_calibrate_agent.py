"""Integration tests: every worker drives the real fake CLI end-to-end.

Sets ``FAKE_AI_PROVIDERS=1`` and does NOT patch ``subprocess.Popen`` — the
workers actually spawn ``src/testing/fake_calibrate_agent.py`` and read its
output (only S3/queue side effects are stubbed). Catches any regression in the
seam or the fake's per-subcommand output contract.
"""

import asyncio
import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

import db


def _make_agent_with_response_test():
    user_uuid = db.create_user("F", "AI", f"fai-{os.urandom(4).hex()}@x.com")
    org_uuid = db.get_personal_org_for_user(user_uuid)["uuid"]
    agent_uuid = db.create_agent(
        name=f"a-{os.urandom(4).hex()}", org_uuid=org_uuid, user_id=user_uuid
    )

    ev_uuid = db.create_evaluator(
        name=f"acc-{os.urandom(4).hex()}",
        evaluator_type="llm",
        output_type="binary",
        owner_user_id=user_uuid,
        org_uuid=org_uuid,
    )
    version = db.create_evaluator_version(
        ev_uuid, judge_model="m", system_prompt="judge this"
    )
    db.set_evaluator_live_version(ev_uuid, version["uuid"])

    test_uuid = db.create_test(
        name=f"t-{os.urandom(4).hex()}",
        type="response",
        config={
            "history": [{"role": "user", "content": "hi"}],
            "evaluation": {"type": "response"},
        },
        org_uuid=org_uuid,
        user_id=user_uuid,
    )
    db.set_test_evaluators(
        test_uuid, [{"evaluator_id": ev_uuid, "variable_values": None}]
    )

    job_uuid = db.create_agent_test_job(
        agent_id=agent_uuid, job_type="llm-unit-test", status="in_progress"
    )
    return db.get_agent(agent_uuid), db.get_test(test_uuid), job_uuid


def test_run_llm_test_task_end_to_end_with_fake_cli():
    from routers.agent_tests import run_llm_test_task

    agent, test, job_uuid = _make_agent_with_response_test()

    with patch.dict(os.environ, {"FAKE_AI_PROVIDERS": "1"}), patch(
        "routers.agent_tests.get_s3_client", return_value=MagicMock()
    ), patch("routers.agent_tests.upload_directory_tree_to_s3"), patch(
        "routers.agent_tests.upload_file_to_s3"
    ), patch(
        "routers.agent_tests.try_start_queued_agent_test_job"
    ), patch(
        "routers.agent_tests.time.sleep"
    ):
        run_llm_test_task(job_uuid, agent, [test], "bucket")

    job = db.get_agent_test_job(job_uuid)
    assert job["status"] == "done", job.get("results")
    results = job["results"]
    assert results["total_tests"] == 1
    assert results["passed"] == results["total_tests"]
    assert results["failed"] == 0
    assert results["test_results"][0]["passed"] is True
    # Aggregate perf blocks must be dicts, not scalars — the run-status reader
    # feeds them into Optional[Dict] response fields, so a scalar 500s the
    # endpoint. Latency is percentiles; cost/tokens are mean/min/max.
    assert set(results["latency_ms"]) == {"p50", "p95", "p99", "count"}
    assert set(results["cost"]) == {"mean", "min", "max", "count"}
    assert set(results["total_tokens"]) == {"mean", "min", "max", "count"}
    from routers.agent_tests import TestRunStatusResponse

    TestRunStatusResponse(
        task_id="t" * 36,
        status="done",
        test_name="x",
        latency_ms=results["latency_ms"],
        cost=results["cost"],
        total_tokens=results["total_tokens"],
    )


def test_run_llm_test_task_agent_connection_mode_with_fake_cli():
    """Agent-connection mode omits -m (agent owns its model); the fake writes a
    single ``default`` folder and the walk-based reader still finds it."""
    from routers.agent_tests import run_llm_test_task

    agent, test, job_uuid = _make_agent_with_response_test()
    agent = {**agent, "config": {**(agent.get("config") or {}), "agent_url": "http://x"}}

    with patch.dict(os.environ, {"FAKE_AI_PROVIDERS": "1"}), patch(
        "routers.agent_tests.get_s3_client", return_value=MagicMock()
    ), patch("routers.agent_tests.upload_directory_tree_to_s3"), patch(
        "routers.agent_tests.upload_file_to_s3"
    ), patch("routers.agent_tests.try_start_queued_agent_test_job"), patch(
        "routers.agent_tests.time.sleep"
    ):
        run_llm_test_task(job_uuid, agent, [test], "bucket")

    job = db.get_agent_test_job(job_uuid)
    assert job["status"] == "done", job.get("results")
    assert job["results"]["passed"] == job["results"]["total_tests"] == 1


def test_run_benchmark_task_multi_model_end_to_end_with_fake_cli():
    """Benchmark path: multiple ``-m`` models → per-model folders + leaderboard,
    all matched back via ``_match_model_to_folder`` / leaderboard normalization."""
    from routers.agent_tests import run_benchmark_task

    agent, test, _ = _make_agent_with_response_test()
    job_uuid = db.create_agent_test_job(
        agent_id=agent["uuid"], job_type="llm-benchmark", status="in_progress"
    )
    models = ["openai/gpt-4.1", "openai/gpt-4o-mini"]

    with patch.dict(os.environ, {"FAKE_AI_PROVIDERS": "1"}), patch(
        "routers.agent_tests.get_s3_client", return_value=MagicMock()
    ), patch("routers.agent_tests.upload_directory_tree_to_s3"), patch(
        "routers.agent_tests.upload_file_to_s3"
    ), patch("routers.agent_tests.try_start_queued_agent_test_job"), patch(
        "routers.agent_tests.time.sleep"
    ):
        run_benchmark_task(job_uuid, agent, [test], models, "bucket")

    job = db.get_agent_test_job(job_uuid)
    assert job["status"] == "done", job.get("results")
    by_model = {m["model"]: m for m in job["results"]["model_results"]}
    assert set(by_model) == set(models)
    for m in models:
        assert by_model[m]["success"] is True
        assert by_model[m]["passed"] == by_model[m]["total_tests"] == 1
        # Per-model aggregate perf blocks are dicts (see unit-test rationale).
        assert set(by_model[m]["latency_ms"]) == {"p50", "p95", "p99", "count"}
        assert set(by_model[m]["cost"]) == {"mean", "min", "max", "count"}
        assert set(by_model[m]["total_tokens"]) == {"mean", "min", "max", "count"}
    assert job["results"]["leaderboard_summary"], "leaderboard should be populated"


def _make_eval_job(job_type, details):
    user_uuid = db.create_user("F", "AI", f"fai-{os.urandom(4).hex()}@x.com")
    org_uuid = db.get_personal_org_for_user(user_uuid)["uuid"]
    return db.create_job(
        job_type=job_type,
        org_uuid=org_uuid,
        user_id=user_uuid,
        status="in_progress",
        details=details,
    )


def test_run_stt_evaluation_task_end_to_end_with_fake_cli():
    from routers.stt import run_evaluation_task, STTEvaluationRequest

    job_uuid = _make_eval_job(
        "stt-eval",
        {
            "audio_paths": ["s3://bucket/key.wav"],
            "texts": ["hi"],
            "providers": ["deepgram", "openai"],
            "language": "en",
            "s3_bucket": "bucket",
            "evaluators": [],
        },
    )

    with patch.dict(os.environ, {"FAKE_AI_PROVIDERS": "1"}), patch(
        "routers.stt.get_s3_client", return_value=MagicMock()
    ), patch("routers.stt.download_file_from_s3"), patch(
        "routers.stt.upload_file_to_s3"
    ), patch("routers.stt.upload_top_level_files_to_s3"), patch(
        "routers.stt.upload_directory_tree_to_s3"
    ), patch("routers.stt.try_start_queued_job"), patch("routers.stt.time.sleep"):
        request = STTEvaluationRequest(
            audio_paths=["s3://bucket/key.wav"],
            texts=["hi"],
            providers=["deepgram", "openai"],
            language="en",
        )
        run_evaluation_task(job_uuid, request, "bucket")

    job = db.get_job(job_uuid)
    assert job["status"] == "done", job.get("results")
    provider_results = job["results"]["provider_results"]
    providers = {r["provider"] for r in provider_results}
    assert providers == {"deepgram", "openai"}
    assert all(r["success"] for r in provider_results)
    # sarvam_judges defaults on, so the LLM-WER/CER scalars ride along.
    assert all("sarvam_llm_wer" in r["metrics"] for r in provider_results)
    assert all("sarvam_llm_cer" in r["metrics"] for r in provider_results)


def test_run_stt_evaluation_task_omits_sarvam_judges_when_disabled():
    from routers.stt import run_evaluation_task, STTEvaluationRequest

    job_uuid = _make_eval_job(
        "stt-eval",
        {
            "audio_paths": ["s3://bucket/key.wav"],
            "texts": ["hi"],
            "providers": ["openai"],
            "language": "en",
            "s3_bucket": "bucket",
            "evaluators": [],
            "sarvam_judges": False,
        },
    )

    with patch.dict(os.environ, {"FAKE_AI_PROVIDERS": "1"}), patch(
        "routers.stt.get_s3_client", return_value=MagicMock()
    ), patch("routers.stt.download_file_from_s3"), patch(
        "routers.stt.upload_file_to_s3"
    ), patch("routers.stt.upload_top_level_files_to_s3"), patch(
        "routers.stt.upload_directory_tree_to_s3"
    ), patch("routers.stt.try_start_queued_job"), patch("routers.stt.time.sleep"):
        request = STTEvaluationRequest(
            audio_paths=["s3://bucket/key.wav"],
            texts=["hi"],
            providers=["openai"],
            language="en",
            sarvam_judges=False,
        )
        run_evaluation_task(job_uuid, request, "bucket")

    job = db.get_job(job_uuid)
    assert job["status"] == "done", job.get("results")
    provider_results = job["results"]["provider_results"]
    assert all("sarvam_llm_wer" not in r["metrics"] for r in provider_results)
    assert all("sarvam_llm_cer" not in r["metrics"] for r in provider_results)


def test_run_tts_evaluation_task_end_to_end_with_fake_cli():
    from routers.tts import run_tts_evaluation_task, TTSEvaluationRequest

    job_uuid = _make_eval_job(
        "tts-eval",
        {
            "texts": ["hi"],
            "providers": ["cartesia"],
            "language": "en",
            "s3_bucket": "bucket",
            "evaluators": [],
        },
    )

    with patch.dict(os.environ, {"FAKE_AI_PROVIDERS": "1"}), patch(
        "routers.tts.get_s3_client", return_value=MagicMock()
    ), patch("routers.tts.upload_file_to_s3"), patch(
        "routers.tts.upload_top_level_files_to_s3"
    ), patch("routers.tts.upload_directory_tree_to_s3"), patch(
        "routers.tts.try_start_queued_job"
    ), patch("routers.tts.time.sleep"):
        request = TTSEvaluationRequest(
            texts=["hi"], providers=["cartesia"], language="en"
        )
        run_tts_evaluation_task(job_uuid, request, "bucket")

    job = db.get_job(job_uuid)
    assert job["status"] == "done", job.get("results")
    providers = {r["provider"] for r in job["results"]["provider_results"]}
    assert providers == {"cartesia"}
    assert all(r["success"] for r in job["results"]["provider_results"])


def test_run_simulation_task_text_end_to_end_with_fake_cli():
    from routers.simulations import run_simulation_task

    user_uuid = db.create_user("F", "S", f"fs-{os.urandom(4).hex()}@x.com")
    org_uuid = db.get_personal_org_for_user(user_uuid)["uuid"]
    sim_uuid = db.create_simulation(
        name=f"sim-{os.urandom(4).hex()}", org_uuid=org_uuid, user_id=user_uuid
    )
    job_uuid = db.create_simulation_job(
        simulation_id=sim_uuid, job_type="text", status="in_progress"
    )

    agent = {"uuid": "a", "name": "Agent", "config": {}}
    personas = [{"uuid": "p1", "name": "Alex", "config": {}}]
    scenarios = [{"uuid": "s1", "name": "Sc", "description": "desc"}]

    with patch.dict(os.environ, {"FAKE_AI_PROVIDERS": "1"}), patch(
        "routers.simulations.get_s3_client", return_value=MagicMock()
    ), patch("routers.simulations.upload_file_to_s3"), patch(
        "routers.simulations.try_start_queued_simulation_job"
    ), patch("routers.simulations.time.sleep"):
        run_simulation_task(
            job_uuid, agent, personas, scenarios, [], "bucket", "text"
        )

    job = db.get_simulation_job(job_uuid)
    assert job["status"] == "done", job.get("results")


def test_run_simulation_task_voice_end_to_end_with_fake_cli():
    """Voice simulations run the same `simulations` subcommand (``--type voice``)
    and read the same per-case output the fake writes."""
    from routers.simulations import run_simulation_task

    user_uuid = db.create_user("F", "V", f"fv-{os.urandom(4).hex()}@x.com")
    org_uuid = db.get_personal_org_for_user(user_uuid)["uuid"]
    sim_uuid = db.create_simulation(
        name=f"sim-{os.urandom(4).hex()}", org_uuid=org_uuid, user_id=user_uuid
    )
    job_uuid = db.create_simulation_job(
        simulation_id=sim_uuid, job_type="voice", status="in_progress"
    )

    agent = {"uuid": "a", "name": "Agent", "config": {}}
    personas = [{"uuid": "p1", "name": "Alex", "config": {}}]
    scenarios = [{"uuid": "s1", "name": "Sc", "description": "desc"}]

    with patch.dict(os.environ, {"FAKE_AI_PROVIDERS": "1"}), patch(
        "routers.simulations.get_s3_client", return_value=MagicMock()
    ), patch("routers.simulations.upload_file_to_s3"), patch(
        "routers.simulations.try_start_queued_simulation_job"
    ), patch("routers.simulations.time.sleep"):
        run_simulation_task(
            job_uuid, agent, personas, scenarios, [], "bucket", "voice"
        )

    job = db.get_simulation_job(job_uuid)
    assert job["status"] == "done", job.get("results")


_ANNOTATION_CASES = {
    "stt": {"predicted_transcript": "pred", "reference_transcript": "ref"},
    "llm": {"chat_history": [{"role": "user", "content": "hi"}], "agent_response": "yes"},
    "llm-general": {"input": "q", "output": "a"},
    "conversation": {
        "transcript": [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello"},
        ]
    },
    "tts": {
        "text": "hello",
        "audio_path": "s3://bucket/tts/media/clip.wav",
    },
}


@pytest.mark.parametrize("task_type", list(_ANNOTATION_CASES))
def test_annotation_eval_run_job_end_to_end_with_fake_cli(task_type):
    """Drive ``_run_job`` for every supported task type with the real fake CLI.
    Asserts the parser lifted a completed evaluator_run for the single item."""
    from annotation_eval_runner import _run_job

    resolved = {
        "uuid": "ev-1",
        "name": "Safety",
        "judge_model": "gpt",
        "system_prompt": "judge",
        "output_type": "binary",
        "output_config": {},
        "variables": [],
        "variable_values": {},
        "kind": "single",
        "data_type": "text",
        "_evaluator_version_id": "ver-1",
    }
    items = [{"uuid": "item-1", "payload": _ANNOTATION_CASES[task_type]}]
    created = []
    updates = []

    with patch.dict(os.environ, {"FAKE_AI_PROVIDERS": "1"}), patch(
        "annotation_eval_runner.get_annotation_task", return_value={"type": task_type}
    ), patch(
        "annotation_eval_runner.get_eval_job_items", return_value=items
    ), patch(
        "annotation_eval_runner.get_job",
        return_value={"updated_at": "2099-01-01 00:00:00"},
    ), patch("annotation_eval_runner._persist_pgid"), patch(
        "annotation_eval_runner.snapshot_eval_job_items"
    ), patch(
        "annotation_eval_runner.create_evaluator_runs",
        side_effect=lambda rows: created.extend(rows),
    ), patch(
        "annotation_eval_runner.update_job",
        side_effect=lambda *a, **k: updates.append(k),
    ), patch("annotation_eval_runner.try_start_queued_job"), patch(
        "annotation_eval_runner.get_s3_client", return_value=MagicMock()
    ), patch(
        "annotation_eval_runner.get_s3_output_config", return_value="bucket"
    ), patch(
        "annotation_eval_runner.download_file_from_s3",
        side_effect=lambda _s3, _b, _k, local: Path(local).write_bytes(b"wav"),
    ), patch(
        "annotation_eval_runner.upload_file_to_s3"
    ), patch(
        "annotation_eval_runner.time.sleep"
    ):
        _run_job("j-1", "task-1", "u-1", [resolved], item_ids=None)

    assert any(u.get("status") == "done" for u in updates), updates
    assert created, f"no evaluator_runs created for task_type={task_type}"
    assert all(r["status"] == "completed" for r in created), created
    assert all(r["item_id"] == "item-1" for r in created)


def test_provider_status_run_check_short_circuits_under_flag():
    from provider_status import ProviderStatusMonitor

    monitor = ProviderStatusMonitor(
        refresh_interval_seconds=60,
        cache_max_age_seconds=60,
        check_timeout_seconds=5,
    )
    with patch.dict(os.environ, {"FAKE_AI_PROVIDERS": "1"}):
        providers = asyncio.run(monitor.run_check())

    assert providers, "expected a non-empty healthy provider set"
    assert all(info.get("status") == "pass" for info in providers.values())
