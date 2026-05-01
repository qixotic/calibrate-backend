"""Job recovery module - restarts in_progress jobs on app startup."""

import os
import signal
import threading
import logging
import time

from db import (
    get_pending_jobs,
    get_agent,
    get_test,
    update_job,
    get_pending_agent_test_jobs,
    get_pending_simulation_jobs,
    update_agent_test_job,
    update_simulation_job,
    get_persona,
    get_scenario,
    get_personas_for_simulation,
    get_scenarios_for_simulation,
    get_evaluators_for_simulation,
    get_simulation,
    get_queued_jobs,
    get_queued_agent_test_jobs,
    get_queued_simulation_jobs,
)
from utils import (
    TaskStatus,
    try_start_queued_job,
    try_start_queued_agent_test_job,
    try_start_queued_simulation_job,
)

logger = logging.getLogger(__name__)

# Job type constants
EVAL_JOB_TYPES = ["stt-eval", "tts-eval", "annotation-eval"]
AGENT_TEST_JOB_TYPES = ["llm-unit-test", "llm-benchmark"]
SIMULATION_JOB_TYPES = ["text", "voice"]


def _start_queued_jobs():
    """Start queued jobs if there's capacity after recovery."""
    # Check for queued generic jobs (stt-eval, tts-eval)
    queued_jobs = get_queued_jobs(EVAL_JOB_TYPES)
    if queued_jobs:
        logger.info(f"Found {len(queued_jobs)} queued eval job(s), attempting to start")
        # Try to start as many as capacity allows
        while try_start_queued_job(EVAL_JOB_TYPES):
            pass

    # Check for queued agent test jobs
    queued_agent_test_jobs = get_queued_agent_test_jobs(AGENT_TEST_JOB_TYPES)
    if queued_agent_test_jobs:
        logger.info(
            f"Found {len(queued_agent_test_jobs)} queued agent test job(s), attempting to start"
        )
        while try_start_queued_agent_test_job(AGENT_TEST_JOB_TYPES):
            pass

    # Check for queued simulation jobs
    queued_simulation_jobs = get_queued_simulation_jobs(SIMULATION_JOB_TYPES)
    if queued_simulation_jobs:
        logger.info(
            f"Found {len(queued_simulation_jobs)} queued simulation job(s), attempting to start"
        )
        while try_start_queued_simulation_job(SIMULATION_JOB_TYPES):
            pass


def _kill_orphaned_processes_from_dict(pids_dict: dict, job_id: str) -> None:
    """
    Kill multiple orphaned processes from a dict mapping (e.g., provider -> PID).

    Args:
        pids_dict: Dict mapping names to PIDs (e.g., {"deepgram": 12345, "openai": 12346})
        job_id: Job ID for logging
    """
    if not pids_dict:
        logger.info(f"Job {job_id}: No running PIDs to kill")
        return

    for name, pid in pids_dict.items():
        if not pid:
            continue
        try:
            # Kill the process group (PID equals PGID when start_new_session=True)
            os.killpg(pid, signal.SIGTERM)
            logger.info(f"Job {job_id}: Sent SIGTERM to process group {pid} ({name})")

            time.sleep(0.5)

            try:
                os.killpg(pid, signal.SIGKILL)
                logger.info(
                    f"Job {job_id}: Sent SIGKILL to process group {pid} ({name})"
                )
            except ProcessLookupError:
                logger.info(
                    f"Job {job_id}: Process group {pid} ({name}) already terminated"
                )
        except ProcessLookupError:
            logger.info(f"Job {job_id}: Process group {pid} ({name}) not found")
        except PermissionError:
            logger.warning(
                f"Job {job_id}: No permission to kill process group {pid} ({name})"
            )
        except Exception as e:
            logger.error(
                f"Job {job_id}: Error killing process group {pid} ({name}): {e}"
            )


def _kill_orphaned_process(details: dict, job_id: str) -> bool:
    """
    Kill an orphaned process from a previous run.

    Args:
        details: Job details containing 'pid' and/or 'pgid'
        job_id: Job ID for logging

    Returns:
        True if process was killed or didn't exist, False on error
    """
    pgid = details.get("pgid")
    pid = details.get("pid")

    if not pgid and not pid:
        logger.info(f"Job {job_id}: No PID/PGID stored, nothing to kill")
        return True

    # Try to kill the process group first (kills all child processes)
    if pgid:
        try:
            # Kill the entire process group
            os.killpg(pgid, signal.SIGTERM)
            logger.info(f"Job {job_id}: Sent SIGTERM to process group {pgid}")

            # Give it a moment to terminate gracefully
            time.sleep(1)

            # Check if still running and force kill
            try:
                os.killpg(pgid, signal.SIGKILL)
                logger.info(f"Job {job_id}: Sent SIGKILL to process group {pgid}")
            except ProcessLookupError:
                logger.info(f"Job {job_id}: Process group {pgid} already terminated")
            except PermissionError:
                logger.warning(
                    f"Job {job_id}: No permission to kill process group {pgid}"
                )

            return True
        except ProcessLookupError:
            logger.info(f"Job {job_id}: Process group {pgid} not found (already dead)")
            return True
        except PermissionError:
            logger.warning(f"Job {job_id}: No permission to kill process group {pgid}")
            # Fall through to try killing by PID
        except Exception as e:
            logger.error(f"Job {job_id}: Error killing process group {pgid}: {e}")
            # Fall through to try killing by PID

    # Fallback: try to kill by PID
    if pid:
        try:
            os.kill(pid, signal.SIGTERM)
            logger.info(f"Job {job_id}: Sent SIGTERM to process {pid}")

            time.sleep(1)

            try:
                os.kill(pid, signal.SIGKILL)
                logger.info(f"Job {job_id}: Sent SIGKILL to process {pid}")
            except ProcessLookupError:
                logger.info(f"Job {job_id}: Process {pid} already terminated")

            return True
        except ProcessLookupError:
            logger.info(f"Job {job_id}: Process {pid} not found (already dead)")
            return True
        except PermissionError:
            logger.warning(f"Job {job_id}: No permission to kill process {pid}")
            return False
        except Exception as e:
            logger.error(f"Job {job_id}: Error killing process {pid}: {e}")
            return False

    return True


def recover_pending_jobs():
    """Check for in_progress jobs and restart them."""
    # Recover generic jobs
    pending_jobs = get_pending_jobs()
    if pending_jobs:
        logger.info(f"Found {len(pending_jobs)} in_progress generic job(s) to recover")
        for job in pending_jobs:
            job_id = job["uuid"]
            job_type = job["type"]
            details = job.get("details")

            if not details:
                logger.warning(f"Job {job_id} has no details, marking as failed")
                update_job(
                    job_id,
                    status=TaskStatus.DONE.value,
                    results={"error": "Job recovery failed: no details available"},
                )
                continue

            try:
                if job_type == "stt-eval":
                    _recover_stt_job(job_id, details)
                elif job_type == "tts-eval":
                    _recover_tts_job(job_id, details)
                elif job_type == "annotation-eval":
                    _recover_annotation_eval_job(job)
                else:
                    logger.warning(f"Unknown job type: {job_type}, marking as failed")
                    update_job(
                        job_id,
                        status=TaskStatus.DONE.value,
                        results={
                            "error": f"Job recovery failed: unknown job type {job_type}"
                        },
                    )
            except Exception as e:
                logger.error(f"Failed to recover job {job_id}: {e}")
                update_job(
                    job_id,
                    status=TaskStatus.DONE.value,
                    results={"error": f"Job recovery failed: {str(e)}"},
                )
    else:
        logger.info("No in_progress generic jobs to recover")

    # Recover agent test jobs
    pending_agent_test_jobs = get_pending_agent_test_jobs()
    if pending_agent_test_jobs:
        logger.info(
            f"Found {len(pending_agent_test_jobs)} in_progress agent test job(s) to recover"
        )
        for job in pending_agent_test_jobs:
            job_id = job["uuid"]
            job_type = job["type"]
            details = job.get("details")

            if not details:
                logger.warning(
                    f"Agent test job {job_id} has no details, marking as failed"
                )
                update_agent_test_job(
                    job_id,
                    status=TaskStatus.DONE.value,
                    results={"error": "Job recovery failed: no details available"},
                )
                continue

            try:
                if job_type == "llm-unit-test":
                    _recover_llm_unit_test_job(job_id, details)
                elif job_type == "llm-benchmark":
                    _recover_llm_benchmark_job(job_id, details)
                else:
                    logger.warning(
                        f"Unknown agent test job type: {job_type}, marking as failed"
                    )
                    update_agent_test_job(
                        job_id,
                        status=TaskStatus.DONE.value,
                        results={
                            "error": f"Job recovery failed: unknown job type {job_type}"
                        },
                    )
            except Exception as e:
                logger.error(f"Failed to recover agent test job {job_id}: {e}")
                update_agent_test_job(
                    job_id,
                    status=TaskStatus.DONE.value,
                    results={"error": f"Job recovery failed: {str(e)}"},
                )
    else:
        logger.info("No in_progress agent test jobs to recover")

    # Recover simulation jobs
    pending_simulation_jobs = get_pending_simulation_jobs()
    if pending_simulation_jobs:
        logger.info(
            f"Found {len(pending_simulation_jobs)} in_progress simulation job(s) to recover"
        )
        for job in pending_simulation_jobs:
            job_id = job["uuid"]
            job_type = job["type"]
            details = job.get("details")

            if not details:
                logger.warning(
                    f"Simulation job {job_id} has no details, marking as failed"
                )
                update_simulation_job(
                    job_id,
                    status=TaskStatus.DONE.value,
                    results={"error": "Job recovery failed: no details available"},
                )
                continue

            try:
                if job_type in ["text", "voice"]:
                    _recover_simulation_job(job_id, details, job_type)
                else:
                    logger.warning(
                        f"Unknown simulation job type: {job_type}, marking as failed"
                    )
                    update_simulation_job(
                        job_id,
                        status=TaskStatus.DONE.value,
                        results={
                            "error": f"Job recovery failed: unknown job type {job_type}"
                        },
                    )
            except Exception as e:
                logger.error(f"Failed to recover simulation job {job_id}: {e}")
                update_simulation_job(
                    job_id,
                    status=TaskStatus.DONE.value,
                    results={"error": f"Job recovery failed: {str(e)}"},
                )
    else:
        logger.info("No in_progress simulation jobs to recover")

    # Start queued jobs if there's capacity
    _start_queued_jobs()


def _recover_stt_job(job_id: str, details: dict):
    """Recover an STT evaluation job."""
    from routers.stt import run_evaluation_task, STTEvaluationRequest

    logger.info(f"Recovering STT job {job_id}")

    # Kill any orphaned processes from previous run
    _kill_orphaned_processes_from_dict(details.get("running_pids", {}), job_id)

    request = STTEvaluationRequest(
        audio_paths=details["audio_paths"],
        texts=details["texts"],
        providers=details["providers"],
        language=details["language"],
    )
    s3_bucket = details["s3_bucket"]

    thread = threading.Thread(
        target=run_evaluation_task,
        args=(job_id, request, s3_bucket),
        daemon=True,
    )
    thread.start()
    logger.info(f"STT job {job_id} recovery started")


def _recover_tts_job(job_id: str, details: dict):
    """Recover a TTS evaluation job."""
    from routers.tts import run_tts_evaluation_task, TTSEvaluationRequest

    logger.info(f"Recovering TTS job {job_id}")

    # Kill any orphaned processes from previous run
    _kill_orphaned_processes_from_dict(details.get("running_pids", {}), job_id)

    request = TTSEvaluationRequest(
        texts=details["texts"],
        providers=details["providers"],
        language=details["language"],
    )
    s3_bucket = details["s3_bucket"]

    thread = threading.Thread(
        target=run_tts_evaluation_task,
        args=(job_id, request, s3_bucket),
        daemon=True,
    )
    thread.start()
    logger.info(f"TTS job {job_id} recovery started")


def _recover_llm_unit_test_job(job_id: str, details: dict):
    """Recover an LLM unit test job."""
    from routers.agent_tests import run_llm_test_task

    logger.info(f"Recovering LLM unit test job {job_id}")

    agent_uuid = details["agent_uuid"]
    test_uuids = details["test_uuids"]
    s3_bucket = details["s3_bucket"]

    # Fetch agent and tests
    agent = get_agent(agent_uuid)
    if not agent:
        raise ValueError(f"Agent {agent_uuid} not found")

    tests = []
    for test_uuid in test_uuids:
        test = get_test(test_uuid)
        if not test:
            raise ValueError(f"Test {test_uuid} not found")
        tests.append(test)

    thread = threading.Thread(
        target=run_llm_test_task,
        args=(job_id, agent, tests, s3_bucket),
        daemon=True,
    )
    thread.start()
    logger.info(f"LLM unit test job {job_id} recovery started")


def _recover_llm_benchmark_job(job_id: str, details: dict):
    """Recover an LLM benchmark job."""
    from routers.agent_tests import run_benchmark_task

    logger.info(f"Recovering LLM benchmark job {job_id}")

    agent_uuid = details["agent_uuid"]
    test_uuids = details["test_uuids"]
    models = details["models"]
    s3_bucket = details["s3_bucket"]

    # Fetch agent and tests
    agent = get_agent(agent_uuid)
    if not agent:
        raise ValueError(f"Agent {agent_uuid} not found")

    tests = []
    for test_uuid in test_uuids:
        test = get_test(test_uuid)
        if not test:
            raise ValueError(f"Test {test_uuid} not found")
        tests.append(test)

    thread = threading.Thread(
        target=run_benchmark_task,
        args=(job_id, agent, tests, models, s3_bucket),
        daemon=True,
    )
    thread.start()
    logger.info(f"LLM benchmark job {job_id} recovery started")


def _recover_annotation_eval_job(job: dict):
    """Recover an annotation evaluator-run job by killing the orphaned
    `calibrate` subprocess (if any) and restarting via the queue starter.
    The runner clears stale `evaluator_runs` rows for the job before
    re-inserting, so a partial previous run does not double-count.
    """
    from annotation_eval_runner import resume_annotation_eval_job

    job_id = job["uuid"]
    details = job.get("details") or {}
    logger.info(f"Recovering annotation eval job {job_id}")

    # Best-effort kill of the orphaned subprocess.
    _kill_orphaned_process(details, job_id)

    if not details.get("evaluators"):
        raise ValueError("details.evaluators missing — cannot reconstruct run")

    resume_annotation_eval_job(job)
    logger.info(f"Annotation eval job {job_id} recovery started")


def _recover_simulation_job(job_id: str, details: dict, job_type: str):
    """Recover a simulation job (text or voice)."""
    from routers.simulations import run_simulation_task

    logger.info(f"Recovering simulation job {job_id} (type: {job_type})")

    # For voice simulations, kill any orphaned processes first
    if job_type == "voice":
        logger.info(f"Killing orphaned processes for voice simulation job {job_id}")
        _kill_orphaned_process(details, job_id)

    simulation_uuid = details["simulation_uuid"]
    agent_uuid = details["agent_uuid"]
    s3_bucket = details["s3_bucket"]

    # Verify simulation exists
    simulation = get_simulation(simulation_uuid)
    if not simulation:
        raise ValueError(f"Simulation {simulation_uuid} not found")

    # Fetch agent
    agent = get_agent(agent_uuid)
    if not agent:
        raise ValueError(f"Agent {agent_uuid} not found")

    # Fetch personas, scenarios, and evaluators
    personas = get_personas_for_simulation(simulation_uuid)
    scenarios = get_scenarios_for_simulation(simulation_uuid)
    evaluators = get_evaluators_for_simulation(simulation_uuid)

    if not personas:
        raise ValueError(f"Simulation {simulation_uuid} has no personas")
    if not scenarios:
        raise ValueError(f"Simulation {simulation_uuid} has no scenarios")

    thread = threading.Thread(
        target=run_simulation_task,
        args=(job_id, agent, personas, scenarios, evaluators, s3_bucket, job_type),
        daemon=True,
    )
    thread.start()
    logger.info(f"Simulation job {job_id} recovery started")
