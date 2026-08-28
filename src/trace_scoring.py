"""Trace-scoring eligibility, plan resolution, and the claim/invoke/settle engine.

Shared by the agent opt-in API, ingest-time run creation, and the scoring
loop. Lives outside `routers/` so `db.py` can call it without importing a
router. There is no lifespan worker here — callers (tests, later PR 6)
drive `claim_and_score_batch`.
"""

from __future__ import annotations

import json
import logging
import random
import shutil
import subprocess
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from llm_judge import build_test_evaluators_payload
from utils import get_calibrate_agent_cli, kill_process_group

# interaction_type → (evaluation.type, required evaluator_type). Kept here
# (not imported from routers.tests) so resolution never creates a db→router
# cycle. Must stay aligned with REQUIRED_EVALUATOR_TYPE_BY_TEST_TYPE for
# `response`/`general`.
TRACE_SCORING_MODE_BY_INTERACTION_TYPE: Dict[str, Tuple[str, str]] = {
    "conversation": ("response", "llm"),
    "general": ("general", "llm-general"),
}

INELIGIBLE_REASON_WRONG_TYPE = "wrong_type_for_agent"
INELIGIBLE_REASON_NO_LIVE_VERSION = "no_live_version"
INELIGIBLE_REASON_DECLARES_VARIABLES = "declares_variables"


@dataclass(frozen=True)
class TraceScoringPin:
    evaluator_uuid: str
    evaluator_version_id: str
    name: str


@dataclass(frozen=True)
class TraceScoringIneligible:
    evaluator_uuid: str
    name: str
    reason: str


@dataclass(frozen=True)
class TraceScoringResolution:
    evaluation_type: Optional[str]
    evaluator_type: Optional[str]
    eligible: List[TraceScoringPin] = field(default_factory=list)
    ineligible: List[TraceScoringIneligible] = field(default_factory=list)

    def as_plan(self) -> Dict[str, Any]:
        """Snapshot envelope for a new run, or a skip reason if nothing can score."""
        if self.evaluation_type is None:
            return {"skip": "unsupported_interaction_type"}
        if not self.eligible:
            return {"skip": "no_usable_evaluators"}
        return {
            "type": self.evaluation_type,
            "evaluators": [
                {
                    "evaluator_uuid": pin.evaluator_uuid,
                    "evaluator_version_id": pin.evaluator_version_id,
                }
                for pin in self.eligible
            ],
        }

    def ineligible_payload(self) -> List[Dict[str, str]]:
        return [
            {
                "evaluator_uuid": item.evaluator_uuid,
                "name": item.name,
                "reason": item.reason,
            }
            for item in self.ineligible
        ]


def partition_trace_scoring_evaluators(
    interaction_type: Optional[str],
    evaluators: List[Dict[str, Any]],
    versions_by_uuid: Dict[str, Dict[str, Any]],
) -> TraceScoringResolution:
    """Split linked evaluators into eligible pins and ineligible-with-reason.

    Filters to the required evaluator type *before* live-version / variable
    checks. A mixed linked set must not be handed to `_validate_evaluators`,
    which raises on the first type mismatch. Never raises.
    """
    mode = TRACE_SCORING_MODE_BY_INTERACTION_TYPE.get(interaction_type or "")
    if mode is None:
        return TraceScoringResolution(
            evaluation_type=None,
            evaluator_type=None,
            eligible=[],
            ineligible=[
                TraceScoringIneligible(
                    evaluator_uuid=ev["uuid"],
                    name=ev.get("name") or ev["uuid"],
                    reason=INELIGIBLE_REASON_WRONG_TYPE,
                )
                for ev in evaluators
            ],
        )

    evaluation_type, required_evaluator_type = mode
    eligible: List[TraceScoringPin] = []
    ineligible: List[TraceScoringIneligible] = []
    for ev in evaluators:
        name = ev.get("name") or ev["uuid"]
        if ev.get("evaluator_type") != required_evaluator_type:
            ineligible.append(
                TraceScoringIneligible(
                    evaluator_uuid=ev["uuid"],
                    name=name,
                    reason=INELIGIBLE_REASON_WRONG_TYPE,
                )
            )
            continue
        live_id = ev.get("live_version_id") or ""
        version = versions_by_uuid.get(live_id)
        if not version:
            ineligible.append(
                TraceScoringIneligible(
                    evaluator_uuid=ev["uuid"],
                    name=name,
                    reason=INELIGIBLE_REASON_NO_LIVE_VERSION,
                )
            )
            continue
        if version.get("variables"):
            ineligible.append(
                TraceScoringIneligible(
                    evaluator_uuid=ev["uuid"],
                    name=name,
                    reason=INELIGIBLE_REASON_DECLARES_VARIABLES,
                )
            )
            continue
        eligible.append(
            TraceScoringPin(
                evaluator_uuid=ev["uuid"],
                evaluator_version_id=version["uuid"],
                name=name,
            )
        )
    return TraceScoringResolution(
        evaluation_type=evaluation_type,
        evaluator_type=required_evaluator_type,
        eligible=eligible,
        ineligible=ineligible,
    )


def resolve_trace_scoring(agent: Dict[str, Any]) -> TraceScoringResolution:
    """Load this agent's linked evaluators and partition them. Never raises."""
    from db import get_evaluator_versions_by_uuids, get_evaluators_for_agent

    evaluators = get_evaluators_for_agent(agent["uuid"])
    live_ids = [ev.get("live_version_id") for ev in evaluators if ev.get("live_version_id")]
    versions = get_evaluator_versions_by_uuids(live_ids)
    return partition_trace_scoring_evaluators(
        agent.get("interaction_type"), evaluators, versions
    )


def resolve_scoring_plan(agent: Dict[str, Any]) -> Dict[str, Any]:
    """Plan written into `trace_evaluations.criteria`, or a skip envelope."""
    return resolve_trace_scoring(agent).as_plan()


logger = logging.getLogger(__name__)

# Lease covers worst-case batch wall-clock (CLI startup ~4s plus judge calls).
# Renewal is out of scope; an expired lease is reclaimed by the same claim scan.
CLAIM_BATCH_SIZE = 20
CLAIM_LEASE_SECONDS = 30 * 60
CLI_TIMEOUT_SECONDS = 25 * 60
CLI_PARALLEL = 4
MAX_ATTEMPTS = 5
ERROR_MAX_CHARS = 2000
_BACKOFF_BASE_SECONDS = 30
_BACKOFF_CAP_SECONDS = 15 * 60
_CORRUPT_SNAPSHOT = "corrupt_snapshot"


@dataclass(frozen=True)
class EvalOnlyCliResult:
    returncode: int
    timed_out: bool
    results: List[Any]
    error: str = ""


@dataclass
class _PreparedRun:
    run: Dict[str, Any]
    snapshot: Dict[str, Any]
    trace: Dict[str, Any]
    hydrated: List[Dict[str, Any]]


def parse_criteria_snapshot(raw: Any) -> Optional[Dict[str, Any]]:
    """Deserialize the immutable run snapshot. None if it cannot be scored."""
    if raw is None or raw == "":
        return None
    if isinstance(raw, str):
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            return None
    elif isinstance(raw, dict):
        data = raw
    else:
        return None
    if data.get("type") not in ("response", "general"):
        return None
    pins_in = data.get("evaluators")
    if not isinstance(pins_in, list) or not pins_in:
        return None
    pins: List[Dict[str, str]] = []
    seen: set = set()
    for item in pins_in:
        if not isinstance(item, dict):
            return None
        evaluator_uuid = item.get("evaluator_uuid")
        version_id = item.get("evaluator_version_id")
        if not evaluator_uuid or not version_id:
            return None
        if evaluator_uuid in seen:
            return None
        seen.add(evaluator_uuid)
        pins.append(
            {
                "evaluator_uuid": evaluator_uuid,
                "evaluator_version_id": version_id,
            }
        )
    return {"type": data["type"], "evaluators": pins}


def hydrate_pinned_evaluators(
    pins: Sequence[Dict[str, str]],
) -> Optional[List[Dict[str, Any]]]:
    """Load execution defs from each pin's version UUID, never the live version.

    Includes soft-deleted historical versions and soft-deleted evaluators.
    Returns None when any pin is missing or `version.evaluator_id` mismatches.
    """
    from db import get_evaluator_versions_by_uuids, get_evaluators_by_uuids

    version_ids = [p["evaluator_version_id"] for p in pins]
    evaluator_ids = [p["evaluator_uuid"] for p in pins]
    versions = get_evaluator_versions_by_uuids(version_ids)
    evaluators = get_evaluators_by_uuids(evaluator_ids, include_deleted=True)
    hydrated: List[Dict[str, Any]] = []
    for pin in pins:
        version = versions.get(pin["evaluator_version_id"])
        evaluator = evaluators.get(pin["evaluator_uuid"])
        if (
            version is None
            or evaluator is None
            or version.get("evaluator_id") != pin["evaluator_uuid"]
        ):
            return None
        hydrated.append(
            {
                "uuid": evaluator["uuid"],
                "name": evaluator.get("name") or evaluator["uuid"],
                "output_type": evaluator.get("output_type") or "binary",
                "judge_model": version.get("judge_model"),
                "system_prompt": version.get("system_prompt") or "",
                "output_config": version.get("output_config"),
                "variables": version.get("variables"),
                "evaluator_version_id": version["uuid"],
                "variable_values": {},
            }
        )
    return hydrated


def build_dataset_item(
    run_uuid: str,
    evaluation_type: str,
    trace: Dict[str, Any],
    criteria_refs: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """One eval-only dataset item. `test_case.id` is the run UUID for mapping."""
    output = trace.get("output") if isinstance(trace.get("output"), dict) else {}
    response = output.get("response")
    response_text = "" if response is None else str(response)
    test_case: Dict[str, Any] = {
        "id": run_uuid,
        "evaluation": {"type": evaluation_type, "criteria": criteria_refs},
    }
    if evaluation_type == "general":
        raw_input = trace.get("input")
        test_case["input"] = raw_input if isinstance(raw_input, str) else str(raw_input or "")
        return {"test_case": test_case, "output": {"response": response_text}}
    history = trace.get("input")
    test_case["history"] = history if isinstance(history, list) else []
    tool_calls = output.get("tool_calls")
    if not isinstance(tool_calls, list):
        tool_calls = []
    return {
        "test_case": test_case,
        "output": {"response": response_text, "tool_calls": tool_calls},
    }


def build_eval_only_batch(
    prepared: Sequence[_PreparedRun],
) -> Tuple[Dict[str, Any], List[Dict[str, Any]], Dict[str, str]]:
    """Union evaluators (deduped by UUID, collision-safe names) plus per-run items.

    Runtime names are suffixed with the evaluator UUID when two display names
    collide. The returned manifest maps that runtime name → evaluator UUID.
    """
    tests_with_evaluators = [
        {"test_uuid": item.run["uuid"], "evaluators": item.hydrated}
        for item in prepared
    ]
    top_level, criteria_per_test = build_test_evaluators_payload(tests_with_evaluators)
    dataset = [
        build_dataset_item(
            item.run["uuid"],
            item.snapshot["type"],
            item.trace,
            criteria_per_test.get(item.run["uuid"]) or [],
        )
        for item in prepared
    ]
    manifest = {ev["name"]: ev["id"] for ev in top_level if ev.get("name") and ev.get("id")}
    return {"evaluators": top_level}, dataset, manifest


def _run_id_from_result(entry: Any) -> Optional[str]:
    if not isinstance(entry, dict):
        return None
    run_id = entry.get("test_case_id")
    if run_id:
        return str(run_id)
    test_case = entry.get("test_case")
    if isinstance(test_case, dict) and test_case.get("id"):
        return str(test_case["id"])
    return None


def index_cli_results(results: Any) -> Dict[str, Optional[Dict[str, Any]]]:
    """Map `test_case_id` → entry. Duplicate ids are stored as None (unsettleable)."""
    indexed: Dict[str, Optional[Dict[str, Any]]] = {}
    if not isinstance(results, list):
        return indexed
    duplicates: set = set()
    for entry in results:
        run_id = _run_id_from_result(entry)
        if not run_id:
            continue
        if run_id in indexed or run_id in duplicates:
            indexed.pop(run_id, None)
            duplicates.add(run_id)
            continue
        indexed[run_id] = entry
    return indexed


def _typed_score_columns(
    judgement: Dict[str, Any], output_type: str
) -> Optional[Dict[str, Any]]:
    reasoning = judgement.get("reasoning")
    if not isinstance(reasoning, str):
        reasoning = None if reasoning is None else str(reasoning)
    if output_type == "rating":
        if "score" not in judgement or judgement["score"] is None:
            return None
        try:
            score = float(judgement["score"])
        except (TypeError, ValueError):
            return None
        return {"match": None, "score": score, "reasoning": reasoning}
    if "match" not in judgement:
        return None
    match = judgement["match"]
    if match is True or match == 1:
        return {"match": 1, "score": None, "reasoning": reasoning}
    if match is False or match == 0:
        return {"match": 0, "score": None, "reasoning": reasoning}
    return None


def map_item_scores(
    entry: Dict[str, Any],
    *,
    pin_uuids: Sequence[str],
    name_to_uuid: Dict[str, str],
    hydrated_by_uuid: Dict[str, Dict[str, Any]],
) -> Optional[List[Dict[str, Any]]]:
    """Map one CLI result onto snapshot evaluators. None if it cannot settle.

    Keys by `evaluator_id` first, then runtime name. Unknown, duplicate,
    malformed, or incomplete sets are rejected — never mapped by position.
    """
    expected = list(pin_uuids)
    expected_set = set(expected)
    metrics = entry.get("metrics")
    if not isinstance(metrics, dict):
        return None
    judge_results = metrics.get("judge_results")
    if not isinstance(judge_results, dict) or not judge_results:
        return None
    mapped: Dict[str, Dict[str, Any]] = {}
    for runtime_name, judgement in judge_results.items():
        if not isinstance(judgement, dict):
            return None
        evaluator_uuid = judgement.get("evaluator_id") or name_to_uuid.get(runtime_name)
        if not evaluator_uuid or evaluator_uuid not in expected_set:
            return None
        if evaluator_uuid in mapped:
            return None
        hydrated = hydrated_by_uuid.get(evaluator_uuid)
        if hydrated is None:
            return None
        columns = _typed_score_columns(judgement, hydrated["output_type"])
        if columns is None:
            return None
        mapped[evaluator_uuid] = {
            "evaluator_uuid": evaluator_uuid,
            "evaluator_version_id": hydrated["evaluator_version_id"],
            **columns,
        }
    if set(mapped) != expected_set:
        return None
    return [mapped[uid] for uid in expected]


def parse_results_json(path: Path) -> List[Any]:
    """Best-effort parse of a possibly partial `results.json`. Invalid → []."""
    try:
        with open(path, encoding="utf-8") as handle:
            parsed = json.load(handle)
    except (OSError, TypeError, ValueError):
        return []
    return parsed if isinstance(parsed, list) else []


def backoff_available_at(
    attempts: int,
    now: int,
    rng: Optional[random.Random] = None,
) -> int:
    """Exponential backoff plus jitter so a failed batch does not reassemble."""
    roller = rng or random
    delay = min(_BACKOFF_BASE_SECONDS * (2 ** max(attempts - 1, 0)), _BACKOFF_CAP_SECONDS)
    jitter = roller.randint(0, max(delay // 2, 1))
    return now + delay + jitter


def _truncate_error(detail: str) -> str:
    text = (detail or "").strip() or "scoring failed"
    if len(text) <= ERROR_MAX_CHARS:
        return text
    return text[: ERROR_MAX_CHARS - 3] + "..."


def _wait_cli_process(proc: subprocess.Popen, timeout: float) -> None:
    try:
        proc.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        pass


def _reap_cli_process(proc: subprocess.Popen) -> None:
    """Kill the process group, then the Popen child, until it has exited.

    Matches the simulation abort fallback (`kill_process_group` then
    `process.kill()`). Callers must not delete the temp dir while `poll()`
    is still None.
    """
    if proc.poll() is not None:
        return
    group_ok = kill_process_group(proc.pid, "trace-scoring")
    _wait_cli_process(proc, 5)
    if proc.poll() is not None:
        return
    logger.warning(
        "trace-scoring: pid %s still alive after process-group kill (ok=%s); "
        "falling back to process.kill()",
        proc.pid,
        group_ok,
    )
    try:
        proc.kill()
    except OSError:
        pass
    _wait_cli_process(proc, 5)


def _cleanup_cli_tempdir(tmp_path: Path, proc: Optional[subprocess.Popen]) -> None:
    if proc is not None and proc.poll() is None:
        logger.error(
            "trace-scoring: leaving temp dir %s because pid %s is still running",
            tmp_path,
            proc.pid,
        )
        return
    shutil.rmtree(tmp_path, ignore_errors=True)


def invoke_eval_only_cli(
    config: Dict[str, Any],
    dataset: List[Dict[str, Any]],
    *,
    timeout_seconds: int = CLI_TIMEOUT_SECONDS,
    parallel: int = CLI_PARALLEL,
) -> EvalOnlyCliResult:
    """Spawn `calibrate-agent llm --eval-only` with temp-file stdio and a timeout."""
    tmp_path = Path(tempfile.mkdtemp(prefix="trace-scoring-"))
    proc: Optional[subprocess.Popen] = None
    timed_out = False
    try:
        config_path = tmp_path / "config.json"
        dataset_path = tmp_path / "dataset.json"
        output_dir = tmp_path / "output"
        output_dir.mkdir()
        config_path.write_text(json.dumps(config), encoding="utf-8")
        dataset_path.write_text(json.dumps(dataset), encoding="utf-8")
        stdout_path = output_dir / "stdout.log"
        stderr_path = output_dir / "stderr.log"
        cli = get_calibrate_agent_cli()
        cmd = [
            cli,
            "llm",
            "-c",
            str(config_path),
            "--eval-only",
            "--dataset",
            str(dataset_path),
            "-o",
            str(output_dir),
            "-n",
            str(parallel),
        ]
        with open(stdout_path, "w") as out_f, open(stderr_path, "w") as err_f:
            proc = subprocess.Popen(
                cmd,
                stdout=out_f,
                stderr=err_f,
                text=True,
                start_new_session=True,
                cwd=str(tmp_path),
            )
            try:
                proc.wait(timeout=timeout_seconds)
            except subprocess.TimeoutExpired:
                timed_out = True
                _reap_cli_process(proc)
        returncode = proc.returncode
        if returncode is None:
            returncode = -9 if timed_out else 0
        try:
            stderr = stderr_path.read_text(encoding="utf-8")
        except OSError:
            stderr = ""
        results = parse_results_json(output_dir / "results.json")
        error = ""
        if timed_out:
            error = f"calibrate-agent llm --eval-only timed out after {timeout_seconds}s"
        elif returncode != 0:
            error = stderr.strip().splitlines()[-1] if stderr.strip() else (
                f"calibrate-agent llm --eval-only exited {returncode}"
            )
        return EvalOnlyCliResult(
            returncode=returncode,
            timed_out=timed_out,
            results=results,
            error=error,
        )
    finally:
        _cleanup_cli_tempdir(tmp_path, proc)


def _fail_run(run: Dict[str, Any], error: str, now: int) -> None:
    from db import settle_trace_evaluation_terminal

    settle_trace_evaluation_terminal(
        run["uuid"], "failed", error=_truncate_error(error), now=now
    )


def _skip_run(run: Dict[str, Any], reason: str, now: int) -> None:
    from db import settle_trace_evaluation_terminal

    settle_trace_evaluation_terminal(run["uuid"], "skipped", error=reason, now=now)


def _defer_or_fail(
    run: Dict[str, Any],
    *,
    now: int,
    error: str,
    rng: Optional[random.Random],
    max_attempts: int,
) -> None:
    from db import defer_trace_evaluation

    attempts = int(run.get("attempts") or 0)
    if attempts >= max_attempts:
        _fail_run(run, error, now)
        return
    defer_trace_evaluation(
        run["uuid"],
        available_at=backoff_available_at(attempts, now, rng),
        now=now,
        error=_truncate_error(error),
    )


def _prepare_claimed_run(run: Dict[str, Any], now: int) -> Optional[_PreparedRun]:
    from db import get_trace, trace_scoring_skip_reason

    skip = trace_scoring_skip_reason(run["org_uuid"], run["trace_uuid"], run["agent_id"])
    if skip:
        _skip_run(run, skip, now)
        return None
    snapshot = parse_criteria_snapshot(run.get("criteria"))
    if snapshot is None:
        _fail_run(run, _CORRUPT_SNAPSHOT, now)
        return None
    hydrated = hydrate_pinned_evaluators(snapshot["evaluators"])
    if hydrated is None:
        _fail_run(run, _CORRUPT_SNAPSHOT, now)
        return None
    trace = get_trace(run["org_uuid"], run["trace_uuid"])
    if trace is None:
        _skip_run(run, "trace_deleted", now)
        return None
    return _PreparedRun(run=run, snapshot=snapshot, trace=trace, hydrated=hydrated)


def process_claimed_runs(
    claimed: Sequence[Dict[str, Any]],
    *,
    now: Optional[int] = None,
    invoke: Optional[Callable[..., EvalOnlyCliResult]] = None,
    rng: Optional[random.Random] = None,
    max_attempts: int = MAX_ATTEMPTS,
    timeout_seconds: int = CLI_TIMEOUT_SECONDS,
) -> None:
    """Hydrate the immutable snapshot, invoke eval-only once, settle by id.

    `now` is claim/prepare time only. Completion, defer/backoff, failure, and
    post-invoke deletion skips use a fresh `time.time()` after the CLI returns
    so a long invoke cannot write `available_at` in the past.
    """
    from db import settle_trace_evaluation_completed

    if not claimed:
        return
    prepare_now = int(now if now is not None else time.time())
    invoke_fn = invoke or invoke_eval_only_cli
    prepared: List[_PreparedRun] = []
    for run in claimed:
        item = _prepare_claimed_run(run, prepare_now)
        if item is not None:
            prepared.append(item)
    if not prepared:
        return

    config, dataset, manifest = build_eval_only_batch(prepared)
    try:
        cli_result = invoke_fn(
            config, dataset, timeout_seconds=timeout_seconds, parallel=CLI_PARALLEL
        )
    except Exception as exc:
        logger.exception("trace scoring CLI invoke failed")
        settle_now = int(time.time())
        for item in prepared:
            _defer_or_fail(
                item.run,
                now=settle_now,
                error=str(exc),
                rng=rng,
                max_attempts=max_attempts,
            )
        return

    indexed = index_cli_results(cli_result.results)
    complete: Dict[str, List[Dict[str, Any]]] = {}
    for item in prepared:
        entry = indexed.get(item.run["uuid"])
        if not isinstance(entry, dict):
            continue
        hydrated_by_uuid = {ev["uuid"]: ev for ev in item.hydrated}
        scores = map_item_scores(
            entry,
            pin_uuids=[pin["evaluator_uuid"] for pin in item.snapshot["evaluators"]],
            name_to_uuid=manifest,
            hydrated_by_uuid=hydrated_by_uuid,
        )
        if scores is not None:
            complete[item.run["uuid"]] = scores

    # Finished peers settle; leftovers always defer-or-fail with jitter so a
    # poison partial/timeout batch cannot sit in `processing` forever.
    default_error = cli_result.error or "incomplete evaluator results"
    settle_now = int(time.time())

    for item in prepared:
        scores = complete.get(item.run["uuid"])
        if scores is not None:
            settle_trace_evaluation_completed(item.run["uuid"], scores, now=settle_now)
            continue
        _defer_or_fail(
            item.run,
            now=settle_now,
            error=default_error,
            rng=rng,
            max_attempts=max_attempts,
        )


def claim_and_score_batch(
    *,
    now: Optional[int] = None,
    batch_size: int = CLAIM_BATCH_SIZE,
    lease_seconds: int = CLAIM_LEASE_SECONDS,
    invoke: Optional[Callable[..., EvalOnlyCliResult]] = None,
    rng: Optional[random.Random] = None,
    max_attempts: int = MAX_ATTEMPTS,
    timeout_seconds: int = CLI_TIMEOUT_SECONDS,
) -> List[Dict[str, Any]]:
    """Claim a batch and score it. No worker loop — the caller drives this."""
    from db import claim_trace_evaluations

    now = int(now if now is not None else time.time())
    claimed = claim_trace_evaluations(
        now=now, lease_seconds=lease_seconds, batch_size=batch_size
    )
    process_claimed_runs(
        claimed,
        now=now,
        invoke=invoke,
        rng=rng,
        max_attempts=max_attempts,
        timeout_seconds=timeout_seconds,
    )
    return claimed

