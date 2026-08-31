"""Trace-scoring eligibility, plan resolution, and the claim/invoke/settle engine.

Shared by agent opt-in, ingest-time run creation, and the scoring loop. Lives
outside `routers/` so `db.py` can import it without a db→router cycle; `db`
imports here are function-local for the same reason.

There is no worker loop here — callers drive `claim_and_score_batch`.
"""

from __future__ import annotations

import json
import logging
import random
import shutil
import subprocess
import tempfile
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Literal

from llm_judge import _scale_bounds, build_test_evaluators_payload
from shared_enums import (
    REQUIRED_EVALUATOR_TYPE_BY_TEST_TYPE,
    AgentInteractionType,
    EvaluatorType,
)
from utils import get_calibrate_agent_cli, kill_process_group

logger = logging.getLogger(__name__)

# Stored on a skipped `trace_eval_runs.error` when ingest cannot build a plan.
TraceEvalSkipReason = Literal["unsupported_interaction_type", "no_usable_evaluators"]


class TraceEvalSettleSkipReason(str, Enum):
    """Why a claimed run was abandoned. Also stored on `error`.

    Separate from `TraceEvalSkipReason`: these are reachable only after a run
    exists, so ingest can never write one and a reader can tell where a skip
    came from.
    """

    TRACE_DELETED = "trace_deleted"
    AGENT_DELETED = "agent_deleted"


# Stored on a failed run whose pinned versions no longer resolve. Every other
# `failed` carries free-text detail, so this is a sentinel, not a vocabulary.
CORRUPT_SNAPSHOT_ERROR = "corrupt_snapshot"

# Subset of TestType that traces can score.
TraceScorableEvaluationType = Literal["response", "general"]

TRACE_SCORING_MODE_BY_INTERACTION_TYPE: dict[
    AgentInteractionType,
    tuple[TraceScorableEvaluationType, EvaluatorType],
] = {
    "conversation": ("response", REQUIRED_EVALUATOR_TYPE_BY_TEST_TYPE["response"]),
    "general": ("general", REQUIRED_EVALUATOR_TYPE_BY_TEST_TYPE["general"]),
}


class IneligibleReason(str, Enum):
    """Why a linked evaluator cannot score this agent's traces."""

    WRONG_TYPE = "wrong_type_for_agent"
    NO_LIVE_VERSION = "no_live_version"
    DECLARES_VARIABLES = "declares_variables"


# Lifecycle of one `trace_eval_runs` row.
class TraceEvalRunStatus(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


# DB partial index ux_trace_eval_active is unique on trace_uuid for only these statuses, so a trace can have many
# completed/failed/skipped runs but at most one still open.
OPEN_TRACE_EVAL_RUN_STATUSES: tuple[TraceEvalRunStatus, ...] = (
    TraceEvalRunStatus.PENDING,
    TraceEvalRunStatus.PROCESSING,
)


def scale_bounds_from_output_config(raw: Any) -> tuple[float | None, float | None]:
    """Numeric min/max of a rating rubric, from the stored JSON TEXT or a dict."""
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except (TypeError, ValueError):
            return (None, None)
    if not isinstance(raw, dict):
        return (None, None)
    return _scale_bounds(raw)


def trace_evaluator_passed(
    output_type: str | None, value: Any, scale_max: Any
) -> bool:
    """CLI pass rule (`_evaluator_passed`): binary passes on 1, rating at scale_max."""
    if value is None:
        return False
    if output_type == "rating":
        if scale_max is None:
            return False
        try:
            return int(value) == int(scale_max)
        except (TypeError, ValueError):
            return False
    return bool(value)


@dataclass(frozen=True)
class ScoringPlanPin:
    """One evaluator pin stored within a `trace_eval_runs.scoring_plan`."""

    evaluator_uuid: str
    evaluator_version_id: str


@dataclass(frozen=True)
class ScoringPlan:
    """JSON envelope pinning evaluators to use in a runnable `trace_eval_runs` row."""

    evaluation_type: TraceScorableEvaluationType
    evaluators: list[ScoringPlanPin]


@dataclass(frozen=True)
class ScoringPlanSkip:
    """Why ingest wrote a `skipped` run instead of a runnable plan."""

    skip: TraceEvalSkipReason


@dataclass(frozen=True)
class TraceScoringEligible:
    """Eligible snapshot pin plus the evaluator name for eligibility responses."""

    pin: ScoringPlanPin
    name: str


@dataclass(frozen=True)
class TraceScoringIneligible:
    evaluator_uuid: str
    name: str
    reason: IneligibleReason


@dataclass(frozen=True)
class TraceScoringResolution:
    evaluation_type: TraceScorableEvaluationType | None
    evaluator_type: EvaluatorType | None
    eligible: list[TraceScoringEligible] = field(default_factory=list)
    ineligible: list[TraceScoringIneligible] = field(default_factory=list)

    def as_plan(self) -> ScoringPlan | ScoringPlanSkip:
        """Snapshot written at ingest, or a skip reason if nothing can score."""
        if self.evaluation_type is None:
            return ScoringPlanSkip(skip="unsupported_interaction_type")
        if not self.eligible:
            return ScoringPlanSkip(skip="no_usable_evaluators")
        return ScoringPlan(
            evaluation_type=self.evaluation_type,
            evaluators=[item.pin for item in self.eligible],
        )


def resolve_trace_scoring(
    interaction_type: AgentInteractionType | None,
    live_evaluators: list[tuple[dict[str, Any], dict[str, Any] | None]],
) -> TraceScoringResolution:
    """Partition linked evaluators for this interaction type.

    `live_evaluators` is `(evaluators row, live evaluator_versions row or None)`
    from `resolve_live_evaluators`. Type is checked before live-version /
    variable checks so a mixed set never reaches `_validate_evaluators`.
    Never raises.
    """
    mode = TRACE_SCORING_MODE_BY_INTERACTION_TYPE.get(interaction_type) if interaction_type is not None else None
    if mode is None:
        return TraceScoringResolution(
            evaluation_type=None,
            evaluator_type=None,
            eligible=[],
            ineligible=[
                TraceScoringIneligible(
                    evaluator_uuid=ev["uuid"],
                    name=ev.get("name") or ev["uuid"],
                    reason=IneligibleReason.WRONG_TYPE,
                )
                for ev, _ in live_evaluators
            ],
        )

    evaluation_type, required_evaluator_type = mode
    eligible: list[TraceScoringEligible] = []
    ineligible: list[TraceScoringIneligible] = []
    for ev, version in live_evaluators:
        name = ev.get("name") or ev["uuid"]
        if ev.get("evaluator_type") != required_evaluator_type:
            ineligible.append(
                TraceScoringIneligible(
                    evaluator_uuid=ev["uuid"],
                    name=name,
                    reason=IneligibleReason.WRONG_TYPE,
                )
            )
            continue
        if not version:
            ineligible.append(
                TraceScoringIneligible(
                    evaluator_uuid=ev["uuid"],
                    name=name,
                    reason=IneligibleReason.NO_LIVE_VERSION,
                )
            )
            continue
        if version.get("variables"):
            ineligible.append(
                TraceScoringIneligible(
                    evaluator_uuid=ev["uuid"],
                    name=name,
                    reason=IneligibleReason.DECLARES_VARIABLES,
                )
            )
            continue
        eligible.append(
            TraceScoringEligible(
                pin=ScoringPlanPin(
                    evaluator_uuid=ev["uuid"],
                    evaluator_version_id=version["uuid"],
                ),
                name=name,
            )
        )
    return TraceScoringResolution(
        evaluation_type=evaluation_type,
        evaluator_type=required_evaluator_type,
        eligible=eligible,
        ineligible=ineligible,
    )


# The lease must outlast the CLI timeout, or a still-running invocation's runs
# are reclaimed and double-scored while the first worker is mid-flight.
CLAIM_BATCH_SIZE = 20
CLI_TIMEOUT_SECONDS = 25 * 60
CLAIM_LEASE_SECONDS = 30 * 60
assert CLAIM_LEASE_SECONDS > CLI_TIMEOUT_SECONDS

CLI_PARALLEL = 4
MAX_ATTEMPTS = 5
ERROR_MAX_CHARS = 2000
_BACKOFF_BASE_SECONDS = 30
_BACKOFF_CAP_SECONDS = 15 * 60


@dataclass(frozen=True)
class EvalOnlyCliResult:
    returncode: int
    timed_out: bool
    results: list[Any]
    error: str = ""


@dataclass(frozen=True)
class PreparedRun:
    """A claimed run that passed liveness, snapshot, and hydration checks."""

    run: dict[str, Any]
    plan: ScoringPlan
    trace: dict[str, Any]
    hydrated: list[dict[str, Any]]


def parse_scoring_plan(raw: Any) -> ScoringPlan | None:
    """Deserialize a stored `scoring_plan`. None means the run cannot be scored.

    Mirrors what `create_trace_with_eval_run` writes (`asdict(ScoringPlan)`).
    Anything else — malformed JSON, an unknown evaluation type, no pins, a
    duplicated evaluator — is a corrupt snapshot, never a partial run.
    """
    if not isinstance(raw, str):
        return None
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return None
    if not isinstance(data, dict):
        return None
    evaluation_type = data.get("evaluation_type")
    if evaluation_type not in ("response", "general"):
        return None
    raw_pins = data.get("evaluators")
    if not isinstance(raw_pins, list) or not raw_pins:
        return None
    pins: list[ScoringPlanPin] = []
    seen: set[str] = set()
    for item in raw_pins:
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
            ScoringPlanPin(
                evaluator_uuid=evaluator_uuid,
                evaluator_version_id=version_id,
            )
        )
    return ScoringPlan(evaluation_type=evaluation_type, evaluators=pins)


def hydrate_pinned_evaluators(
    pins: Sequence[ScoringPlanPin],
) -> list[dict[str, Any]] | None:
    """Load execution defs from each pin's version, never the evaluator's live one.

    Version rows carry no `deleted_at` and the evaluator is read with
    `include_deleted`, so a version pinned months ago still resolves after the
    evaluator was edited or deleted — the run reproduces what it pinned. None
    when any pin is missing or its version belongs to another evaluator.
    """
    from db import get_evaluator_versions_by_uuids, get_evaluators_by_uuids

    versions = get_evaluator_versions_by_uuids(
        [pin.evaluator_version_id for pin in pins]
    )
    evaluators = get_evaluators_by_uuids(
        [pin.evaluator_uuid for pin in pins], include_deleted=True
    )
    hydrated: list[dict[str, Any]] = []
    for pin in pins:
        version = versions.get(pin.evaluator_version_id)
        evaluator = evaluators.get(pin.evaluator_uuid)
        if (
            version is None
            or evaluator is None
            or version.get("evaluator_id") != pin.evaluator_uuid
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
                "evaluator_version_id": version["uuid"],
            }
        )
    return hydrated


def build_dataset_item(
    run_uuid: str,
    evaluation_type: TraceScorableEvaluationType,
    trace: dict[str, Any],
    criteria_refs: list[dict[str, Any]],
) -> dict[str, Any]:
    """One eval-only dataset item. `test_case.id` is the run uuid, which the
    runner echoes back as `test_case_id` — the only thing results are mapped
    by, so a reordered or partial results file still lands correctly."""
    output = trace.get("output") if isinstance(trace.get("output"), dict) else {}
    response = output.get("response")
    test_case: dict[str, Any] = {
        "id": run_uuid,
        "evaluation": {"type": evaluation_type, "criteria": criteria_refs},
    }
    if evaluation_type == "general":
        raw_input = trace.get("input")
        test_case["input"] = (
            raw_input if isinstance(raw_input, str) else str(raw_input or "")
        )
        return {
            "test_case": test_case,
            "output": {"response": "" if response is None else str(response)},
        }
    history = trace.get("input")
    test_case["history"] = history if isinstance(history, list) else []
    tool_calls = output.get("tool_calls")
    return {
        "test_case": test_case,
        "output": {
            "response": "" if response is None else str(response),
            "tool_calls": tool_calls if isinstance(tool_calls, list) else [],
        },
    }


def build_eval_only_batch(
    prepared: Sequence[PreparedRun],
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, str]]:
    """Assemble one invocation covering the whole batch.

    Returns `(config, dataset, manifest)`. The config's evaluator list is the
    union across runs deduped by uuid, so two runs sharing an evaluator ship one
    definition; the manifest maps the runtime name calibrate keys its output by
    (uuid-suffixed on a display-name collision) back to the evaluator uuid.
    """
    top_level, criteria_per_run = build_test_evaluators_payload(
        [
            {"test_uuid": item.run["uuid"], "evaluators": item.hydrated}
            for item in prepared
        ]
    )
    dataset = [
        build_dataset_item(
            item.run["uuid"],
            item.plan.evaluation_type,
            item.trace,
            criteria_per_run.get(item.run["uuid"]) or [],
        )
        for item in prepared
    ]
    manifest = {
        ev["name"]: ev["id"] for ev in top_level if ev.get("name") and ev.get("id")
    }
    return {"evaluators": top_level}, dataset, manifest


def _run_uuid_from_result(entry: Any) -> str | None:
    if not isinstance(entry, dict):
        return None
    run_uuid = entry.get("test_case_id")
    if run_uuid:
        return str(run_uuid)
    test_case = entry.get("test_case")
    if isinstance(test_case, dict) and test_case.get("id"):
        return str(test_case["id"])
    return None


def index_cli_results(results: Any) -> dict[str, dict[str, Any]]:
    """Map `test_case_id` → entry. A duplicated id drops both copies: there is
    no way to tell which one belongs to the run, and guessing would persist a
    score against the wrong trace."""
    indexed: dict[str, dict[str, Any]] = {}
    if not isinstance(results, list):
        return indexed
    duplicated: set[str] = set()
    for entry in results:
        run_uuid = _run_uuid_from_result(entry)
        if not run_uuid:
            continue
        if run_uuid in indexed or run_uuid in duplicated:
            indexed.pop(run_uuid, None)
            duplicated.add(run_uuid)
            continue
        indexed[run_uuid] = entry
    return indexed


def _typed_score(judgement: dict[str, Any], output_type: str) -> dict[str, Any] | None:
    """One judge verdict as the `value` + `output_type` the schema stores.

    None for anything the CHECK constraints would reject — a null rating, a
    non-numeric one, a non-boolean binary. Rejecting here rather than at INSERT
    keeps a malformed verdict a deferrable run instead of an IntegrityError
    raised inside the settle transaction.
    """
    reasoning = judgement.get("reasoning")
    if reasoning is not None and not isinstance(reasoning, str):
        reasoning = str(reasoning)
    if output_type == "rating":
        score = judgement.get("score")
        if score is None or isinstance(score, bool):
            return None
        try:
            value = float(score)
        except (TypeError, ValueError):
            return None
        return {"value": value, "output_type": "rating", "reasoning": reasoning}
    match = judgement.get("match")
    if match is True or match == 1:
        return {"value": 1, "output_type": "binary", "reasoning": reasoning}
    if match is False or match == 0:
        return {"value": 0, "output_type": "binary", "reasoning": reasoning}
    return None


def map_item_scores(
    entry: dict[str, Any],
    *,
    pins: Sequence[ScoringPlanPin],
    name_to_uuid: dict[str, str],
    hydrated_by_uuid: dict[str, dict[str, Any]],
) -> list[dict[str, Any]] | None:
    """Map one result onto the run's pinned evaluators, or None if it cannot settle.

    Keys by `evaluator_id`, falling back to the runtime name — never by
    position. A result that names an evaluator the run did not pin, repeats one,
    carries an unreadable verdict, or covers only part of the snapshot leaves
    the run open for a retry rather than settling it half-scored.
    """
    expected = [pin.evaluator_uuid for pin in pins]
    metrics = entry.get("metrics")
    if not isinstance(metrics, dict):
        return None
    judge_results = metrics.get("judge_results")
    if not isinstance(judge_results, dict) or not judge_results:
        return None
    mapped: dict[str, dict[str, Any]] = {}
    for runtime_name, judgement in judge_results.items():
        if not isinstance(judgement, dict):
            return None
        evaluator_uuid = judgement.get("evaluator_id") or name_to_uuid.get(runtime_name)
        if evaluator_uuid not in expected or evaluator_uuid in mapped:
            return None
        hydrated = hydrated_by_uuid[evaluator_uuid]
        score = _typed_score(judgement, hydrated["output_type"])
        if score is None:
            return None
        mapped[evaluator_uuid] = {
            "evaluator_uuid": evaluator_uuid,
            "evaluator_version_id": hydrated["evaluator_version_id"],
            **score,
        }
    if set(mapped) != set(expected):
        return None
    return [mapped[uid] for uid in expected]


def parse_results_json(path: Path) -> list[Any]:
    """Best-effort read of a possibly partial `results.json`.

    The runner rewrites the whole array after each item with no atomic replace,
    so a read landing mid-rewrite hits invalid JSON. Unreadable means "nothing
    finished yet" — the next attempt picks it up.
    """
    try:
        parsed = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return []
    return parsed if isinstance(parsed, list) else []


def backoff_available_at(
    attempts: int,
    now: int,
    rng: random.Random | None = None,
) -> int:
    """Exponential backoff plus jitter. The jitter is not decorative: without it
    a whole-invocation failure defers every run in the batch to the same
    instant, and the next claim reassembles and re-fails the identical batch."""
    roller = rng or random
    delay = min(
        _BACKOFF_BASE_SECONDS * (2 ** max(attempts - 1, 0)), _BACKOFF_CAP_SECONDS
    )
    return now + delay + roller.randint(0, max(delay // 2, 1))


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
    """Kill the process group, then the child directly if it outlives that.

    Mirrors the simulation abort fallback. Callers must not delete the temp dir
    until this leaves `poll()` non-None — a surviving child still writes there.
    """
    if proc.poll() is not None:
        return
    group_killed = kill_process_group(proc.pid, "trace-scoring")
    _wait_cli_process(proc, 5)
    if proc.poll() is not None:
        return
    logger.warning(
        "trace-scoring: pid %s survived the process-group kill (ok=%s); "
        "falling back to process.kill()",
        proc.pid,
        group_killed,
    )
    try:
        proc.kill()
    except OSError:
        pass
    _wait_cli_process(proc, 5)


def _cleanup_cli_tempdir(tmp_path: Path, proc: subprocess.Popen | None) -> None:
    if proc is not None and proc.poll() is None:
        logger.error(
            "trace-scoring: leaving temp dir %s behind because pid %s is still running",
            tmp_path,
            proc.pid,
        )
        return
    shutil.rmtree(tmp_path, ignore_errors=True)


def invoke_eval_only_cli(
    config: dict[str, Any],
    dataset: list[dict[str, Any]],
    *,
    timeout_seconds: int = CLI_TIMEOUT_SECONDS,
    parallel: int = CLI_PARALLEL,
) -> EvalOnlyCliResult:
    """Score one batch with `calibrate-agent llm --eval-only`.

    stdio goes to files rather than pipes so a chatty run cannot deadlock on a
    full buffer. Results are read even on timeout or a non-zero exit: the runner
    rewrites `results.json` after every item, so the runs that finished before
    the failure can still settle.
    """
    tmp_path = Path(tempfile.mkdtemp(prefix="trace-scoring-"))
    proc: subprocess.Popen | None = None
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
            stderr = stderr_path.read_text(encoding="utf-8").strip()
        except OSError:
            stderr = ""
        error = ""
        if timed_out:
            error = f"calibrate-agent llm --eval-only timed out after {timeout_seconds}s"
        elif returncode != 0:
            error = (
                stderr.splitlines()[-1]
                if stderr
                else f"calibrate-agent llm --eval-only exited {returncode}"
            )
        return EvalOnlyCliResult(
            returncode=returncode,
            timed_out=timed_out,
            results=parse_results_json(output_dir / "results.json"),
            error=error,
        )
    finally:
        _cleanup_cli_tempdir(tmp_path, proc)


def _fail_run(run: dict[str, Any], error: str, now: int) -> None:
    from db import settle_trace_eval_run_terminal

    settle_trace_eval_run_terminal(
        run["uuid"],
        status=TraceEvalRunStatus.FAILED,
        error=_truncate_error(error),
        now=now,
    )


def _defer_or_fail(
    run: dict[str, Any],
    *,
    now: int,
    error: str,
    rng: random.Random | None,
    max_attempts: int,
) -> None:
    """Hand an unfinished run back for a retry, or bury it at the ceiling.

    Never leaves the run `processing`: a leftover there is invisible to the
    attempt ceiling and reassembles into the same poison batch every lease.
    """
    from db import defer_trace_eval_run

    attempts = int(run.get("attempts") or 0)
    if attempts >= max_attempts:
        _fail_run(run, error, now)
        return
    defer_trace_eval_run(
        run["uuid"],
        available_at=backoff_available_at(attempts, now, rng),
        now=now,
        error=_truncate_error(error),
    )


def _prepare_claimed_run(run: dict[str, Any], now: int) -> PreparedRun | None:
    """Settle everything a claimed run can be settled by without a judge call."""
    from db import get_trace, settle_trace_eval_run_terminal, trace_scoring_skip_reason

    def skip(reason: TraceEvalSettleSkipReason) -> None:
        settle_trace_eval_run_terminal(
            run["uuid"],
            status=TraceEvalRunStatus.SKIPPED,
            error=reason.value,
            now=now,
        )

    deleted = trace_scoring_skip_reason(
        run["org_uuid"], run["trace_uuid"], run["agent_id"]
    )
    if deleted is not None:
        skip(deleted)
        return None
    plan = parse_scoring_plan(run.get("scoring_plan"))
    if plan is None:
        _fail_run(run, CORRUPT_SNAPSHOT_ERROR, now)
        return None
    hydrated = hydrate_pinned_evaluators(plan.evaluators)
    if hydrated is None:
        _fail_run(run, CORRUPT_SNAPSHOT_ERROR, now)
        return None
    trace = get_trace(run["org_uuid"], run["trace_uuid"])
    if trace is None:
        skip(TraceEvalSettleSkipReason.TRACE_DELETED)
        return None
    return PreparedRun(run=run, plan=plan, trace=trace, hydrated=hydrated)


def process_claimed_runs(
    claimed: Sequence[dict[str, Any]],
    *,
    now: int | None = None,
    invoke: Callable[..., EvalOnlyCliResult] | None = None,
    rng: random.Random | None = None,
    max_attempts: int = MAX_ATTEMPTS,
    timeout_seconds: int = CLI_TIMEOUT_SECONDS,
) -> None:
    """Hydrate each snapshot, score the batch in one invocation, settle by id.

    `now` stamps preparation only. Everything after the CLI returns re-reads the
    clock, because an invocation can run for many minutes and reusing the claim
    time would write a backoff `available_at` that is already in the past.
    """
    from db import settle_trace_eval_run_completed

    prepare_now = int(now if now is not None else time.time())
    prepared: list[PreparedRun] = []
    for run in claimed:
        try:
            item = _prepare_claimed_run(run, prepare_now)
        except Exception as exc:
            # A run left in `processing` is invisible to the attempt ceiling, so
            # one whose preparation always raises (a trace holding unreadable
            # JSON, say) would be reclaimed forever. Defer it like any other.
            logger.exception("trace-scoring: preparing run %s raised", run["uuid"])
            _defer_or_fail(
                run,
                now=prepare_now,
                error=str(exc),
                rng=rng,
                max_attempts=max_attempts,
            )
            continue
        if item is not None:
            prepared.append(item)
    if not prepared:
        return

    config, dataset, manifest = build_eval_only_batch(prepared)
    invoke_fn = invoke or invoke_eval_only_cli
    try:
        cli_result = invoke_fn(
            config, dataset, timeout_seconds=timeout_seconds, parallel=CLI_PARALLEL
        )
    except Exception as exc:
        logger.exception("trace-scoring: eval-only invocation raised")
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
    scored: dict[str, list[dict[str, Any]]] = {}
    for item in prepared:
        entry = indexed.get(item.run["uuid"])
        if entry is None:
            continue
        scores = map_item_scores(
            entry,
            pins=item.plan.evaluators,
            name_to_uuid=manifest,
            hydrated_by_uuid={ev["uuid"]: ev for ev in item.hydrated},
        )
        if scores is not None:
            scored[item.run["uuid"]] = scores

    settle_now = int(time.time())
    leftover_error = cli_result.error or "incomplete evaluator results"
    for item in prepared:
        scores = scored.get(item.run["uuid"])
        if scores is None:
            _defer_or_fail(
                item.run,
                now=settle_now,
                error=leftover_error,
                rng=rng,
                max_attempts=max_attempts,
            )
            continue
        settle_trace_eval_run_completed(item.run["uuid"], scores, now=settle_now)


def claim_and_score_batch(
    *,
    now: int | None = None,
    batch_size: int = CLAIM_BATCH_SIZE,
    lease_seconds: int = CLAIM_LEASE_SECONDS,
    invoke: Callable[..., EvalOnlyCliResult] | None = None,
    rng: random.Random | None = None,
    max_attempts: int = MAX_ATTEMPTS,
    timeout_seconds: int = CLI_TIMEOUT_SECONDS,
) -> list[dict[str, Any]]:
    """Claim one batch and score it. No loop — the caller decides when to run."""
    from db import claim_trace_eval_runs

    now = int(now if now is not None else time.time())
    claimed = claim_trace_eval_runs(
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
