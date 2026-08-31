"""Trace-scoring eligibility and plan resolution.

Shared by agent opt-in and ingest-time run creation. Lives outside `routers/`
so `db.py` can import it without a db→router cycle.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Literal

from shared_enums import (
    REQUIRED_EVALUATOR_TYPE_BY_TEST_TYPE,
    AgentInteractionType,
    EvaluatorType,
)

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
