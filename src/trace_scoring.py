"""Trace-scoring eligibility and plan resolution.

Shared by the agent opt-in API and ingest-time run creation. Lives outside
`routers/` so `db.py` can call it without importing a router.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

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
