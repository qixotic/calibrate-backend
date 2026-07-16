import os
import json
import subprocess
import time
import traceback
import threading
import logging
import tempfile
from pathlib import Path
from typing import Optional, List, Dict, Any, Literal

from fastapi import APIRouter, HTTPException, Depends, Path as PathParam
from pydantic import BaseModel, ConfigDict, Field

from db import (
    create_simulation,
    ensure_name_unique,
    get_simulation,
    get_all_simulations,
    update_simulation,
    delete_simulation,
    get_persona,
    get_scenario,
    get_evaluator,
    get_evaluator_uuid_for_legacy_metric,
    legacy_metric_uuid_exists,
    add_persona_to_simulation,
    add_scenario_to_simulation,
    add_evaluator_to_simulation,
    remove_persona_from_simulation,
    remove_scenario_from_simulation,
    remove_evaluator_from_simulation,
    get_personas_for_simulation,
    get_scenarios_for_simulation,
    get_evaluators_for_simulation,
    get_agent,
    get_agents_by_uuids,
    get_tools_for_agent,
    create_simulation_job,
    get_simulation_job,
    update_simulation_job,
    update_simulation_job_visibility,
    get_simulation_jobs_for_simulation,
    get_simulation_jobs_summary,
    delete_simulation_job,
)
from llm_judge import build_evaluator_cli_payload
from utils import (
    AGENT_TYPE_DESCRIPTION,
    TaskStatus,
    TaskCreateResponse,
    SimulationRunType,
    EvaluatorTypeLiteral,
    DataTypeLiteral,
    OutputTypeLiteral,
    EvaluatorKindLiteral,
    DATA_TYPE_DESCRIPTION,
    OUTPUT_TYPE_DESCRIPTION,
    get_s3_client,
    get_s3_output_config,
    list_object_keys,
    can_start_simulation_job,
    try_start_queued_simulation_job,
    register_job_starter,
    generate_presigned_download_url,
    kill_process_group,
    is_job_timed_out,
    capture_exception_to_sentry,
    build_tool_configs,
    get_calibrate_agent_cli,
    PRESIGNED_URL_EXPIRY_SECONDS,
    PRESIGNED_URL_REFRESH_BUFFER_SECONDS,
    upload_file_to_s3,
    env_bool,
    env_int,
    env_str,
)
from auth_utils import get_current_org, OrgContext
from datetime import datetime

# Job types that share the same queue
SIMULATION_JOB_TYPES = ["text", "voice"]

_EXAMPLE_ID = "f47ac10b-58cc-4372-a567-0e02b2c3d479"


def _is_job_aborted(task_id: str) -> bool:
    """Check if a simulation job was aborted by the user."""
    job = get_simulation_job(task_id)
    return bool(job and (job.get("details") or {}).get("aborted"))


def _start_simulation_job_from_queue(job: dict) -> bool:
    """Start a simulation job from the queue."""
    job_id = job["uuid"]
    job_type = job.get("type")  # 'text' or 'voice'
    details = job.get("details", {})

    simulation_uuid = details.get("simulation_uuid")
    agent_uuid = details.get("agent_uuid")
    s3_bucket = details.get("s3_bucket", "")

    # Get simulation details
    simulation = get_simulation(simulation_uuid)
    if not simulation:
        return False

    # Get agent
    agent = get_agent(agent_uuid)
    if not agent:
        return False

    # Get linked entities
    personas = get_personas_for_simulation(simulation_uuid)
    scenarios = get_scenarios_for_simulation(simulation_uuid)
    evaluators = get_evaluators_for_simulation(simulation_uuid)

    if not personas or not scenarios:
        return False

    # Start background task in a separate thread
    thread = threading.Thread(
        target=run_simulation_task,
        args=(job_id, agent, personas, scenarios, evaluators, s3_bucket, job_type),
        daemon=True,
    )
    thread.start()

    return True


# Register the job starters for simulation jobs
register_job_starter("text", _start_simulation_job_from_queue)
register_job_starter("voice", _start_simulation_job_from_queue)

logger = logging.getLogger(__name__)


router = APIRouter(prefix="/simulations", tags=["simulations"])


def _should_regenerate_presigned_urls(
    presigned_urls_generated_at: Optional[str],
) -> bool:
    """
    Check if presigned URLs need to be regenerated based on when they were created.

    Args:
        presigned_urls_generated_at: ISO timestamp when URLs were generated, or None

    Returns:
        True if URLs should be regenerated (expired or about to expire or never generated)
    """
    if not presigned_urls_generated_at:
        return True

    try:
        generated_at = datetime.fromisoformat(
            presigned_urls_generated_at.replace("Z", "+00:00")
        )
        # Remove timezone info for comparison with utcnow
        if generated_at.tzinfo is not None:
            generated_at = generated_at.replace(tzinfo=None)

        now = datetime.utcnow()
        elapsed_seconds = (now - generated_at).total_seconds()

        # Regenerate if elapsed time exceeds expiry minus buffer
        threshold = PRESIGNED_URL_EXPIRY_SECONDS - PRESIGNED_URL_REFRESH_BUFFER_SECONDS
        return elapsed_seconds >= threshold
    except Exception as e:
        logger.warning(f"Failed to parse presigned_urls_generated_at: {e}")
        return True


def _get_audio_urls_from_s3_key(
    s3_key_prefix: str,
    s3_bucket: str,
    transcript: Optional[List[Dict[str, Any]]] = None,
) -> List[str]:
    """
    List all audio files in an S3 key prefix and generate presigned URLs for them.

    Audio files are sorted in conversation order: for each exchange number N,
    the transcript is consulted to determine whether bot or user spoke first.
    Files are returned as [1_bot, 1_user, 2_bot, 2_user, ...] or
    [1_user, 1_bot, ...] depending on the transcript.

    Args:
        s3_key_prefix: S3 key prefix (e.g., "simulations/runs/task_id/simulation_persona_1_scenario_1/audios")
        s3_bucket: S3 bucket name
        transcript: Optional transcript to determine bot/user ordering per exchange

    Returns:
        List of presigned URLs for audio files, sorted in conversation order
    """
    try:
        s3 = get_s3_client()

        # Ensure prefix ends with / for directory listing
        if s3_key_prefix and not s3_key_prefix.endswith("/"):
            s3_key_prefix += "/"

        # List objects in the S3 prefix
        audio_extensions = {".wav", ".mp3", ".ogg"}
        audio_files = []

        for key in list_object_keys(s3, s3_bucket, s3_key_prefix):
            # Skip if it's a directory marker
            if key.endswith("/"):
                continue
            # Check if it's an audio file
            file_ext = Path(key).suffix.lower()
            if file_ext in audio_extensions:
                audio_files.append(key)

        # Group audio files by exchange number
        # Files are named like: 1_bot.wav, 1_user.wav, 2_bot.wav, 2_user.wav
        from collections import defaultdict

        exchanges: Dict[int, Dict[str, str]] = defaultdict(dict)
        ungrouped = []

        for key in audio_files:
            filename = Path(key).stem  # e.g., "1_bot"
            parts = filename.split("_", 1)
            if parts[0].isdigit():
                num = int(parts[0])
                role = parts[1]  # "bot" or "user"
                exchanges[num][role] = key
            else:
                ungrouped.append(key)

        # Determine speaker order from transcript (spoken turns only)
        # Spoken turns are those with "content" and role "assistant" or "user"
        # (tool_calls-only messages have no audio)
        spoken_order: List[str] = []  # ["bot", "user", "bot", "user", ...]
        if transcript:
            for turn in transcript:
                role = turn.get("role", "")
                if role in ("assistant", "user") and turn.get("content"):
                    spoken_order.append("bot" if role == "assistant" else "user")

        # Build sorted list: for each exchange, use transcript to determine order
        sorted_files = []
        for num in sorted(exchanges.keys()):
            group = exchanges[num]
            # Exchange N maps to spoken turns at indices (N-1)*2 and (N-1)*2+1
            exchange_idx = (num - 1) * 2
            if (
                exchange_idx < len(spoken_order)
                and spoken_order[exchange_idx] == "user"
            ):
                # User speaks first in this exchange
                order = ["user", "bot"]
            else:
                # Bot speaks first (default)
                order = ["bot", "user"]

            for role in order:
                if role in group:
                    sorted_files.append(group[role])

        # Append any ungrouped files at the end
        ungrouped.sort()
        sorted_files.extend(ungrouped)

        print(sorted_files)

        # Generate presigned URLs
        presigned_urls = []
        for audio_key in sorted_files:
            presigned_url = generate_presigned_download_url(audio_key, bucket=s3_bucket)
            if presigned_url:
                presigned_urls.append(presigned_url)
                logger.info(f"Generated presigned URL for {audio_key}")
            else:
                # Fallback to S3 path if presigned URL generation fails
                presigned_urls.append(f"s3://{s3_bucket}/{audio_key}")

        return presigned_urls

    except Exception as e:
        logger.error(
            f"Error listing audio files from S3 key prefix {s3_key_prefix}: {str(e)}"
        )
        return []


class EvaluatorRef(BaseModel):
    """Reference to an evaluator linked to a simulation."""

    model_config = ConfigDict(extra="forbid")

    evaluator_uuid: str = Field(
        min_length=36,
        max_length=36,
        description="Evaluator to link. Must be a `conversation`-type evaluator with a live version in your workspace",
        examples=[_EXAMPLE_ID],
    )
    variable_values: Optional[Dict[str, Any]] = Field(
        None,
        description="Values for the evaluator's `{{placeholder}}` variables, pinned at link time. Omit to use version defaults",
    )


class SimulationCreate(BaseModel):
    """Create body must use `evaluators` with evaluator IDs, not legacy metric IDs or aliases."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(description="Simulation name, unique within the workspace")
    agent_uuid: Optional[str] = Field(
        None,
        min_length=36,
        max_length=36,
        description="Agent under test. Omit to create without an agent",
        examples=[_EXAMPLE_ID],
    )
    persona_uuids: Optional[List[str]] = Field(
        None, description="Personas to link. Omit to link none"
    )
    scenario_uuids: Optional[List[str]] = Field(
        None, description="Scenarios to link. Omit to link none"
    )
    evaluators: Optional[List[EvaluatorRef]] = Field(
        None, description="`conversation` evaluators to link. Omit to link none"
    )


class SimulationUpdate(BaseModel):
    """Update body must use `evaluators` with evaluator IDs, not legacy metric IDs or aliases."""

    model_config = ConfigDict(extra="forbid")

    name: Optional[str] = Field(None, description="New name. Omit to leave unchanged")
    agent_uuid: Optional[str] = Field(
        None, description="Agent to link. Empty string (`\"\"`) clears the agent. Omit to leave unchanged"
    )
    persona_uuids: Optional[List[str]] = Field(
        None, description="The personas to use, replacing the current set. Omit to leave unchanged"
    )
    scenario_uuids: Optional[List[str]] = Field(
        None, description="The scenarios to use, replacing the current set. Omit to leave unchanged"
    )
    evaluators: Optional[List[EvaluatorRef]] = Field(
        None, description="The evaluators to use, replacing the current set. Omit to leave unchanged"
    )


def _resolve_simulation_evaluator_ref(
    ref: EvaluatorRef, org_uuid: str
) -> Dict[str, Any]:
    """Resolve and validate one simulation evaluator link. Rejects legacy `metrics.uuid` values."""
    evaluator_uuid = ref.evaluator_uuid.strip()
    evaluator = get_evaluator(evaluator_uuid)
    if not evaluator:
        migrated = get_evaluator_uuid_for_legacy_metric(evaluator_uuid)
        if migrated:
            raise HTTPException(
                status_code=400,
                detail=(
                    "Simulations accept evaluator UUIDs only, not legacy metric UUIDs. "
                    f"Replace {evaluator_uuid} with evaluator UUID {migrated}."
                ),
            )
        if legacy_metric_uuid_exists(evaluator_uuid):
            raise HTTPException(
                status_code=400,
                detail=(
                    "Simulations accept evaluator UUIDs only, not legacy metric UUIDs. "
                    "This UUID matches a legacy metric row that has no migrated evaluator."
                ),
            )
        raise HTTPException(
            status_code=404,
            detail=f"Evaluator {evaluator_uuid} not found",
        )
    if evaluator.get("org_uuid") is not None and evaluator["org_uuid"] != org_uuid:
        raise HTTPException(
            status_code=404,
            detail=f"Evaluator {evaluator_uuid} not found",
        )
    if evaluator.get("evaluator_type") != "conversation":
        raise HTTPException(
            status_code=400,
            detail=(
                f"Evaluator {evaluator_uuid} has evaluator_type="
                f"'{evaluator.get('evaluator_type')}'. Simulations only accept "
                f"'conversation' evaluators."
            ),
        )
    version_uuid = evaluator.get("live_version_id")
    if not version_uuid:
        raise HTTPException(
            status_code=400,
            detail=f"Evaluator {evaluator_uuid} has no live version",
        )
    return {
        "evaluator_uuid": evaluator_uuid,
        "version_uuid": version_uuid,
        "variable_values": ref.variable_values,
    }


class PersonaResponse(BaseModel):
    uuid: str = Field(
        min_length=36,
        max_length=36,
        description="Persona ID",
        examples=[_EXAMPLE_ID],
    )
    name: str = Field(description="Persona name")
    description: Optional[str] = Field(None, description="Persona description")
    config: Optional[Dict[str, Any]] = Field(
        None, description="Persona config (e.g. gender, language, interruption sensitivity)"
    )
    created_at: str = Field(description="When the persona was created (ISO 8601 UTC)")
    updated_at: str = Field(description="When the persona was last updated (ISO 8601 UTC)")


class ScenarioResponse(BaseModel):
    uuid: str = Field(
        min_length=36,
        max_length=36,
        description="Scenario ID",
        examples=[_EXAMPLE_ID],
    )
    name: str = Field(description="Scenario name")
    description: Optional[str] = Field(None, description="Scenario description")
    created_at: str = Field(description="When the scenario was created (ISO 8601 UTC)")
    updated_at: str = Field(description="When the scenario was last updated (ISO 8601 UTC)")


class EvaluatorResponse(BaseModel):
    uuid: str = Field(
        min_length=36,
        max_length=36,
        description="Evaluator ID",
        examples=[_EXAMPLE_ID],
    )
    name: str = Field(description="Evaluator name")
    description: Optional[str] = Field(None, description="Evaluator description")
    evaluator_type: EvaluatorTypeLiteral = Field(
        "conversation", description="What the evaluator judges (always `conversation` for simulations)"
    )
    data_type: DataTypeLiteral = Field("text", description=DATA_TYPE_DESCRIPTION)
    kind: EvaluatorKindLiteral = Field(
        "single",
        description=(
            "How the evaluator scores:\n\n"
            "- `single`: judges one output\n"
            "- `side_by_side`: compares two outputs and picks a winner\n"
        ),
    )
    output_type: OutputTypeLiteral = Field("binary", description=OUTPUT_TYPE_DESCRIPTION)
    output_config: Optional[Dict[str, Any]] = Field(None, description="The rubric, pinned at link time")
    evaluator_version_id: str = Field(
        min_length=36,
        max_length=36,
        description="Version ID pinned at link time",
        examples=[_EXAMPLE_ID],
    )
    version_number: int = Field(description="Number of the pinned version. The first version is 1")
    judge_model: str = Field(description="Judge model for the pinned version")
    variables: Optional[List[Dict[str, Any]]] = Field(None, description="Declared prompt variables")
    variable_values: Optional[Dict[str, Any]] = Field(None, description="Values pinned for this link")


class AgentSummaryResponse(BaseModel):
    uuid: str = Field(
        min_length=36,
        max_length=36,
        description="Agent ID",
        examples=[_EXAMPLE_ID],
    )
    name: str = Field(description="Agent name")
    type: Literal["agent", "connection"] = Field(
        description=AGENT_TYPE_DESCRIPTION
    )
    config: Optional[Dict[str, Any]] = Field(None, description="Agent config")
    created_at: str = Field(description="When the agent was created (ISO 8601 UTC)")
    updated_at: str = Field(description="When the agent was last updated (ISO 8601 UTC)")


class SimulationListResponse(BaseModel):
    uuid: str = Field(
        min_length=36,
        max_length=36,
        description="Simulation ID",
        examples=[_EXAMPLE_ID],
    )
    name: str = Field(description="Simulation name")
    agent: Optional[AgentSummaryResponse] = Field(None, description="Linked agent summary")
    created_at: str = Field(description="When the simulation was created (ISO 8601 UTC)")
    updated_at: str = Field(description="When the simulation was last updated (ISO 8601 UTC)")


class SimulationDetailResponse(BaseModel):
    uuid: str = Field(
        min_length=36,
        max_length=36,
        description="Simulation ID",
        examples=[_EXAMPLE_ID],
    )
    name: str = Field(description="Simulation name")
    agent: Optional[AgentSummaryResponse] = Field(None, description="Linked agent summary")
    created_at: str = Field(description="When the simulation was created (ISO 8601 UTC)")
    updated_at: str = Field(description="When the simulation was last updated (ISO 8601 UTC)")
    personas: List[PersonaResponse] = Field(description="Linked personas")
    scenarios: List[ScenarioResponse] = Field(description="Linked scenarios")
    evaluators: List[EvaluatorResponse] = Field(description="Linked evaluators with their pinned versions")


class SimulationCreateResponse(BaseModel):
    uuid: str = Field(
        min_length=36,
        max_length=36,
        description="ID of the newly created simulation",
        examples=[_EXAMPLE_ID],
    )
    message: str = Field(description="Confirmation message")


class RunSimulationRequest(BaseModel):
    type: SimulationRunType = Field(
        ...,
        description="Run mode: `text` for chat simulations, `voice` for spoken simulations (**unsupported with `connection` agents**)",
    )


class EvaluationCriterionResult(BaseModel):
    name: str = Field(description="Evaluator name at run time")
    value: float = Field(description="Judge score (0/1 for binary, or the rating value)")
    reasoning: str = Field(description="Judge's explanation for the score")
    evaluator_uuid: Optional[str] = Field(
        None,
        min_length=36,
        max_length=36,
        description="Source evaluator ID, echoed from the run",
        examples=[_EXAMPLE_ID],
    )
    description: Optional[str] = Field(None, description="Evaluator's current description")


class SimulationEvaluatorRef(BaseModel):
    """Evaluator snapshot included with run status."""

    evaluator_uuid: str = Field(
        min_length=36,
        max_length=36,
        description="Evaluator ID for stable reference across renames",
        examples=[_EXAMPLE_ID],
    )
    name: str = Field(description="Evaluator's current DB name at response time")
    description: Optional[str] = Field(None, description="Evaluator's current DB description at response time")


class SimulationCaseResult(BaseModel):
    """Result for a single persona-scenario simulation"""

    simulation_name: str = Field(description="Case identifier within the run, e.g. `simulation_persona_1_scenario_1`")
    persona: Optional[Dict[str, Any]] = Field(
        None, description="Full persona object (label, characteristics, gender, language)"
    )
    scenario: Optional[Dict[str, Any]] = Field(
        None, description="Full scenario object (name/label and description)"
    )
    evaluation_results: Optional[List[EvaluationCriterionResult]] = Field(
        None, description="Judge results for each evaluator"
    )
    transcript: Optional[List[Dict[str, Any]]] = Field(None, description="Ordered conversation turns")
    audio_urls: Optional[List[str]] = Field(
        None, description="Presigned URLs for audio of each turn, in conversation order (voice runs only)"
    )
    conversation_wav_url: Optional[str] = Field(
        None, description="Presigned URL for the combined conversation.wav (voice runs only)"
    )
    aborted: Optional[bool] = Field(None, description="`true` if this case was aborted before completing")


class SimulationRunStatusResponse(BaseModel):
    task_id: str = Field(
        min_length=36,
        max_length=36,
        description="Simulation run ID",
        examples=[_EXAMPLE_ID],
    )
    name: str = Field(description='Display name in `Run {index}` form (creation order)')
    status: TaskStatus = Field(description="Run status")
    type: SimulationRunType = Field(description="Run mode")
    updated_at: str = Field(description="When the run was last updated (ISO 8601 UTC)")
    total_simulations: Optional[int] = Field(
        None, description="Expected number of persona x scenario cases"
    )
    completed_simulations: Optional[int] = Field(
        None, description="Number of cases finished so far"
    )
    metrics: Optional[Dict[str, Any]] = Field(None, description="Aggregated metrics")
    simulation_results: Optional[List[SimulationCaseResult]] = Field(
        None, description="Results for each case"
    )
    evaluators: Optional[List[SimulationEvaluatorRef]] = Field(
        None, description="Evaluators used for this run, in link order"
    )
    error: Optional[str] = Field(None, description="Failure message")
    is_public: bool = Field(False, description="Whether the run is shared via a public link")
    share_token: Optional[str] = Field(None, description="Share token for the public view")


class SimulationRunListItem(BaseModel):
    uuid: str = Field(
        min_length=36,
        max_length=36,
        description="Simulation run ID",
        examples=[_EXAMPLE_ID],
    )
    name: str = Field(description='Display name in `Run {index}` form (creation order)')
    status: TaskStatus = Field(description="Run status")
    type: SimulationRunType = Field(description="Run mode")
    updated_at: str = Field(description="When the run was last updated (ISO 8601 UTC)")


class SimulationRunsResponse(BaseModel):
    runs: List[SimulationRunListItem] = Field(description="Runs for the simulation, most recently updated first")


def _snapshot_evaluators_for_job_details(
    evaluators: List[Dict[str, Any]],
) -> List[Dict[str, str]]:
    """Persist ordered evaluator UUIDs on the job (same order as calibrate config)."""
    return [
        {
            "uuid": ev["uuid"],
            "name": ev.get("name") or "",
        }
        for ev in evaluators
        if ev.get("uuid")
    ]


def apply_simulation_job_evaluator_enrichment(
    details: Dict[str, Any],
    simulation_results: Optional[List[Any]],
) -> tuple[Optional[List[SimulationEvaluatorRef]], Optional[List[Any]]]:
    """Attach evaluator_uuid from each evaluation_results row's echoed evaluator_id."""
    snaps = details.get("evaluators") or []
    refs: List[SimulationEvaluatorRef] = []
    current_by_uuid: Dict[str, Optional[Dict[str, Any]]] = {}
    for s in snaps:
        if isinstance(s, dict) and s.get("uuid"):
            uid = s["uuid"]
            ev = get_evaluator(uid)
            current_by_uuid[uid] = ev
            refs.append(
                SimulationEvaluatorRef(
                    evaluator_uuid=uid,
                    name=(ev["name"] if ev else None) or s.get("name") or "",
                    description=ev.get("description") if ev else None,
                )
            )
    if simulation_results:
        for sim in simulation_results:
            if not isinstance(sim, dict):
                continue
            er = sim.get("evaluation_results")
            if not isinstance(er, list):
                continue
            for row in er:
                if not isinstance(row, dict):
                    continue
                echoed_id = row.get("evaluator_id")
                if echoed_id:
                    row["evaluator_uuid"] = echoed_id
                    ev = current_by_uuid.get(echoed_id)
                    if echoed_id not in current_by_uuid:
                        ev = get_evaluator(echoed_id)
                        current_by_uuid[echoed_id] = ev
                    row["description"] = ev.get("description") if ev else None
    top = refs if refs else None
    return top, simulation_results


@router.post("", response_model=SimulationCreateResponse, summary="Create simulation")
async def create_simulation_endpoint(
    simulation: SimulationCreate, ctx: OrgContext = Depends(get_current_org)
):
    """Create a simulation, optionally linking an agent, personas, scenarios, and `conversation` evaluators"""
    if simulation.agent_uuid:
        agent = get_agent(simulation.agent_uuid)
        if not agent or agent.get("org_uuid") != ctx.org_uuid:
            raise HTTPException(status_code=404, detail="Agent not found")

    if simulation.persona_uuids:
        for persona_uuid in simulation.persona_uuids:
            persona = get_persona(persona_uuid)
            if not persona or persona.get("org_uuid") != ctx.org_uuid:
                raise HTTPException(
                    status_code=404, detail=f"Persona {persona_uuid} not found"
                )

    if simulation.scenario_uuids:
        for scenario_uuid in simulation.scenario_uuids:
            scenario = get_scenario(scenario_uuid)
            if not scenario or scenario.get("org_uuid") != ctx.org_uuid:
                raise HTTPException(
                    status_code=404, detail=f"Scenario {scenario_uuid} not found"
                )

    resolved_evaluator_refs: List[Dict[str, Any]] = []
    if simulation.evaluators:
        for ref in simulation.evaluators:
            resolved_evaluator_refs.append(
                _resolve_simulation_evaluator_ref(ref, ctx.org_uuid)
            )

    with ensure_name_unique(
        "simulations", simulation.name, ctx.org_uuid, entity="Simulation"
    ):
        simulation_uuid = create_simulation(
            name=simulation.name,
            agent_id=simulation.agent_uuid,
            org_uuid=ctx.org_uuid,
            user_id=ctx.user_id,
        )

    # Add personas to simulation
    if simulation.persona_uuids:
        for persona_uuid in simulation.persona_uuids:
            add_persona_to_simulation(simulation_uuid, persona_uuid)

    # Add scenarios to simulation
    if simulation.scenario_uuids:
        for scenario_uuid in simulation.scenario_uuids:
            add_scenario_to_simulation(simulation_uuid, scenario_uuid)

    for ref in resolved_evaluator_refs:
        add_evaluator_to_simulation(
            simulation_uuid,
            evaluator_id=ref["evaluator_uuid"],
            evaluator_version_id=ref["version_uuid"],
            variable_values=ref["variable_values"],
        )

    return SimulationCreateResponse(
        uuid=simulation_uuid, message="Simulation created successfully"
    )


@router.get("", response_model=List[SimulationListResponse], summary="List simulations")
async def list_simulations(ctx: OrgContext = Depends(get_current_org)):
    """List all simulations"""
    simulations = get_all_simulations(org_uuid=ctx.org_uuid)
    # Hydrate each simulation's agent summary from ONE batched query instead of
    # a per-simulation `get_agent` (N+1).
    agents_by_id = get_agents_by_uuids(
        [sim["agent_id"] for sim in simulations if sim.get("agent_id")]
    )
    result = []
    for sim in simulations:
        agent = None
        if sim.get("agent_id"):
            agent_data = agents_by_id.get(sim["agent_id"])
            if agent_data:
                agent = AgentSummaryResponse(
                    uuid=agent_data["uuid"],
                    name=agent_data["name"],
                    type=agent_data.get("type", "agent"),
                    config=agent_data.get("config"),
                    created_at=agent_data["created_at"],
                    updated_at=agent_data["updated_at"],
                )
        result.append(
            SimulationListResponse(
                uuid=sim["uuid"],
                name=sim["name"],
                agent=agent,
                created_at=sim["created_at"],
                updated_at=sim["updated_at"],
            )
        )
    return result


class VisibilityRequest(BaseModel):
    is_public: bool = Field(description="`true` to publish the run via a share link. `false` to make it private")


class VisibilityResponse(BaseModel):
    is_public: bool = Field(description="Resulting public/private state of the run")
    share_token: str | None = Field(None, description="Share token when public")


@router.patch("/run/{task_id}/visibility", response_model=VisibilityResponse, summary="Update simulation run visibility")
async def update_simulation_run_visibility(
    body: VisibilityRequest,
    task_id: str = PathParam(
        description="The simulation run to update",
        examples=["f47ac10b-58cc-4372-a567-0e02b2c3d479"],
    ),
    ctx: OrgContext = Depends(get_current_org),
):
    """Update public sharing for a simulation run"""
    job = get_simulation_job(task_id)
    if not job:
        raise HTTPException(status_code=404, detail="Task not found")

    simulation_id = job.get("simulation_id")
    if simulation_id:
        simulation = get_simulation(simulation_id)
        if not simulation or simulation.get("org_uuid") != ctx.org_uuid:
            raise HTTPException(status_code=404, detail="Task not found")
    else:
        raise HTTPException(status_code=404, detail="Task not found")

    if body.is_public:
        import uuid as _uuid
        share_token = job.get("share_token") or str(_uuid.uuid4())
    else:
        share_token = None

    update_simulation_job_visibility(task_id, body.is_public, share_token)
    return VisibilityResponse(is_public=body.is_public, share_token=share_token)


@router.get("/run/{task_id}", response_model=SimulationRunStatusResponse, summary="Get simulation run status")
async def get_simulation_run_status(
    task_id: str = PathParam(
        description="The simulation run to poll",
        examples=["f47ac10b-58cc-4372-a567-0e02b2c3d479"],
    ),
    ctx: OrgContext = Depends(get_current_org),
):
    """Get the status and results of a simulation run"""
    job = get_simulation_job(task_id)
    if not job:
        raise HTTPException(status_code=404, detail="Task not found")

    simulation_id = job.get("simulation_id")
    if simulation_id:
        simulation = get_simulation(simulation_id)
        if not simulation or simulation.get("org_uuid") != ctx.org_uuid:
            raise HTTPException(status_code=404, detail="Task not found")

    status = job["status"]
    results = job.get("results") or {}
    details = job.get("details") or {}

    # Check for timeout on in-progress jobs
    if status == TaskStatus.IN_PROGRESS.value:
        updated_at = job.get("updated_at")
        if updated_at and is_job_timed_out(updated_at, timeout_seconds=15 * 60):
            logger.warning(f"Simulation job {task_id} timed out, marking as failed")

            # Kill running process
            pid = details.get("pid") or details.get("pgid")
            if pid:
                kill_process_group(pid, task_id)

            # Mark job as failed (preserve existing results, add error)
            results["error"] = "Job timed out after 5 minutes of inactivity"
            update_simulation_job(
                task_id,
                status=TaskStatus.FAILED.value,
                results=results,
            )
            status = TaskStatus.FAILED.value

            # Try to start the next queued job
            try_start_queued_simulation_job(SIMULATION_JOB_TYPES)

    # Calculate run index based on creation order
    run_name = "Run 1"  # Default
    if simulation_id:
        all_jobs = get_simulation_jobs_for_simulation(simulation_id)
        # Sort by created_at ASC to get oldest first (Run 1 is the oldest)
        sorted_jobs = sorted(all_jobs, key=lambda j: j.get("created_at", ""))
        # Find the index of current job (1-indexed)
        for idx, j in enumerate(sorted_jobs, start=1):
            if j["uuid"] == task_id:
                run_name = f"Run {idx}"
                break

    simulation_results = results.get("simulation_results") or []

    # If this is a voice simulation, handle presigned URLs based on status
    if job.get("type") == "voice" and simulation_results:
        if status == TaskStatus.DONE.value:
            # For done status: generate presigned URLs on-the-fly from S3 paths
            # Don't cache them in the database
            try:
                s3_bucket = get_s3_output_config()

                for sim_result in simulation_results:
                    # Generate audio URLs from S3 path
                    audios_s3_key_prefix = sim_result.get("audios_s3_path")
                    if audios_s3_key_prefix:
                        audio_urls = _get_audio_urls_from_s3_key(
                            audios_s3_key_prefix,
                            s3_bucket,
                            transcript=sim_result.get("transcript"),
                        )
                        sim_result["audio_urls"] = audio_urls
                        logger.info(
                            f"Generated {len(audio_urls)} presigned URLs on-the-fly for simulation {sim_result.get('simulation_name')}"
                        )

                    # Generate presigned URL for conversation.wav
                    conversation_wav_s3_key = sim_result.get("conversation_wav_s3_key")
                    if conversation_wav_s3_key:
                        conversation_wav_url = generate_presigned_download_url(
                            conversation_wav_s3_key, bucket=s3_bucket
                        )
                        sim_result["conversation_wav_url"] = (
                            conversation_wav_url if conversation_wav_url else ""
                        )
                        logger.info(
                            f"Generated presigned URL on-the-fly for conversation.wav for simulation {sim_result.get('simulation_name')}"
                        )
                    else:
                        sim_result["conversation_wav_url"] = ""

            except Exception as e:
                logger.warning(f"Failed to generate audio URLs: {str(e)}")
                # Continue without audio URLs if generation fails
        # For in-progress status: presigned URLs are already stored in results during monitoring
        # Just return them as-is (they were generated when the audio files were uploaded)

    evaluators_out, simulation_results = apply_simulation_job_evaluator_enrichment(
        details, simulation_results
    )

    return SimulationRunStatusResponse(
        task_id=task_id,
        name=run_name,
        status=status,
        type=job["type"],
        updated_at=job["updated_at"],
        total_simulations=results.get("total_simulations"),
        completed_simulations=results.get("completed_simulations"),
        metrics=results.get("metrics"),
        simulation_results=simulation_results,
        evaluators=evaluators_out,
        error=results.get("error"),
        is_public=bool(job.get("is_public")),
        share_token=job.get("share_token"),
    )


@router.get("/{simulation_uuid}/runs", response_model=SimulationRunsResponse, summary="List simulation runs")
async def get_simulation_runs(
    simulation_uuid: str = PathParam(
        description="The simulation whose runs to list",
        examples=["f47ac10b-58cc-4372-a567-0e02b2c3d479"],
    ),
    ctx: OrgContext = Depends(get_current_org),
):
    """List runs for a simulation, most recently updated first"""
    simulation = get_simulation(simulation_uuid)
    if not simulation or simulation.get("org_uuid") != ctx.org_uuid:
        raise HTTPException(status_code=404, detail="Simulation not found")

    # Get all jobs for this simulation (slim run-list headers only)
    jobs = get_simulation_jobs_summary(simulation_uuid)

    # Sort by created_at ASC to calculate run index (Run 1 is the oldest)
    sorted_by_created = sorted(jobs, key=lambda j: j.get("created_at", ""))

    # Create a mapping of job UUID to run index
    job_to_index = {
        job["uuid"]: idx for idx, job in enumerate(sorted_by_created, start=1)
    }

    # Sort by updated_at DESC for response (most recently updated first)
    sorted_by_updated = sorted(
        jobs, key=lambda j: j.get("updated_at", ""), reverse=True
    )

    runs = [
        SimulationRunListItem(
            uuid=job["uuid"],
            name=f"Run {job_to_index[job['uuid']]}",  # Use the index from creation order
            status=job["status"],
            type=job["type"],
            updated_at=job["updated_at"],
        )
        for job in sorted_by_updated
    ]

    return SimulationRunsResponse(runs=runs)


@router.get("/{simulation_uuid}", response_model=SimulationDetailResponse, summary="Get simulation")
async def get_simulation_endpoint(
    simulation_uuid: str = PathParam(
        description="The simulation to retrieve",
        examples=["f47ac10b-58cc-4372-a567-0e02b2c3d479"],
    ),
    ctx: OrgContext = Depends(get_current_org),
):
    """Get a simulation with its linked agent, personas, scenarios, and evaluators"""
    simulation = get_simulation(simulation_uuid)
    if not simulation or simulation.get("org_uuid") != ctx.org_uuid:
        raise HTTPException(status_code=404, detail="Simulation not found")

    # Get linked agent
    agent = None
    if simulation.get("agent_id"):
        agent_data = get_agent(simulation["agent_id"])
        if agent_data:
            agent = AgentSummaryResponse(
                uuid=agent_data["uuid"],
                name=agent_data["name"],
                type=agent_data.get("type", "agent"),
                config=agent_data.get("config"),
                created_at=agent_data["created_at"],
                updated_at=agent_data["updated_at"],
            )

    # Get linked entities
    personas = get_personas_for_simulation(simulation_uuid)
    scenarios = get_scenarios_for_simulation(simulation_uuid)
    evaluators = get_evaluators_for_simulation(simulation_uuid)

    return SimulationDetailResponse(
        uuid=simulation["uuid"],
        name=simulation["name"],
        agent=agent,
        created_at=simulation["created_at"],
        updated_at=simulation["updated_at"],
        personas=personas,
        scenarios=scenarios,
        evaluators=evaluators,
    )


@router.put("/{simulation_uuid}", response_model=SimulationDetailResponse, summary="Update simulation")
async def update_simulation_endpoint(
    simulation: SimulationUpdate,
    simulation_uuid: str = PathParam(
        description="The simulation to update",
        examples=["f47ac10b-58cc-4372-a567-0e02b2c3d479"],
    ),
    ctx: OrgContext = Depends(get_current_org),
):
    """Update a simulation's name, agent, and linked personas, scenarios, and evaluators"""
    existing_simulation = get_simulation(simulation_uuid)
    if (
        not existing_simulation
        or existing_simulation.get("org_uuid") != ctx.org_uuid
    ):
        raise HTTPException(status_code=404, detail="Simulation not found")

    if simulation.agent_uuid is not None and simulation.agent_uuid != "":
        agent = get_agent(simulation.agent_uuid)
        if not agent or agent.get("org_uuid") != ctx.org_uuid:
            raise HTTPException(status_code=404, detail="Agent not found")

    if simulation.persona_uuids is not None:
        for persona_uuid in simulation.persona_uuids:
            persona = get_persona(persona_uuid)
            if not persona or persona.get("org_uuid") != ctx.org_uuid:
                raise HTTPException(
                    status_code=404, detail=f"Persona {persona_uuid} not found"
                )

    if simulation.scenario_uuids is not None:
        for scenario_uuid in simulation.scenario_uuids:
            scenario = get_scenario(scenario_uuid)
            if not scenario or scenario.get("org_uuid") != ctx.org_uuid:
                raise HTTPException(
                    status_code=404, detail=f"Scenario {scenario_uuid} not found"
                )

    resolved_evaluator_refs: List[Dict[str, Any]] = []
    if simulation.evaluators is not None:
        for ref in simulation.evaluators:
            resolved_evaluator_refs.append(
                _resolve_simulation_evaluator_ref(ref, ctx.org_uuid)
            )

    if simulation.name is not None or simulation.agent_uuid is not None:
        with ensure_name_unique(
            "simulations",
            simulation.name,
            ctx.org_uuid,
            entity="Simulation",
            exclude_uuid=simulation_uuid,
        ):
            # Empty string means clear the agent
            if simulation.agent_uuid == "":
                update_simulation(
                    simulation_uuid=simulation_uuid,
                    name=simulation.name,
                    clear_agent=True,
                )
            else:
                update_simulation(
                    simulation_uuid=simulation_uuid,
                    name=simulation.name,
                    agent_id=simulation.agent_uuid,
                )

    # Update personas if provided (replace existing)
    if simulation.persona_uuids is not None:
        # Remove existing personas
        existing_personas = get_personas_for_simulation(simulation_uuid)
        for persona in existing_personas:
            remove_persona_from_simulation(simulation_uuid, persona["uuid"])
        # Add new personas
        for persona_uuid in simulation.persona_uuids:
            add_persona_to_simulation(simulation_uuid, persona_uuid)

    # Update scenarios if provided (replace existing)
    if simulation.scenario_uuids is not None:
        # Remove existing scenarios
        existing_scenarios = get_scenarios_for_simulation(simulation_uuid)
        for scenario in existing_scenarios:
            remove_scenario_from_simulation(simulation_uuid, scenario["uuid"])
        # Add new scenarios
        for scenario_uuid in simulation.scenario_uuids:
            add_scenario_to_simulation(simulation_uuid, scenario_uuid)

    # Update evaluators if provided (replace existing)
    if simulation.evaluators is not None:
        existing_evaluators = get_evaluators_for_simulation(simulation_uuid)
        for evaluator in existing_evaluators:
            remove_evaluator_from_simulation(simulation_uuid, evaluator["uuid"])
        for ref in resolved_evaluator_refs:
            add_evaluator_to_simulation(
                simulation_uuid,
                evaluator_id=ref["evaluator_uuid"],
                evaluator_version_id=ref["version_uuid"],
                variable_values=ref["variable_values"],
            )

    # Return full detail response
    updated_simulation = get_simulation(simulation_uuid)

    # Get linked agent
    agent = None
    if updated_simulation.get("agent_id"):
        agent_data = get_agent(updated_simulation["agent_id"])
        if agent_data:
            agent = AgentSummaryResponse(
                uuid=agent_data["uuid"],
                name=agent_data["name"],
                type=agent_data.get("type", "agent"),
                config=agent_data.get("config"),
                created_at=agent_data["created_at"],
                updated_at=agent_data["updated_at"],
            )

    personas = get_personas_for_simulation(simulation_uuid)
    scenarios = get_scenarios_for_simulation(simulation_uuid)
    evaluators = get_evaluators_for_simulation(simulation_uuid)

    return SimulationDetailResponse(
        uuid=updated_simulation["uuid"],
        name=updated_simulation["name"],
        agent=agent,
        created_at=updated_simulation["created_at"],
        updated_at=updated_simulation["updated_at"],
        personas=personas,
        scenarios=scenarios,
        evaluators=evaluators,
    )


@router.delete("/{simulation_uuid}", summary="Delete simulation")
async def delete_simulation_endpoint(
    simulation_uuid: str = PathParam(
        description="The simulation to delete",
        examples=["f47ac10b-58cc-4372-a567-0e02b2c3d479"],
    ),
    ctx: OrgContext = Depends(get_current_org),
):
    """Delete a simulation from your workspace"""
    existing_simulation = get_simulation(simulation_uuid)
    if (
        not existing_simulation
        or existing_simulation.get("org_uuid") != ctx.org_uuid
    ):
        raise HTTPException(status_code=404, detail="Simulation not found")

    deleted = delete_simulation(simulation_uuid)
    if not deleted:
        raise HTTPException(status_code=404, detail="Simulation not found")
    return {"message": "Simulation deleted successfully"}


# ============ Run Simulation API ============


def _build_calibrate_simulation_config(
    agent: Dict[str, Any],
    personas: List[Dict[str, Any]],
    scenarios: List[Dict[str, Any]],
    evaluators: List[Dict[str, Any]],
    simulation_type: str = "text",
) -> Dict[str, Any]:
    """
    Build the calibrate simulation config from agent, personas, scenarios, and evaluators.

    Args:
        agent: Agent dict with config containing system_prompt and llm.model
        personas: List of persona dicts with description and config (containing gender, language)
        scenarios: List of scenario dicts with description
        evaluators: List of evaluator link dicts (from get_evaluators_for_simulation)
        simulation_type: Type of simulation - "text" or "voice"
    """
    agent_config = agent.get("config") or {}

    # Build personas list (same for both modes)
    default_gender = env_str("DEFAULT_PERSONA_GENDER", "female")
    default_language = env_str("DEFAULT_PERSONA_LANGUAGE", "english")
    default_interruption = env_str("DEFAULT_PERSONA_INTERRUPTION_SENSITIVITY", "medium")

    persona_list = []
    for persona in personas:
        persona_config = persona.get("config") or {}
        persona_obj = {
            "label": persona.get("name", ""),
            "characteristics": persona.get("description") or persona.get("name"),
            "gender": persona_config.get("gender", default_gender),
            "language": persona_config.get("language", default_language),
        }
        if simulation_type == "voice":
            persona_obj["interruption_sensitivity"] = persona_config.get(
                "interruption_sensitivity", default_interruption
            )
        if persona_obj["characteristics"]:
            persona_list.append(persona_obj)

    # Build scenarios list (same for both modes)
    scenario_list = [
        {"name": s.get("name", ""), "description": s.get("description", "")}
        for s in scenarios
    ]

    # Build full evaluator payload for the calibrate CLI (minimal shape: name, system_prompt,
    # judge_model, type, scale_min/scale_max for rating). Variables in the system prompt are
    # pre-substituted because simulations don't have a per-row arguments mechanism.
    evaluators_payload = build_evaluator_cli_payload(evaluators)

    settings_config = agent_config.get("settings", {})
    # Fallbacks share the same env var as `_default_agent_config()` in agents.py
    # so an operator can pin a single value across both new-agent creation and
    # the simulation-time fallback for legacy agents whose config is missing
    # these fields. Hardcoded fallbacks preserve the historical sim-runtime
    # values (True / 50) for old data; new agents always have these set explicitly.
    shared_settings = {
        "agent_speaks_first": settings_config.get(
            "agent_speaks_first", env_bool("DEFAULT_AGENT_SPEAKS_FIRST", True)
        ),
        "max_turns": settings_config.get(
            "max_assistant_turns", env_int("DEFAULT_AGENT_MAX_TURNS", 50)
        ),
    }

    if agent_config.get("agent_url"):
        # Agent connection mode — agent owns its LLM; no system_prompt/tools/params
        # Only supported for text simulations (caller must guard voice)
        config: Dict[str, Any] = {
            "agent_url": agent_config["agent_url"],
            "personas": persona_list,
            "scenarios": scenario_list,
            "evaluators": evaluators_payload,
            "settings": shared_settings,
        }
        if agent_config.get("agent_headers"):
            config["agent_headers"] = agent_config["agent_headers"]
        return config

    # Calibrate agent mode
    llm_config = agent_config.get("llm", {})
    # Shares DEFAULT_AGENT_LLM_MODEL with new-agent creation. Hardcoded fallback
    # stays "gpt-4.1" to preserve runtime behavior for legacy agents created
    # before agent-create defaults were wired up.
    model = llm_config.get("model", env_str("DEFAULT_AGENT_LLM_MODEL", "gpt-4.1"))

    agent_tools = get_tools_for_agent(agent["uuid"])
    tool_configs = build_tool_configs(agent_tools)

    config = {
        "system_prompt": agent_config.get("system_prompt", ""),
        "tools": tool_configs,
        "personas": persona_list,
        "scenarios": scenario_list,
        "evaluators": evaluators_payload,
        "settings": shared_settings,
    }

    if simulation_type == "text":
        config["params"] = {"model": model}
    else:
        stt_config = agent_config.get("stt", {})
        if stt_config:
            config["stt"] = stt_config
        tts_config = agent_config.get("tts", {})
        if tts_config:
            config["tts"] = tts_config
        if llm_config:
            config["llm"] = llm_config

    return config


def _extract_persona_scenario_indices(sim_name: str) -> tuple:
    """
    Extract persona and scenario indices from simulation directory name.
    Format: simulation_persona_N_scenario_M (1-based indices)
    Returns (persona_index, scenario_index) as 0-based indices, or (None, None) if parsing fails.
    """
    import re

    match = re.match(r"simulation_persona_(\d+)_scenario_(\d+)", sim_name)
    if match:
        # Convert from 1-based (in folder name) to 0-based (for list indexing)
        return int(match.group(1)) - 1, int(match.group(2)) - 1
    return None, None


def _parse_text_simulation_directory(
    sim_dir: Path,
    personas_list: Optional[List[Dict[str, Any]]] = None,
    scenarios_list: Optional[List[Dict[str, Any]]] = None,
) -> Optional[Dict[str, Any]]:
    """
    Parse a single text simulation directory.

    Returns results for both complete (has evaluation_results.csv) and
    in-progress (has transcript.json but no evaluation_results.csv) simulations.

    Args:
        sim_dir: Path to the simulation directory
        personas_list: Optional list of personas from calibrate config (used as fallback)
        scenarios_list: Optional list of scenarios from calibrate config (used as fallback)

    Returns:
        Dict with simulation result data, or None if directory doesn't exist
    """
    import csv

    if not sim_dir.exists():
        return None

    sim_name = sim_dir.name
    eval_results_file = sim_dir / "evaluation_results.csv"
    transcript_file = sim_dir / "transcript.json"
    config_file = sim_dir / "config.json"

    # Check if simulation is complete
    is_complete = eval_results_file.exists()

    eval_results = []
    if eval_results_file.exists():
        try:
            with open(eval_results_file, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    eval_results.append(
                        {
                            "evaluator_id": row.get("evaluator_id"),
                            "name": row.get("name"),
                            "type": row.get("type"),
                            "value": row.get("value"),
                            "reasoning": row.get("reasoning", ""),
                        }
                    )
        except Exception as e:
            logger.warning(
                f"Failed to parse evaluation_results.csv for {sim_name}: {e}"
            )

    # Parse transcript.json if it exists
    transcript = None
    if transcript_file.exists():
        try:
            with open(transcript_file, "r", encoding="utf-8") as f:
                transcript = json.load(f)
        except Exception as e:
            logger.warning(f"Failed to parse transcript.json for {sim_name}: {e}")

    # Parse config.json to get persona and scenario data
    persona_data = None
    scenario_data = None
    if config_file.exists():
        try:
            with open(config_file, "r", encoding="utf-8") as f:
                config_data = json.load(f)
                persona_data = config_data.get("persona")
                scenario_data = config_data.get("scenario")
        except Exception as e:
            logger.warning(f"Failed to parse config.json for {sim_name}: {e}")

    # Fallback: if persona/scenario not in config.json, extract from directory name
    if (persona_data is None or scenario_data is None) and (
        personas_list or scenarios_list
    ):
        persona_idx, scenario_idx = _extract_persona_scenario_indices(sim_name)
        if persona_data is None and personas_list and persona_idx is not None:
            if 0 <= persona_idx < len(personas_list):
                persona_data = personas_list[persona_idx]
        if scenario_data is None and scenarios_list and scenario_idx is not None:
            if 0 <= scenario_idx < len(scenarios_list):
                scenario_data = scenarios_list[scenario_idx]

    # Only return if we have at least config.json or transcript.json (simulation has started)
    if not config_file.exists() and not transcript_file.exists():
        return None

    return {
        "simulation_name": sim_name,
        "persona": persona_data,
        "scenario": scenario_data,
        "evaluation_results": eval_results if is_complete else None,
        "transcript": transcript,
        "is_complete": is_complete,
    }


def _get_text_simulation_directories(output_dir: Path) -> List[Path]:
    """Get all simulation directories from output directory."""
    sim_dirs = []
    if not output_dir.exists():
        return sim_dirs
    for root, dirs, files in os.walk(output_dir):
        for dir_name in dirs:
            if dir_name.startswith("simulation_persona_"):
                sim_dirs.append(Path(root) / dir_name)
    return sim_dirs


def _update_text_simulation_intermediate_results(
    task_id: str,
    output_dir: Path,
    expected_total: int,
    s3_prefix: str,
    personas_list: Optional[List[Dict[str, Any]]] = None,
    scenarios_list: Optional[List[Dict[str, Any]]] = None,
    prev_state: Optional[tuple] = None,
) -> Optional[tuple]:
    """Update intermediate results for a text simulation job.

    Args:
        prev_state: Previous state tuple for change detection

    Returns:
        Current state tuple (to be passed as prev_state in next call)
    """
    simulation_results = []
    completed_count = 0
    transcript_lengths = []  # For change detection

    for sim_dir in _get_text_simulation_directories(output_dir):
        sim_result = _parse_text_simulation_directory(
            sim_dir, personas_list, scenarios_list
        )
        if sim_result:
            # Remove is_complete field before storing (internal use only)
            is_complete = sim_result.pop("is_complete", False)
            if is_complete:
                completed_count += 1
            simulation_results.append(sim_result)
            # Track transcript length for change detection
            transcript = sim_result.get("transcript") or []
            transcript_lengths.append((sim_dir.name, len(transcript)))

    if not simulation_results:
        return prev_state

    # Build current state for change detection
    # Note: Don't read metrics.json during in-progress - calibrate creates it incrementally
    # and reading it before all simulations complete will give incomplete metrics.
    # The final metrics are read after the process completes in _run_calibrate_text_simulation.
    current_state = (
        completed_count,
        tuple(sorted(transcript_lengths)),
    )

    # Only update DB if state changed and job hasn't been aborted
    if current_state != prev_state:
        if _is_job_aborted(task_id):
            return current_state
        update_simulation_job(
            task_id,
            status=TaskStatus.IN_PROGRESS.value,
            results={
                "total_simulations": expected_total,
                "completed_simulations": completed_count,
                "simulation_results": simulation_results,
                "results_s3_prefix": s3_prefix,
                "metrics": None,  # Don't include metrics during in-progress
            },
        )

    return current_state


def _run_calibrate_text_simulation(
    model: Optional[str],
    calibrate_config: Dict[str, Any],
    input_dir: Path,
    output_dir: Path,
    s3_bucket: str,
    s3_prefix: str,
    task_id: Optional[str] = None,
    log_prefix: str = "LLM simulation",
) -> Dict[str, Any]:
    """
    Run calibrate llm simulations run command and return parsed results.
    Updates the database incrementally as each simulation completes.

    Args:
        model: Model name to use
        calibrate_config: The calibrate config dict
        input_dir: Directory to write config files
        output_dir: Directory to write output files
        s3_bucket: S3 bucket name
        s3_prefix: S3 key prefix for uploading results
        task_id: Optional task ID for intermediate updates
        log_prefix: Prefix for log messages

    Returns:
        Dict with keys: success, total_simulations, metrics, simulation_results, error
    """
    s3 = get_s3_client()

    # Update config with model only in Calibrate agent mode
    config = calibrate_config.copy()
    if model:
        config["params"] = {"model": model}

    # Resolve directories to absolute paths
    input_dir = input_dir.resolve()
    output_dir = output_dir.resolve()

    # Create directories
    input_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Write config to input directory
    config_file_name = "simulation_config"
    config_file = input_dir / f"{config_file_name}.json"
    with open(config_file, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

    # Get personas and scenarios lists for intermediate results
    personas_list = calibrate_config.get("personas", [])
    scenarios_list = calibrate_config.get("scenarios", [])
    expected_total = len(personas_list) * len(scenarios_list)

    # Build CLI command — agent connection mode omits -m (agent owns its model).
    # Parallelism is left to calibrate, which reads CALIBRATE_SIMULATION_PARALLEL
    # from the inherited env (we don't pass -n), mirroring the LLM-test path.
    run_cmd = [
        get_calibrate_agent_cli(),
        "simulations",
        "--type",
        "text",
        "-c",
        str(config_file),
        "-o",
        str(output_dir),
    ]
    if model:
        run_cmd += ["-m", model]
    else:
        run_cmd += ["--skip-verify"]

    logger.info(f"{log_prefix} command: {' '.join(run_cmd)}")

    # Use Popen with polling for intermediate updates
    stdout_file = tempfile.NamedTemporaryFile(mode="w+", delete=False, suffix=".log")
    stderr_file = tempfile.NamedTemporaryFile(mode="w+", delete=False, suffix=".log")

    try:
        process = subprocess.Popen(
            run_cmd,
            stdout=stdout_file,
            stderr=stderr_file,
            start_new_session=True,
        )

        # Poll for process completion while updating intermediate results
        poll_interval = 2  # seconds
        prev_state = None  # Track state to avoid unnecessary DB updates

        while process.poll() is None:
            if task_id:
                if _is_job_aborted(task_id):
                    logger.info(f"Text simulation {task_id} aborted, killing process group and stopping")
                    kill_process_group(process.pid, task_id)
                    try:
                        process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        logger.warning(f"Text simulation {task_id}: process {process.pid} did not exit within 5s after kill")
                        process.kill()
                        process.wait(timeout=5)
                    break
                prev_state = _update_text_simulation_intermediate_results(
                    task_id,
                    output_dir,
                    expected_total,
                    s3_prefix,
                    personas_list,
                    scenarios_list,
                    prev_state,
                )
            time.sleep(poll_interval)

        # Final update after process completes (skip if job was aborted by user)
        if task_id:
            if _is_job_aborted(task_id):
                logger.info(
                    f"Text simulation {task_id} was aborted by user, skipping final processing"
                )
                return {
                    "success": False,
                    "total_simulations": 0,
                    "metrics": None,
                    "simulation_results": [],
                    "error": "user_aborted",
                }
            _update_text_simulation_intermediate_results(
                task_id,
                output_dir,
                expected_total,
                s3_prefix,
                personas_list,
                scenarios_list,
                prev_state,
            )

        # Read stdout/stderr
        stdout_file.seek(0)
        stderr_file.seek(0)
        stdout_content = stdout_file.read()
        stderr_content = stderr_file.read()

        if stdout_content:
            logger.info(f"{log_prefix} stdout: {stdout_content}")
        if stderr_content:
            logger.info(f"{log_prefix} stderr: {stderr_content}")

    finally:
        stdout_file.close()
        stderr_file.close()
        os.unlink(stdout_file.name)
        os.unlink(stderr_file.name)

    # Parse final results
    metrics_data = None
    simulation_results = []

    # Find metrics.json file
    metrics_file = output_dir / "metrics.json"
    if metrics_file.exists():
        with open(metrics_file, "r", encoding="utf-8") as f:
            metrics_data = json.load(f)

    # Parse all simulation directories
    for sim_dir in _get_text_simulation_directories(output_dir):
        sim_result = _parse_text_simulation_directory(
            sim_dir, personas_list, scenarios_list
        )
        if sim_result:
            # Remove is_complete field (internal use only)
            sim_result.pop("is_complete", None)
            simulation_results.append(sim_result)

    # Upload results to S3
    for root, dirs, files in os.walk(output_dir):
        for file in files:
            local_file_path = Path(root) / file
            relative_path = local_file_path.relative_to(output_dir)
            s3_key = f"{s3_prefix}/{relative_path}"
            upload_file_to_s3(s3, local_file_path, s3_bucket, s3_key)

    # Upload the config file to S3
    if config_file.exists():
        config_s3_key = f"{s3_prefix}/simulation_config.json"
        upload_file_to_s3(s3, config_file, s3_bucket, config_s3_key)
        logger.info(f"Uploaded config file to S3: {config_s3_key}")

    error = None
    if process.returncode != 0:
        is_failure = True
        error = f"Command failed with exit code {process.returncode}: {stderr_content}"
    elif not simulation_results:
        is_failure = True
        error = "Simulation produced no output (no simulation directories found)"
    else:
        is_failure = False

    if is_failure:
        logger.error(error)
        capture_exception_to_sentry(RuntimeError(error))

    return {
        "success": not is_failure,
        "total_simulations": len(simulation_results),
        "metrics": metrics_data,
        "simulation_results": simulation_results,
        "error": error,
    }


def _parse_simulation_directory(
    sim_dir: Path,
    output_dir: Path,
    s3_bucket: str,
    s3_prefix: str,
    uploaded_audio_files: set,
    include_presigned_urls: bool = False,
) -> Optional[Dict[str, Any]]:
    """
    Parse a single simulation directory and upload its audio files to S3.

    Args:
        sim_dir: Path to the simulation directory
        output_dir: Base output directory
        s3_bucket: S3 bucket name
        s3_prefix: S3 key prefix for uploading results
        uploaded_audio_files: Set to track uploaded audio files (modified in place)
        include_presigned_urls: If True, include presigned URLs in the result (for in-progress status)

    Returns:
        Dict with simulation result data, or None if parsing failed
    """
    sim_name = sim_dir.name
    eval_results_file = sim_dir / "evaluation_results.csv"
    transcript_file = sim_dir / "transcript.json"
    config_file = sim_dir / "config.json"

    eval_results = []
    if eval_results_file.exists():
        import csv

        with open(eval_results_file, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                eval_results.append(
                    {
                        "evaluator_id": row.get("evaluator_id"),
                        "name": row.get("name"),
                        "type": row.get("type"),
                        "value": row.get("value"),
                        "reasoning": row.get("reasoning", ""),
                    }
                )

    # Parse transcript.json if it exists
    transcript = None
    if transcript_file.exists():
        try:
            with open(transcript_file, "r", encoding="utf-8") as f:
                transcript = json.load(f)
        except Exception as e:
            logger.warning(f"Failed to parse transcript.json for {sim_name}: {e}")

    # Parse config.json to get persona and scenario data
    persona_data = None
    scenario_data = None
    if config_file.exists():
        try:
            with open(config_file, "r", encoding="utf-8") as f:
                config_data = json.load(f)
                persona_data = config_data.get("persona")
                scenario_data = config_data.get("scenario")
        except Exception as e:
            logger.warning(f"Failed to parse config.json for {sim_name}: {e}")

    # Upload audio files and optionally generate presigned URLs
    audios_s3_path, conversation_wav_s3_key, audio_urls, conversation_wav_url = (
        _upload_audio_and_generate_urls(
            sim_dir, output_dir, s3_bucket, s3_prefix, uploaded_audio_files
        )
    )

    result = {
        "simulation_name": sim_name,
        "persona": persona_data,
        "scenario": scenario_data,
        "evaluation_results": eval_results,
        "transcript": transcript,
        "audios_s3_path": audios_s3_path,
        "conversation_wav_s3_key": conversation_wav_s3_key,
    }

    # Include presigned URLs only during in-progress status
    if include_presigned_urls:
        result["audio_urls"] = audio_urls if audio_urls else None
        result["conversation_wav_url"] = conversation_wav_url

    return result


def _is_simulation_complete(sim_dir: Path) -> bool:
    """
    Check if a simulation directory is complete.
    A simulation is considered complete when it has an evaluation_results.csv file,
    which is created after the evaluation step finishes.
    """
    eval_results_file = sim_dir / "evaluation_results.csv"
    return eval_results_file.exists()


def _is_simulation_started(sim_dir: Path) -> bool:
    """
    Check if a simulation has started (has config.json or transcript.json).
    """
    config_file = sim_dir / "config.json"
    transcript_file = sim_dir / "transcript.json"
    return config_file.exists() or transcript_file.exists()


def _upload_audio_and_generate_urls(
    sim_dir: Path,
    output_dir: Path,
    s3_bucket: str,
    s3_prefix: str,
    uploaded_audio_files: set,
) -> tuple:
    """
    Upload audio files for a simulation directory and generate presigned URLs.

    Args:
        sim_dir: Path to the simulation directory
        output_dir: Base output directory
        s3_bucket: S3 bucket name
        s3_prefix: S3 key prefix for uploading results
        uploaded_audio_files: Set to track uploaded audio files (modified in place)

    Returns:
        Tuple of (audios_s3_path, conversation_wav_s3_key, audio_urls, conversation_wav_url)
    """
    s3 = get_s3_client()
    sim_name = sim_dir.name
    audios_dir = sim_dir / "audios"

    audios_s3_path = None
    conversation_wav_s3_key = None
    audio_urls = []
    conversation_wav_url = None

    # Upload audios folder for this simulation to S3
    if audios_dir.exists() and audios_dir.is_dir():
        audios_s3_prefix = f"{s3_prefix}/{sim_name}/audios"
        audio_files_to_upload = []

        for audio_file in audios_dir.iterdir():
            if audio_file.is_file() and audio_file.suffix in {".wav", ".mp3", ".ogg"}:
                audio_files_to_upload.append(audio_file)

        # Sort audio files for consistent URL ordering
        def natural_sort_key(path: Path) -> tuple:
            filename = path.name
            parts = filename.split("_", 1)
            if len(parts) > 1 and parts[0].isdigit():
                return (int(parts[0]), parts[1])
            return (float("inf"), filename)

        audio_files_to_upload.sort(key=natural_sort_key)

        for audio_file in audio_files_to_upload:
            # Upload if not already uploaded
            if str(audio_file) not in uploaded_audio_files:
                relative_audio_path = audio_file.relative_to(output_dir)
                audio_s3_key = f"{s3_prefix}/{relative_audio_path}"
                upload_file_to_s3(s3, audio_file, s3_bucket, audio_s3_key)
                uploaded_audio_files.add(str(audio_file))
                logger.info(
                    f"Uploaded audio file {audio_file.name} to S3: {audio_s3_key}"
                )

            # Generate presigned URL
            relative_audio_path = audio_file.relative_to(output_dir)
            audio_s3_key = f"{s3_prefix}/{relative_audio_path}"
            presigned_url = generate_presigned_download_url(
                audio_s3_key, bucket=s3_bucket
            )
            if presigned_url:
                audio_urls.append(presigned_url)

        if audio_files_to_upload:
            audios_s3_path = audios_s3_prefix

    # Upload conversation.wav if it exists
    conversation_wav_file = sim_dir / "conversation.wav"
    if conversation_wav_file.exists() and conversation_wav_file.is_file():
        conversation_wav_s3_key = f"{s3_prefix}/{sim_name}/conversation.wav"
        if str(conversation_wav_file) not in uploaded_audio_files:
            upload_file_to_s3(
                s3, conversation_wav_file, s3_bucket, conversation_wav_s3_key
            )
            uploaded_audio_files.add(str(conversation_wav_file))
            logger.info(
                f"Uploaded conversation.wav for {sim_name} to s3://{s3_bucket}/{conversation_wav_s3_key}"
            )

        # Generate presigned URL
        conversation_wav_url = generate_presigned_download_url(
            conversation_wav_s3_key, bucket=s3_bucket
        )

    return audios_s3_path, conversation_wav_s3_key, audio_urls, conversation_wav_url


def _parse_voice_simulation_in_progress(
    sim_dir: Path,
    personas_list: Optional[List[Dict[str, Any]]] = None,
    scenarios_list: Optional[List[Dict[str, Any]]] = None,
) -> Optional[Dict[str, Any]]:
    """
    Parse an in-progress voice simulation directory for intermediate results.
    Does NOT upload audio or generate presigned URLs - those are only available
    after evaluation_results are ready.

    Args:
        sim_dir: Path to the simulation directory
        personas_list: Optional list of personas from calibrate config (used as fallback)
        scenarios_list: Optional list of scenarios from calibrate config (used as fallback)

    Returns:
        Dict with simulation data (no audio URLs), or None if simulation hasn't started
    """
    if not sim_dir.exists():
        return None

    sim_name = sim_dir.name
    transcript_file = sim_dir / "transcript.json"
    config_file = sim_dir / "config.json"

    # Only return if simulation has started
    if not config_file.exists() and not transcript_file.exists():
        return None

    # Parse transcript.json if it exists
    transcript = None
    if transcript_file.exists():
        try:
            with open(transcript_file, "r", encoding="utf-8") as f:
                transcript = json.load(f)
        except Exception as e:
            logger.warning(f"Failed to parse transcript.json for {sim_name}: {e}")

    # Parse config.json to get persona and scenario data
    persona_data = None
    scenario_data = None
    if config_file.exists():
        try:
            with open(config_file, "r", encoding="utf-8") as f:
                config_data = json.load(f)
                persona_data = config_data.get("persona")
                scenario_data = config_data.get("scenario")
        except Exception as e:
            logger.warning(f"Failed to parse config.json for {sim_name}: {e}")

    # Fallback: if persona/scenario not in config.json, extract from directory name
    if (persona_data is None or scenario_data is None) and (
        personas_list or scenarios_list
    ):
        persona_idx, scenario_idx = _extract_persona_scenario_indices(sim_name)
        if persona_data is None and personas_list and persona_idx is not None:
            if 0 <= persona_idx < len(personas_list):
                persona_data = personas_list[persona_idx]
        if scenario_data is None and scenarios_list and scenario_idx is not None:
            if 0 <= scenario_idx < len(scenarios_list):
                scenario_data = scenarios_list[scenario_idx]

    # Don't upload audio or generate URLs for in-progress simulations
    # Audio URLs are only returned after evaluation_results are available
    return {
        "simulation_name": sim_name,
        "persona": persona_data,
        "scenario": scenario_data,
        "evaluation_results": None,  # In-progress, no evaluation yet
        "transcript": transcript,
        "audios_s3_path": None,
        "conversation_wav_s3_key": None,
        "audio_urls": None,
        "conversation_wav_url": None,
    }


def _run_calibrate_voice_simulation(
    calibrate_config: Dict[str, Any],
    input_dir: Path,
    output_dir: Path,
    s3_bucket: str,
    s3_prefix: str,
    task_id: str,
    log_prefix: str = "Voice simulation",
) -> Dict[str, Any]:
    """
    Run calibrate agent simulation command and return parsed results.
    Updates the database incrementally as each simulation completes.

    Args:
        calibrate_config: The calibrate config dict (for voice simulations)
        input_dir: Directory to write config files
        output_dir: Directory to write output files
        s3_bucket: S3 bucket name
        s3_prefix: S3 key prefix for uploading results
        task_id: The task ID for updating the database with incremental results
        log_prefix: Prefix for log messages

    Returns:
        Dict with keys: success, total_simulations, metrics, simulation_results, error, audios_s3_path
    """
    import time

    s3 = get_s3_client()

    # Resolve directories to absolute paths
    input_dir = input_dir.resolve()
    output_dir = output_dir.resolve()

    # Create directories
    input_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Write config to input directory
    config_file_name = "simulation_config"
    config_file = input_dir / f"{config_file_name}.json"
    with open(config_file, "w", encoding="utf-8") as f:
        json.dump(calibrate_config, f, indent=2)

    # Run calibrate agent simulation command as a non-blocking process.
    # Parallelism is left to calibrate via the inherited CALIBRATE_SIMULATION_PARALLEL
    # env (we don't pass -n), mirroring the LLM-test path.
    run_cmd = [
        get_calibrate_agent_cli(),
        "simulations",
        "--type",
        "voice",
        "-c",
        str(config_file),
        "-o",
        str(output_dir),
    ]

    logger.info(f"{log_prefix} command: {' '.join(run_cmd)}")

    # Open log files for stdout and stderr
    stdout_log_path = output_dir / "stdout.log"
    stderr_log_path = output_dir / "stderr.log"

    with (
        open(stdout_log_path, "w") as stdout_file,
        open(stderr_log_path, "w") as stderr_file,
    ):
        # Start the process without blocking, writing output to files
        process = subprocess.Popen(
            run_cmd,
            stdout=stdout_file,
            stderr=stderr_file,
            start_new_session=True,  # Detach from parent process group
            cwd=str(output_dir),
        )

        # Store the process PID and process group ID in the job for cleanup on restart
        # The process group ID (pgid) equals the PID when start_new_session=True
        logger.info(f"{log_prefix}: Started process with PID {process.pid}")
        update_simulation_job(
            task_id,
            status=TaskStatus.IN_PROGRESS.value,
            details={
                "pid": process.pid,
                "pgid": process.pid,  # Same as PID when start_new_session=True
            },
        )

        # Track processed simulations and uploaded files
        completed_simulations = set()
        uploaded_audio_files = set()
        completed_results = []  # Results for completed simulations

        # Get personas and scenarios lists for intermediate results
        personas_list = calibrate_config.get("personas", [])
        scenarios_list = calibrate_config.get("scenarios", [])
        expected_total = len(personas_list) * len(scenarios_list)
        logger.info(
            f"{log_prefix}: Expecting {expected_total} simulations ({len(personas_list)} personas x {len(scenarios_list)} scenarios)"
        )

        # Monitor for new simulation directories while the process runs
        poll_interval = 2  # seconds between checks
        prev_state = None  # Track state to avoid unnecessary DB updates

        while process.poll() is None:
            if _is_job_aborted(task_id):
                logger.info(f"Voice simulation {task_id} aborted, killing process group and stopping")
                kill_process_group(process.pid, task_id)
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    logger.warning(f"Voice simulation {task_id}: process {process.pid} did not exit within 5s after kill")
                    process.kill()
                    process.wait(timeout=5)
                break

            in_progress_results = []  # Rebuilt each iteration
            in_progress_transcript_lengths = (
                []
            )  # Track transcript lengths for change detection

            # Find all simulation directories
            for item in output_dir.iterdir():
                if item.is_dir() and item.name.startswith("simulation_persona_"):
                    if _is_simulation_complete(item):
                        # Simulation is complete
                        if item.name not in completed_simulations:
                            logger.info(
                                f"{log_prefix}: Found completed simulation directory: {item.name}"
                            )
                            # Parse and upload the completed simulation with presigned URLs (for in-progress display)
                            sim_result = _parse_simulation_directory(
                                sim_dir=item,
                                output_dir=output_dir,
                                s3_bucket=s3_bucket,
                                s3_prefix=s3_prefix,
                                uploaded_audio_files=uploaded_audio_files,
                                include_presigned_urls=True,  # Include URLs during in-progress
                            )
                            if sim_result:
                                completed_results.append(sim_result)
                                completed_simulations.add(item.name)
                    elif _is_simulation_started(item):
                        # Simulation in progress - get intermediate data (no audio URLs)
                        if item.name not in completed_simulations:
                            sim_result = _parse_voice_simulation_in_progress(
                                sim_dir=item,
                                personas_list=personas_list,
                                scenarios_list=scenarios_list,
                            )
                            if sim_result:
                                in_progress_results.append(sim_result)
                                # Track transcript length for change detection
                                transcript = sim_result.get("transcript") or []
                                in_progress_transcript_lengths.append(
                                    (item.name, len(transcript))
                                )

            # Build current state for change detection
            current_state = (
                len(completed_results),
                tuple(sorted(in_progress_transcript_lengths)),
            )

            # Only update DB if state changed
            if current_state != prev_state:
                all_results = completed_results + in_progress_results
                if all_results:
                    results_dict = {
                        "total_simulations": expected_total,
                        "completed_simulations": len(completed_results),
                        "simulation_results": all_results,
                        "results_s3_prefix": s3_prefix,
                    }
                    update_simulation_job(
                        task_id,
                        status=TaskStatus.IN_PROGRESS.value,
                        results=results_dict,
                    )
                    logger.info(
                        f"{log_prefix}: Updated DB with {len(completed_results)} completed + {len(in_progress_results)} in-progress simulations"
                    )
                prev_state = current_state

            time.sleep(poll_interval)

        # Process finished, wait for it to complete (skip if aborted — process is already killed)
        if not (task_id and _is_job_aborted(task_id)):
            process.wait()

    # Check if job was aborted by user before doing final processing
    if task_id:
        if _is_job_aborted(task_id):
            logger.info(
                f"Voice simulation {task_id} was aborted by user, skipping final processing"
            )
            return {
                "success": False,
                "total_simulations": 0,
                "metrics": None,
                "simulation_results": [],
                "error": "user_aborted",
            }

    # Read logs from files
    stdout = ""
    stderr = ""
    if stdout_log_path.exists():
        with open(stdout_log_path, "r") as f:
            stdout = f.read()
        if stdout:
            logger.info(f"{log_prefix} stdout: {stdout}")
    if stderr_log_path.exists():
        with open(stderr_log_path, "r") as f:
            stderr = f.read()
        if stderr:
            logger.info(f"{log_prefix} stderr: {stderr}")

    # Final pass: check for any remaining simulation directories that weren't processed
    # Don't include presigned URLs since status will be done
    for item in output_dir.iterdir():
        if (
            item.is_dir()
            and item.name.startswith("simulation_persona_")
            and item.name not in completed_simulations
        ):
            if _is_simulation_complete(item):
                logger.info(
                    f"{log_prefix}: Found remaining completed simulation directory: {item.name}"
                )
                sim_result = _parse_simulation_directory(
                    sim_dir=item,
                    output_dir=output_dir,
                    s3_bucket=s3_bucket,
                    s3_prefix=s3_prefix,
                    uploaded_audio_files=uploaded_audio_files,
                    include_presigned_urls=False,  # Don't include URLs for final done status
                )
                if sim_result:
                    completed_results.append(sim_result)
                    completed_simulations.add(item.name)

    # Strip presigned URLs from all completed_results before storing (for done status)
    # Only keep S3 paths for on-the-fly URL generation when status is fetched
    for sim_result in completed_results:
        sim_result.pop("audio_urls", None)
        sim_result.pop("conversation_wav_url", None)

    # Parse final results (metrics.json and results.csv)
    metrics_data = None
    metrics_file = output_dir / "metrics.json"
    results_file = output_dir / "results.csv"

    if metrics_file.exists():
        with open(metrics_file, "r", encoding="utf-8") as f:
            metrics_data = json.load(f)

    # Parse results.csv for aggregated scores
    if results_file.exists():
        import csv

        with open(results_file, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            results_data = list(reader)

    # Upload all other results to S3 (excluding audios which are already uploaded)
    for root, dirs, files in os.walk(output_dir):
        # Skip audios directories as they're already uploaded
        if "audios" in root.split(os.sep):
            continue
        for file in files:
            local_file_path = Path(root) / file
            # Skip audio files that were already uploaded
            if str(local_file_path) in uploaded_audio_files:
                continue
            relative_path = local_file_path.relative_to(output_dir)
            s3_key = f"{s3_prefix}/{relative_path}"
            upload_file_to_s3(s3, local_file_path, s3_bucket, s3_key)

    # Upload the config file to S3
    if config_file.exists():
        config_s3_key = f"{s3_prefix}/simulation_config.json"
        upload_file_to_s3(s3, config_file, s3_bucket, config_s3_key)
        logger.info(f"Uploaded config file to S3: {config_s3_key}")

    error = None
    if process.returncode != 0:
        is_failure = True
        error = f"Command failed with exit code {process.returncode}: {stderr}"
    elif not completed_results:
        is_failure = True
        error = "Simulation produced no output (no simulation directories found)"
    else:
        is_failure = False

    if is_failure:
        logger.error(error)
        capture_exception_to_sentry(RuntimeError(error))

    return {
        "success": not is_failure,
        "total_simulations": len(completed_results),
        "metrics": metrics_data,
        "simulation_results": completed_results,
        "error": error,
    }


def run_simulation_task(
    task_id: str,
    agent: Dict[str, Any],
    personas: List[Dict[str, Any]],
    scenarios: List[Dict[str, Any]],
    evaluators: List[Dict[str, Any]],
    s3_bucket: str,
    simulation_type: str = "text",
):
    """Run the simulation in the background (text or voice)."""
    try:
        logger.info(
            f"Running {simulation_type} simulation task {task_id} for agent {agent['uuid']} "
            f"with {len(personas)} persona(s), {len(scenarios)} scenario(s), "
            f"and {len(evaluators)} evaluator(s)"
        )
        update_simulation_job(task_id, status=TaskStatus.IN_PROGRESS.value)

        # Create temporary directory for processing (automatically cleaned up after use)
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            try:
                # Build calibrate config
                calibrate_config = _build_calibrate_simulation_config(
                    agent, personas, scenarios, evaluators, simulation_type=simulation_type
                )

                # Create input and output directories
                input_dir = temp_path / "input"
                output_dir = temp_path / "output"

                # Run calibrate simulation based on type
                results_prefix = f"simulations/runs/{task_id}"
                if simulation_type == "voice":
                    result = _run_calibrate_voice_simulation(
                        calibrate_config=calibrate_config,
                        input_dir=input_dir,
                        output_dir=output_dir,
                        s3_bucket=s3_bucket,
                        s3_prefix=results_prefix,
                        task_id=task_id,
                        log_prefix=f"Voice simulation {task_id}",
                    )
                else:
                    # Agent connection mode has no params.model — pass None
                    agent_cfg = agent.get("config") or {}
                    model_to_use = (
                        None
                        if agent_cfg.get("agent_url")
                        else calibrate_config["params"]["model"]
                    )
                    result = _run_calibrate_text_simulation(
                        model=model_to_use,
                        calibrate_config=calibrate_config,
                        input_dir=input_dir,
                        output_dir=output_dir,
                        s3_bucket=s3_bucket,
                        s3_prefix=results_prefix,
                        task_id=task_id,
                        log_prefix=f"Chat simulation {task_id}",
                    )

                # Check if job was aborted by user - don't overwrite abort results
                if _is_job_aborted(task_id):
                    logger.info(
                        f"Simulation task {task_id} was aborted by user, skipping final update"
                    )
                    return

                # Prepare results dict
                results_dict = {
                    "total_simulations": result["total_simulations"],
                    "metrics": result["metrics"],
                    "simulation_results": result["simulation_results"],
                    "results_s3_prefix": results_prefix,
                    "error": result.get("error"),
                }

                # Determine final status based on success
                final_status = (
                    TaskStatus.DONE.value
                    if result["success"]
                    else TaskStatus.FAILED.value
                )

                # Update job with results
                update_simulation_job(
                    task_id,
                    status=final_status,
                    results=results_dict,
                )

                logger.info(
                    f"{simulation_type.capitalize()} simulation task {task_id} completed: "
                    f"{result['total_simulations']} simulation(s) run, status={final_status}"
                )

            except Exception as e:
                # Check if job was aborted - don't overwrite abort results
                if _is_job_aborted(task_id):
                    logger.info(
                        f"Simulation task {task_id} was aborted by user, skipping error update"
                    )
                    return

                traceback.print_exc()
                capture_exception_to_sentry(e)
                update_simulation_job(
                    task_id,
                    status=TaskStatus.FAILED.value,
                    results={"error": f"Unexpected error during simulation: {str(e)}"},
                )
        # Temporary directory is automatically cleaned up here

    except Exception as e:
        # Check if job was aborted - don't overwrite abort results
        if _is_job_aborted(task_id):
            logger.info(
                f"Simulation task {task_id} was aborted by user, skipping error update"
            )
            return

        traceback.print_exc()
        capture_exception_to_sentry(e)
        update_simulation_job(
            task_id,
            status=TaskStatus.FAILED.value,
            results={"error": f"Task failed: {str(e)}"},
        )
    finally:
        # Try to start the next queued job
        try_start_queued_simulation_job(SIMULATION_JOB_TYPES)


@router.post("/{simulation_uuid}/run", response_model=TaskCreateResponse, summary="Run simulation")
async def run_simulation_endpoint(
    request: RunSimulationRequest,
    simulation_uuid: str = PathParam(
        description="The simulation to run",
        examples=["f47ac10b-58cc-4372-a567-0e02b2c3d479"],
    ),
    ctx: OrgContext = Depends(get_current_org),
):
    """Run a simulation as a background job"""
    simulation = get_simulation(simulation_uuid)
    if not simulation or simulation.get("org_uuid") != ctx.org_uuid:
        raise HTTPException(status_code=404, detail="Simulation not found")

    agent_uuid = simulation.get("agent_id")
    if not agent_uuid:
        raise HTTPException(
            status_code=400,
            detail="No agent linked to this simulation. Link an agent to the simulation first.",
        )

    # Verify agent exists
    agent = get_agent(agent_uuid)
    if not agent:
        raise HTTPException(status_code=404, detail="Agent not found")

    # Guard: agent connection is not supported for voice simulations
    agent_config = agent.get("config") or {}
    if agent_config.get("agent_url"):
        if request.type == "voice":
            raise HTTPException(
                status_code=400,
                detail="Voice simulations are not supported for agent connection mode. Use a Calibrate agent instead.",
            )
        if not agent_config.get("connection_verified"):
            raise HTTPException(
                status_code=400,
                detail="Agent connection not verified. Call POST /agents/{agent_uuid}/verify-connection first.",
            )

    # Get linked entities
    personas = get_personas_for_simulation(simulation_uuid)
    scenarios = get_scenarios_for_simulation(simulation_uuid)
    evaluators = get_evaluators_for_simulation(simulation_uuid)

    if not personas:
        raise HTTPException(
            status_code=400,
            detail="Simulation has no personas. Add at least one persona.",
        )

    if not scenarios:
        raise HTTPException(
            status_code=400,
            detail="Simulation has no scenarios. Add at least one scenario.",
        )

    # Get S3 configuration
    try:
        s3_bucket = get_s3_output_config()
    except ValueError as e:
        raise HTTPException(status_code=500, detail=str(e))

    can_start = can_start_simulation_job(SIMULATION_JOB_TYPES, ctx.org_uuid)
    initial_status = (
        TaskStatus.IN_PROGRESS.value if can_start else TaskStatus.QUEUED.value
    )

    # Create job in database with details for recovery
    job_id = create_simulation_job(
        simulation_id=simulation_uuid,
        job_type=request.type,
        status=initial_status,
        details={
            "simulation_uuid": simulation_uuid,
            "agent_uuid": agent_uuid,
            "s3_bucket": s3_bucket,
            "evaluators": _snapshot_evaluators_for_job_details(evaluators),
        },
        results=None,
    )

    if can_start:
        # Start background task in a separate thread
        thread = threading.Thread(
            target=run_simulation_task,
            args=(job_id, agent, personas, scenarios, evaluators, s3_bucket, request.type),
            daemon=True,
        )
        thread.start()
        logger.info(f"Started {request.type} simulation job {job_id} immediately")
    else:
        logger.info(f"Queued {request.type} simulation job {job_id}")

    return TaskCreateResponse(task_id=job_id, status=initial_status)


@router.post("/run/{job_uuid}/abort", response_model=SimulationRunStatusResponse, summary="Abort simulation run")
async def abort_simulation_run(
    job_uuid: str = PathParam(
        description="The in-progress simulation run to abort",
        examples=["f47ac10b-58cc-4372-a567-0e02b2c3d479"],
    ),
    ctx: OrgContext = Depends(get_current_org),
):
    """Abort an in-progress simulation run, keeping partial results collected so far"""
    simulation_job = get_simulation_job(job_uuid)
    if not simulation_job:
        raise HTTPException(status_code=404, detail="Job not found")

    simulation_id = simulation_job.get("simulation_id")
    if simulation_id:
        simulation = get_simulation(simulation_id)
        if not simulation or simulation.get("org_uuid") != ctx.org_uuid:
            raise HTTPException(status_code=404, detail="Job not found")
    else:
        raise HTTPException(status_code=404, detail="Job not found")

    # Only allow aborting in-progress jobs
    if simulation_job.get("status") != TaskStatus.IN_PROGRESS.value:
        raise HTTPException(status_code=400, detail="Can only abort in-progress jobs")

    details = simulation_job.get("details") or {}
    results = simulation_job.get("results") or {}

    # Kill running process
    pid = details.get("pid") or details.get("pgid")
    if pid:
        kill_process_group(pid, job_uuid)

    # Mark each simulation result as aborted or not
    # Complete simulations (have evaluation_results) are not aborted
    simulation_results = results.get("simulation_results") or []
    for sim_result in simulation_results:
        sim_result["aborted"] = sim_result.get("evaluation_results") is None

    results["simulation_results"] = simulation_results

    # Mark as aborted in details and save with done status
    update_simulation_job(
        job_uuid,
        status=TaskStatus.DONE.value,
        results=results,
        details={"aborted": True},
    )

    # Try to start the next queued job
    try_start_queued_simulation_job(SIMULATION_JOB_TYPES)

    # Re-read the job to get the updated_at timestamp and merged details
    updated_job = get_simulation_job(job_uuid)
    updated_at = updated_job["updated_at"] if updated_job else ""
    results = (updated_job or {}).get("results") or {}
    abort_details = (updated_job or {}).get("details") or {}
    abort_simulation_results = results.get("simulation_results") or []
    evaluators_out, abort_simulation_results = apply_simulation_job_evaluator_enrichment(
        abort_details, abort_simulation_results
    )

    # Calculate run name
    run_name = "Run 1"
    if simulation_id:
        all_jobs = get_simulation_jobs_for_simulation(simulation_id)
        sorted_jobs = sorted(all_jobs, key=lambda j: j.get("created_at", ""))
        for idx, j in enumerate(sorted_jobs, start=1):
            if j["uuid"] == job_uuid:
                run_name = f"Run {idx}"
                break

    return SimulationRunStatusResponse(
        task_id=job_uuid,
        name=run_name,
        status=TaskStatus.DONE.value,
        type=simulation_job["type"],
        updated_at=updated_at,
        total_simulations=results.get("total_simulations"),
        completed_simulations=results.get("completed_simulations"),
        metrics=results.get("metrics"),
        simulation_results=abort_simulation_results,
        evaluators=evaluators_out,
        error=results.get("error"),
        is_public=bool(updated_job.get("is_public")) if updated_job else False,
        share_token=updated_job.get("share_token") if updated_job else None,
    )


@router.delete("/run/{job_uuid}", summary="Delete simulation run")
async def delete_simulation_job_endpoint(
    job_uuid: str = PathParam(
        description="The simulation run to delete",
        examples=["f47ac10b-58cc-4372-a567-0e02b2c3d479"],
    ),
    ctx: OrgContext = Depends(get_current_org),
):
    """Delete a simulation run and its results"""
    simulation_job = get_simulation_job(job_uuid)
    if not simulation_job:
        raise HTTPException(status_code=404, detail="Job not found")

    simulation_id = simulation_job.get("simulation_id")
    if simulation_id:
        simulation = get_simulation(simulation_id)
        if not simulation or simulation.get("org_uuid") != ctx.org_uuid:
            raise HTTPException(status_code=404, detail="Job not found")
    else:
        raise HTTPException(status_code=404, detail="Job not found")

    # Check if this was a running job (to trigger next queued job after delete)
    was_running = simulation_job.get("status") == TaskStatus.IN_PROGRESS.value
    details = simulation_job.get("details") or {}

    # Kill running process if job is in progress
    if was_running:
        pid = details.get("pid") or details.get("pgid")
        if pid:
            kill_process_group(pid, job_uuid)

    # Delete the job
    deleted = delete_simulation_job(job_uuid)
    if not deleted:
        raise HTTPException(status_code=404, detail="Job not found")

    # If the deleted job was running, try to start the next queued job
    if was_running:
        try_start_queued_simulation_job(SIMULATION_JOB_TYPES)

    return {"message": "Simulation job deleted successfully"}
