"""Org-scoped entity ownership guards for routers.

Every helper returns 404 (never 403) when the caller's org cannot see the
resource, so existence is not leaked across tenants.
"""

from typing import Any, Dict

from fastapi import HTTPException

from db import get_agent, get_evaluator


def ensure_owned_agent(agent_uuid: str, org_uuid: str) -> Dict[str, Any]:
    """Return the agent when it belongs to `org_uuid`, else 404."""
    agent = get_agent(agent_uuid)
    if not agent or agent.get("org_uuid") != org_uuid:
        raise HTTPException(status_code=404, detail="Agent not found")
    return agent


def ensure_owned_evaluator(evaluator_uuid: str, org_uuid: str) -> Dict[str, Any]:
    """Return the evaluator when visible to `org_uuid`, else 404.

    Seeded defaults (`org_uuid IS NULL`) are visible to every org.
    """
    evaluator = get_evaluator(evaluator_uuid)
    if not evaluator:
        raise HTTPException(status_code=404, detail="Evaluator not found")
    owner_org = evaluator.get("org_uuid")
    if owner_org is not None and owner_org != org_uuid:
        raise HTTPException(status_code=404, detail="Evaluator not found")
    return evaluator
