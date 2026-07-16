"""Broad-coverage CRUD tests for src/db.py.

Each entity has its own block: create → read (by uuid, list, bulk) →
update → soft-delete → list-after-delete. We also exercise the pivot
tables (agent_tools, agent_tests, simulation_personas, simulation_scenarios,
simulation_evaluators, test_evaluators) and the queue accounting helpers.

Tests share one initialized DB (conftest fixture). Every row uses a
freshly minted name/uuid so tests are order-independent.
"""

from __future__ import annotations

import sqlite3
import time
import uuid as _uuid
from unittest.mock import MagicMock, patch

import pytest

import db
from db import NameAlreadyExistsError


def _u(prefix: str = "x") -> str:
    """Short unique suffix to avoid name collisions across tests."""
    return f"{prefix}-{_uuid.uuid4().hex[:8]}"


# ---------------------------------------------------------------------------
# init / safety helpers
# ---------------------------------------------------------------------------


def test_init_db_is_idempotent():
    # The session fixture already called init_db(); calling again must
    # not raise (migrations are wrapped in try/except, seeds use IF NOT EXISTS).
    db.init_db()
    db.init_db()


def test_llm_general_seed_and_vocabulary(user):
    """`llm-general` is a first-class type across all three axes: the
    default-prompt purpose, the seeded evaluator, and the annotation-task
    type allowlist."""
    assert "llm-general" in db.ANNOTATION_TASK_TYPES
    purpose = db.DEFAULT_PROMPTS_BY_PURPOSE["llm-general"]
    assert purpose["evaluator_type"] == "llm-general"
    assert purpose["data_type"] == "text"

    db.init_db()  # idempotent — seeds default-llm-general
    seeded = db.get_evaluator_by_slug("default-llm-general")
    assert seeded is not None
    assert seeded["evaluator_type"] == "llm-general"
    assert seeded["data_type"] == "text"
    # Carries a real {{criteria}} variable so per-item criteria substitute.
    version = db.get_evaluator_version(seeded["live_version_id"])
    assert any(v["name"] == "criteria" for v in (version["variables"] or []))

    # `llm-general` is an accepted custom evaluator_type, not just a seed slug.
    assert "llm-general" in db.VALID_EVALUATOR_TYPES
    ev_uuid = db.create_evaluator(
        name=_u("custom-llm-general"),
        evaluator_type="llm-general",
        owner_user_id=user["uuid"],
        org_uuid=user["org_uuid"],
    )
    assert db.get_evaluator(ev_uuid)["evaluator_type"] == "llm-general"


def test_simulation_to_conversation_migration():
    """init_db() converts the legacy `simulation` value to `conversation` for
    both `evaluators.evaluator_type` and `annotation_tasks.type`. Rows are
    inserted raw to bypass the API-layer validators that now reject the old
    value."""
    user_uuid = db.create_user("M", "G", _u("mig") + "@example.com")
    ev_uuid = str(_uuid.uuid4())
    task_uuid = str(_uuid.uuid4())
    with db.get_db_connection() as conn:
        cur = conn.cursor()
        cur.execute(
            "INSERT INTO evaluators (uuid, name, evaluator_type) "
            "VALUES (?, ?, 'simulation')",
            (ev_uuid, _u("legacy-ev")),
        )
        cur.execute(
            "INSERT INTO annotation_tasks (uuid, user_id, name, type) "
            "VALUES (?, ?, ?, 'simulation')",
            (task_uuid, user_uuid, _u("legacy-task")),
        )
        conn.commit()

    db.init_db()  # idempotent — runs the rename migrations

    with db.get_db_connection() as conn:
        cur = conn.cursor()
        ev_type = cur.execute(
            "SELECT evaluator_type FROM evaluators WHERE uuid = ?", (ev_uuid,)
        ).fetchone()[0]
        task_type = cur.execute(
            "SELECT type FROM annotation_tasks WHERE uuid = ?", (task_uuid,)
        ).fetchone()[0]
    assert ev_type == "conversation"
    assert task_type == "conversation"


def test_legacy_api_keys_table_is_dropped_and_recreated():
    """init_db() drops a legacy user-scoped `api_keys` table (the abandoned
    `user_id` shape, no `org_uuid`) and recreates the current org-scoped one,
    since `CREATE TABLE IF NOT EXISTS` can't reshape an existing table."""
    with db.get_db_connection() as conn:
        cur = conn.cursor()
        cur.execute("DROP TABLE IF EXISTS api_keys")
        cur.execute(
            """
            CREATE TABLE api_keys (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                uuid TEXT NOT NULL UNIQUE,
                user_id TEXT NOT NULL,
                key_hash TEXT NOT NULL,
                key_prefix TEXT NOT NULL,
                name TEXT NOT NULL
            )
            """
        )
        cur.execute(
            "INSERT INTO api_keys (uuid, user_id, key_hash, key_prefix, name) "
            "VALUES (?, 'u1', 'h', 'sk_abc', 'legacy')",
            (str(_uuid.uuid4()),),
        )
        conn.commit()

    db.init_db()  # idempotent — runs the api_keys reshape migration

    with db.get_db_connection() as conn:
        cur = conn.cursor()
        cols = {r[1] for r in cur.execute("PRAGMA table_info(api_keys)").fetchall()}
        rows = cur.execute("SELECT COUNT(*) FROM api_keys").fetchone()[0]
    assert "org_uuid" in cols and "key_last_four" in cols
    assert "user_id" not in cols
    assert rows == 0  # legacy row dropped with the table


def test_is_name_taken_whitelist_only():
    with pytest.raises(ValueError):
        db.is_name_taken("not_a_table", "n", "u")


# ---------------------------------------------------------------------------
# Users
# ---------------------------------------------------------------------------


@pytest.fixture
def user():
    email = f"{_u('user')}@example.com"
    user_uuid = db.create_user("Alice", "Smith", email)
    # `create_user` auto-provisions a personal org and an owner membership.
    org = db.get_personal_org_for_user(user_uuid)
    yield {
        "uuid": user_uuid,
        "email": email,
        "org_uuid": org["uuid"] if org else None,
    }


def test_user_crud_and_get_or_create(user):
    fetched = db.get_user(user["uuid"])
    assert fetched["email"] == user["email"]
    assert fetched["first_name"] == "Alice"

    by_email = db.get_user_by_email(user["email"])
    assert by_email["uuid"] == user["uuid"]

    assert db.update_user(user["uuid"], first_name="Alicia") is True
    assert db.get_user(user["uuid"])["first_name"] == "Alicia"
    assert db.update_user(user["uuid"]) is False  # no fields

    # get_or_create with same email but a different name → updates
    existing = db.get_or_create_user(user["email"], "Aly", "S")
    assert existing["first_name"] == "Aly"

    # get_or_create with a new email → creates
    new_email = f"{_u('new')}@example.com"
    created = db.get_or_create_user(new_email, "Bob", "Builder")
    assert created["email"] == new_email
    assert db.get_user_by_email(new_email) is not None

    assert db.get_user_by_email("missing-email@example.com") is None

    all_users = db.get_all_users()
    assert any(u["uuid"] == user["uuid"] for u in all_users)


def test_create_user_with_password_then_delete():
    email = f"{_u('pw')}@example.com"
    pw_uuid = db.create_user_with_password("Pass", "Word", email, "hash$abc")
    fetched = db.get_user(pw_uuid)
    assert fetched["email"] == email
    assert db.delete_user(pw_uuid) is True
    assert db.get_user(pw_uuid) is None
    assert db.delete_user(pw_uuid) is False  # already gone


# ---------------------------------------------------------------------------
# Agents + Tools + agent_tools pivot
# ---------------------------------------------------------------------------


def test_agents_tools_and_pivot(user):
    agent_uuid = db.create_agent(
        name=_u("agent"),
        agent_type="agent",
        config={"llm": "gpt-4"},
        user_id=user["uuid"], org_uuid=user["org_uuid"],
    )
    fetched = db.get_agent(agent_uuid)
    assert fetched["config"] == {"llm": "gpt-4"}

    # update / list / delete
    assert db.update_agent(agent_uuid, name=_u("agent2"), config={"llm": "gpt-5"})
    assert db.update_agent(agent_uuid) is False
    assert any(a["uuid"] == agent_uuid for a in db.get_all_agents(org_uuid=user["org_uuid"]))
    assert any(a["uuid"] == agent_uuid for a in db.get_all_agents())

    with pytest.raises(ValueError):
        db.create_agent(name=_u("no-owner"), org_uuid=None)

    tool_uuid = db.create_tool(
        name=_u("tool"),
        description="desc",
        config={"type": "structured_output", "parameters": []},
        user_id=user["uuid"], org_uuid=user["org_uuid"],
    )
    assert db.get_tool(tool_uuid)["description"] == "desc"
    assert db.update_tool(tool_uuid, description="updated")
    assert db.update_tool(tool_uuid) is False
    assert any(t["uuid"] == tool_uuid for t in db.get_all_tools(org_uuid=user["org_uuid"]))
    assert any(t["uuid"] == tool_uuid for t in db.get_all_tools())

    with pytest.raises(ValueError):
        db.create_tool(name=_u("no-owner-tool"), description="d", org_uuid=None)

    # link / unlink / re-link the agent_tools pivot
    link_id = db.add_tool_to_agent(agent_uuid, tool_uuid)
    assert isinstance(link_id, int)
    assert db.get_agent_tool_link(agent_uuid, tool_uuid) is not None
    assert any(t["uuid"] == tool_uuid for t in db.get_tools_for_agent(agent_uuid))
    assert any(a["uuid"] == agent_uuid for a in db.get_agents_for_tool(tool_uuid))
    assert db.remove_tool_from_agent(agent_uuid, tool_uuid) is True
    assert db.get_agent_tool_link(agent_uuid, tool_uuid) is None
    # re-add restores the soft-deleted row
    relinked = db.add_tool_to_agent(agent_uuid, tool_uuid)
    assert relinked == link_id
    assert db.get_all_agent_tools()

    # soft delete cascade — link already active from re-add above
    assert db.delete_agent(agent_uuid) is True
    assert db.get_agent(agent_uuid) is None
    assert db.delete_agent(agent_uuid) is False
    assert db.get_tools_for_agent(agent_uuid) == []  # link cascaded

    assert db.delete_tool(tool_uuid) is True
    assert db.delete_tool(tool_uuid) is False


# ---------------------------------------------------------------------------
# Tests entity + bulk
# ---------------------------------------------------------------------------


def test_tests_crud_and_bulk(user):
    t_uuid = db.create_test(name=_u("test"), type="llm", config={"k": "v"}, user_id=user["uuid"], org_uuid=user["org_uuid"])
    assert db.get_test(t_uuid)["config"] == {"k": "v"}
    assert db.update_test(t_uuid, name=_u("renamed"), type="llm", config={"k": "v2"})
    assert db.update_test(t_uuid) is False
    assert any(t["uuid"] == t_uuid for t in db.get_all_tests_summary(org_uuid=user["org_uuid"]))
    assert any(t["uuid"] == t_uuid for t in db.get_all_tests_summary())

    with pytest.raises(ValueError):
        db.create_test(name=_u("no-owner-test"), type="llm", org_uuid=None)

    # bulk_create_tests + collision
    bulk_uuids = db.bulk_create_tests(
        [
            {"name": _u("bulk1"), "type": "llm", "config": None},
            {"name": _u("bulk2"), "type": "llm", "config": {"x": 1}},
        ],
        user_id=user["uuid"], org_uuid=user["org_uuid"],
    )
    assert len(bulk_uuids) == 2

    # collide via pre-existing name in DB
    dup_name = _u("dup-pre")
    db.create_test(name=dup_name, type="llm", user_id=user["uuid"], org_uuid=user["org_uuid"])
    with pytest.raises(ValueError):
        db.bulk_create_tests(
            [{"name": dup_name, "type": "llm", "config": None}],
            user_id=user["uuid"], org_uuid=user["org_uuid"],
        )
    with pytest.raises(ValueError):
        db.bulk_create_tests([{"name": "n", "type": "llm", "config": None}], org_uuid=None)

    # single delete + bulk delete
    assert db.delete_test(t_uuid) is True
    assert db.delete_test(t_uuid) is False
    assert db.bulk_delete_tests(bulk_uuids, user["org_uuid"]) == 2
    assert db.bulk_delete_tests([], user["org_uuid"]) == 0
    # unowned UUIDs → 0
    assert db.bulk_delete_tests([str(_uuid.uuid4())], user["org_uuid"]) == 0


def test_bulk_delete_agents_cascade_and_scoping(user):
    def _agent(name):
        return db.create_agent(
            name=_u(name), agent_type="agent", config={},
            user_id=user["uuid"], org_uuid=user["org_uuid"],
        )

    a1, a2 = _agent("bd1"), _agent("bd2")

    tool = db.create_tool(
        name=_u("bd-tool"), description="d",
        config={"type": "structured_output", "parameters": []},
        user_id=user["uuid"], org_uuid=user["org_uuid"],
    )
    test = db.create_test(name=_u("bd-test"), type="llm", user_id=user["uuid"], org_uuid=user["org_uuid"])
    ev = db.get_evaluator_by_slug("default-safety")
    db.add_tool_to_agent(a1, tool)
    db.add_test_to_agent(a1, test)
    db.add_evaluator_to_agent(a1, ev["uuid"])

    # Empty / dedupe.
    assert db.bulk_delete_agents([], user["org_uuid"]) == {"deleted_count": 0, "not_found": []}

    # An unknown UUID makes it all-or-nothing — nothing is deleted.
    ghost = str(_uuid.uuid4())
    res = db.bulk_delete_agents([a1, ghost], user["org_uuid"])
    assert res == {"deleted_count": 0, "not_found": [ghost]}
    assert db.get_agent(a1) is not None

    # Happy path deletes both and cascades the pivots.
    res = db.bulk_delete_agents([a1, a1, a2], user["org_uuid"])
    assert res == {"deleted_count": 2, "not_found": []}
    assert db.get_agent(a1) is None and db.get_agent(a2) is None
    assert db.get_tools_for_agent(a1) == []
    assert db.get_tests_for_agent(a1) == []
    assert db.get_evaluators_for_agent(a1) == []
    # Pre-existing entities themselves are untouched.
    assert db.get_tool(tool) is not None
    assert db.get_test(test) is not None


def test_bulk_delete_agents_is_org_scoped(user):
    agent = db.create_agent(
        name=_u("scoped"), agent_type="agent", config={},
        user_id=user["uuid"], org_uuid=user["org_uuid"],
    )
    other = db.create_user("Bob", "B", f"{_u('other')}@example.com")
    other_org = db.get_personal_org_for_user(other)["uuid"]

    # Caller in another org can't reach it → reported not_found, not deleted.
    res = db.bulk_delete_agents([agent], other_org)
    assert res == {"deleted_count": 0, "not_found": [agent]}
    assert db.get_agent(agent) is not None


# ---------------------------------------------------------------------------
# Personas + Scenarios
# ---------------------------------------------------------------------------


def test_personas_scenarios(user):
    p_uuid = db.create_persona(
        name=_u("persona"),
        description="some persona",
        config={"language": "en"},
        user_id=user["uuid"], org_uuid=user["org_uuid"],
    )
    assert db.get_persona(p_uuid)["description"] == "some persona"
    assert db.update_persona(p_uuid, name=_u("persona-new"), description="d2", config={"k": 1})
    assert db.update_persona(p_uuid) is False
    assert any(p["uuid"] == p_uuid for p in db.get_all_personas(org_uuid=user["org_uuid"]))
    assert any(p["uuid"] == p_uuid for p in db.get_all_personas())
    assert db.delete_persona(p_uuid) is True
    assert db.delete_persona(p_uuid) is False
    with pytest.raises(ValueError):
        db.create_persona(name=_u("no-owner-p"), org_uuid=None)

    s_uuid = db.create_scenario(
        name=_u("scen"),
        description="scenario",
        user_id=user["uuid"], org_uuid=user["org_uuid"],
    )
    assert db.get_scenario(s_uuid)["description"] == "scenario"
    assert db.update_scenario(s_uuid, name=_u("scen-new"), description="d2")
    assert db.update_scenario(s_uuid) is False
    assert any(s["uuid"] == s_uuid for s in db.get_all_scenarios(org_uuid=user["org_uuid"]))
    assert any(s["uuid"] == s_uuid for s in db.get_all_scenarios())
    assert db.delete_scenario(s_uuid) is True
    assert db.delete_scenario(s_uuid) is False
    with pytest.raises(ValueError):
        db.create_scenario(name=_u("no-owner-s"), org_uuid=None)


# ---------------------------------------------------------------------------
# Evaluators (top-level rows + versions + duplicate)
# ---------------------------------------------------------------------------


def test_evaluator_crud_versions_duplicate(user):
    ev_uuid = db.create_evaluator(
        name=_u("ev"),
        description="d",
        evaluator_type="llm",
        data_type="text",
        kind="single",
        output_type="binary",
        owner_user_id=user["uuid"], org_uuid=user["org_uuid"],
        slug=None,
    )
    row = db.get_evaluator(ev_uuid)
    assert row["evaluator_type"] == "llm"

    by_uuids = db.get_evaluators_by_uuids([ev_uuid, "not-real"])
    assert ev_uuid in by_uuids
    assert db.get_evaluators_by_uuids([]) == {}
    assert db.get_evaluators_by_uuids([None]) == {}  # type: ignore[arg-type]

    # validation errors
    with pytest.raises(ValueError):
        db.create_evaluator(name=_u("bad-type"), evaluator_type="nope", owner_user_id=user["uuid"], org_uuid=user["org_uuid"])
    with pytest.raises(ValueError):
        db.create_evaluator(name=_u("bad-data"), data_type="nope", owner_user_id=user["uuid"], org_uuid=user["org_uuid"])
    with pytest.raises(ValueError):
        db.create_evaluator(name=_u("bad-kind"), kind="nope", owner_user_id=user["uuid"], org_uuid=user["org_uuid"])
    with pytest.raises(ValueError):
        db.create_evaluator(name=_u("bad-output"), output_type="nope", owner_user_id=user["uuid"], org_uuid=user["org_uuid"])

    # versions
    v1 = db.create_evaluator_version(
        ev_uuid,
        judge_model="openai/gpt-4",
        system_prompt="Judge this",
    )
    assert v1["version_number"] == 1
    v2 = db.create_evaluator_version(
        ev_uuid,
        judge_model="openai/gpt-4",
        system_prompt="Judge this again",
    )
    assert v2["version_number"] == 2
    assert db.get_evaluator_version(v1["uuid"])["uuid"] == v1["uuid"]
    assert db.get_evaluator_version("nope") is None
    versions = db.get_evaluator_versions(ev_uuid)
    assert {v["uuid"] for v in versions} == {v1["uuid"], v2["uuid"]}
    assert db.set_evaluator_live_version(ev_uuid, v2["uuid"]) is True
    assert db.set_evaluator_live_version(ev_uuid, "missing") is False

    # rating evaluator path (output_config required)
    rating_uuid = db.create_evaluator(
        name=_u("rating"),
        output_type="rating",
        owner_user_id=user["uuid"], org_uuid=user["org_uuid"],
    )
    with pytest.raises(ValueError):
        db.create_evaluator_version(rating_uuid, judge_model="m", system_prompt="p")
    with pytest.raises(ValueError):
        db.create_evaluator_version(
            rating_uuid,
            judge_model="m",
            system_prompt="p",
            output_config={"scale": "not a list"},
        )
    with pytest.raises(ValueError):
        db.create_evaluator_version(
            rating_uuid,
            judge_model="m",
            system_prompt="p",
            output_config={"scale": [{"value": 1}]},  # need >=2
        )
    good = db.create_evaluator_version(
        rating_uuid,
        judge_model="m",
        system_prompt="p",
        output_config={"scale": [{"value": 1, "name": "a"}, {"value": 2, "name": "b"}]},
        variables=[{"name": "x", "default": ""}],
    )
    assert good["variables"] == [{"name": "x", "default": ""}]

    # create_evaluator_version on non-existent parent
    with pytest.raises(ValueError):
        db.create_evaluator_version("not-real-uuid", judge_model="m", system_prompt="p")

    # name lookup helpers
    assert db.evaluator_name_exists(row["name"], org_uuid=user["org_uuid"]) is True
    assert db.evaluator_name_exists(row["name"], org_uuid=user["org_uuid"], exclude_uuid=ev_uuid) is False
    assert db.evaluator_name_exists("nonexistent-name-zzz", org_uuid=None) is False
    assert db.get_evaluator_by_slug("default-safety") is not None
    assert db.get_evaluator_by_slug("nope") is None
    assert db.legacy_metric_uuid_exists("non-existent") is False
    assert db.get_evaluator_uuid_for_legacy_metric("non-existent") is None

    # listing
    user_visible = db.get_all_evaluators(org_uuid=user["org_uuid"])
    assert any(e["uuid"] == ev_uuid for e in user_visible)
    own_only = db.get_all_evaluators(org_uuid=user["org_uuid"], include_defaults=False)
    assert all(e["owner_user_id"] for e in own_only)
    filtered = db.get_all_evaluators(org_uuid=user["org_uuid"], evaluator_type="llm", data_type="text")
    assert all(e["evaluator_type"] == "llm" for e in filtered)
    assert db.get_all_evaluators() is not None  # admin view

    # update + validations
    assert db.update_evaluator(ev_uuid, description="new desc")
    assert db.update_evaluator(ev_uuid) is False
    with pytest.raises(ValueError):
        db.update_evaluator(ev_uuid, evaluator_type="bogus")
    with pytest.raises(ValueError):
        db.update_evaluator(ev_uuid, data_type="bogus")
    with pytest.raises(ValueError):
        db.update_evaluator(ev_uuid, kind="bogus")
    with pytest.raises(ValueError):
        db.update_evaluator(ev_uuid, output_type="bogus")

    # duplicate -> new evaluator + cloned live version
    dup_uuid = db.duplicate_evaluator(ev_uuid, new_name=_u("dup"), owner_user_id=user["uuid"], org_uuid=user["org_uuid"])
    assert dup_uuid is not None
    assert db.get_evaluator(dup_uuid) is not None
    assert db.duplicate_evaluator("does-not-exist", new_name="x", owner_user_id=user["uuid"], org_uuid=user["org_uuid"]) is None

    # delete (custom evaluators only)
    assert db.delete_evaluator(ev_uuid) is True
    assert db.delete_evaluator(ev_uuid) is False
    # cannot delete a seeded default evaluator
    seeded = db.get_evaluator_by_slug("default-safety")
    assert db.delete_evaluator(seeded["uuid"]) is False


# ---------------------------------------------------------------------------
# Simulations + pivots (personas/scenarios/evaluators)
# ---------------------------------------------------------------------------


def test_simulations_and_pivots(user):
    agent_uuid = db.create_agent(name=_u("a-sim"), user_id=user["uuid"], org_uuid=user["org_uuid"])
    sim_uuid = db.create_simulation(name=_u("sim"), agent_id=agent_uuid, user_id=user["uuid"], org_uuid=user["org_uuid"])
    assert db.get_simulation(sim_uuid)["agent_id"] == agent_uuid
    assert any(s["uuid"] == sim_uuid for s in db.get_all_simulations(org_uuid=user["org_uuid"]))
    assert any(s["uuid"] == sim_uuid for s in db.get_all_simulations())

    assert db.update_simulation(sim_uuid, name=_u("sim2"))
    assert db.update_simulation(sim_uuid) is False
    new_agent = db.create_agent(name=_u("a2"), user_id=user["uuid"], org_uuid=user["org_uuid"])
    assert db.update_simulation(sim_uuid, agent_id=new_agent)
    assert db.update_simulation(sim_uuid, clear_agent=True)
    assert db.get_simulation(sim_uuid)["agent_id"] is None

    with pytest.raises(ValueError):
        db.create_simulation(name="no-owner", org_uuid=None)

    # persona pivot
    persona_uuid = db.create_persona(name=_u("p"), user_id=user["uuid"], org_uuid=user["org_uuid"])
    pid = db.add_persona_to_simulation(sim_uuid, persona_uuid)
    assert isinstance(pid, int)
    assert db.get_simulation_persona_link(sim_uuid, persona_uuid)
    assert any(p["uuid"] == persona_uuid for p in db.get_personas_for_simulation(sim_uuid))
    assert db.get_all_simulation_personas()
    assert db.remove_persona_from_simulation(sim_uuid, persona_uuid) is True
    # re-add restores soft-deleted link
    pid2 = db.add_persona_to_simulation(sim_uuid, persona_uuid)
    assert pid2 == pid

    # scenario pivot
    scenario_uuid = db.create_scenario(name=_u("s"), user_id=user["uuid"], org_uuid=user["org_uuid"])
    sid = db.add_scenario_to_simulation(sim_uuid, scenario_uuid)
    assert db.get_simulation_scenario_link(sim_uuid, scenario_uuid)
    assert any(s["uuid"] == scenario_uuid for s in db.get_scenarios_for_simulation(sim_uuid))
    assert db.get_all_simulation_scenarios()
    assert db.remove_scenario_from_simulation(sim_uuid, scenario_uuid) is True
    sid2 = db.add_scenario_to_simulation(sim_uuid, scenario_uuid)
    assert sid2 == sid

    # evaluator pivot — uses a seeded default + its live version
    seeded = db.get_evaluator_by_slug("default-safety")
    v = db.get_evaluator_versions(seeded["uuid"])[0]
    eid = db.add_evaluator_to_simulation(
        sim_uuid, seeded["uuid"], v["uuid"], variable_values={}
    )
    assert any(e["uuid"] == seeded["uuid"] for e in db.get_evaluators_for_simulation(sim_uuid))
    assert db.remove_evaluator_from_simulation(sim_uuid, seeded["uuid"]) is True
    eid2 = db.add_evaluator_to_simulation(sim_uuid, seeded["uuid"], v["uuid"])
    assert eid2 == eid

    # delete the simulation — cascades to persona/scenario/metric pivots
    # (simulation_evaluators is not part of the cascade per `delete_simulation`).
    assert db.delete_simulation(sim_uuid) is True
    assert db.delete_simulation(sim_uuid) is False
    assert db.get_personas_for_simulation(sim_uuid) == []
    assert db.get_scenarios_for_simulation(sim_uuid) == []


# ---------------------------------------------------------------------------
# Agent-Tests pivot + set_test_evaluators
# ---------------------------------------------------------------------------


def test_agent_tests_pivot_and_test_evaluators(user):
    agent_uuid = db.create_agent(name=_u("a-tests"), user_id=user["uuid"], org_uuid=user["org_uuid"])
    test_uuid = db.create_test(name=_u("t"), type="llm", user_id=user["uuid"], org_uuid=user["org_uuid"])
    other_test = db.create_test(name=_u("t2"), type="llm", user_id=user["uuid"], org_uuid=user["org_uuid"])

    db.add_test_to_agent(agent_uuid, test_uuid)
    db.add_test_to_agent(agent_uuid, other_test)
    assert db.get_agent_test_link(agent_uuid, test_uuid)
    assert len(db.get_tests_for_agent(agent_uuid)) == 2
    assert any(a["uuid"] == agent_uuid for a in db.get_agents_for_test(test_uuid))
    assert db.get_all_agent_tests()

    # remove one + bulk remove rest
    assert db.remove_test_from_agent(agent_uuid, test_uuid) is True
    assert db.bulk_remove_tests_from_agent(agent_uuid, [other_test]) == 1
    assert db.bulk_remove_tests_from_agent(agent_uuid, []) == 0

    # re-add (restores soft-deleted link)
    db.add_test_to_agent(agent_uuid, test_uuid)

    # set_test_evaluators with an evaluator that has a live version
    seeded = db.get_evaluator_by_slug("default-safety")
    db.set_test_evaluators(
        test_uuid,
        [{"evaluator_id": seeded["uuid"], "variable_values": {"k": "v"}}],
    )
    assert any(e["uuid"] == seeded["uuid"] for e in db.get_evaluators_for_test(test_uuid))

    # explicit version_id branch
    v = db.get_evaluator_versions(seeded["uuid"])[0]
    db.set_test_evaluators(
        test_uuid,
        [{"evaluator_id": seeded["uuid"], "evaluator_version_id": v["uuid"]}],
    )
    # add direct pivot + soft-delete restore
    db.remove_evaluator_from_test(test_uuid, seeded["uuid"])
    eid = db.add_evaluator_to_test(test_uuid, seeded["uuid"], v["uuid"], variable_values={"a": 1})
    assert isinstance(eid, int)
    db.remove_evaluator_from_test(test_uuid, seeded["uuid"])
    eid2 = db.add_evaluator_to_test(test_uuid, seeded["uuid"], v["uuid"])
    assert eid2 == eid

    # evaluator with no live version → set_test_evaluators raises
    no_live = db.create_evaluator(name=_u("no-live"), owner_user_id=user["uuid"], org_uuid=user["org_uuid"])
    with pytest.raises(ValueError):
        db.set_test_evaluators(test_uuid, [{"evaluator_id": no_live}])


def test_backfill_agent_evaluator_links_from_test_evaluators(user):
    """On first deploy the agent_evaluators pivot is backfilled once from
    test_evaluators; later init_db() runs skip it."""
    agent_uuid = db.create_agent(
        name=_u("a-backfill-ev"), user_id=user["uuid"], org_uuid=user["org_uuid"]
    )
    test_uuid = db.create_test(
        name=_u("t-backfill-ev"), type="llm", user_id=user["uuid"], org_uuid=user["org_uuid"]
    )
    other_test = db.create_test(
        name=_u("t-backfill-ev2"), type="llm", user_id=user["uuid"], org_uuid=user["org_uuid"]
    )
    db.add_test_to_agent(agent_uuid, test_uuid)
    db.add_test_to_agent(agent_uuid, other_test)

    seeded_a = db.get_evaluator_by_slug("default-safety")
    seeded_b = db.get_evaluator_by_slug("default-helpfulness")
    v_a = db.get_evaluator_versions(seeded_a["uuid"])[0]
    v_b = db.get_evaluator_versions(seeded_b["uuid"])[0]
    db.add_evaluator_to_test(test_uuid, seeded_a["uuid"], v_a["uuid"])
    db.add_evaluator_to_test(other_test, seeded_b["uuid"], v_b["uuid"])

    # Simulate a pre-feature DB: pivots exist, but agent_evaluators and its
    # migration flag have not been applied yet.
    with db.get_db_connection() as conn:
        conn.execute("DROP TABLE IF EXISTS agent_evaluators")
        conn.execute(
            "DELETE FROM _schema_migrations WHERE name = ?",
            (db.AGENT_EVALUATORS_BACKFILL_MIGRATION,),
        )
        conn.commit()

    db.init_db()
    linked = {e["uuid"] for e in db.get_evaluators_for_agent(agent_uuid)}
    assert linked == {seeded_a["uuid"], seeded_b["uuid"]}

    # Migration flag is set — a later init_db() does not re-run the backfill.
    assert db.remove_evaluator_from_agent(agent_uuid, seeded_a["uuid"]) is True
    db.init_db()
    assert seeded_a["uuid"] not in {e["uuid"] for e in db.get_evaluators_for_agent(agent_uuid)}
    assert seeded_b["uuid"] in {e["uuid"] for e in db.get_evaluators_for_agent(agent_uuid)}


def test_backfill_agent_evaluator_links_retries_after_crash_mid_migration(user):
    """If deploy creates agent_evaluators but crashes before the backfill
    commits, the next init_db() still runs the migration."""
    agent_uuid = db.create_agent(
        name=_u("a-crash-ev"), user_id=user["uuid"], org_uuid=user["org_uuid"]
    )
    test_uuid = db.create_test(
        name=_u("t-crash-ev"), type="llm", user_id=user["uuid"], org_uuid=user["org_uuid"]
    )
    db.add_test_to_agent(agent_uuid, test_uuid)
    seeded = db.get_evaluator_by_slug("default-safety")
    v = db.get_evaluator_versions(seeded["uuid"])[0]
    db.add_evaluator_to_test(test_uuid, seeded["uuid"], v["uuid"])

    with db.get_db_connection() as conn:
        conn.execute("DELETE FROM agent_evaluators")
        conn.execute(
            "DELETE FROM _schema_migrations WHERE name = ?",
            (db.AGENT_EVALUATORS_BACKFILL_MIGRATION,),
        )
        conn.commit()

    assert db.get_evaluators_for_agent(agent_uuid) == []

    db.init_db()
    linked = {e["uuid"] for e in db.get_evaluators_for_agent(agent_uuid)}
    assert linked == {seeded["uuid"]}


# ---------------------------------------------------------------------------
# Per-org default-evaluator provisioning (fork-on-provision)
# ---------------------------------------------------------------------------

_DEFAULT_SLUGS = {s["slug"] for s in db.DEFAULT_EVALUATORS_SEED}


def _fresh_org():
    """A brand-new user + personal org. Signup auto-forks the seeded defaults."""
    user_uuid = db.create_user("P", "V", _u("prov") + "@x.com")
    return user_uuid, db.get_personal_org_for_user(user_uuid)["uuid"]


def _forks_by_slug(org_uuid):
    return {
        e["source_default_slug"]: e
        for e in db.get_all_evaluators(org_uuid=org_uuid)
        if e.get("source_default_slug")
    }


def test_new_org_is_provisioned_editable_forks_of_every_default():
    user_uuid, org_uuid = _fresh_org()
    forks = _forks_by_slug(org_uuid)
    assert set(forks) == _DEFAULT_SLUGS

    safety = forks["default-safety"]
    assert safety["org_uuid"] == org_uuid  # scoped to the org
    assert safety["owner_user_id"] == user_uuid  # owned ⇒ editable (reads is_default True at the API)
    assert safety["slug"] is None  # the globally-unique slug stays with the template
    assert safety["live_version_id"]  # live version copied

    # The seed templates themselves are never returned to the org.
    assert all(
        e.get("slug") is None for e in db.get_all_evaluators(org_uuid=org_uuid)
    )


def test_provision_default_evaluators_is_idempotent():
    _user, org_uuid = _fresh_org()
    before = len(db.get_all_evaluators(org_uuid=org_uuid))
    assert db.provision_default_evaluators_for_org(org_uuid) == 0
    assert len(db.get_all_evaluators(org_uuid=org_uuid)) == before


def test_provision_never_resurrects_deleted_or_renamed_fork():
    _user, org_uuid = _fresh_org()
    forks = _forks_by_slug(org_uuid)

    assert db.delete_evaluator(forks["default-safety"]["uuid"]) is True
    assert (
        db.update_evaluator(forks["default-conciseness"]["uuid"], name="My Renamed")
        is True
    )

    # Re-provisioning must not re-create the deleted one or duplicate the renamed one.
    assert db.provision_default_evaluators_for_org(org_uuid) == 0
    after = _forks_by_slug(org_uuid)
    assert "default-safety" not in after
    assert after["default-conciseness"]["name"] == "My Renamed"
    assert (
        len(
            [
                e
                for e in db.get_all_evaluators(org_uuid=org_uuid)
                if e.get("source_default_slug") == "default-conciseness"
            ]
        )
        == 1
    )


def test_provision_picks_up_a_newly_added_default():
    _user, org_uuid = _fresh_org()
    new_slug = _u("default-new")
    tmpl = db.create_evaluator(
        name=_u("New Default"),
        evaluator_type="llm",
        output_type="binary",
        slug=new_slug,
    )
    v = db.create_evaluator_version(tmpl, judge_model="m", system_prompt="p")
    db.set_evaluator_live_version(tmpl, v["uuid"])
    try:
        assert db.provision_default_evaluators_for_org(org_uuid) == 1
        assert new_slug in _forks_by_slug(org_uuid)
    finally:
        # Keep the shared session DB clean: this template would otherwise be
        # forked into every org created by a later test.
        with db.get_db_connection() as conn:
            conn.execute(
                "DELETE FROM evaluator_versions WHERE evaluator_id IN "
                "(SELECT uuid FROM evaluators WHERE slug = ? OR source_default_slug = ?)",
                (new_slug, new_slug),
            )
            conn.execute(
                "DELETE FROM evaluators WHERE slug = ? OR source_default_slug = ?",
                (new_slug, new_slug),
            )
            conn.execute(
                "DELETE FROM org_default_evaluators WHERE source_default_slug = ?",
                (new_slug,),
            )
            conn.commit()


def test_provision_skips_default_whose_name_collides_with_a_custom():
    user_uuid, org_uuid = _fresh_org()
    safety = _forks_by_slug(org_uuid)["default-safety"]

    # Simulate a not-yet-provisioned org that already has a custom evaluator
    # named like the default: drop the fork + its receipt, then add the collision.
    with db.get_db_connection() as conn:
        conn.execute("DELETE FROM evaluators WHERE uuid = ?", (safety["uuid"],))
        conn.execute(
            "DELETE FROM org_default_evaluators "
            "WHERE org_uuid = ? AND source_default_slug = ?",
            (org_uuid, "default-safety"),
        )
        conn.commit()
    db.create_evaluator(
        name="Safety",
        evaluator_type="llm",
        output_type="binary",
        owner_user_id=user_uuid,
        org_uuid=org_uuid,
    )

    assert db.provision_default_evaluators_for_org(org_uuid) == 0  # skipped
    safeties = [
        e for e in db.get_all_evaluators(org_uuid=org_uuid) if e["name"] == "Safety"
    ]
    assert len(safeties) == 1  # only the custom, no second copy
    assert safeties[0].get("source_default_slug") is None

    # A receipt is still written (NULL evaluator_uuid) so it isn't retried.
    with db.get_db_connection() as conn:
        row = conn.execute(
            "SELECT evaluator_uuid FROM org_default_evaluators "
            "WHERE org_uuid = ? AND source_default_slug = ?",
            (org_uuid, "default-safety"),
        ).fetchone()
    assert row is not None and row["evaluator_uuid"] is None


def test_backfill_forks_defaults_into_existing_orgs_and_runs_once():
    """One-time fork backfill provisions every existing org; once the migration
    flag is set, a later init_db() does not resurrect a fork the user deleted."""
    _user, org_uuid = _fresh_org()

    # Simulate a pre-feature DB: this org has no forks / receipts, flag unset.
    with db.get_db_connection() as conn:
        conn.execute(
            "DELETE FROM evaluators "
            "WHERE org_uuid = ? AND source_default_slug IS NOT NULL",
            (org_uuid,),
        )
        conn.execute(
            "DELETE FROM org_default_evaluators WHERE org_uuid = ?", (org_uuid,)
        )
        conn.execute(
            "DELETE FROM _schema_migrations WHERE name = ?",
            (db.FORK_DEFAULT_EVALUATORS_MIGRATION,),
        )
        conn.commit()
    assert _forks_by_slug(org_uuid) == {}

    db.init_db()
    assert set(_forks_by_slug(org_uuid)) == _DEFAULT_SLUGS

    # Flag now set + receipt on file ⇒ deleting a fork and re-running init_db
    # does NOT bring it back.
    assert db.delete_evaluator(_forks_by_slug(org_uuid)["default-safety"]["uuid"]) is True
    db.init_db()
    assert "default-safety" not in _forks_by_slug(org_uuid)


def test_create_organization_provisions_default_forks():
    """An explicitly-created workspace (not just the personal org) is forked."""
    owner = db.create_user("O", "W", _u("orgowner") + "@x.com")
    org_uuid = db.create_organization(_u("Workspace"), owner)
    forks = _forks_by_slug(org_uuid)
    assert set(forks) == _DEFAULT_SLUGS
    assert forks["default-safety"]["owner_user_id"] == owner


def _run_repoint():
    with db.get_db_connection() as conn:
        n = db._backfill_repoint_default_links_to_forks(conn.cursor())
        conn.commit()
        return n


def _active_test_link(test_uuid):
    with db.get_db_connection() as conn:
        return conn.execute(
            "SELECT evaluator_id, evaluator_version_id FROM test_evaluators "
            "WHERE test_id = ? AND deleted_at IS NULL",
            (test_uuid,),
        ).fetchall()


def test_repoint_relinks_test_template_link_to_fork_and_repins_version():
    user_uuid, org_uuid = _fresh_org()
    tmpl = db.get_evaluator_by_slug("default-safety")
    test_uuid = db.create_test(
        name=_u("t-repoint"), type="llm", user_id=user_uuid, org_uuid=org_uuid
    )
    db.add_evaluator_to_test(test_uuid, tmpl["uuid"], tmpl["live_version_id"])

    _run_repoint()

    fork = _forks_by_slug(org_uuid)["default-safety"]
    rows = _active_test_link(test_uuid)
    assert [r["evaluator_id"] for r in rows] == [fork["uuid"]]
    # the pinned version follows the fork (the template's version id doesn't
    # exist under the fork)
    assert rows[0]["evaluator_version_id"] == fork["live_version_id"]


def test_repoint_relinks_agent_template_link_to_fork():
    user_uuid, org_uuid = _fresh_org()
    tmpl = db.get_evaluator_by_slug("default-helpfulness")
    agent_uuid = db.create_agent(
        name=_u("a-repoint"), user_id=user_uuid, org_uuid=org_uuid
    )
    db.add_evaluator_to_agent(agent_uuid, tmpl["uuid"])

    _run_repoint()

    fork = _forks_by_slug(org_uuid)["default-helpfulness"]
    with db.get_db_connection() as conn:
        row = conn.execute(
            "SELECT evaluator_id FROM agent_evaluators "
            "WHERE agent_id = ? AND deleted_at IS NULL",
            (agent_uuid,),
        ).fetchone()
    assert row["evaluator_id"] == fork["uuid"]


def test_repoint_soft_deletes_redundant_template_link_on_collision():
    user_uuid, org_uuid = _fresh_org()
    tmpl = db.get_evaluator_by_slug("default-safety")
    fork = _forks_by_slug(org_uuid)["default-safety"]
    test_uuid = db.create_test(
        name=_u("t-collide"), type="llm", user_id=user_uuid, org_uuid=org_uuid
    )
    # The test already links the fork AND (legacy) the template.
    db.add_evaluator_to_test(test_uuid, fork["uuid"], fork["live_version_id"])
    db.add_evaluator_to_test(test_uuid, tmpl["uuid"], tmpl["live_version_id"])

    _run_repoint()

    # Only the fork link survives; the redundant template link is soft-deleted —
    # no UNIQUE(test_id, evaluator_id) collision, no duplicate.
    rows = _active_test_link(test_uuid)
    assert [r["evaluator_id"] for r in rows] == [fork["uuid"]]


def test_repoint_leaves_link_on_template_when_org_deleted_its_fork():
    user_uuid, org_uuid = _fresh_org()
    tmpl = db.get_evaluator_by_slug("default-conciseness")
    fork = _forks_by_slug(org_uuid)["default-conciseness"]
    test_uuid = db.create_test(
        name=_u("t-nofork"), type="llm", user_id=user_uuid, org_uuid=org_uuid
    )
    db.add_evaluator_to_test(test_uuid, tmpl["uuid"], tmpl["live_version_id"])
    assert db.delete_evaluator(fork["uuid"]) is True  # org deleted its fork

    _run_repoint()

    # No fork to move to ⇒ the link stays on the template (still resolvable).
    rows = _active_test_link(test_uuid)
    assert [r["evaluator_id"] for r in rows] == [tmpl["uuid"]]


def test_repoint_default_links_runs_via_init_db_and_is_flag_gated():
    user_uuid, org_uuid = _fresh_org()
    tmpl = db.get_evaluator_by_slug("default-instruction-following")
    agent_uuid = db.create_agent(
        name=_u("a-repoint-initdb"), user_id=user_uuid, org_uuid=org_uuid
    )
    db.add_evaluator_to_agent(agent_uuid, tmpl["uuid"])

    # Simulate a pre-migration DB: clear the flag so init_db re-runs the re-point.
    with db.get_db_connection() as conn:
        conn.execute(
            "DELETE FROM _schema_migrations WHERE name = ?",
            (db.REPOINT_DEFAULT_LINKS_MIGRATION,),
        )
        conn.commit()

    db.init_db()

    fork = _forks_by_slug(org_uuid)["default-instruction-following"]
    with db.get_db_connection() as conn:
        row = conn.execute(
            "SELECT evaluator_id FROM agent_evaluators "
            "WHERE agent_id = ? AND deleted_at IS NULL",
            (agent_uuid,),
        ).fetchone()
    assert row["evaluator_id"] == fork["uuid"]


# ---------------------------------------------------------------------------
# Finished-run snapshot re-point (past runs' evaluator click → org fork)
# ---------------------------------------------------------------------------


def _run_snapshot_repoint():
    with db.get_db_connection() as conn:
        n = db._backfill_repoint_default_job_snapshots(conn.cursor())
        conn.commit()
        return n


def test_repoint_job_snapshots_rewrites_template_uuid_in_json():
    _user, org_uuid = _fresh_org()
    tmpl = db.get_evaluator_by_slug("default-safety")
    fork = _forks_by_slug(org_uuid)["default-safety"]
    job_uuid = _u("job")
    details = f'{{"evaluators": [{{"uuid": "{tmpl["uuid"]}", "name": "Safety"}}]}}'
    results = f'{{"rows": [{{"evaluator_id": "{tmpl["uuid"]}"}}]}}'
    with db.get_db_connection() as conn:
        conn.execute(
            "INSERT INTO jobs (uuid, type, status, org_uuid, details, results) "
            "VALUES (?, 'stt-eval', 'done', ?, ?, ?)",
            (job_uuid, org_uuid, details, results),
        )
        conn.commit()

    _run_snapshot_repoint()

    with db.get_db_connection() as conn:
        row = conn.execute(
            "SELECT details, results FROM jobs WHERE uuid = ?", (job_uuid,)
        ).fetchone()
    # Both the details snapshot AND the results are rewritten to the fork; no
    # trace of the template uuid remains.
    assert fork["uuid"] in row["details"] and tmpl["uuid"] not in row["details"]
    assert fork["uuid"] in row["results"] and tmpl["uuid"] not in row["results"]


def test_repoint_job_snapshots_updates_evaluator_runs_column_and_repins_version():
    _user, org_uuid = _fresh_org()
    tmpl = db.get_evaluator_by_slug("default-helpfulness")
    fork = _forks_by_slug(org_uuid)["default-helpfulness"]
    job_uuid, run_uuid = _u("job"), _u("run")
    with db.get_db_connection() as conn:
        conn.execute(
            "INSERT INTO jobs (uuid, type, status, org_uuid) "
            "VALUES (?, 'annotation-eval', 'done', ?)",
            (job_uuid, org_uuid),
        )
        conn.execute(
            "INSERT INTO evaluator_runs "
            "(uuid, job_id, item_id, evaluator_id, evaluator_version_id) "
            "VALUES (?, ?, ?, ?, ?)",
            (run_uuid, job_uuid, _u("item"), tmpl["uuid"], tmpl["live_version_id"]),
        )
        conn.commit()

    _run_snapshot_repoint()

    with db.get_db_connection() as conn:
        row = conn.execute(
            "SELECT evaluator_id, evaluator_version_id FROM evaluator_runs "
            "WHERE uuid = ?",
            (run_uuid,),
        ).fetchone()
    assert row["evaluator_id"] == fork["uuid"]
    assert row["evaluator_version_id"] == fork["live_version_id"]


def test_repoint_job_snapshots_updates_annotation_job_evaluators_column():
    user_uuid, org_uuid = _fresh_org()
    tmpl = db.get_evaluator_by_slug("default-conciseness")
    fork = _forks_by_slug(org_uuid)["default-conciseness"]
    task_uuid, job_uuid = _u("task"), _u("ajob")
    with db.get_db_connection() as conn:
        conn.execute(
            "INSERT INTO annotation_tasks (uuid, user_id, name, org_uuid) "
            "VALUES (?, ?, 'T', ?)",
            (task_uuid, user_uuid, org_uuid),
        )
        conn.execute(
            "INSERT INTO annotation_jobs (uuid, task_id, annotator_id, public_token) "
            "VALUES (?, ?, ?, ?)",
            (job_uuid, task_uuid, _u("ann"), _u("tok")),
        )
        conn.execute(
            "INSERT INTO annotation_job_evaluators (job_id, evaluator_id, position) "
            "VALUES (?, ?, 0)",
            (job_uuid, tmpl["uuid"]),
        )
        conn.commit()

    _run_snapshot_repoint()

    with db.get_db_connection() as conn:
        row = conn.execute(
            "SELECT evaluator_id FROM annotation_job_evaluators WHERE job_id = ?",
            (job_uuid,),
        ).fetchone()
    assert row["evaluator_id"] == fork["uuid"]


def test_repoint_job_snapshots_leaves_template_when_org_deleted_fork():
    _user, org_uuid = _fresh_org()
    tmpl = db.get_evaluator_by_slug("default-faithfulness")
    fork = _forks_by_slug(org_uuid)["default-faithfulness"]
    assert db.delete_evaluator(fork["uuid"]) is True  # org deleted its fork
    job_uuid = _u("job")
    details = f'{{"evaluators": [{{"uuid": "{tmpl["uuid"]}"}}]}}'
    with db.get_db_connection() as conn:
        conn.execute(
            "INSERT INTO jobs (uuid, type, status, org_uuid, details) "
            "VALUES (?, 'stt-eval', 'done', ?, ?)",
            (job_uuid, org_uuid, details),
        )
        conn.commit()

    _run_snapshot_repoint()

    with db.get_db_connection() as conn:
        row = conn.execute(
            "SELECT details FROM jobs WHERE uuid = ?", (job_uuid,)
        ).fetchone()
    assert tmpl["uuid"] in row["details"]  # no fork to move to → unchanged


def test_repoint_job_snapshots_runs_via_init_db_and_is_flag_gated():
    _user, org_uuid = _fresh_org()
    tmpl = db.get_evaluator_by_slug("default-instruction-following")
    fork = _forks_by_slug(org_uuid)["default-instruction-following"]
    job_uuid = _u("job")
    details = f'{{"evaluators": [{{"uuid": "{tmpl["uuid"]}"}}]}}'
    with db.get_db_connection() as conn:
        conn.execute(
            "INSERT INTO jobs (uuid, type, status, org_uuid, details) "
            "VALUES (?, 'stt-eval', 'done', ?, ?)",
            (job_uuid, org_uuid, details),
        )
        conn.execute(
            "DELETE FROM _schema_migrations WHERE name = ?",
            (db.REPOINT_DEFAULT_JOB_SNAPSHOTS_MIGRATION,),
        )
        conn.commit()

    db.init_db()

    with db.get_db_connection() as conn:
        row = conn.execute(
            "SELECT details FROM jobs WHERE uuid = ?", (job_uuid,)
        ).fetchone()
    assert fork["uuid"] in row["details"]


def test_repoint_job_snapshots_agent_test_jobs_org_via_agent_join():
    """agent_test_jobs resolves its org through agents.org_uuid (JOIN, not a
    direct column) — exercise that path explicitly."""
    user_uuid, org_uuid = _fresh_org()
    tmpl = db.get_evaluator_by_slug("default-safety")
    fork = _forks_by_slug(org_uuid)["default-safety"]
    agent_uuid = db.create_agent(
        name=_u("a-snap"), user_id=user_uuid, org_uuid=org_uuid
    )
    job_uuid = _u("atj")
    details = f'{{"evaluators_by_test_id": {{"t1": [{{"uuid": "{tmpl["uuid"]}"}}]}}}}'
    results = f'{{"judge": {{"evaluator_id": "{tmpl["uuid"]}"}}}}'
    with db.get_db_connection() as conn:
        conn.execute(
            "INSERT INTO agent_test_jobs (uuid, agent_id, type, status, details, results) "
            "VALUES (?, ?, 'llm-unit-test', 'done', ?, ?)",
            (job_uuid, agent_uuid, details, results),
        )
        conn.commit()

    _run_snapshot_repoint()

    with db.get_db_connection() as conn:
        row = conn.execute(
            "SELECT details, results FROM agent_test_jobs WHERE uuid = ?", (job_uuid,)
        ).fetchone()
    assert fork["uuid"] in row["details"] and tmpl["uuid"] not in row["details"]
    assert fork["uuid"] in row["results"] and tmpl["uuid"] not in row["results"]


def test_repoint_job_snapshots_simulation_jobs_org_via_simulation_join():
    """simulation_jobs resolves its org through simulations.org_uuid (JOIN)."""
    _user, org_uuid = _fresh_org()
    tmpl = db.get_evaluator_by_slug("default-sim-goal-completion")
    fork = _forks_by_slug(org_uuid)["default-sim-goal-completion"]
    sim_uuid, job_uuid = _u("sim"), _u("simjob")
    details = f'{{"evaluators": [{{"uuid": "{tmpl["uuid"]}"}}]}}'
    with db.get_db_connection() as conn:
        conn.execute(
            "INSERT INTO simulations (uuid, name, org_uuid) VALUES (?, 'S', ?)",
            (sim_uuid, org_uuid),
        )
        conn.execute(
            "INSERT INTO simulation_jobs (uuid, simulation_id, type, status, details) "
            "VALUES (?, ?, 'text', 'done', ?)",
            (job_uuid, sim_uuid, details),
        )
        conn.commit()

    _run_snapshot_repoint()

    with db.get_db_connection() as conn:
        row = conn.execute(
            "SELECT details FROM simulation_jobs WHERE uuid = ?", (job_uuid,)
        ).fetchone()
    assert fork["uuid"] in row["details"] and tmpl["uuid"] not in row["details"]


def test_repoint_job_snapshots_rewrites_only_templates_not_customs():
    """A snapshot mixing two templates and a custom evaluator: both templates
    move to their forks; the custom uuid is left exactly as-is."""
    user_uuid, org_uuid = _fresh_org()
    t1 = db.get_evaluator_by_slug("default-safety")
    t2 = db.get_evaluator_by_slug("default-helpfulness")
    f1 = _forks_by_slug(org_uuid)["default-safety"]
    f2 = _forks_by_slug(org_uuid)["default-helpfulness"]
    custom = db.create_evaluator(
        name=_u("Custom"),
        evaluator_type="llm",
        output_type="binary",
        owner_user_id=user_uuid,
        org_uuid=org_uuid,
    )
    job_uuid = _u("job")
    details = (
        f'{{"evaluators": [{{"uuid": "{t1["uuid"]}"}}, '
        f'{{"uuid": "{t2["uuid"]}"}}, {{"uuid": "{custom}"}}]}}'
    )
    with db.get_db_connection() as conn:
        conn.execute(
            "INSERT INTO jobs (uuid, type, status, org_uuid, details) "
            "VALUES (?, 'stt-eval', 'done', ?, ?)",
            (job_uuid, org_uuid, details),
        )
        conn.commit()

    _run_snapshot_repoint()

    with db.get_db_connection() as conn:
        d = conn.execute(
            "SELECT details FROM jobs WHERE uuid = ?", (job_uuid,)
        ).fetchone()["details"]
    assert f1["uuid"] in d and t1["uuid"] not in d
    assert f2["uuid"] in d and t2["uuid"] not in d
    assert custom in d  # custom evaluator untouched


def test_repoint_job_snapshots_is_idempotent():
    _user, org_uuid = _fresh_org()
    tmpl = db.get_evaluator_by_slug("default-instruction-following")
    fork = _forks_by_slug(org_uuid)["default-instruction-following"]
    job_uuid = _u("job")
    details = f'{{"evaluators": [{{"uuid": "{tmpl["uuid"]}"}}]}}'
    with db.get_db_connection() as conn:
        conn.execute(
            "INSERT INTO jobs (uuid, type, status, org_uuid, details) "
            "VALUES (?, 'stt-eval', 'done', ?, ?)",
            (job_uuid, org_uuid, details),
        )
        conn.commit()

    _run_snapshot_repoint()  # first pass rewrites it
    with db.get_db_connection() as conn:
        after_first = conn.execute(
            "SELECT details FROM jobs WHERE uuid = ?", (job_uuid,)
        ).fetchone()["details"]
    assert fork["uuid"] in after_first

    second = _run_snapshot_repoint()  # nothing left to rewrite
    with db.get_db_connection() as conn:
        after_second = conn.execute(
            "SELECT details FROM jobs WHERE uuid = ?", (job_uuid,)
        ).fetchone()["details"]
    assert second == 0
    assert after_second == after_first  # byte-identical, no double-rewrite


def test_get_evaluators_for_test_resolves_live_version(user):
    """A test run must always use the evaluator's CURRENT live version, not the
    version pinned on the pivot at link time. Editing the evaluator (new live
    version) after linking changes what get_evaluators_for_test returns, even
    though the pivot's evaluator_version_id still points at the old version."""
    test_uuid = db.create_test(
        name=_u("t-live"), type="llm", user_id=user["uuid"], org_uuid=user["org_uuid"]
    )
    ev_uuid = db.create_evaluator(
        name=_u("ev-live"), owner_user_id=user["uuid"], org_uuid=user["org_uuid"]
    )
    v1 = db.create_evaluator_version(ev_uuid, judge_model="m", system_prompt="PROMPT V1")
    db.set_evaluator_live_version(ev_uuid, v1["uuid"])

    # Link at v1 (set_test_evaluators pins live-at-link-time = v1 on the pivot).
    db.set_test_evaluators(test_uuid, [{"evaluator_id": ev_uuid}])
    linked = db.get_evaluators_for_test(test_uuid)
    assert len(linked) == 1
    assert linked[0]["evaluator_version_id"] == v1["uuid"]  # live == v1 here
    assert linked[0]["system_prompt"] == "PROMPT V1"
    assert linked[0]["version_number"] == 1

    # Edit the evaluator: new version becomes live. Do NOT re-link.
    v2 = db.create_evaluator_version(ev_uuid, judge_model="m", system_prompt="PROMPT V2")
    db.set_evaluator_live_version(ev_uuid, v2["uuid"])

    relinked = db.get_evaluators_for_test(test_uuid)
    # All three describe the SAME (live) version v2 — the row is internally
    # consistent even though the pivot still pins v1 in the DB.
    assert relinked[0]["evaluator_version_id"] == v2["uuid"]
    assert relinked[0]["system_prompt"] == "PROMPT V2"
    assert relinked[0]["version_number"] == 2


def test_get_evaluators_for_simulation_resolves_live_version(user):
    """A simulation run must always use the evaluator's CURRENT live version, not
    the version pinned on the simulation_evaluators pivot at link time. Editing the
    evaluator (new live version) after linking changes what
    get_evaluators_for_simulation returns, even though the pivot's
    evaluator_version_id still points at the old version. Mirrors
    test_get_evaluators_for_test_resolves_live_version."""
    sim_uuid = db.create_simulation(
        name=_u("sim-live"), user_id=user["uuid"], org_uuid=user["org_uuid"]
    )
    ev_uuid = db.create_evaluator(
        name=_u("ev-sim-live"), owner_user_id=user["uuid"], org_uuid=user["org_uuid"]
    )
    v1 = db.create_evaluator_version(ev_uuid, judge_model="m", system_prompt="PROMPT V1")
    db.set_evaluator_live_version(ev_uuid, v1["uuid"])

    # Link pinning v1 on the pivot.
    db.add_evaluator_to_simulation(sim_uuid, ev_uuid, v1["uuid"])
    linked = db.get_evaluators_for_simulation(sim_uuid)
    linked = [e for e in linked if e["uuid"] == ev_uuid]
    assert len(linked) == 1
    assert linked[0]["evaluator_version_id"] == v1["uuid"]  # live == v1 here
    assert linked[0]["system_prompt"] == "PROMPT V1"
    assert linked[0]["version_number"] == 1

    # Edit the evaluator: new version becomes live. Do NOT re-link.
    v2 = db.create_evaluator_version(ev_uuid, judge_model="m", system_prompt="PROMPT V2")
    db.set_evaluator_live_version(ev_uuid, v2["uuid"])

    relinked = db.get_evaluators_for_simulation(sim_uuid)
    relinked = [e for e in relinked if e["uuid"] == ev_uuid]
    # Now reports v2 even though the pivot still pins v1 in the DB.
    assert relinked[0]["evaluator_version_id"] == v2["uuid"]
    assert relinked[0]["system_prompt"] == "PROMPT V2"
    assert relinked[0]["version_number"] == 2


def test_evaluator_variables_are_immutable_across_versions(user):
    """Variable names are frozen after the first version. A new version must
    declare the same variable name set; renaming/adding/removing one is rejected.
    This guards the live-resolution path: a test's pinned variable_values must
    always still fill the live prompt's {{placeholders}}."""
    ev_uuid = db.create_evaluator(
        name=_u("ev-immut"), owner_user_id=user["uuid"], org_uuid=user["org_uuid"]
    )
    db.create_evaluator_version(
        ev_uuid,
        judge_model="m",
        system_prompt="Check {{criteria}}",
        variables=[{"name": "criteria"}],
    )

    # Same name set → allowed (description/default may change).
    v2 = db.create_evaluator_version(
        ev_uuid,
        judge_model="m",
        system_prompt="Strictly check {{criteria}}",
        variables=[{"name": "criteria", "description": "now with a description"}],
    )
    assert v2["version_number"] == 2

    # Renamed variable → rejected.
    with pytest.raises(ValueError, match="immutable"):
        db.create_evaluator_version(
            ev_uuid,
            judge_model="m",
            system_prompt="Check {{requirement}}",
            variables=[{"name": "requirement"}],
        )

    # Added variable → rejected.
    with pytest.raises(ValueError, match="immutable"):
        db.create_evaluator_version(
            ev_uuid,
            judge_model="m",
            system_prompt="Check {{criteria}} and {{extra}}",
            variables=[{"name": "criteria"}, {"name": "extra"}],
        )

    # Removed all variables → rejected.
    with pytest.raises(ValueError, match="immutable"):
        db.create_evaluator_version(
            ev_uuid, judge_model="m", system_prompt="Check it", variables=None
        )


def test_add_test_to_agent_restore_refreshes_created_at(user):
    """Re-adding a soft-deleted link reuses the row but refreshes created_at
    to now, so the re-added test sorts as recently-added rather than inheriting
    its original first-add time."""
    agent_uuid = db.create_agent(name=_u("a-restore"), user_id=user["uuid"], org_uuid=user["org_uuid"])
    test_uuid = db.create_test(name=_u("t-restore"), type="llm", user_id=user["uuid"], org_uuid=user["org_uuid"])

    link_id = db.add_test_to_agent(agent_uuid, test_uuid)

    # Backdate created_at so we can detect whether re-add refreshes it.
    with db.get_db_connection() as conn:
        conn.execute(
            "UPDATE agent_tests SET created_at = '2000-01-01 00:00:00' WHERE id = ?",
            (link_id,),
        )
        conn.commit()

    assert db.remove_test_from_agent(agent_uuid, test_uuid) is True

    # Re-add restores the same row...
    assert db.add_test_to_agent(agent_uuid, test_uuid) == link_id
    # ...and created_at is no longer the backdated value.
    restored = db.get_agent_test_link(agent_uuid, test_uuid)
    assert restored is not None
    assert restored["created_at"] != "2000-01-01 00:00:00"


# ---------------------------------------------------------------------------
# Generic jobs + queue accounting
# ---------------------------------------------------------------------------


def test_generic_jobs_and_queue(user):
    j_uuid = db.create_job(
        job_type="stt-eval",
        user_id=user["uuid"], org_uuid=user["org_uuid"],
        status="in_progress",
        details={"provider": "openai"},
    )
    assert db.get_job(j_uuid)["details"] == {"provider": "openai"}
    assert db.get_job(j_uuid, org_uuid=user["org_uuid"]) is not None
    assert db.get_job("does-not-exist") is None

    queued_uuid = db.create_job(
        job_type="stt-eval", user_id=user["uuid"], org_uuid=user["org_uuid"], status="queued"
    )
    assert any(j["uuid"] == queued_uuid for j in db.get_queued_jobs(["stt-eval"]))
    assert db.get_queued_jobs() is not None
    assert any(j["uuid"] == j_uuid for j in db.get_pending_jobs())

    assert db.count_running_jobs(["stt-eval"]) >= 1
    assert db.count_running_jobs() >= 1
    assert db.count_running_jobs_for_org(user["org_uuid"], ["stt-eval"]) >= 1
    assert db.count_running_jobs_for_org(user["org_uuid"]) >= 1

    # update — status + results
    assert db.update_job(j_uuid, status="done", results={"r": 1}) is True
    # update — details merges
    assert db.update_job(j_uuid, details={"new_key": 2}) is True
    fetched = db.get_job(j_uuid)
    assert fetched["details"]["provider"] == "openai"
    assert fetched["details"]["new_key"] == 2
    # update — details replace
    assert db.update_job(
        j_uuid, details={"provider": "deepgram"}, replace_details=True
    ) is True
    replaced = db.get_job(j_uuid)
    assert replaced["details"] == {"provider": "deepgram"}
    # merge when stored details are NULL
    null_details_uuid = db.create_job(
        job_type="stt-eval",
        user_id=user["uuid"],
        org_uuid=user["org_uuid"],
        status="queued",
        details=None,
    )
    assert db.update_job(null_details_uuid, details={"provider": "openai"}) is True
    assert db.get_job(null_details_uuid)["details"] == {"provider": "openai"}
    assert db.update_job(j_uuid) is False

    # visibility / share token
    assert db.update_job_visibility(j_uuid, True, "tok-123") is True
    assert db.get_job_by_share_token("tok-123") is not None
    assert db.get_job_by_share_token("tok-123", job_type="stt-eval") is not None
    assert db.get_job_by_share_token("missing-tok") is None
    assert db.update_job_visibility(j_uuid, False, None) is True
    assert db.get_job_by_share_token("tok-123") is None

    # all jobs / filters
    assert any(j["uuid"] == j_uuid for j in db.get_all_jobs(org_uuid=user["org_uuid"]))
    assert any(
        j["uuid"] == j_uuid for j in db.get_all_jobs(org_uuid=user["org_uuid"], job_type="stt-eval")
    )

    # generic-jobs-for-task lookup (annotation-eval shape)
    task_id = str(_uuid.uuid4())
    anno_uuid = db.create_job(
        job_type="annotation-eval",
        user_id=user["uuid"], org_uuid=user["org_uuid"],
        details={"task_id": task_id},
    )
    assert any(j["uuid"] == anno_uuid for j in db.get_generic_jobs_for_task(task_id, "annotation-eval"))

    # soft-delete + hard-delete
    assert db.soft_delete_job(j_uuid) is True
    assert db.soft_delete_job(j_uuid) is False
    assert db.delete_job(queued_uuid) is True
    assert db.delete_job(queued_uuid) is False


# ---------------------------------------------------------------------------
# Jobs-summary denormalized header columns + backfill
# ---------------------------------------------------------------------------


def _summary_item(org_uuid, job_uuid, job_type=None):
    for item in db.get_all_jobs_summary(org_uuid, job_type=job_type):
        if item["uuid"] == job_uuid:
            return item
    return None


def test_create_job_populates_summary_columns(user):
    job_uuid = db.create_job(
        job_type="stt-eval",
        user_id=user["uuid"],
        org_uuid=user["org_uuid"],
        details={
            "providers": ["deepgram", "openai"],
            "language": "hindi",
            "texts": ["a", "b", "c"],
            "evaluators": [{"heavy": "x" * 500}],
        },
    )

    item = _summary_item(user["org_uuid"], job_uuid)
    assert item is not None
    assert item["providers"] == ["deepgram", "openai"]
    assert item["language"] == "hindi"
    assert item["sample_count"] == 3

    # Prove it's a real column, not derived from the details blob at read time.
    with db.get_db_connection() as conn:
        row = conn.execute(
            "SELECT providers, sample_count FROM jobs WHERE uuid = ?", (job_uuid,)
        ).fetchone()
    import json as _json

    assert _json.loads(row["providers"]) == ["deepgram", "openai"]
    assert row["sample_count"] == 3


def test_jobs_summary_uses_live_dataset_name_via_join(user):
    ds_uuid = db.create_dataset(_u("live-ds"), "stt", user["org_uuid"], user["uuid"])
    job_uuid = db.create_job(
        job_type="stt-eval",
        user_id=user["uuid"],
        org_uuid=user["org_uuid"],
        details={
            "dataset_id": ds_uuid,
            "dataset_name": "STALE-FROZEN-NAME",
            "providers": ["openai"],
            "texts": ["a"],
        },
    )

    item = _summary_item(user["org_uuid"], job_uuid)
    assert item["dataset_id"] == ds_uuid
    assert item["dataset_name"] != "STALE-FROZEN-NAME"

    new_name = _u("renamed-ds")
    assert db.update_dataset_name(ds_uuid, user["org_uuid"], new_name) is True
    item = _summary_item(user["org_uuid"], job_uuid)
    assert item["dataset_name"] == new_name


def test_jobs_summary_nulls_dataset_when_soft_deleted(user):
    ds_uuid = db.create_dataset(_u("del-ds"), "stt", user["org_uuid"], user["uuid"])
    job_uuid = db.create_job(
        job_type="stt-eval",
        user_id=user["uuid"],
        org_uuid=user["org_uuid"],
        details={"dataset_id": ds_uuid, "providers": ["openai"], "texts": ["a"]},
    )

    item = _summary_item(user["org_uuid"], job_uuid)
    assert item["dataset_id"] == ds_uuid
    assert item["dataset_name"] is not None

    assert db.delete_dataset(ds_uuid, user["org_uuid"]) is True
    item = _summary_item(user["org_uuid"], job_uuid)
    assert item["dataset_id"] is None
    assert item["dataset_name"] is None


def test_jobs_summary_inline_run_has_no_dataset(user):
    job_uuid = db.create_job(
        job_type="stt-eval",
        user_id=user["uuid"],
        org_uuid=user["org_uuid"],
        details={"providers": ["openai", "deepgram"], "texts": ["a", "b"]},
    )

    item = _summary_item(user["org_uuid"], job_uuid)
    assert item["dataset_id"] is None
    assert item["dataset_name"] is None
    assert item["sample_count"] == 2


def test_jobs_summary_active_dataset_with_blank_name_still_shows(user):
    """An active dataset with an empty name must still surface — nulling is
    keyed on the joined row existing, not on the name being truthy."""
    ds_uuid = db.create_dataset(
        name=_u("blank-name-ds"), dataset_type="stt", org_uuid=user["org_uuid"]
    )
    with db.get_db_connection() as conn:
        conn.execute("UPDATE datasets SET name = '' WHERE uuid = ?", (ds_uuid,))
        conn.commit()

    job_uuid = db.create_job(
        job_type="stt-eval",
        user_id=user["uuid"],
        org_uuid=user["org_uuid"],
        details={"providers": ["openai"], "texts": ["a"], "dataset_id": ds_uuid},
    )

    item = _summary_item(user["org_uuid"], job_uuid)
    assert item["dataset_id"] == ds_uuid
    assert item["dataset_name"] == ""


def test_backfill_jobs_summary_columns_fills_legacy_rows(user):
    job_uuid = db.create_job(
        job_type="stt-eval",
        user_id=user["uuid"],
        org_uuid=user["org_uuid"],
        details={
            "providers": ["deepgram"],
            "language": "tamil",
            "texts": ["a", "b", "c", "d"],
        },
    )

    # Simulate a legacy row that predates the columns and the migration.
    with db.get_db_connection() as conn:
        conn.execute(
            "UPDATE jobs SET dataset_id=NULL, language=NULL, providers=NULL, "
            "sample_count=NULL WHERE uuid=?",
            (job_uuid,),
        )
        conn.execute(
            "DELETE FROM _schema_migrations WHERE name = ?",
            (db.JOBS_SUMMARY_BACKFILL_MIGRATION,),
        )
        conn.commit()

    db.init_db()

    item = _summary_item(user["org_uuid"], job_uuid)
    assert item["providers"] == ["deepgram"]
    assert item["language"] == "tamil"
    assert item["sample_count"] == 4


def test_backfill_jobs_summary_runs_once_and_never_again(user):
    job_uuid = db.create_job(
        job_type="stt-eval",
        user_id=user["uuid"],
        org_uuid=user["org_uuid"],
        details={"providers": ["openai"], "language": "hindi", "texts": ["a"]},
    )

    # The backfill flag is already set (created via create_job / prior init_db).
    # A user hand-edits a column afterwards; the migration must not re-run and
    # clobber it.
    with db.get_db_connection() as conn:
        flag = conn.execute(
            "SELECT 1 FROM _schema_migrations WHERE name = ?",
            (db.JOBS_SUMMARY_BACKFILL_MIGRATION,),
        ).fetchone()
        assert flag is not None
        conn.execute(
            "UPDATE jobs SET language = 'MANUALLY_EDITED' WHERE uuid = ?",
            (job_uuid,),
        )
        conn.commit()

    db.init_db()

    with db.get_db_connection() as conn:
        language = conn.execute(
            "SELECT language FROM jobs WHERE uuid = ?", (job_uuid,)
        ).fetchone()["language"]
    assert language == "MANUALLY_EDITED"


# ---------------------------------------------------------------------------
# Agent Test Jobs
# ---------------------------------------------------------------------------


def test_agent_test_jobs(user):
    agent_uuid = db.create_agent(name=_u("a-atj"), user_id=user["uuid"], org_uuid=user["org_uuid"])
    j_uuid = db.create_agent_test_job(
        agent_id=agent_uuid,
        job_type="llm-unit-test",
        details={"x": 1},
    )
    assert db.get_agent_test_job(j_uuid)["details"] == {"x": 1}
    assert any(
        j["uuid"] == j_uuid for j in db.get_agent_test_jobs_for_agent_summary(agent_uuid)
    )
    assert any(
        j["uuid"] == j_uuid
        for j in db.get_agent_test_jobs_for_agent_summary(agent_uuid, job_type="llm-unit-test")
    )
    assert any(j["uuid"] == j_uuid for j in db.get_all_agent_test_jobs())
    assert any(
        j["uuid"] == j_uuid for j in db.get_all_agent_test_jobs(job_type="llm-unit-test")
    )
    assert any(j["uuid"] == j_uuid for j in db.get_agent_test_jobs_for_org_summary(user["org_uuid"]))
    assert any(
        j["uuid"] == j_uuid
        for j in db.get_agent_test_jobs_for_org_summary(user["org_uuid"], job_type="llm-unit-test")
    )

    queued_uuid = db.create_agent_test_job(
        agent_id=agent_uuid, job_type="llm-benchmark", status="queued"
    )
    assert any(j["uuid"] == queued_uuid for j in db.get_queued_agent_test_jobs())
    assert any(
        j["uuid"] == queued_uuid
        for j in db.get_queued_agent_test_jobs(["llm-benchmark"])
    )
    assert any(j["uuid"] == j_uuid for j in db.get_pending_agent_test_jobs())

    assert db.count_running_agent_test_jobs(["llm-unit-test"]) >= 1
    assert db.count_running_agent_test_jobs() >= 1
    assert db.count_running_agent_test_jobs_for_org(user["org_uuid"], ["llm-unit-test"]) >= 1
    assert db.count_running_agent_test_jobs_for_org(user["org_uuid"]) >= 1

    assert db.update_agent_test_job(j_uuid, status="done", results={"k": 1}) is True
    assert db.update_agent_test_job(j_uuid) is False
    assert db.update_agent_test_job_visibility(j_uuid, True, "agtok") is True
    assert db.get_agent_test_job_by_share_token("agtok") is not None
    assert db.get_agent_test_job_by_share_token("agtok", "llm-unit-test") is not None
    assert db.get_agent_test_job_by_share_token("missing") is None

    assert db.delete_agent_test_job(j_uuid) is True
    assert db.delete_agent_test_job(j_uuid) is False


# ---------------------------------------------------------------------------
# Simulation Jobs
# ---------------------------------------------------------------------------


def test_simulation_jobs(user):
    sim_uuid = db.create_simulation(name=_u("sim-jobs"), user_id=user["uuid"], org_uuid=user["org_uuid"])
    j_uuid = db.create_simulation_job(
        simulation_id=sim_uuid,
        job_type="text",
        details={"task_id": "t"},
    )
    assert db.get_simulation_job(j_uuid)["details"] == {"task_id": "t"}
    assert any(
        j["uuid"] == j_uuid for j in db.get_simulation_jobs_for_simulation(sim_uuid)
    )
    assert any(
        j["uuid"] == j_uuid
        for j in db.get_simulation_jobs_for_simulation(sim_uuid, job_type="text")
    )
    assert any(j["uuid"] == j_uuid for j in db.get_all_simulation_jobs())
    assert any(j["uuid"] == j_uuid for j in db.get_all_simulation_jobs(job_type="text"))
    assert any(j["uuid"] == j_uuid for j in db.get_pending_simulation_jobs())

    queued_uuid = db.create_simulation_job(
        simulation_id=sim_uuid, job_type="voice", status="queued"
    )
    assert any(j["uuid"] == queued_uuid for j in db.get_queued_simulation_jobs())
    assert any(
        j["uuid"] == queued_uuid for j in db.get_queued_simulation_jobs(["voice"])
    )

    assert db.count_running_simulation_jobs(["text"]) >= 1
    assert db.count_running_simulation_jobs() >= 1
    assert db.count_running_simulation_jobs_for_org(user["org_uuid"], ["text"]) >= 1
    assert db.count_running_simulation_jobs_for_org(user["org_uuid"]) >= 1

    assert db.update_simulation_job(j_uuid, status="done", results={"ok": True}) is True
    assert db.update_simulation_job(j_uuid, details={"more": 1}) is True
    fetched = db.get_simulation_job(j_uuid)
    assert fetched["details"]["task_id"] == "t"
    assert fetched["details"]["more"] == 1
    assert db.update_simulation_job(j_uuid) is False

    assert db.update_simulation_job_visibility(j_uuid, True, "stok") is True
    assert db.get_simulation_job_by_share_token("stok") is not None
    assert db.get_simulation_job_by_share_token("missing") is None

    assert db.delete_simulation_job(j_uuid) is True
    assert db.delete_simulation_job(j_uuid) is False


# ---------------------------------------------------------------------------
# Datasets + items
# ---------------------------------------------------------------------------


def test_datasets_and_items(user):
    ds_uuid = db.create_dataset(name=_u("ds"), dataset_type="stt", user_id=user["uuid"], org_uuid=user["org_uuid"])
    with pytest.raises(ValueError):
        db.create_dataset(name=_u("bad-type"), dataset_type="bogus", user_id=user["uuid"], org_uuid=user["org_uuid"])

    assert db.get_dataset(ds_uuid, user["org_uuid"])["type"] == "stt"
    assert db.get_dataset(ds_uuid, "other-org") is None
    assert any(d["uuid"] == ds_uuid for d in db.get_all_datasets(user["org_uuid"]))
    assert any(
        d["uuid"] == ds_uuid for d in db.get_all_datasets(user["org_uuid"], dataset_type="stt")
    )

    item_ids = db.add_dataset_items(
        ds_uuid,
        [
            {"text": "hello", "audio_path": "s3://b/k1"},
            {"text": "world", "audio_path": "s3://b/k2"},
        ],
    )
    assert len(item_ids) == 2
    assert db.add_dataset_items(ds_uuid, []) == []
    items = db.get_dataset_items(ds_uuid)
    assert [i["text"] for i in items] == ["hello", "world"]
    assert db.get_dataset_item(item_ids[0], ds_uuid)["text"] == "hello"
    by_ids = db.get_dataset_items_by_uuids(item_ids)
    assert {i["uuid"] for i in by_ids} == set(item_ids)
    assert db.get_dataset_items_by_uuids([]) == []

    counts = db.get_dataset_item_counts([ds_uuid])
    assert counts[ds_uuid] == 2
    assert db.get_dataset_item_counts([]) == {}
    assert db.get_dataset_eval_counts([ds_uuid])[ds_uuid] == 0
    assert db.get_dataset_eval_counts([]) == {}
    assert ds_uuid in db.get_active_dataset_ids([ds_uuid])
    assert db.get_active_dataset_ids([]) == set()

    assert db.update_dataset_name(ds_uuid, user["org_uuid"], _u("renamed-ds")) is True
    assert db.update_dataset_name(ds_uuid, "wrong-org", "x") is False

    assert db.update_dataset_item(item_ids[0], ds_uuid, text="updated") is True
    assert db.update_dataset_item(item_ids[0], ds_uuid, audio_path=None) is True
    assert db.update_dataset_item(item_ids[0], ds_uuid) is False

    assert db.delete_dataset_item(item_ids[0], ds_uuid) is True
    assert db.delete_dataset_item(item_ids[0], ds_uuid) is False

    assert db.delete_dataset(ds_uuid, user["org_uuid"]) is True
    assert db.delete_dataset(ds_uuid, user["org_uuid"]) is False


# ---------------------------------------------------------------------------
# User Limits
# ---------------------------------------------------------------------------


def test_org_limits(user):
    from routers.org_limits import OrgLimits

    limits = OrgLimits(max_rows_per_eval=42)
    row_uuid = db.create_org_limits(user["org_uuid"], limits)
    assert isinstance(row_uuid, str)

    fetched = db.get_org_limits(user["org_uuid"])
    assert fetched["limits"]["max_rows_per_eval"] == 42

    new_limits = OrgLimits(max_rows_per_eval=99)
    updated = db.update_org_limits(user["org_uuid"], new_limits)
    assert updated["limits"]["max_rows_per_eval"] == 99
    assert db.update_org_limits("nope", new_limits) is None

    assert db.delete_org_limits(user["org_uuid"]) is True
    assert db.delete_org_limits(user["org_uuid"]) is False
    assert db.get_org_limits(user["org_uuid"]) is None


# ---------------------------------------------------------------------------
# Annotation Tasks + Items + Evaluators + Annotator + Jobs
# ---------------------------------------------------------------------------


def test_annotation_pipeline(user):
    # task
    task_uuid = db.create_annotation_task(
        name=_u("anno-task"),
        user_id=user["uuid"], org_uuid=user["org_uuid"],
        type="llm",
        description="d",
    )
    assert db.get_annotation_task(task_uuid)["item_count"] == 0
    assert any(t["uuid"] == task_uuid for t in db.get_all_annotation_tasks(user["org_uuid"]))
    by = db.get_annotation_tasks_by_uuids([task_uuid])
    assert task_uuid in by
    assert db.get_annotation_tasks_by_uuids([]) == {}
    assert db.get_annotation_tasks_by_uuids([None]) == {}  # type: ignore[arg-type]

    with pytest.raises(ValueError):
        db.create_annotation_task(name="x", org_uuid=None, type="llm")
    with pytest.raises(ValueError):
        db.create_annotation_task(name="x", user_id=user["uuid"], org_uuid=user["org_uuid"], type="bogus")

    assert db.update_annotation_task(task_uuid, name=_u("t2"), description="d2")
    assert db.update_annotation_task(task_uuid) is False

    # items
    item_ids = db.create_annotation_items(
        task_uuid,
        [{"payload": {"text": "row1"}}, {"payload": {"text": "row2"}}],
    )
    assert len(item_ids) == 2
    assert db.create_annotation_items(task_uuid, []) == []
    fresh = db.get_annotation_item(item_ids[0])
    assert fresh["payload"] == {"text": "row1"}
    # New rows get updated_at populated (equal to created_at on insert).
    assert fresh.get("updated_at") is not None
    assert fresh["updated_at"] == fresh["created_at"]
    assert len(db.get_annotation_items_for_task(task_uuid)) == 2

    pre_edit = db.get_annotation_item(item_ids[0])
    time.sleep(1.1)  # SQLite CURRENT_TIMESTAMP has 1s resolution
    updated = db.bulk_update_annotation_items(
        task_uuid,
        [{"uuid": item_ids[0], "payload": {"text": "edited"}}],
    )
    assert updated == 1
    post_edit = db.get_annotation_item(item_ids[0])
    # Edit bumps updated_at past the row's created_at and previous updated_at.
    assert post_edit["updated_at"] > pre_edit["updated_at"]
    assert post_edit["updated_at"] > post_edit["created_at"]
    assert db.bulk_update_annotation_items(task_uuid, []) == 0

    with pytest.raises(ValueError):
        db.create_annotation_items(task_uuid, [{"payload": None}])
    with pytest.raises(ValueError):
        db.bulk_update_annotation_items(task_uuid, [{"payload": {"x": 1}}])
    with pytest.raises(ValueError):
        db.bulk_update_annotation_items(task_uuid, [{"uuid": item_ids[0]}])

    # evaluator link to task
    seeded = db.get_evaluator_by_slug("default-safety")
    eid = db.add_evaluator_to_annotation_task(task_uuid, seeded["uuid"])
    assert isinstance(eid, int)
    assert any(
        e["uuid"] == seeded["uuid"] for e in db.get_evaluators_for_annotation_task(task_uuid)
    )
    assert db.remove_evaluator_from_annotation_task(task_uuid, seeded["uuid"]) is True
    # restore
    eid2 = db.add_evaluator_to_annotation_task(task_uuid, seeded["uuid"])
    assert eid2 == eid

    # Re-linking refreshes created_at so the evaluator moves to the end of
    # the task's evaluator ordering (which feeds the labelling-form column
    # order). Link a second evaluator AFTER the restore, then unlink and
    # re-link the first — it should now sort last.
    other = db.get_evaluator_by_slug("default-helpfulness")
    db.add_evaluator_to_annotation_task(task_uuid, other["uuid"])
    ordered_before = [
        e["uuid"] for e in db.get_evaluators_for_annotation_task(task_uuid)
    ]
    assert ordered_before == [seeded["uuid"], other["uuid"]]
    assert db.remove_evaluator_from_annotation_task(task_uuid, seeded["uuid"]) is True
    # Ensure a measurable timestamp delta — SQLite CURRENT_TIMESTAMP has
    # 1-second resolution.
    time.sleep(1.1)
    db.add_evaluator_to_annotation_task(task_uuid, seeded["uuid"])
    ordered_after = [
        e["uuid"] for e in db.get_evaluators_for_annotation_task(task_uuid)
    ]
    assert ordered_after == [other["uuid"], seeded["uuid"]]

    # Explicit reorder: caller-supplied order wins over insertion order, and
    # the change is visible to every read path that lists task evaluators.
    db.reorder_evaluators_for_annotation_task(
        task_uuid, [seeded["uuid"], other["uuid"]]
    )
    ordered_explicit = [
        e["uuid"] for e in db.get_evaluators_for_annotation_task(task_uuid)
    ]
    assert ordered_explicit == [seeded["uuid"], other["uuid"]]
    positions = [
        e["position"] for e in db.get_evaluators_for_annotation_task(task_uuid)
    ]
    assert positions == [1, 2]

    # Reorder validates membership — extra/missing IDs raise.
    with pytest.raises(ValueError):
        db.reorder_evaluators_for_annotation_task(task_uuid, [seeded["uuid"]])
    with pytest.raises(ValueError):
        db.reorder_evaluators_for_annotation_task(
            task_uuid, [seeded["uuid"], other["uuid"], other["uuid"]]
        )
    bogus = "00000000-0000-0000-0000-000000000000"
    with pytest.raises(ValueError):
        db.reorder_evaluators_for_annotation_task(
            task_uuid, [seeded["uuid"], bogus]
        )

    # Job snapshot freezes the order at creation time. Snapshot now while
    # order is [seeded, other], then flip the task order — the job keeps
    # its frozen view.
    snapshot_items = db.create_annotation_items(
        task_uuid, [{"payload": {"name": _u("snap-item")}}]
    )
    snapshot_ann = db.create_annotator(
        name=_u("snap-ann"), user_id=user["uuid"], org_uuid=user["org_uuid"]
    )
    snap_job = db.create_annotation_job(
        task_id=task_uuid,
        annotator_id=snapshot_ann,
        item_uuids=snapshot_items,
        public_token=_u("snap-token"),
    )
    job_order_before = db.get_evaluator_ids_for_job(snap_job)
    assert job_order_before == [seeded["uuid"], other["uuid"]]
    db.reorder_evaluators_for_annotation_task(
        task_uuid, [other["uuid"], seeded["uuid"]]
    )
    assert db.get_evaluator_ids_for_job(snap_job) == [seeded["uuid"], other["uuid"]]
    # Task-level read reflects the new order.
    assert [
        e["uuid"] for e in db.get_evaluators_for_annotation_task(task_uuid)
    ] == [other["uuid"], seeded["uuid"]]
    # Restore for the rest of the test.
    db.reorder_evaluators_for_annotation_task(
        task_uuid, [seeded["uuid"], other["uuid"]]
    )

    # Reorder must IGNORE pivot rows whose evaluator has been soft-deleted on
    # the `evaluators` table. `delete_evaluator` doesn't cascade to the pivot
    # (the link's `deleted_at` stays NULL), but `get_evaluators_for_annotation_task`
    # JOINs and filters `e.deleted_at IS NULL` — so clients can't see those
    # UUIDs. If the validator counted them, every task with a deleted-but-still-
    # linked evaluator would 400 forever. Custom evaluator (org-owned) so
    # delete_evaluator is allowed; seeded defaults are protected.
    custom_ev = db.create_evaluator(
        name=_u("ev-soft-delete"),
        owner_user_id=user["uuid"],
        org_uuid=user["org_uuid"],
    )
    db.add_evaluator_to_annotation_task(task_uuid, custom_ev)
    visible_before = [
        e["uuid"] for e in db.get_evaluators_for_annotation_task(task_uuid)
    ]
    assert custom_ev in visible_before
    assert db.delete_evaluator(custom_ev) is True
    visible_after = [
        e["uuid"] for e in db.get_evaluators_for_annotation_task(task_uuid)
    ]
    # FE no longer sees the deleted evaluator on the task.
    assert custom_ev not in visible_after
    # Reorder using only the FE-visible ids must succeed (validator must
    # exclude the deleted evaluator from its "current" set).
    db.reorder_evaluators_for_annotation_task(task_uuid, visible_after)

    # annotator
    ann_uuid = db.create_annotator(name=_u("ann"), user_id=user["uuid"], org_uuid=user["org_uuid"])
    assert db.get_annotator(ann_uuid)
    assert any(a["uuid"] == ann_uuid for a in db.get_all_annotators(user["org_uuid"]))
    ann_map = db.get_annotators_by_uuids([ann_uuid])
    assert ann_uuid in ann_map
    assert db.get_annotators_by_uuids([]) == {}
    assert db.get_annotators_by_uuids([None]) == {}  # type: ignore[arg-type]
    assert db.update_annotator(ann_uuid, name=_u("ann2")) is True
    assert db.update_annotator(ann_uuid) is False
    with pytest.raises(ValueError):
        db.update_annotator(ann_uuid, name="   ")
    with pytest.raises(ValueError):
        db.create_annotator(name="", user_id=user["uuid"], org_uuid=user["org_uuid"])
    with pytest.raises(ValueError):
        db.create_annotator(name="x", org_uuid=None)

    # soft-delete + restore (create_annotator restores on name match)
    assert db.delete_annotator(ann_uuid) is True
    assert db.delete_annotator(ann_uuid) is False
    same_name = db.get_annotator(ann_uuid) or {"name": _u("ann-recreate")}
    restored = db.create_annotator(name=same_name.get("name", _u("ann-recreate")), user_id=user["uuid"], org_uuid=user["org_uuid"])
    assert restored

    # annotation job
    job_uuid = db.create_annotation_job(
        task_id=task_uuid,
        annotator_id=restored,
        item_uuids=item_ids,
        public_token="pub-tok",
    )
    assert db.get_annotation_job(job_uuid)
    assert any(j["uuid"] == job_uuid for j in db.get_jobs_for_task(task_uuid))
    assert any(j["uuid"] == job_uuid for j in db.get_jobs_for_task_detailed(task_uuid))
    assert any(j["uuid"] == job_uuid for j in db.get_jobs_for_annotator(restored))
    assert any(
        j["uuid"] == job_uuid for j in db.get_jobs_for_annotator_detailed(restored)
    )
    counts = db.get_job_counts_for_org_annotators(user["org_uuid"])
    assert counts.get(restored, 0) >= 1
    assert db.get_annotation_job_by_token("pub-tok")
    assert db.get_annotation_job_by_token(None) is None
    assert db.get_annotation_job_by_token("import:abc") is None
    assert db.get_evaluator_ids_for_job(job_uuid)
    assert db.get_evaluators_for_job(job_uuid)
    job_items = db.get_job_items(job_uuid)
    assert len(job_items) == 2

    with pytest.raises(ValueError):
        db.create_annotation_job(
            task_id=task_uuid, annotator_id=restored, item_uuids=[], public_token="x"
        )
    with pytest.raises(ValueError):
        db.create_annotation_job(
            task_id=task_uuid,
            annotator_id=restored,
            item_uuids=[item_ids[0], item_ids[0]],
            public_token="x",
        )
    with pytest.raises(ValueError):
        db.create_annotation_job(
            task_id=task_uuid,
            annotator_id=restored,
            item_uuids=["does-not-exist"],
            public_token="x",
        )

    # visibility
    assert db.update_annotation_job_visibility(job_uuid, True, "view-tok") is True
    assert db.get_annotation_job_by_view_token("view-tok")
    assert db.get_annotation_job_by_view_token(None) is None
    assert db.get_annotation_job_by_view_token("nope") is None

    # status
    assert db.update_annotation_job_status(job_uuid, "completed", set_completed_at=True) is True

    # annotations
    annotation_uuid = db.upsert_annotation(
        job_id=job_uuid,
        item_id=item_ids[0],
        value={"pass": True, "reasoning": "ok"},
        evaluator_id=seeded["uuid"],
    )
    assert annotation_uuid
    # upsert same slot — should reuse uuid
    annotation_uuid_again = db.upsert_annotation(
        job_id=job_uuid,
        item_id=item_ids[0],
        value={"pass": False, "reasoning": "updated"},
        evaluator_id=seeded["uuid"],
    )
    assert annotation_uuid == annotation_uuid_again
    # row-level (evaluator_id=None)
    db.upsert_annotation(job_id=job_uuid, item_id=item_ids[0], value=None)
    db.upsert_annotation(job_id=job_uuid, item_id=item_ids[0], value={"x": 1})

    assert db.get_annotations_for_job(job_uuid)
    assert db.get_annotated_item_ids(restored, item_ids)
    assert db.get_annotated_item_ids(restored, []) == []
    assert db.get_annotations_for_item(item_ids[0])
    assert db.get_annotations_for_slots(task_uuid, item_ids, [seeded["uuid"]])
    assert db.get_annotations_for_slots(task_uuid, [], []) == []
    assert db.get_annotations_for_task(task_uuid)
    assert db.get_annotations_for_task(
        task_uuid, since="2000-01-01 00:00:00", until="2999-01-01 00:00:00"
    )
    assert db.get_annotations_for_org(user["org_uuid"])
    assert db.get_annotations_for_org(
        user["org_uuid"], since="2000-01-01 00:00:00", until="2999-01-01 00:00:00"
    )
    # Page-scoped bulk fetch matches the single-task fetch for this task.
    tasks_anns = db.get_annotations_for_tasks([task_uuid])
    assert tasks_anns
    assert all(a["task_id"] == task_uuid for a in tasks_anns)
    assert len(tasks_anns) == len(db.get_annotations_for_task(task_uuid))
    assert db.get_annotations_for_tasks([]) == []
    overlap = db.get_annotations_for_annotator_overlap_slots(user["org_uuid"], restored)
    assert isinstance(overlap, list)

    # evaluator_runs
    v = db.get_evaluator_versions(seeded["uuid"])[0]
    run_ids = db.create_evaluator_runs(
        [
            {
                "job_id": job_uuid,
                "item_id": item_ids[0],
                "evaluator_id": seeded["uuid"],
                "evaluator_version_id": v["uuid"],
                "value": {"pass": True, "reasoning": "x"},
                "status": "completed",
            }
        ]
    )
    assert len(run_ids) == 1
    assert db.get_evaluator_runs_for_job(job_uuid)
    assert db.get_evaluator_runs_for_task(task_uuid)
    org_runs = db.get_evaluator_runs_for_org(user["org_uuid"])
    assert org_runs
    assert all(r["task_id"] == task_uuid for r in org_runs)
    tasks_runs = db.get_evaluator_runs_for_tasks([task_uuid])
    assert tasks_runs
    assert all(r["task_id"] == task_uuid for r in tasks_runs)
    assert len(tasks_runs) == len(db.get_evaluator_runs_for_task(task_uuid))
    assert db.get_evaluator_runs_for_tasks([]) == []
    assert db.get_evaluator_runs_for_evaluator_org_scoped(
        seeded["uuid"], user["org_uuid"]
    )
    assert db.get_evaluator_runs_for_evaluator_org_scoped(
        seeded["uuid"], user["org_uuid"], task_id=task_uuid, version_id=v["uuid"]
    )
    assert db.get_evaluator_runs_for_item(item_ids[0])

    cleared = db.clear_evaluator_runs_for_job(job_uuid)
    assert cleared >= 1

    # eval-job snapshot helpers
    eval_job_uuid = db.create_job(
        job_type="annotation-eval",
        user_id=user["uuid"], org_uuid=user["org_uuid"],
        details={"task_id": task_uuid},
    )
    snapshot_items = [
        {"uuid": item_ids[0], "payload": {"text": "snap1"}},
        {"uuid": item_ids[1], "payload": {"text": "snap2"}},
    ]
    db.snapshot_eval_job_items(eval_job_uuid, snapshot_items)
    db.snapshot_eval_job_items(eval_job_uuid, [])  # no-op
    snap = db.get_eval_job_items(eval_job_uuid)
    assert len(snap) == 2

    # soft-delete items
    deleted = db.soft_delete_annotation_items(task_uuid, [item_ids[0]])
    assert deleted == 1
    assert db.soft_delete_annotation_items(task_uuid, []) == 0
    assert db.soft_delete_annotation_job(job_uuid) is True
    assert db.soft_delete_annotation_job(job_uuid) is False

    # finally — delete the task (full cascade)
    assert db.delete_annotation_task(task_uuid) is True
    assert db.delete_annotation_task(task_uuid) is False


# ---------------------------------------------------------------------------
# Name uniqueness machinery
# ---------------------------------------------------------------------------


def test_name_uniqueness_helpers(user):
    name = _u("uniq")
    db.create_persona(name=name, user_id=user["uuid"], org_uuid=user["org_uuid"])
    assert db.is_name_taken("personas", name, user["org_uuid"]) is True
    assert db.is_name_taken("personas", "definitely-not-taken-name", user["org_uuid"]) is False

    # ensure_name_unique pre-check raises a typed error
    with pytest.raises(NameAlreadyExistsError):
        with db.ensure_name_unique(
            "personas", name, user["org_uuid"], entity="Persona"
        ):
            pass  # would do an insert


def test_add_evaluator_to_agent_recovers_from_concurrent_insert():
    """If a racing insert trips the UNIQUE(agent_id, evaluator_id) constraint
    between the existence check and our INSERT, add_evaluator_to_agent recovers
    to the winning row and returns its id instead of raising a 500."""
    cursor = MagicMock()
    # SELECT (miss) -> INSERT (UNIQUE conflict) -> re-SELECT (winning row).
    cursor.execute.side_effect = [
        None,
        sqlite3.IntegrityError(
            "UNIQUE constraint failed: agent_evaluators.agent_id, "
            "agent_evaluators.evaluator_id"
        ),
        None,
    ]
    cursor.fetchone.side_effect = [
        None,  # initial existence check: nothing yet
        {"id": 4242, "deleted_at": None},  # the racer's committed row
    ]
    conn = MagicMock()
    conn.cursor.return_value = cursor
    cm = MagicMock()
    cm.__enter__.return_value = conn
    cm.__exit__.return_value = False

    with patch("db.get_db_connection", return_value=cm):
        link_id = db.add_evaluator_to_agent("agent-x", "eval-y")

    assert link_id == 4242
    conn.rollback.assert_called_once()


def test_add_evaluator_to_agent_race_restores_soft_deleted_winner():
    """If the racing row landed soft-deleted, the recovery path restores it."""
    cursor = MagicMock()
    cursor.execute.side_effect = [
        None,  # initial SELECT
        sqlite3.IntegrityError("UNIQUE constraint failed"),  # INSERT
        None,  # re-SELECT
        None,  # UPDATE restore
    ]
    cursor.fetchone.side_effect = [
        None,
        {"id": 77, "deleted_at": "2026-07-11 00:00:00"},
    ]
    conn = MagicMock()
    conn.cursor.return_value = cursor
    cm = MagicMock()
    cm.__enter__.return_value = conn
    cm.__exit__.return_value = False

    with patch("db.get_db_connection", return_value=cm):
        link_id = db.add_evaluator_to_agent("agent-x", "eval-y")

    assert link_id == 77
    # The restore UPDATE ran (4 execute calls total incl. the restore).
    assert cursor.execute.call_count == 4


# ---------------------------------------------------------------------------
# Dataset name de-dup migration (guards idx_datasets_org_name_active build)
# ---------------------------------------------------------------------------


def _dataset_dedupe_conn():
    """Minimal in-memory DB with just the columns the de-dup pass touches."""
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.execute(
        "CREATE TABLE datasets (id INTEGER PRIMARY KEY AUTOINCREMENT, "
        "uuid TEXT, name TEXT, org_uuid TEXT, deleted_at TIMESTAMP DEFAULT NULL)"
    )
    return conn


def _insert_ds(conn, name, org, deleted=None):
    conn.execute(
        "INSERT INTO datasets (uuid, name, org_uuid, deleted_at) VALUES (?, ?, ?, ?)",
        (_u("ds"), name, org, deleted),
    )


def test_dedupe_active_dataset_names_renames_newer_collisions():
    conn = _dataset_dedupe_conn()
    cur = conn.cursor()
    # Two active collisions in org A (oldest keeps the name), one unrelated row.
    _insert_ds(conn, "Marathi TTS", "orgA")
    _insert_ds(conn, "Marathi TTS", "orgA")
    _insert_ds(conn, "Marathi TTS", "orgA")
    _insert_ds(conn, "Hindi TTS", "orgA")
    # Same name in a different org must NOT be touched.
    _insert_ds(conn, "Marathi TTS", "orgB")

    renamed = db._dedupe_active_dataset_names(cur)
    assert renamed == 2

    rows = cur.execute(
        "SELECT name FROM datasets WHERE org_uuid = 'orgA' ORDER BY id"
    ).fetchall()
    assert [r["name"] for r in rows] == [
        "Marathi TTS",
        "Marathi TTS (2)",
        "Marathi TTS (3)",
        "Hindi TTS",
    ]
    # Other org untouched.
    other = cur.execute(
        "SELECT name FROM datasets WHERE org_uuid = 'orgB'"
    ).fetchone()
    assert other["name"] == "Marathi TTS"

    # The unique index the migration protects now builds without error.
    cur.execute(
        "CREATE UNIQUE INDEX idx ON datasets(org_uuid, name) WHERE deleted_at IS NULL"
    )

    # Idempotent: a second pass finds nothing to rename.
    assert db._dedupe_active_dataset_names(cur) == 0


def test_dedupe_active_dataset_names_skips_taken_suffix():
    conn = _dataset_dedupe_conn()
    cur = conn.cursor()
    # "X (2)" already exists, so the renamed collision must jump to "X (3)".
    _insert_ds(conn, "X", "orgA")
    _insert_ds(conn, "X", "orgA")
    _insert_ds(conn, "X (2)", "orgA")

    renamed = db._dedupe_active_dataset_names(cur)
    assert renamed == 1
    names = {
        r["name"]
        for r in cur.execute("SELECT name FROM datasets WHERE org_uuid = 'orgA'")
    }
    assert names == {"X", "X (2)", "X (3)"}


def test_dedupe_active_dataset_names_ignores_soft_deleted():
    conn = _dataset_dedupe_conn()
    cur = conn.cursor()
    # A soft-deleted duplicate is not a collision (the index only covers active).
    _insert_ds(conn, "Y", "orgA")
    _insert_ds(conn, "Y", "orgA", deleted="2020-01-01")

    assert db._dedupe_active_dataset_names(cur) == 0


# ---------------------------------------------------------------------------
# Slim run-list job summaries (never read `details`, drop heavy `results` subtrees)
# ---------------------------------------------------------------------------


def _heavy_unit_results():
    return {
        "total_tests": 2,
        "passed": 1,
        "failed": 1,
        "latency_ms": {"p50": 12.0, "p95": 30.0, "count": 2},
        "cost": {"mean": 0.01, "count": 2},
        "total_tokens": {"mean": 42, "count": 2},
        "error": None,
        "test_results": [
            {
                "name": "case-a",
                "passed": True,
                # Heavy per-case detail that must NOT survive into the summary.
                "output": {"response": "X" * 500, "cost": 0.01},
                "judge_results": {"quality": {"reasoning": "Y" * 500}},
                "reasoning": "Z" * 500,
                "test_case": {"name": "ignored-when-name-present", "history": [1, 2]},
            },
            # No top-level name -> falls back to test_case.name.
            {"passed": False, "test_case": {"name": "case-b"}},
        ],
    }


def _heavy_benchmark_results():
    return {
        "error": None,
        "model_results": [
            {
                "model": "openai/gpt-4.1",
                "success": True,
                "message": "ok",
                "total_tests": 2,
                "passed": 2,
                "failed": 0,
                # Heavy nested per-case detail that must be dropped.
                "test_results": [{"output": "H" * 500} for _ in range(3)],
            }
        ],
    }


def test_agent_test_jobs_summary_slims_results_and_omits_details(user):
    agent_uuid = db.create_agent(
        name=_u("a-runs"), user_id=user["uuid"], org_uuid=user["org_uuid"]
    )
    heavy_details = {"config": {"blob": "D" * 1000}}
    unit_uuid = db.create_agent_test_job(
        agent_uuid, "llm-unit-test", status="done",
        details=heavy_details, results=_heavy_unit_results(),
    )
    bench_uuid = db.create_agent_test_job(
        agent_uuid, "llm-benchmark", status="done",
        details=heavy_details, results=_heavy_benchmark_results(),
    )

    rows = db.get_agent_test_jobs_for_agent_summary(agent_uuid)
    by_uuid = {r["uuid"]: r for r in rows}
    assert set(by_uuid) == {unit_uuid, bench_uuid}

    unit = by_uuid[unit_uuid]
    # `details` is never read into the summary.
    assert "details" not in unit or unit.get("details") is None
    res = unit["results"]
    # Scalar aggregates are extracted verbatim (dict aggregates round-trip).
    assert res["total_tests"] == 2
    assert res["passed"] == 1
    assert res["failed"] == 1
    assert res["latency_ms"] == {"p50": 12.0, "p95": 30.0, "count": 2}
    assert res["cost"] == {"mean": 0.01, "count": 2}
    assert res["total_tokens"] == {"mean": 42, "count": 2}
    # test_results are slimmed to {name, passed} only — heavy keys gone,
    # and the missing top-level name falls back to test_case.name.
    assert [(c["name"], bool(c["passed"])) for c in res["test_results"]] == [
        ("case-a", True),
        ("case-b", False),
    ]
    for case in res["test_results"]:
        assert set(case) == {"name", "passed"}

    bench = by_uuid[bench_uuid]
    mr = bench["results"]["model_results"]
    assert len(mr) == 1
    assert set(mr[0]) == {"model", "success", "message", "total_tests", "passed", "failed"}
    assert mr[0]["model"] == "openai/gpt-4.1"
    assert "test_results" not in mr[0]


def test_agent_test_jobs_summary_filters_by_type_and_orders(user):
    agent_uuid = db.create_agent(
        name=_u("a-runs2"), user_id=user["uuid"], org_uuid=user["org_uuid"]
    )
    u1 = db.create_agent_test_job(agent_uuid, "llm-unit-test", status="done")
    b1 = db.create_agent_test_job(agent_uuid, "llm-benchmark", status="done")

    only_unit = db.get_agent_test_jobs_for_agent_summary(agent_uuid, job_type="llm-unit-test")
    assert [r["uuid"] for r in only_unit] == [u1]

    both = db.get_agent_test_jobs_for_agent_summary(agent_uuid)
    # Newest-first (b1 created after u1).
    assert [r["uuid"] for r in both] == [b1, u1]
    # Null-results job yields empty slim arrays, not an error.
    assert both[0]["results"]["test_results"] in (None, [])


def test_agent_test_jobs_summary_skips_non_object_result_rows(user):
    # Non-object array elements would make json_extract raise malformed-JSON;
    # the `je.type = 'object'` guard skips them (mirrors the old isinstance check).
    agent_uuid = db.create_agent(
        name=_u("a-junk"), user_id=user["uuid"], org_uuid=user["org_uuid"]
    )
    results = {
        "test_results": ["not-a-dict", None, 5, {"name": "real", "passed": True}],
        "model_results": ["junk", {"model": "gpt", "passed": 1}],
    }
    job_uuid = db.create_agent_test_job(
        agent_uuid, "llm-unit-test", status="done", results=results
    )
    row = next(
        r for r in db.get_agent_test_jobs_for_agent_summary(agent_uuid)
        if r["uuid"] == job_uuid
    )
    assert [c["name"] for c in row["results"]["test_results"]] == ["real"]
    assert [m["model"] for m in row["results"]["model_results"]] == ["gpt"]


def test_agent_test_jobs_for_org_summary_joins_agent_name(user):
    agent_uuid = db.create_agent(
        name=_u("a-org"), user_id=user["uuid"], org_uuid=user["org_uuid"]
    )
    job_uuid = db.create_agent_test_job(
        agent_uuid, "llm-unit-test", status="done", results=_heavy_unit_results()
    )
    rows = db.get_agent_test_jobs_for_org_summary(user["org_uuid"])
    row = next(r for r in rows if r["uuid"] == job_uuid)
    assert row["agent_id"] == agent_uuid
    assert row["agent_name"]  # populated from the joined agent
    assert row["results"]["total_tests"] == 2


def test_tests_summary_extracts_only_description(user):
    agent_uuid = db.create_agent(
        name=_u("a-tl"), user_id=user["uuid"], org_uuid=user["org_uuid"]
    )
    heavy_config = {
        "description": "short desc",
        # Heavy blocks the slim projection must never read.
        "history": [{"role": "user", "content": "H" * 1000}],
        "evaluation": {"criteria": "E" * 1000},
        "settings": {"temperature": 0.7},
    }
    test_uuid = db.create_test(
        name=_u("t-heavy"), type="response", config=heavy_config,
        user_id=user["uuid"], org_uuid=user["org_uuid"],
    )
    db.add_test_to_agent(agent_uuid, test_uuid)

    for rows in (
        db.get_all_tests_summary(org_uuid=user["org_uuid"]),
        db.get_tests_for_agent_summary(agent_uuid),
    ):
        row = next(r for r in rows if r["uuid"] == test_uuid)
        assert row["name"]
        assert row["type"] == "response"
        # Only the description survives; heavy blocks are absent.
        assert row["config"] == {"description": "short desc"}


def test_tests_summary_handles_missing_description(user):
    test_uuid = db.create_test(
        name=_u("t-nodesc"), type="response", config={"history": []},
        user_id=user["uuid"], org_uuid=user["org_uuid"],
    )
    row = next(
        r for r in db.get_all_tests_summary(org_uuid=user["org_uuid"])
        if r["uuid"] == test_uuid
    )
    assert row["config"] == {"description": None}


def test_new_indexes_exist():
    with db.get_db_connection() as conn:
        names = {
            r[0]
            for r in conn.execute(
                "SELECT name FROM sqlite_master WHERE type = 'index'"
            ).fetchall()
        }
    expected = {
        "idx_annotation_items_task",
        "idx_evaluator_runs_job",
        "idx_dataset_items_dataset",
        "idx_jobs_status_type_created",
        "idx_agent_test_jobs_agent_created",
        "idx_agent_test_jobs_status",
        "idx_agent_test_jobs_share",
        "idx_simulation_jobs_sim_created",
        "idx_simulation_jobs_status",
        "idx_simulation_jobs_share",
        "idx_annotation_jobs_task",
        "idx_annotation_jobs_annotator",
    }
    assert expected <= names


def test_simulation_jobs_summary_returns_headers_only(user):
    agent_uuid = db.create_agent(
        name=_u("a-simrun"), user_id=user["uuid"], org_uuid=user["org_uuid"]
    )
    sim_uuid = db.create_simulation(
        name=_u("sim-run"), agent_id=agent_uuid,
        user_id=user["uuid"], org_uuid=user["org_uuid"],
    )
    heavy = {"transcript": [{"role": "user", "content": "H" * 1000}]}
    job_uuid = db.create_simulation_job(
        sim_uuid, "text", status="done",
        details={"config": "D" * 1000}, results=heavy,
    )
    rows = db.get_simulation_jobs_summary(sim_uuid)
    row = next(r for r in rows if r["uuid"] == job_uuid)
    assert set(row) >= {"uuid", "status", "type", "created_at", "updated_at"}
    # Neither heavy blob is fetched.
    assert "results" not in row
    assert "details" not in row
