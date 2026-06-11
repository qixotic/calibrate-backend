"""Broad-coverage CRUD tests for src/db.py.

Each entity has its own block: create → read (by uuid, list, bulk) →
update → soft-delete → list-after-delete. We also exercise the pivot
tables (agent_tools, agent_tests, simulation_personas, simulation_scenarios,
simulation_evaluators, test_evaluators) and the queue accounting helpers.

Tests share one initialized DB (conftest fixture). Every row uses a
freshly minted name/uuid so tests are order-independent.
"""

from __future__ import annotations

import time
import uuid as _uuid

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
    ev_uuid = _uuid.uuid4().hex
    task_uuid = _uuid.uuid4().hex
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
            (_uuid.uuid4().hex,),
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
    assert any(t["uuid"] == t_uuid for t in db.get_all_tests(org_uuid=user["org_uuid"]))
    assert any(t["uuid"] == t_uuid for t in db.get_all_tests())

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
        j["uuid"] == j_uuid for j in db.get_agent_test_jobs_for_agent(agent_uuid)
    )
    assert any(
        j["uuid"] == j_uuid
        for j in db.get_agent_test_jobs_for_agent(agent_uuid, job_type="llm-unit-test")
    )
    assert any(j["uuid"] == j_uuid for j in db.get_all_agent_test_jobs())
    assert any(
        j["uuid"] == j_uuid for j in db.get_all_agent_test_jobs(job_type="llm-unit-test")
    )
    assert any(j["uuid"] == j_uuid for j in db.get_agent_test_jobs_for_org(user["org_uuid"]))
    assert any(
        j["uuid"] == j_uuid
        for j in db.get_agent_test_jobs_for_org(user["org_uuid"], job_type="llm-unit-test")
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
    assert db.get_evaluator_runs_for_org(user["org_uuid"])
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
