"""Unit tests for the shared list-endpoint deps in `src/pagination.py`.

These cover the factories' contracts (allowlist enforcement, default values,
apply() semantics) directly so future endpoints adopting them don't have to
rediscover the edge cases via their own integration tests.
"""

from __future__ import annotations

import pytest
from fastapi import HTTPException

from pagination import (
    OptionalPaginationParams,
    PaginationParams,
    count_and_page,
    make_projection_params,
    make_search_params,
    make_sort_params,
    page_envelope,
    paginate,
)


# ---------------------------------------------------------------------------
# PaginationParams
# ---------------------------------------------------------------------------


def test_pagination_assigns_values():
    # Called directly (outside FastAPI), the Query(...) defaults aren't
    # resolved — FastAPI does that. Pass explicit values; the unit test
    # is just checking the constructor wires them through.
    p = PaginationParams(limit=25, offset=10)
    assert p.limit == 25
    assert p.offset == 10


def test_pagination_max_limit_is_export_friendly():
    """The cap is intentionally huge (1M) so FE "export all" flows can
    pass `limit=total` in a single request. If you ever lower this,
    update the CLAUDE.md bullet and warn FE consumers that depend on
    single-shot fetches — there are real ones (CSV export from the
    annotation-task summary view)."""
    from pagination import MAX_LIMIT

    assert MAX_LIMIT >= 1_000_000


# ---------------------------------------------------------------------------
# OptionalPaginationParams + paginate
# ---------------------------------------------------------------------------


def test_optional_pagination_defaults_to_no_limit():
    # Omitting the params (limit=None) means "return everything" — the slice is
    # a no-op so adding the dep to an endpoint keeps every item.
    p = OptionalPaginationParams(limit=None, offset=0)
    items = list(range(10))
    env = paginate(items, p)
    assert env == {"items": items, "total": 10, "limit": None, "offset": 0}


def test_optional_pagination_slices_and_reports_total():
    items = list(range(10))
    env = paginate(items, OptionalPaginationParams(limit=3, offset=2))
    assert env["items"] == [2, 3, 4]
    # Total is the PRE-slice length, so a client knows more pages exist.
    assert env["total"] == 10
    assert env["limit"] == 3 and env["offset"] == 2


def test_optional_pagination_offset_only_returns_tail():
    items = list(range(5))
    env = paginate(items, OptionalPaginationParams(limit=None, offset=3))
    assert env["items"] == [3, 4]
    assert env["total"] == 5


def test_optional_pagination_offset_past_end_is_empty():
    items = list(range(3))
    env = paginate(items, OptionalPaginationParams(limit=10, offset=100))
    assert env["items"] == []
    # Total still reflects the full set even when the page is empty.
    assert env["total"] == 3


def test_count_and_page_and_envelope_compose():
    # count_and_page returns (page, total) for endpoints that transform the
    # page before wrapping; page_envelope builds the same dict paginate does.
    items = list(range(10))
    p = OptionalPaginationParams(limit=2, offset=0)
    page, total = count_and_page(items, p)
    assert page == [0, 1] and total == 10
    transformed = [x * 10 for x in page]
    assert page_envelope(transformed, total, p) == {
        "items": [0, 10],
        "total": 10,
        "limit": 2,
        "offset": 0,
    }


# ---------------------------------------------------------------------------
# make_sort_params
# ---------------------------------------------------------------------------


def test_sort_params_factory_validates_default_at_module_load():
    # Happy path: default in sortable, factory succeeds.
    Sort = make_sort_params(
        sortable=["created_at", "updated_at"], default="created_at"
    )
    # Constructor wires the values through (FastAPI normally fills the
    # `Query(...)` defaults; here we pass them explicitly).
    s = Sort(sort_by="created_at", order="desc")
    assert s.sort_by == "created_at"
    assert s.order == "desc"

    # default not in sortable → factory raises at module-load time, not
    # request time. Catches typos before they reach production.
    with pytest.raises(ValueError):
        make_sort_params(sortable=["a", "b"], default="c")


def test_sort_params_rejects_disallowed_column_at_request_time():
    Sort = make_sort_params(sortable=["created_at"], default="created_at")
    with pytest.raises(HTTPException) as exc:
        Sort(sort_by="password", order="asc")
    assert exc.value.status_code == 422


def test_sort_params_apply_orders_with_secondary_tiebreaker():
    Sort = make_sort_params(
        sortable=["updated_at"], default="updated_at", default_order="desc"
    )
    items = [
        {"uuid": "a", "updated_at": "2024-01-01"},
        {"uuid": "c", "updated_at": "2024-01-02"},
        # Identical timestamps — secondary key (uuid) breaks the tie
        # deterministically so paging is stable.
        {"uuid": "b", "updated_at": "2024-01-02"},
    ]
    desc = Sort(sort_by="updated_at", order="desc").apply(items)
    assert [it["uuid"] for it in desc] == ["c", "b", "a"]

    asc = Sort(sort_by="updated_at", order="asc").apply(items)
    assert [it["uuid"] for it in asc] == ["a", "b", "c"]


def test_sort_params_apply_handles_missing_or_null_sort_values():
    Sort = make_sort_params(
        sortable=["updated_at"], default="updated_at", default_order="asc"
    )
    items = [
        {"uuid": "a", "updated_at": "2024-01-01"},
        {"uuid": "b", "updated_at": None},  # would crash on `None < "2024..."`
        {"uuid": "c"},  # missing entirely — same coercion path
    ]
    # Should not raise; null/missing coerce to "" and sort to the front on asc.
    result = Sort(sort_by="updated_at", order="asc").apply(items)
    assert {it["uuid"] for it in result[:2]} == {"b", "c"}
    assert result[-1]["uuid"] == "a"


# ---------------------------------------------------------------------------
# make_search_params
# ---------------------------------------------------------------------------


def test_search_params_factory_rejects_empty_searchable():
    with pytest.raises(ValueError):
        make_search_params(searchable=[])


def test_search_params_noop_for_none_and_blank():
    Search = make_search_params(searchable=["name"])
    items = [{"name": "alpha"}, {"name": "beta"}]
    assert Search(q=None).apply(items) == items
    assert Search(q="   ").apply(items) == items
    assert Search(q="").apply(items) == items


def test_search_params_case_insensitive_substring():
    Search = make_search_params(searchable=["name"])
    items = [{"name": "Alpha"}, {"name": "BETA"}, {"name": "gamma"}]
    assert [it["name"] for it in Search(q="ALP").apply(items)] == ["Alpha"]
    assert [it["name"] for it in Search(q="a").apply(items)] == [
        "Alpha",
        "BETA",
        "gamma",
    ]


def test_search_params_nested_paths():
    """Dotted paths reach into nested JSON columns (common pattern: items
    whose user-facing label lives at `payload.name`)."""
    Search = make_search_params(searchable=["payload.name"])
    items = [
        {"payload": {"name": "alpha"}},
        {"payload": {"name": "beta"}},
        {"payload": {"other": "alpha"}},  # `name` missing — must not match
        {"payload": None},  # null payload — must not crash
        {},  # missing payload — must not crash
    ]
    matched = Search(q="alpha").apply(items)
    assert matched == [{"payload": {"name": "alpha"}}]


def test_search_params_matches_any_listed_path():
    """Multiple `searchable` entries OR together — a hit in any path counts."""
    Search = make_search_params(searchable=["name", "description"])
    items = [
        {"name": "alpha", "description": "x"},
        {"name": "y", "description": "alpha-thing"},
        {"name": "z", "description": "z"},
    ]
    assert len(Search(q="alpha").apply(items)) == 2


# ---------------------------------------------------------------------------
# make_projection_params (?compact=)
# ---------------------------------------------------------------------------


def test_projection_requires_non_empty_heavy_fields():
    with pytest.raises(ValueError):
        make_projection_params(heavy_fields=[])


def test_projection_compact_description_is_docs_clean():
    """The generated `?compact` help text must read cleanly in the public docs:
    no internal walk markers (`[]`/`*`) and no em-dashes (which the static
    api-writing-style checker can't see through this runtime string)."""
    from fastapi import params

    Proj = make_projection_params(
        heavy_fields=["config", "rows[].annotations.*.reasoning"]
    )
    # The Query default lives on the constructor signature's `compact` param.
    import inspect

    default = inspect.signature(Proj.__init__).parameters["compact"].default
    desc = default.description if isinstance(default, params.Query) else ""
    assert "—" not in desc  # no em-dash
    assert "[]" not in desc and ".*" not in desc  # no internal markers
    # Readable dotted names survive.
    assert "`config`" in desc
    assert "`rows.annotations.reasoning`" in desc


def test_projection_noop_when_not_compact():
    Proj = make_projection_params(heavy_fields=["config"])
    data = {"uuid": "u1", "config": {"big": "blob"}}
    # compact defaults to False → untouched.
    assert Proj(compact=False).apply(data) == {"uuid": "u1", "config": {"big": "blob"}}


def test_projection_nulls_top_level_key():
    Proj = make_projection_params(heavy_fields=["config"])
    data = {"uuid": "u1", "config": {"big": "blob"}}
    out = Proj(compact=True).apply(data)
    # Key stays, value nulled — shape stays response_model-compatible.
    assert out == {"uuid": "u1", "config": None}


def test_projection_nulls_list_item_fields():
    Proj = make_projection_params(
        heavy_fields=["results[].output", "results[].judge_results"]
    )
    data = {
        "results": [
            {"name": "a", "passed": True, "output": {"resp": "x"}, "judge_results": [1]},
            {"name": "b", "passed": False, "output": {"resp": "y"}, "judge_results": [2]},
        ]
    }
    out = Proj(compact=True).apply(data)
    assert out["results"] == [
        {"name": "a", "passed": True, "output": None, "judge_results": None},
        {"name": "b", "passed": False, "output": None, "judge_results": None},
    ]


def test_projection_nulls_dict_value_wildcard():
    # `rows[].annotations.*.reasoning` — annotations is a dict keyed by
    # annotator uuid; null each value's `reasoning` while keeping `value`.
    Proj = make_projection_params(heavy_fields=["rows[].annotations.*.reasoning"])
    data = {
        "rows": [
            {
                "item_id": "i1",
                "annotations": {
                    "ann-1": {"value": 1, "reasoning": "long text"},
                    "ann-2": {"value": 0, "reasoning": "more text"},
                },
            }
        ]
    }
    out = Proj(compact=True).apply(data)
    anns = out["rows"][0]["annotations"]
    assert anns["ann-1"] == {"value": 1, "reasoning": None}
    assert anns["ann-2"] == {"value": 0, "reasoning": None}


def test_projection_nested_list_in_list():
    Proj = make_projection_params(
        heavy_fields=["evaluators[].versions[].system_prompt"]
    )
    data = {
        "evaluators": [
            {"uuid": "e1", "versions": [{"version_number": 1, "system_prompt": "P1"}]},
        ]
    }
    out = Proj(compact=True).apply(data)
    assert out["evaluators"][0]["versions"][0] == {
        "version_number": 1,
        "system_prompt": None,
    }


def test_projection_missing_paths_are_noops():
    # Paths that don't exist / wrong types must not raise — a projection that
    # doesn't apply to a given payload silently skips.
    Proj = make_projection_params(
        heavy_fields=["config", "results[].output", "rows[].annotations.*.reasoning"]
    )
    data = {"uuid": "u1", "results": None, "rows": "not-a-list"}
    out = Proj(compact=True).apply(data)
    assert out == {"uuid": "u1", "results": None, "rows": "not-a-list"}
