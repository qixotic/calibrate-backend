"""Shared list-endpoint primitives: pagination, sort, search.

All three are FastAPI dependencies that handlers depend on; they don't touch
SQL directly — they take an already-loaded list of dicts and return a
filtered/sorted/sliced view. That's a deliberate trade: rebuilding queries
across the many entity types in this app would mean a parser per shape, and
the entity tables currently fit in memory. If a particular endpoint outgrows
that, push the predicate down into SQL inside the handler — these deps still
own the request-parameter parsing.

Each endpoint customizes sort/search via a factory that bakes in the
per-endpoint allowlist of columns:

    SummarySort = make_sort_params(
        sortable=["created_at", "updated_at"], default="created_at",
    )
    SummarySearch = make_search_params(searchable=["payload.name"])

    @router.get(...)
    async def handler(
        ...,
        search: SummarySearch = Depends(),
        sort: SummarySort = Depends(),
        pagination: PaginationParams = Depends(),
    ):
        items = search.apply(items)
        items = sort.apply(items)
        page = items[pagination.offset : pagination.offset + pagination.limit]

The factories return a fresh class per call so FastAPI's dependency cache
treats each endpoint's sort/search as a distinct type. `sortable` is enforced
as a strict allowlist — anything else gets a 422 from FastAPI (no
SQL-injection surface even though sort runs post-fetch in Python).
"""

from typing import Any, Dict, Generic, List, Literal, Optional, Tuple, Type, TypeVar

from fastapi import HTTPException, Query
from pydantic import BaseModel, Field

DEFAULT_LIMIT = 50
# Cap is intentionally very high (1M) so "give me everything" use cases like
# CSV export can pass `limit=<total>` without a multi-request loop on the FE.
# The cap exists only as a guard against pathological values (e.g. integer
# overflow attempts); it is not a per-request payload budget — handlers
# remain responsible for their own size/perf characteristics.
MAX_LIMIT = 1_000_000


class PaginationParams:
    """FastAPI dependency for `?limit=&offset=` query params."""

    def __init__(
        self,
        limit: int = Query(
            DEFAULT_LIMIT,
            ge=1,
            le=MAX_LIMIT,
            description="Maximum number of items to return",
        ),
        offset: int = Query(
            0,
            ge=0,
            description="Number of items to skip before returning results",
        ),
    ):
        self.limit = limit
        self.offset = offset


class OptionalPaginationParams:
    """FastAPI dependency for OPTIONAL `?limit=&offset=` on list endpoints.

    Unlike `PaginationParams` (whose `limit` defaults to 50), here `limit`
    defaults to `None` meaning "no limit" — omitting the params returns the full
    list unchanged, so adding this dep to an existing endpoint is
    backwards-compatible. `offset` alone (with no `limit`) skips a prefix and
    returns the rest.

    Pair with `paginate(items, pagination)` (or `count_and_page` + `page_envelope`
    when the page needs a per-item transform), which returns a
    `PaginatedResponse`-shaped `{items, total, limit, offset}` body. `total` is
    the pre-slice count so a client can tell whether more pages exist.
    """

    def __init__(
        self,
        limit: Optional[int] = Query(
            None,
            ge=1,
            le=MAX_LIMIT,
            description="Maximum number of items to return. Omit for no limit (all items)",
        ),
        offset: int = Query(
            0,
            ge=0,
            description="Number of items to skip before returning results",
        ),
    ):
        self.limit = limit
        self.offset = offset


T = TypeVar("T")


class PaginatedResponse(BaseModel, Generic[T]):
    """Uniform envelope for paginated list endpoints. `items` is the page;
    `total` is the count BEFORE slicing so a client knows whether more pages
    exist. `limit`/`offset` echo the applied window (`limit=null` = unbounded).

    Every paginated list returns this same shape, so a client reads the array
    via `.items` and the count via `.total` the same way on every endpoint."""

    items: List[T] = Field(description="The page of results")
    total: int = Field(
        description="Total number of items matching the query, before pagination"
    )
    limit: Optional[int] = Field(
        None, description="The applied `limit` (null when unbounded)"
    )
    offset: int = Field(0, description="The applied `offset`")


def count_and_page(
    items: List[Any], pagination: "OptionalPaginationParams"
) -> Tuple[List[Any], int]:
    """Return `(page, total)`: the sliced page plus the pre-slice length. Use
    this when the page needs a per-item transform before going into the
    envelope; wrap the transformed page with `page_envelope`.

    With `pagination.limit is None` the slice keeps everything from `offset` on.
    """
    total = len(items)
    start = pagination.offset
    if pagination.limit is None:
        page = items[start:]
    else:
        page = items[start : start + pagination.limit]
    return page, total


def page_envelope(
    items: List[Any], total: int, pagination: "OptionalPaginationParams"
) -> Dict[str, Any]:
    """Build the `PaginatedResponse` body dict from an already-sliced page and
    its pre-slice total. FastAPI coerces the dict to the endpoint's
    `PaginatedResponse[T]` response_model."""
    return {
        "items": items,
        "total": total,
        "limit": pagination.limit,
        "offset": pagination.offset,
    }


def paginate(
    items: List[Any], pagination: "OptionalPaginationParams"
) -> Dict[str, Any]:
    """Slice `items` and return the `PaginatedResponse` envelope in one call.
    For endpoints that transform each item first, use `count_and_page` +
    `page_envelope` instead."""
    page, total = count_and_page(items, pagination)
    return page_envelope(page, total, pagination)


def make_sort_params(
    *,
    sortable: List[str],
    default: str,
    default_order: Literal["asc", "desc"] = "desc",
) -> Type:
    """Build a FastAPI `Depends`-compatible sort class for one endpoint.

    `sortable` is the allowlist of column names the endpoint will accept for
    `?sort_by=`. `default` must be a member. The returned class has an
    `apply(items, *, secondary_key="uuid")` method that returns a sorted copy
    using the standard `(sort_value, secondary_key)` key — secondary key
    breaks ties so paging is stable when timestamps collide (sqlite
    `CURRENT_TIMESTAMP` is second-resolution).

    Empty/missing sort values coerce to `""` so they sort to one end
    deterministically rather than raising `TypeError` on `None < str`.
    """
    if default not in sortable:
        raise ValueError(
            f"default sort_by={default!r} must be one of sortable={sortable!r}"
        )
    allowed = list(sortable)  # frozen copy
    description = "Sort key for the results"

    class SortParams:
        def __init__(
            self,
            sort_by: str = Query(default, description=description),
            order: Literal["asc", "desc"] = Query(
                default_order, description="Sort direction"
            ),
        ):
            if sort_by not in allowed:
                # FastAPI auto-422s on Literal mismatch — we can't put a
                # dynamic Literal on the annotation, so validate manually.
                raise HTTPException(
                    status_code=422,
                    detail=(
                        f"sort_by={sort_by!r} not allowed; expected one of "
                        f"{allowed!r}"
                    ),
                )
            self.sort_by = sort_by
            self.order = order

        def apply(
            self,
            items: List[Dict[str, Any]],
            *,
            secondary_key: str = "uuid",
        ) -> List[Dict[str, Any]]:
            reverse = self.order == "desc"
            return sorted(
                items,
                key=lambda it: (
                    it.get(self.sort_by) or "",
                    it.get(secondary_key) or "",
                ),
                reverse=reverse,
            )

    SortParams.__name__ = f"SortParams[{'|'.join(allowed)}]"
    return SortParams


def make_search_params(*, searchable: List[str]) -> Type:
    """Build a FastAPI `Depends`-compatible search class for one endpoint.

    `searchable` is the list of dotted paths the search will match against
    (e.g. `"payload.name"` for a nested JSON field, `"name"` for a top-level
    column). The query parameter is always `?q=` — standardize across the
    codebase rather than letting each endpoint pick its own name.

    `q` is case-insensitive substring; empty/whitespace-only `q` is a no-op
    so a FE search-input binding doesn't have to special-case the cleared
    state. Returns `apply(items) -> filtered list`.
    """
    if not searchable:
        raise ValueError("searchable must be non-empty")
    paths = [p.split(".") for p in searchable]
    description = (
        f"Case-insensitive substring search on {', '.join(f'`{s}`' for s in searchable)}. "
        "Blank is a no-op"
    )

    class SearchParams:
        def __init__(
            self,
            q: Optional[str] = Query(None, description=description),
        ):
            self.q: Optional[str] = (
                q.strip().lower() if isinstance(q, str) and q.strip() else None
            )

        def apply(
            self, items: List[Dict[str, Any]]
        ) -> List[Dict[str, Any]]:
            if self.q is None:
                return items
            needle = self.q
            return [it for it in items if _matches(it, paths, needle)]

    SearchParams.__name__ = f"SearchParams[{'|'.join(searchable)}]"
    return SearchParams


def make_projection_params(*, heavy_fields: List[str]) -> Type:
    """Build a FastAPI `Depends`-compatible `?compact=` class for one endpoint.

    `heavy_fields` is the endpoint's list of payload-dominating fields to drop
    when the caller passes `?compact=true`. Each entry is a dotted path with an
    optional `[]` (iterate a list) or `*` (iterate a dict's values) at any
    segment, e.g.:

        "config"                         # top-level key
        "results[].output"              # `output` on every element of `results`
        "rows[].annotations.*.reasoning" # `reasoning` on every value of each
                                         # row's `annotations` dict
        "evaluators[].versions[].system_prompt"

    Projection **nulls the value in place, keeping the key** — the response
    stays shape-compatible with the endpoint's `response_model` (so the
    generated SDK/MCP types are unchanged and the heavy fields must already be
    `Optional`). The weight being shed is the values (transcripts, rubrics,
    prompts), not the ~10-char key names. Full detail is fetched by omitting
    the flag.

    The returned class exposes `apply(data: dict) -> dict`, meant to run on a
    handler's `model_dump()` output (a fresh dict, safe to mutate) right before
    returning. A no-op when `compact` is false.
    """
    if not heavy_fields:
        raise ValueError("heavy_fields must be non-empty")
    tokenized = [_tokenize_projection(p) for p in heavy_fields]
    # Docs-facing: readable dotted names, no walk markers or em-dashes. This
    # runtime-built string bypasses the static api-writing-style checker.
    display = ", ".join(f"`{_display_projection_path(p)}`" for p in heavy_fields)
    description = (
        f"Return a compact response that omits heavy detail fields ({display}), "
        "keeping only the lightweight decision fields. Omit for full detail"
    )

    class ProjectionParams:
        def __init__(
            self,
            compact: bool = Query(False, description=description),
        ):
            self.compact = compact

        def apply(self, data: Dict[str, Any]) -> Dict[str, Any]:
            if self.compact and isinstance(data, dict):
                for tokens in tokenized:
                    _null_at(data, tokens)
            return data

    ProjectionParams.__name__ = "ProjectionParams"
    return ProjectionParams


def _display_projection_path(path: str) -> str:
    """Human-readable form of a projection path for docs: drop the internal
    `[]` (iterate list) and `*` (iterate dict values) markers so a reader sees
    plain dotted field names (`rows[].annotations.*.reasoning` → `rows.annotations.reasoning`)."""
    return path.replace("[]", "").replace(".*", "")


def _tokenize_projection(path: str) -> List[str]:
    """Split a projection path into walk tokens, exploding a trailing `[]` on a
    segment into its own token so the walker can iterate that list."""
    tokens: List[str] = []
    for seg in path.split("."):
        if seg.endswith("[]"):
            tokens.append(seg[:-2])
            tokens.append("[]")
        else:
            tokens.append(seg)
    return tokens


def _null_at(obj: Any, tokens: List[str]) -> None:
    """Walk `tokens` into `obj`, nulling the leaf. `[]` iterates a list; `*`
    iterates a dict's values; any other token is a dict key. Missing keys /
    type mismatches are silently skipped so a projection path that doesn't
    apply to a given payload is a no-op rather than an error."""
    if not tokens:
        return
    tok, rest = tokens[0], tokens[1:]
    if tok == "[]":
        if isinstance(obj, list):
            for el in obj:
                _null_at(el, rest)
        return
    if tok == "*":
        if isinstance(obj, dict):
            for val in obj.values():
                _null_at(val, rest)
        return
    if not isinstance(obj, dict):
        return
    if not rest:
        if tok in obj:
            obj[tok] = None
    else:
        _null_at(obj.get(tok), rest)


def _matches(item: Dict[str, Any], paths: List[List[str]], needle: str) -> bool:
    for path in paths:
        value = _get_path(item, path)
        if isinstance(value, str) and needle in value.lower():
            return True
    return False


def _get_path(item: Dict[str, Any], path: List[str]) -> Any:
    cur: Any = item
    for part in path:
        if not isinstance(cur, dict):
            return None
        cur = cur.get(part)
    return cur


__all__ = [
    "DEFAULT_LIMIT",
    "MAX_LIMIT",
    "PaginationParams",
    "OptionalPaginationParams",
    "PaginatedResponse",
    "count_and_page",
    "page_envelope",
    "paginate",
    "make_sort_params",
    "make_search_params",
    "make_projection_params",
]
