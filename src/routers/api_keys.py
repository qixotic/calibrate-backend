"""API keys router — credentials for programmatic API access.

A key is scoped to the caller's active org (resolved via `get_current_org`, i.e.
the `X-Org-UUID` header or the personal org). The raw `sk_…` key is returned
exactly once, on creation; afterwards only its prefix and bcrypt hash are stored,
so it can be listed/revoked but never re-displayed. Authenticate downstream
requests with `Authorization: Bearer sk_…` or `X-API-Key: sk_…` — see
`auth_utils.get_org_jwt_or_api_key`.
"""

import re
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field, field_validator

from auth_utils import (
    API_KEY_PREFIX,
    OrgContext,
    generate_api_key,
    get_current_org,
    hash_api_key,
)
from db import (
    create_api_key,
    get_api_key,
    list_api_keys_for_org,
    soft_delete_api_key,
)

router = APIRouter(prefix="/api-keys", tags=["api-keys"])


class CreateApiKeyRequest(BaseModel):
    name: str = Field(..., min_length=1, max_length=200)


def _masked(last_four: str) -> str:
    """Display form once the raw key is gone, e.g. `sk_••••1a2b`."""
    return f"{API_KEY_PREFIX}••••{last_four}"


_TZ_SUFFIX = re.compile(r"(Z|[+-]\d{2}:?\d{2})$")


def _to_utc_iso(ts: Optional[str]) -> Optional[str]:
    """Normalize a SQLite UTC timestamp to explicit ISO-8601 UTC.

    SQLite `CURRENT_TIMESTAMP` is naive UTC (`2026-06-05 10:11:00`). Emitting it
    without a zone makes browsers parse it as local time, skewing "Last used" by
    the viewer's offset. We swap the space for `T` and append `Z` so the FE can
    `new Date(...)` it directly. No-op if a zone is already present or value is
    None/empty.
    """
    if not ts:
        return ts
    s = str(ts).strip().replace(" ", "T")
    return s if _TZ_SUFFIX.search(s) else s + "Z"


class ApiKeyResponse(BaseModel):
    """Listing shape — never includes the raw key.

    `last_four` is the only fragment of the key kept after creation;
    `masked_key` is a ready-to-render display string built from it.
    """

    uuid: str
    name: str
    last_four: str
    masked_key: str
    last_used_at: Optional[str] = None
    created_at: str
    updated_at: str

    # Stamp timestamps as explicit UTC (…Z) so the FE doesn't read them as local.
    @field_validator("created_at", "updated_at", "last_used_at")
    @classmethod
    def _stamp_utc(cls, v: Optional[str]) -> Optional[str]:
        return _to_utc_iso(v)

    @classmethod
    def from_row(cls, row: dict, **extra) -> "ApiKeyResponse":
        """Build the response (any subclass) from a DB row, deriving the display
        fields. `extra` carries subclass-only fields, e.g. the raw `key`."""
        last_four = row.get("key_last_four", "")
        return cls(last_four=last_four, masked_key=_masked(last_four), **extra, **row)


class CreateApiKeyResponse(ApiKeyResponse):
    """Creation shape — carries the raw `key` exactly once. Show it, then never
    again; subsequent reads only ever return `masked_key` / `last_four`."""

    key: str


@router.post("", response_model=CreateApiKeyResponse, status_code=201)
async def create_key(
    request: CreateApiKeyRequest,
    ctx: OrgContext = Depends(get_current_org),
):
    """Mint a new API key for the caller's active org. Returns the raw key once."""
    raw_key, key_prefix = generate_api_key()
    row = create_api_key(
        org_uuid=ctx.org_uuid,
        owner_user_id=ctx.user_id,
        name=request.name,
        key_prefix=key_prefix,
        key_last_four=raw_key[-4:],
        key_hash=hash_api_key(raw_key),
    )
    return CreateApiKeyResponse.from_row(row, key=raw_key)


@router.get("", response_model=List[ApiKeyResponse])
async def list_keys(ctx: OrgContext = Depends(get_current_org)):
    """List active API keys for the caller's active org (no raw keys)."""
    return [ApiKeyResponse.from_row(k) for k in list_api_keys_for_org(ctx.org_uuid)]


@router.delete("/{key_uuid}", status_code=204)
async def revoke_key(
    key_uuid: str,
    ctx: OrgContext = Depends(get_current_org),
):
    """Revoke (soft-delete) an API key. 404 if it isn't in the caller's org."""
    if get_api_key(key_uuid, ctx.org_uuid) is None:
        raise HTTPException(status_code=404, detail="API key not found")
    soft_delete_api_key(key_uuid, ctx.org_uuid)
