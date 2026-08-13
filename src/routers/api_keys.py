"""API keys for programmatic access to your workspace.

Each key is scoped to your active workspace. The raw key is returned exactly
once at creation. Later reads show only a masked display form.
"""

import re
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, Path
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
    name: str = Field(
        ...,
        min_length=1,
        max_length=200,
        description="Label shown in your key listings",
    )


def _masked(last_four: str) -> str:
    """Display form once the raw key is gone, e.g. `••••1a2b`."""
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
    uuid: str = Field(
        min_length=36,
        max_length=36,
        description="API key ID",
    )
    name: str = Field(description="Label for the key")
    last_four: str = Field(
        description="Last four characters of the key"
    )
    masked_key: str = Field(
        description="Masked display form of the key for listings"
    )
    last_used_at: Optional[str] = Field(
        None,
        description="When the key last authenticated a request",
    )
    created_at: str = Field(description="When the key was created (ISO 8601 UTC)")
    updated_at: str = Field(description="When the key was last updated (ISO 8601 UTC)")

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
    key: str = Field(
        description="The API key. **Returned exactly once at creation.** Store it now. It cannot be retrieved again"
    )


@router.post(
    "", response_model=CreateApiKeyResponse, status_code=201, summary="Create API key"
)
def create_key(
    request: CreateApiKeyRequest,
    ctx: OrgContext = Depends(get_current_org),
):
    """Create an API key for programmatic access to the API"""
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


@router.get("", response_model=List[ApiKeyResponse], summary="List API keys")
def list_keys(ctx: OrgContext = Depends(get_current_org)):
    """List active API keys"""
    return [ApiKeyResponse.from_row(k) for k in list_api_keys_for_org(ctx.org_uuid)]


@router.delete("/{key_uuid}", status_code=204, summary="Revoke API key")
def revoke_key(
    key_uuid: str = Path(
        description="The API key to revoke",
        examples=["f47ac10b-58cc-4372-a567-0e02b2c3d479"],
    ),
    ctx: OrgContext = Depends(get_current_org),
):
    """Revoke an API key so it can no longer authenticate requests"""
    if get_api_key(key_uuid, ctx.org_uuid) is None:
        raise HTTPException(status_code=404, detail="API key not found")
    soft_delete_api_key(key_uuid, ctx.org_uuid)
