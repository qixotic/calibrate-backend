"""JWT Authentication Utilities.

This module provides JWT token creation/validation for securing API endpoints.
"""

import os
import secrets
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Optional

import bcrypt
from jose import JWTError, jwt
from fastapi import HTTPException, Depends, Header
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

logger = logging.getLogger(__name__)

# JWT Configuration
JWT_SECRET_KEY = os.getenv("JWT_SECRET_KEY", "change-this-secret-key-in-production")
JWT_ALGORITHM = "HS256"
JWT_EXPIRATION_HOURS = int(os.getenv("JWT_EXPIRATION_HOURS", "168"))  # 7 days default

# Security scheme for Bearer token authentication
security = HTTPBearer(auto_error=True)


def create_access_token(user_uuid: str, email: str) -> str:
    """
    Create a JWT access token containing the user's UUID.

    Args:
        user_uuid: The unique identifier of the user
        email: The user's email (for logging/debugging)

    Returns:
        Encoded JWT token string
    """
    expire = datetime.utcnow() + timedelta(hours=JWT_EXPIRATION_HOURS)
    payload = {
        "sub": user_uuid,  # subject = user UUID
        "email": email,
        "exp": expire,
        "iat": datetime.utcnow(),
    }
    token = jwt.encode(payload, JWT_SECRET_KEY, algorithm=JWT_ALGORITHM)
    logger.debug(f"Created access token for user {user_uuid} (expires: {expire})")
    return token


def decode_token(token: str) -> Optional[dict]:
    """
    Decode and validate a JWT token.

    Args:
        token: The JWT token string

    Returns:
        Decoded payload dict if valid, None if invalid/expired
    """
    try:
        payload = jwt.decode(token, JWT_SECRET_KEY, algorithms=[JWT_ALGORITHM])
        return payload
    except JWTError as e:
        logger.debug(f"Token decode failed: {e}")
        return None


async def get_current_user_id(
    credentials: HTTPAuthorizationCredentials = Depends(security),
) -> str:
    """
    FastAPI dependency to extract and validate user_id from JWT token.

    Usage:
        @router.post("/endpoint")
        async def endpoint(user_id: str = Depends(get_current_user_id)):
            # user_id is now available and validated
            pass

    Args:
        credentials: HTTP Authorization header credentials (injected by FastAPI)

    Returns:
        The user UUID extracted from the token

    Raises:
        HTTPException: 401 if token is missing, invalid, or expired
    """
    token = credentials.credentials
    payload = decode_token(token)

    if not payload:
        raise HTTPException(
            status_code=401,
            detail="Invalid or expired token",
            headers={"WWW-Authenticate": "Bearer"},
        )

    user_id = payload.get("sub")
    if not user_id:
        raise HTTPException(
            status_code=401,
            detail="Token missing user information",
            headers={"WWW-Authenticate": "Bearer"},
        )

    return user_id


SUPERADMIN_EMAIL = os.getenv("SUPERADMIN_EMAIL", "").strip()


def is_superadmin_user(user_id: str) -> bool:
    """Return True if `user_id` belongs to the configured superadmin email."""
    if not SUPERADMIN_EMAIL:
        return False
    from db import get_user

    user = get_user(user_id)
    return bool(user) and user.get("email", "") == SUPERADMIN_EMAIL


async def require_superadmin(
    credentials: HTTPAuthorizationCredentials = Depends(security),
) -> str:
    """
    FastAPI dependency that requires the caller to be a superadmin.

    Extracts the email from the JWT and checks it against SUPERADMIN_EMAIL.
    Returns the user UUID if authorized.

    Raises:
        HTTPException: 401 if token invalid, 403 if not superadmin
    """
    user_id = await get_current_user_id(credentials)

    payload = decode_token(credentials.credentials)
    email = payload.get("email", "")
    if not SUPERADMIN_EMAIL or email != SUPERADMIN_EMAIL:
        raise HTTPException(status_code=403, detail="Superadmin access required")

    return user_id


@dataclass
class OrgContext:
    """Resolved multi-tenant context for an authenticated request.

    `user_id`  — the JWT subject (who is making the request; used for audit /
                 created_by stamping).
    `org_uuid` — the org all queries should be scoped to. Resolved from the
                 `X-Org-UUID` request header if present; otherwise falls back
                 to the caller's personal org (auto-provisioned at signup).
    `role`     — the caller's role inside `org_uuid` ('owner' or 'admin').
                 Both have full access; surfaced for endpoints that branch on
                 it (e.g. the future "transfer ownership" UI).
    """

    user_id: str
    org_uuid: str
    role: str


async def get_current_org(
    credentials: HTTPAuthorizationCredentials = Depends(security),
    x_org_uuid: Optional[str] = Header(default=None, alias="X-Org-UUID"),
) -> OrgContext:
    """Resolve `(user_id, org_uuid, role)` for the current request.

    - With `X-Org-UUID` header: verify the caller is an active member of that
      org. If not, 404 (existence-leak parity).
    - Without the header: fall back to the caller's personal org. This keeps
      pre-multi-tenant frontends working without changes — they always
      operate inside the personal workspace which contains everything the
      user owned before the migration.
    """
    user_id = await get_current_user_id(credentials)

    payload = decode_token(credentials.credentials) or {}
    is_superadmin = (
        bool(SUPERADMIN_EMAIL) and payload.get("email", "") == SUPERADMIN_EMAIL
    )

    # Import lazily to dodge the circular dependency (db imports nothing from
    # auth_utils, but routers import both — keeping db out of module-load
    # makes the dep graph robust against future moves).
    from db import get_member_role, get_personal_org_for_user, get_organization

    if x_org_uuid:
        role = get_member_role(x_org_uuid, user_id)
        if role is None:
            # Superadmin bypass: grant owner-level access to any existing org
            # without requiring membership.
            if is_superadmin and get_organization(x_org_uuid) is not None:
                return OrgContext(user_id=user_id, org_uuid=x_org_uuid, role="owner")
            raise HTTPException(status_code=404, detail="Organization not found")
        return OrgContext(user_id=user_id, org_uuid=x_org_uuid, role=role)

    personal = get_personal_org_for_user(user_id)
    if personal is None:
        # Should never happen post-PR-1 backfill, but fall back gracefully.
        raise HTTPException(
            status_code=500,
            detail="No personal organization for user; backfill missing",
        )
    return OrgContext(user_id=user_id, org_uuid=personal["uuid"], role="owner")


# Optional dependency that doesn't require authentication
# Useful for endpoints that work with or without auth
async def get_optional_user_id(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(
        HTTPBearer(auto_error=False)
    ),
) -> Optional[str]:
    """
    FastAPI dependency to optionally extract user_id from JWT token.

    Returns None if no token is provided or token is invalid.
    Useful for endpoints that should work for both authenticated and anonymous users.

    Args:
        credentials: Optional HTTP Authorization header credentials

    Returns:
        The user UUID if token is valid, None otherwise
    """
    if not credentials:
        return None

    payload = decode_token(credentials.credentials)
    if not payload:
        return None

    return payload.get("sub")


# ---------------------------------------------------------------------------
# API keys (programmatic API access)
# ---------------------------------------------------------------------------
#
# Keys are prefixed `sk_` so they can coexist with JWTs in the same
# `Authorization: Bearer …` header (JWTs are dotted base64, never start with
# `sk_`), and are also accepted via the dedicated `X-API-Key` header. The
# raw key is returned exactly once at creation; the DB keeps only a bcrypt hash
# plus the first `API_KEY_PREFIX_LEN` chars for candidate lookup.

API_KEY_PREFIX = "sk_"
API_KEY_PREFIX_LEN = 12  # chars of the raw key stored verbatim for lookup
_API_KEY_RANDOM_BYTES = 32

# Optional bearer so a missing/!=API-key Authorization header doesn't 401 here
# — these endpoints fall back to JWT/anonymous handling elsewhere.
_optional_bearer = HTTPBearer(auto_error=False)


def generate_api_key() -> tuple[str, str]:
    """Return `(raw_key, key_prefix)`. The raw key is shown to the user once."""
    raw = API_KEY_PREFIX + secrets.token_urlsafe(_API_KEY_RANDOM_BYTES)
    return raw, raw[:API_KEY_PREFIX_LEN]


def hash_api_key(raw_key: str) -> str:
    """bcrypt-hash a raw API key for storage."""
    return bcrypt.hashpw(raw_key.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")


def _resolve_api_key(raw_key: str) -> Optional[OrgContext]:
    """Validate a presented raw key → OrgContext, or None if it doesn't match.

    API keys grant owner-level access within the org they're scoped to (they can
    do anything their creator could). `last_used_at` is touched on success.
    """
    from db import (
        find_active_api_keys_by_prefix,
        get_organization,
        touch_api_key_last_used,
    )

    prefix = raw_key[:API_KEY_PREFIX_LEN]
    for row in find_active_api_keys_by_prefix(prefix):
        if bcrypt.checkpw(raw_key.encode("utf-8"), row["key_hash"].encode("utf-8")):
            # A revoked org (soft-deleted) invalidates its keys.
            if get_organization(row["org_uuid"]) is None:
                return None
            touch_api_key_last_used(row["uuid"])
            return OrgContext(
                user_id=row["owner_user_id"],
                org_uuid=row["org_uuid"],
                role="owner",
            )
    return None


async def get_org_jwt_or_api_key(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(_optional_bearer),
    x_api_key: Optional[str] = Header(default=None, alias="X-API-Key"),
    x_org_uuid: Optional[str] = Header(default=None, alias="X-Org-UUID"),
) -> OrgContext:
    """Resolve an OrgContext from EITHER an `sk_` API key OR a JWT.

    Lets a single endpoint serve both the frontend (JWT, optionally with an
    `X-Org-UUID`) and programmatic clients (an API key via `X-API-Key` or an
    `sk_`-prefixed bearer). The API key wins when present. Raises 401 if neither path yields a
    valid context — i.e. this is a *required*-auth dependency; unlike the
    JWT-only `get_current_org`, it just additionally accepts keys.
    """
    raw_key: Optional[str] = None
    if x_api_key and x_api_key.strip():
        raw_key = x_api_key.strip()
    elif credentials and credentials.credentials.startswith(API_KEY_PREFIX):
        raw_key = credentials.credentials.strip()

    if raw_key:
        ctx = _resolve_api_key(raw_key)
        if ctx is None:
            raise HTTPException(
                status_code=401,
                detail="Invalid or revoked API key",
                headers={"WWW-Authenticate": "Bearer"},
            )
        return ctx

    # No credentials at all → 403, matching FastAPI's HTTPBearer(auto_error=True)
    # default used by get_current_org elsewhere (a present-but-bad JWT 401s below).
    if credentials is None:
        raise HTTPException(
            status_code=403,
            detail="Not authenticated",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return await get_current_org(credentials=credentials, x_org_uuid=x_org_uuid)
