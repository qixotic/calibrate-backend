"""Wakeup signal from ingest to the trace-scoring worker pool.

A standalone module so `db.py` (sync, after commit) and the asyncio worker can
both import it without a cycle.

Exposed as functions, not a module-level `asyncio.Event()`. Event binds to
whichever loop first calls `.wait()` on it and raises `RuntimeError` if a later
`.wait()` arrives from a different loop. Production starts the lifespan once,
but the test suite starts many `TestClient`s — each with its own loop — over
one process lifetime. `reset()` at each lifespan start binds a fresh Event.
"""

from __future__ import annotations

import asyncio
from typing import Optional

_event: Optional[asyncio.Event] = None


def reset() -> asyncio.Event:
    """Bind a fresh Event to the current context. Call once per lifespan start."""
    global _event
    _event = asyncio.Event()
    return _event


def get() -> asyncio.Event:
    """Return the current Event, creating one if no lifespan has reset it yet."""
    global _event
    if _event is None:
        _event = asyncio.Event()
    return _event


def set() -> None:
    get().set()
