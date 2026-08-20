"""Shared wakeup signal between trace ingestion and the trace-scoring worker
pool. A standalone module with no other imports, so db.py (called from a sync
context) and workers/trace_scoring.py (an asyncio consumer) can both import it
without creating a circular import between them.

Exposed as functions, not a bare module-level `asyncio.Event()`, because
`Event` binds to whatever running loop first calls `.wait()` on it and raises
`RuntimeError` if a later `.wait()` arrives from a different loop. A single
process only ever starts the FastAPI lifespan once in production, but any
test process that spins up multiple `TestClient` instances runs a fresh event
loop per lifespan -- `reset()` must be called at the start of each one so the
Event is never reused across loops.
"""

import asyncio
from typing import Optional

_event: Optional[asyncio.Event] = None


def reset() -> asyncio.Event:
    """Bind a fresh Event to the current context. Call once per app lifespan
    startup, before anything calls wait()/set()."""
    global _event
    _event = asyncio.Event()
    return _event


def get() -> asyncio.Event:
    """Return the current Event, creating one if no lifespan has reset it
    yet (e.g. a unit test calling db.create_trace_with_eval_queue directly,
    with no worker pool around to wait on it)."""
    global _event
    if _event is None:
        _event = asyncio.Event()
    return _event


def set() -> None:
    get().set()
