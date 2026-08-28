"""Bounded trace-scoring worker pool, started from the FastAPI lifespan.

Not `BackgroundTasks` (unbounded, request-lifecycle-bound) and not
`MAX_CONCURRENT_JOBS_PER_ORG` (defaults to 1; a backfill would block an org's
live scoring). One worker, one subprocess, bounded batch, explicit CLI
`--parallel` via `claim_and_score_batch`. Subprocess work runs in a thread so
it never blocks the event loop.
"""

from __future__ import annotations

import asyncio
import logging
from typing import List, Optional

import trace_scoring_nudge
from trace_scoring import (
    CLAIM_BATCH_SIZE,
    CLAIM_LEASE_SECONDS,
    claim_and_score_batch,
)
from utils import capture_exception_to_sentry

logger = logging.getLogger(__name__)

# Conservative defaults — not env vars. Batch size / lease / `--parallel` are
# the claim-engine constants; this module only owns pool size and idle poll.
POOL_SIZE = 1
POLL_SECONDS = 5.0
_ERROR_BACKOFF_SECONDS = 1.0

_pool_enabled = True
_active_pool: Optional["TraceScoringPool"] = None


def set_pool_enabled(enabled: bool) -> None:
    """Test seam: the suite disables the pool so ingest assertions are not raced."""
    global _pool_enabled
    _pool_enabled = enabled


def _run_batch() -> list:
    return claim_and_score_batch(
        batch_size=CLAIM_BATCH_SIZE,
        lease_seconds=CLAIM_LEASE_SECONDS,
    )


async def _worker_loop(
    worker_id: int, stop_event: asyncio.Event, nudge: asyncio.Event
) -> None:
    while not stop_event.is_set():
        try:
            claimed = await asyncio.to_thread(_run_batch)
        except Exception as exc:
            logger.exception("trace-scoring worker %s failed", worker_id)
            capture_exception_to_sentry(exc)
            try:
                await asyncio.wait_for(
                    stop_event.wait(), timeout=_ERROR_BACKOFF_SECONDS
                )
            except asyncio.TimeoutError:
                pass
            continue
        if stop_event.is_set():
            break
        if claimed:
            continue
        try:
            await asyncio.wait_for(nudge.wait(), timeout=POLL_SECONDS)
        except asyncio.TimeoutError:
            pass
        else:
            nudge.clear()


class TraceScoringPool:
    """Owns the worker tasks; started and stopped from the app lifespan."""

    def __init__(self, size: int = POOL_SIZE):
        self._size = size
        self._stop_event: Optional[asyncio.Event] = None
        self._tasks: List[asyncio.Task] = []
        self._leases = 0

    @property
    def is_running(self) -> bool:
        return bool(self._tasks) and not all(t.done() for t in self._tasks)

    def start(self) -> None:
        if not _pool_enabled:
            return
        if self.is_running:
            return
        # Fresh Event per lifespan: asyncio.Event binds to the loop that first
        # waits on it and raises if a later wait arrives from a different loop.
        nudge = trace_scoring_nudge.reset()
        self._stop_event = asyncio.Event()
        self._tasks = [
            asyncio.create_task(
                _worker_loop(i, self._stop_event, nudge),
                name=f"trace-scoring-{i}",
            )
            for i in range(self._size)
        ]
        logger.info("trace-scoring started %s worker(s)", self._size)

    async def shutdown(self) -> None:
        if self._stop_event is not None:
            self._stop_event.set()
        # Wake idle waiters so they see stop without waiting out the poll.
        trace_scoring_nudge.set()
        if self._tasks:
            await asyncio.gather(*self._tasks, return_exceptions=True)
        self._tasks = []
        self._stop_event = None
        logger.info("trace-scoring workers stopped")


def start_trace_scoring_pool() -> TraceScoringPool:
    """Start the process-wide pool, or join one already running.

    Overlapping TestClient lifespans share one pool so a second start cannot
    spawn a second set of workers against the same DB.
    """
    global _active_pool
    if _active_pool is not None:
        _active_pool._leases += 1
        return _active_pool
    pool = TraceScoringPool()
    pool._leases = 1
    pool.start()
    _active_pool = pool
    return pool


async def shutdown_trace_scoring_pool(pool: TraceScoringPool) -> None:
    """Release one lifespan lease; stop workers only when the last lease drops."""
    global _active_pool
    pool._leases = max(0, pool._leases - 1)
    if pool._leases > 0:
        return
    await pool.shutdown()
    if _active_pool is pool:
        _active_pool = None
