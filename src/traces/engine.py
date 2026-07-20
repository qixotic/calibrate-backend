"""SQLAlchemy engine and sessions for the traces store.

Traces are machine-ingested at production rates, so they get their own
database (by default a dedicated SQLite file next to pense.db): SQLite
serializes all writers on one file-level lock, and a trace stream sharing
pense.db's lock would degrade every interactive endpoint. The engine is built
from TRACES_DATABASE_URL so the store can move to Postgres by swapping the
DSN — everything in this package must stay portable across both dialects
(JSON vs JSONB, partial indexes, no dialect-only SQL outside marked spots).
"""

import os
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator, Optional

from sqlalchemy import create_engine, event
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker

_engine: Optional[Engine] = None
_session_factory: Optional[sessionmaker] = None


def get_traces_database_url() -> str:
    """Resolve the traces DSN: TRACES_DATABASE_URL, else traces.db under DB_ROOT_DIR."""
    url = os.getenv("TRACES_DATABASE_URL")
    if url:
        return url
    return f"sqlite:///{Path(os.getenv('DB_ROOT_DIR', '.')) / 'traces.db'}"


def _set_sqlite_pragmas(dbapi_connection, connection_record):
    # WAL keeps readers from blocking on the single writer; busy_timeout makes
    # a second writer wait instead of failing instantly with "database is
    # locked". Applied per-connection; journal_mode itself is persistent.
    cursor = dbapi_connection.cursor()
    cursor.execute("PRAGMA journal_mode=WAL")
    cursor.execute("PRAGMA busy_timeout=5000")
    cursor.execute("PRAGMA synchronous=NORMAL")
    cursor.close()


def get_traces_engine() -> Engine:
    global _engine, _session_factory
    if _engine is None:
        url = get_traces_database_url()
        connect_args = {}
        if url.startswith("sqlite"):
            # FastAPI runs sync handlers in a threadpool; pooled connections
            # can be handed to a different thread than the one that made them.
            connect_args["check_same_thread"] = False
        engine = create_engine(url, connect_args=connect_args)
        if engine.dialect.name == "sqlite":
            event.listens_for(engine, "connect")(_set_sqlite_pragmas)
        _engine = engine
        _session_factory = sessionmaker(bind=engine, expire_on_commit=False)
    return _engine


@contextmanager
def traces_session() -> Iterator[Session]:
    """Session context manager mirroring db.get_db_connection() ergonomics:
    commit on clean exit, rollback on exception, always close."""
    get_traces_engine()
    session = _session_factory()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()
