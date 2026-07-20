"""Alembic environment for the traces store only — never point this at pense.db."""

import sys
from pathlib import Path

from alembic import context
from sqlalchemy import create_engine

# For programmatic runs (traces.migrate builds a Config without reading
# alembic.ini, so its prepend_sys_path never applies).
_SRC = str(Path(__file__).resolve().parents[2])
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from traces.engine import get_traces_database_url  # noqa: E402
from traces.models import TracesBase  # noqa: E402

config = context.config
target_metadata = TracesBase.metadata

# Own version table so the store can later share a Postgres database with
# other schemas without version-table collisions.
VERSION_TABLE = "alembic_version_traces"


def _url() -> str:
    return config.get_main_option("sqlalchemy.url") or get_traces_database_url()


def run_migrations_offline() -> None:
    context.configure(
        url=_url(),
        target_metadata=target_metadata,
        literal_binds=True,
        render_as_batch=True,
        version_table=VERSION_TABLE,
    )
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    engine = create_engine(_url())
    with engine.connect() as connection:
        context.configure(
            connection=connection,
            target_metadata=target_metadata,
            compare_type=True,
            # SQLite can't ALTER in place; batch mode recreates the table.
            render_as_batch=True,
            version_table=VERSION_TABLE,
        )
        with context.begin_transaction():
            context.run_migrations()
    engine.dispose()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
