"""Run Alembic migrations for the traces store.

Called at app startup (main.py lifespan) and by the test-session fixture — the
same slot init_db() occupies for pense.db. The Config is built programmatically
so no ini file is read at runtime; src/traces/alembic.ini exists only for the
Alembic CLI (revision --autogenerate).
"""

from pathlib import Path

from alembic import command
from alembic.config import Config

from traces.engine import get_traces_database_url


def run_traces_migrations() -> None:
    cfg = Config()
    cfg.set_main_option(
        "script_location", str(Path(__file__).resolve().parent / "migrations")
    )
    # Config is a configparser: a literal % in a DSN must be doubled or it's
    # misread as interpolation.
    cfg.set_main_option("sqlalchemy.url", get_traces_database_url().replace("%", "%%"))
    command.upgrade(cfg, "head")
