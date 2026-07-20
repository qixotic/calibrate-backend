"""Shared pytest fixtures and environment setup for the test suite.

Sets `DB_ROOT_DIR` to a tmp dir BEFORE importing anything from `src/`,
because `db.py` resolves `DB_PATH` at import time. Also seeds JWT/S3
env vars that are required for assorted module-level reads.
"""

import os
import sys
import tempfile
from pathlib import Path

# Valid 36-char UUID v4 placeholders for tests (min_length=36 on API models).
NONEXISTENT_UUID = "00000000-0000-4000-8000-000000000001"
NONEXISTENT_UUID_2 = "00000000-0000-4000-8000-000000000002"
EXAMPLE_UUID = "f47ac10b-58cc-4372-a567-0e02b2c3d479"
EXAMPLE_UUID_2 = "6ba7b810-9dad-11d1-80b4-00c04fd430c8"
EXAMPLE_VERSION_UUID = "6ba7b811-9dad-11d1-80b4-00c04fd430c8"

# These must be set before any `src/` module is imported.
_TEST_DB_ROOT = tempfile.mkdtemp(prefix="pense-test-db-")
os.environ.setdefault("DB_ROOT_DIR", _TEST_DB_ROOT)
os.environ.setdefault("JWT_SECRET_KEY", "test-secret-key-for-unit-tests-32-chars-min")
os.environ.setdefault("JWT_EXPIRATION_HOURS", "1")
os.environ.setdefault("S3_OUTPUT_BUCKET", "test-bucket")
os.environ.setdefault("MAX_CONCURRENT_JOBS", "1")
os.environ.setdefault("MAX_CONCURRENT_JOBS_PER_ORG", "1")
os.environ.setdefault("DEFAULT_MAX_ROWS_PER_EVAL", "20")
os.environ.setdefault("DEFAULT_MAX_TRACES", "50000")
os.environ.setdefault("SUPERADMIN_EMAIL", "admin@example.com")
# Pin docs creds so a developer's local src/.env (which overrides via load_dotenv)
# doesn't change the basic-auth check that test_docs_endpoints_require_basic_auth
# relies on. setdefault is intentional — explicit env wins.
os.environ.setdefault("DOCS_USERNAME", "admin")
os.environ.setdefault("DOCS_PASSWORD", "changeme")

_SRC = str(Path(__file__).resolve().parents[1] / "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

import pytest  # noqa: E402

import db  # noqa: E402
from traces.migrate import run_traces_migrations  # noqa: E402


@pytest.fixture(scope="session", autouse=True)
def initialized_db():
    """Initialize the schema once per test session.

    Tests share the schema but each test that mutates rows uses its own
    UUIDs so there's no cross-test contamination via row collisions.
    """
    db.init_db()
    run_traces_migrations()
    yield
