"""Self-contained persistence for production traces.

Traces live in their own database behind SQLAlchemy + Alembic, deliberately
separate from the raw-sqlite3 pense.db layer in db.py — see engine.py for why.
Routers import from traces.store; nothing outside this package touches its
engine, sessions, or models.
"""
