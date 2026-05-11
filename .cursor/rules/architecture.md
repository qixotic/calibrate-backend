---
description: "Calibrate Backend Architecture - AI Agent Testing & Evaluation Platform"
alwaysApply: true
---

# Calibrate Backend Architecture

## Project Overview

**Calibrate Backend** is a FastAPI-based REST API that serves as the backend for an AI agent testing and evaluation platform. It provides capabilities for:

1. **Speech-to-Text (STT) Evaluation** - Benchmark multiple STT providers against ground truth transcriptions
2. **Text-to-Speech (TTS) Evaluation** - Benchmark multiple TTS providers for quality metrics
3. **LLM Agent Testing** - Run unit tests and benchmarks on LLM-based agents
4. **Voice/Chat Simulations** - Run simulated conversations between AI agents and personas across various scenarios

The backend wraps the `calibrate` CLI tool and orchestrates evaluation jobs while providing a RESTful API interface.

---

## Technology Stack

| Component            | Technology                          | Purpose                                                                                                                                                                      |
| -------------------- | ----------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Framework**        | FastAPI                             | Async REST API framework                                                                                                                                                     |
| **Database**         | SQLite                              | Persistent data storage                                                                                                                                                      |
| **Storage**          | AWS S3                              | File/result storage                                                                                                                                                          |
| **Authentication**   | Google OAuth + Email/Password + JWT | User authentication via Google ID tokens or email/password credentials, API access via JWT. API docs (`/docs`, `/redoc`, `/openapi.json`) are protected with HTTP Basic Auth |
| **Monitoring**       | Sentry                              | Error tracking and performance monitoring                                                                                                                                    |
| **Tracing**          | Langfuse (via OTEL)                 | LLM observability and tracing                                                                                                                                                |
| **Package Manager**  | uv                                  | Python dependency management                                                                                                                                                 |
| **Containerization** | Docker                              | Deployment                                                                                                                                                                   |
| **CLI Tool**         | calibrate                           | Core evaluation/simulation engine                                                                                                                                            |

---

## Project Structure

```
calibrate-backend/
├── src/
│   ├── main.py              # FastAPI app entry point, lifespan management
│   ├── db.py                # SQLite database layer (~5000 lines)
│   ├── utils.py             # Shared utilities (S3 client/upload helpers, STT/TTS evaluator run pairing, tool config building)
│   ├── dataset_utils.py     # Dataset resolution helpers for STT/TTS evaluations
│   ├── job_recovery.py      # Restart in-progress jobs on app startup
│   └── routers/
│       ├── auth.py          # Authentication (Google OAuth, username/password signup & login)
│       ├── users.py         # User management endpoints
│       ├── agents.py        # Agent CRUD operations
│       ├── tools.py         # Tool CRUD operations
│       ├── agent_tools.py   # Agent-Tool relationship management
│       ├── tests.py         # Test case CRUD and bulk upload operations
│       ├── agent_tests.py   # Agent test execution & benchmarking
│       ├── personas.py      # Persona CRUD operations
│       ├── scenarios.py     # Scenario CRUD operations
│       ├── evaluators.py    # Evaluator CRUD, invoke, default prompts
│       ├── simulations.py   # Simulation orchestration (chat/voice)
│       ├── datasets.py      # Dataset CRUD and item management
│       ├── stt.py           # STT provider evaluation
│       ├── tts.py           # TTS provider evaluation
│       ├── jobs.py          # Job listing API (STT/TTS eval jobs)
│       ├── public.py        # Share-token read-only endpoints (STT/TTS/tests/benchmarks/simulations)
│       ├── api_keys.py      # API key CRUD (evaluator invoke / automation)
│       └── user_limits.py   # Per-user limits CRUD and limit queries
├── db/
│   └── calibrate.db         # SQLite database file
├── pyproject.toml           # Python project configuration
├── Dockerfile               # Container build configuration
└── docker-compose.yml       # Container orchestration
```

---

## Database Schema

### Entity Relationship

```
users
  ├── agents (user_id FK)
  ├── tools (user_id FK)
  ├── tests (user_id FK)
  ├── personas (user_id FK)
  ├── scenarios (user_id FK)
  ├── metrics (user_id FK)
  ├── evaluators (owner_user_id FK, nullable for seeded defaults)
  ├── api_keys (user_id FK)
  ├── simulations (user_id FK)
  ├── datasets (user_id FK)
  ├── jobs (user_id FK)
  └── user_limits (user_id FK, UNIQUE)

datasets
  └── dataset_items (dataset_id FK → datasets.uuid)

agents
  ├── agent_tools (many-to-many with tools)
  ├── agent_tests (many-to-many with tests)
  ├── agent_test_jobs
  └── simulations (agent_id FK)

simulations
  ├── simulation_personas (many-to-many with personas)
  ├── simulation_scenarios (many-to-many with scenarios)
  ├── simulation_evaluators (many-to-many: simulations ↔ evaluator versions)
  └── simulation_jobs
```

### Core Tables

| Table                               | Purpose                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 |
| ----------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `users`                             | User accounts (Google OAuth or email/password credentials)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              |
| `agents`                            | AI agent configurations; `type` column distinguishes `agent` (platform-managed) from `connection` (external HTTP endpoint)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              |
| `tools`                             | Tool/function definitions for agents                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    |
| `tests`                             | Test cases with evaluation criteria                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     |
| `personas`                          | Simulated user personas (characteristics, gender, language)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             |
| `scenarios`                         | Conversation scenarios/contexts                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         |
| `metrics`                           | Legacy evaluation criteria rows (frozen for old data); **new work uses `evaluators` + `evaluator_versions`** and pivots (`simulation_evaluators`, `test_evaluators`, STT/TTS job `details.evaluators`). Migration assigns **new** evaluator UUIDs (`evaluators.source_metric_uuid` stores the old `metrics.uuid`); legacy metrics actively linked to simulations become **`evaluator_type='simulation'`** evaluators whose version prompt is the canonical simulation default prompt with `<ENTER EVALUATION CRITERIA HERE>` replaced by the metric description/name. APIs resolve legacy ids only to produce **400** hints, not to accept them as links.                                                                                                                                                               |
| `evaluators` / `evaluator_versions` | User-defined and seeded judges (prompt, rubric, judge model); versions pin rubric per run                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               |
| `simulations`                       | Simulation configurations linking agents, personas, scenarios, and evaluators (via `simulation_evaluators`)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             |
| `datasets`                          | Named collections of evaluation inputs (STT or TTS), user_id FK                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         |
| `dataset_items`                     | Individual items within a dataset (text, optional audio_path, `updated_at` — nullable for pre-migration rows)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           |
| `jobs`                              | Generic STT/TTS evaluation jobs (user_id FK to users)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   |
| `agent_test_jobs`                   | LLM unit test and benchmark jobs                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        |
| `simulation_jobs`                   | Chat/voice simulation jobs                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              |
| `user_limits`                       | Per-user limits/quotas as a JSON blob (one row per user, `user_id` UNIQUE). No soft delete — hard delete only. The `limits` JSON is validated at the API layer via the `UserLimits` Pydantic model in `user_limits.py` (with `Field(gt=0, le=10000)` constraints); currently only `max_rows_per_eval` is permitted. The `create_user_limits` endpoint validates user existence explicitly via `get_user()` (returns 404) because SQLite FK constraints are not enforced (`PRAGMA foreign_keys` is off), and handles race conditions via `sqlite3.IntegrityError` catch (returns 409). `update_user_limits()` in `db.py` returns the updated row directly (update + select in one connection) to avoid extra round-trips. DB functions accept `UserLimits` type (imported via `TYPE_CHECKING` to avoid circular imports) |
| `api_keys`                          | Per-user API keys for **`POST /evaluators/{uuid}/invoke`** (and similar automation). Stores bcrypt hash + short `key_prefix` for lookup; raw key never persisted. Soft-deleted rows excluded from lookup                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                |

**Seeded default evaluators** (`db.py`): `init_db()` runs **`_seed_default_evaluators()`**, which walks **`DEFAULT_EVALUATORS_SEED`** (stable `slug`s such as `default-llm-next-reply`, `default-stt-transcription`, `default-faithfulness`, simulation defaults, etc.). It is **idempotent** on every startup: missing rows are created; existing rows get **`evaluators` metadata** (`name`, **`description`**, `evaluator_type`, …) **UPDATE**d when the seed text differs; if the **live** `evaluator_versions` row’s prompt/rubric/`variables` no longer matches the seed, a **new version** is inserted and promoted to **live** so pinned links stay reproducible. **`DEFAULT_PROMPTS_BY_PURPOSE`** stays aligned with **`GET /evaluators/default-prompt`** for LLM/STT/TTS/simulation prefills. **Copy convention**: the seed’s **evaluator-level `description`** (the short blurb in evaluator lists) intentionally **omits a trailing full stop**; nested rubric lines under **`output_config.scale`** may still end with a period where the judge wording calls for it.

**Evaluator name uniqueness**: Custom evaluator create, update, and duplicate requests validate `name` with `evaluator_name_exists()` before writing. Names must be unique within the current user's visible evaluator namespace: the user's own non-deleted evaluators plus seeded defaults (`owner_user_id IS NULL`). Different users can reuse the same custom name, but a user cannot create or rename a custom evaluator to collide with a seeded default or another custom evaluator they own. Collisions return **409** from `routers/evaluators.py`. There is no database-level unique constraint for this because soft deletes and default visibility make the namespace user-relative.

**Evaluator metadata on job results**: Jobs snapshot stable evaluator linkage and pinned-version execution metadata at submission time, but evaluator descriptions are **not** snapshotted for result display. Result APIs resolve descriptions from the current `evaluators.description` row every time so public/private result pages see the live evaluator description without extra frontend calls. STT/TTS jobs store hydrated evaluator dicts in `jobs.details.evaluators` with `uuid`, `name`, type fields, pinned version id, prompt/model/rubric, and variable values; API responses enrich `provider_results[].evaluator_runs[]` with the current DB `description`. LLM unit-test and benchmark jobs store `agent_test_jobs.details.evaluators_by_test_id`, where each snapshot entry includes `uuid`, calibrate `name`, `output_type`, `variable_values`, and rating bounds; `judge_results[].description` is resolved live from DB. **Benchmark-only (`llm-benchmark`)**: when each model’s calibrate `metrics.json` is present, the job’s stored `results.model_results[].evaluator_summary` lists per-evaluator aggregates copied from `metrics.json.criteria` (binary: `passed` / `total` / `pass_rate`; rating: `mean` / `min` / `max` / `count` / `scale_min` / `scale_max`), keyed by the same metric keys the CLI uses. At read time, `_enrich_evaluator_summary()` refreshes each entry’s `name` and `description` from the current `evaluators` row when `evaluator_uuid` is present — same “live label” pattern as per-row `judge_results`. Both row-level and summary enrichment share **`_get_evaluator_cached_for_enrichment()`** in `routers/agent_tests.py`: a mutable **`evaluator_cache: Dict[str, Optional[dict]]`** (same idea as simulation’s **`current_by_uuid`** in **`apply_simulation_job_evaluator_enrichment()`**) so a single HTTP response performs **at most one `get_evaluator` SQLite read per distinct evaluator UUID**, regardless of how many `test_results` rows or benchmark models repeat that UUID (important for frontend polling). **`GET /agent-tests/agent/{uuid}/runs`** and **`GET /agent-tests/runs`** allocate **one cache dict per listed job** and pass it into **`_enrich_test_results_with_evaluators`** (top-level unit-test `test_results`) **and** **`_enrich_model_results_with_evaluators`** (nested benchmark `model_results`) so the same UUID is not loaded twice across those two shapes. Simulation jobs store `simulation_jobs.details.evaluators` as ordered `{uuid, name}` refs; `SimulationEvaluatorRef.description` and per-row `evaluation_results[].description` are resolved live from DB. Older jobs or deleted evaluators may yield nullable descriptions.

### Users Table Schema

The `users` table supports two authentication methods. Columns:

| Column          | Type      | Notes                                                              |
| --------------- | --------- | ------------------------------------------------------------------ |
| `id`            | INTEGER   | Auto-increment primary key                                         |
| `uuid`          | TEXT      | Unique user identifier used across the app                         |
| `first_name`    | TEXT      | NOT NULL                                                           |
| `last_name`     | TEXT      | NOT NULL                                                           |
| `email`         | TEXT      | NOT NULL UNIQUE. Used as the user identifier for both auth methods |
| `password_hash` | TEXT      | Nullable. bcrypt hash. Only set for email/password users           |
| `created_at`    | TIMESTAMP | Default CURRENT_TIMESTAMP                                          |
| `updated_at`    | TIMESTAMP | Default CURRENT_TIMESTAMP                                          |

**Two auth paths for the same table**: The `email` column is the shared identifier. Google OAuth users have `password_hash` as NULL (authenticated via Google tokens). Email/password users have `password_hash` set (authenticated via bcrypt). Both use the same `email` column — there is no separate `username` column.

### Design Patterns

1. **Soft Deletes**: All entity tables use `deleted_at` timestamp for soft deletion
2. **UUIDs**: All entities use UUID as primary identifier (separate from auto-increment id)
3. **JSON Config**: Complex configurations stored as JSON strings in `config` columns
4. **Pivot Tables**: Many-to-many relationships use dedicated pivot tables with soft delete support
5. **Parent `updated_at` cascade**: When child rows are mutated the parent's `updated_at` is bumped in the same transaction. Currently applies to `datasets` ← `dataset_items` (add, update, delete) and `tests` ← `test_evaluators` (replace evaluator links via `set_test_evaluators()`, including evaluator-only `PUT /tests/{uuid}`)
6. **Schema migrations**: New columns on existing tables are added via `ALTER TABLE ADD COLUMN` wrapped in `try/except sqlite3.OperationalError: pass` inside `init_db()`. **Gotcha**: SQLite does not allow `DEFAULT CURRENT_TIMESTAMP` (or any non-constant expression) in `ALTER TABLE ADD COLUMN` — the statement silently fails and the `except` swallows it. Always use `DEFAULT NULL` for migration `ADD COLUMN` statements. The `CREATE TABLE` definition can still use `DEFAULT CURRENT_TIMESTAMP` for new databases.

---

## API Architecture

### Router Organization

Each router follows a consistent pattern:

- Pydantic models for request/response validation
- CRUD endpoints following REST conventions
- Background job execution for long-running tasks

### API Endpoints

#### Authentication

- `POST /auth/google` - Google OAuth login (returns JWT access token)
- `POST /auth/signup` - Register with first name, last name, email, and password (returns JWT access token)
- `POST /auth/login` - Login with email and password (returns JWT access token)

All three auth endpoints return the same `LoginResponse` format (`access_token`, `token_type`, `user`, `message`).

**JWT Token Usage**: After login/signup, all API requests must include the JWT token in the Authorization header:

```
Authorization: Bearer <jwt_token>
```

The JWT token contains the user's UUID and is validated on every protected endpoint.

#### Entity Management (CRUD)

- `/agents` - Agent management
- `/tools` - Tool definitions
- `/tests` - Test cases (CRUD + bulk upload + bulk delete)
- `/personas` - User personas
- `/scenarios` - Conversation scenarios
- `/evaluators` - Evaluator CRUD, default prompts (`GET /evaluators/default-prompt`). **`POST /evaluators/{uuid}/versions`** creates a new prompt/rubric version without changing `live_version_id` by default; clients can include **`make_live: true`** in that same request to promote the new version immediately, or call **`POST /evaluators/{uuid}/versions/live`** later. **`POST /evaluators/{uuid}/invoke`** uses **`get_user_from_api_key`** (API key — `X-API-Key` or `Authorization: Bearer <calib_…>`), not JWT; all other evaluator routes use JWT. Replaces the removed `/metrics` router for new work.
- `/api-keys` - Create (returns raw key once), list metadata, soft-delete keys (JWT only). Keys authenticate the evaluator invoke endpoint.
- `/simulations` - Simulation configurations (CRUD + run/status)
- `/datasets` - Dataset CRUD, item management (add/update/delete items), `eval_count` per dataset (number of linked STT/TTS eval jobs via `json_extract` on jobs `details`)
- `/users` - User management (read-only)
- `/user-limits` - Per-user limits CRUD + `/me/max-rows-per-eval` query endpoint. Mutating endpoints (`POST`, `PUT`, `DELETE`) require superadmin (`require_superadmin` dependency composes `get_current_user_id` for token validation, then checks JWT email against `SUPERADMIN_EMAIL` env var); read endpoints (`GET`) require only standard JWT auth

#### Relationship Management

- `/agent-tools` - Link/unlink tools to agents
- `/agent-tests` - Link/unlink tests to agents (single + bulk; required for benchmarks; optional for run — run can also accept explicit test_uuids)

#### Datasets

- `POST /datasets` - Create a new dataset (`name`, `dataset_type`: `stt`|`tts`)
- `GET /datasets` - List datasets for the current user (optional `?dataset_type=stt|tts` filter)
- `GET /datasets/{dataset_id}` - Get dataset with all items
- `PATCH /datasets/{dataset_id}` - Rename a dataset
- `DELETE /datasets/{dataset_id}` - Soft delete a dataset and all its items
- `POST /datasets/{dataset_id}/items` - Add items to a dataset
- `DELETE /datasets/{dataset_id}/items/{item_uuid}` - Soft delete a single item

All dataset endpoints require JWT auth. Every DB operation is scoped to the authenticated user (`user_id` is always required, never optional).

#### Evaluation & Testing (all require JWT auth)

- `GET /jobs` - List all STT/TTS evaluation jobs for authenticated user (each item includes top-level `dataset_id` and `dataset_name` extracted from job details; both are `null` when the associated dataset has been deleted)
- `DELETE /jobs/{job_uuid}` - Delete a job (kills processes, triggers next queued job)
- `POST /stt/evaluate` - Start STT evaluation task
- `GET /stt/evaluate/{task_id}` - Get STT evaluation status (includes timeout detection)
- `POST /tts/evaluate` - Start TTS evaluation task
- `GET /tts/evaluate/{task_id}` - Get TTS evaluation status (includes timeout detection)
- `POST /agent-tests/agent/{uuid}/run` - Run agent unit tests (optional `test_uuids` in body; if omitted, runs all linked tests)
- `POST /agent-tests/agent/{uuid}/benchmark` - Run multi-model benchmark (always runs all linked tests; body only requires `models` list)
- `GET /agent-tests/agent/{uuid}/runs` - List all test runs for an agent
- `GET /agent-tests/run/{task_id}` - Get test run status (includes timeout detection)
- `GET /agent-tests/benchmark/{task_id}` - Get benchmark status (includes timeout detection)
- `DELETE /agent-tests/job/{job_uuid}` - Delete an agent test job (triggers next queued job)

#### Simulations

- `GET /simulations` - List simulations for the current user
- `POST /simulations` - Create simulation (`name`, optional `agent_uuid`, `persona_uuids`, `scenario_uuids`, `evaluators`). **`SimulationCreate` / `SimulationUpdate`** and nested **`EvaluatorRef`** (defined in **`routers/simulations.py`**, distinct from **`routers/tests.py`**'s same-named model) use **`extra='forbid'`** — unknown JSON keys (e.g. `metric_uuids`, `metrics` as substitutes for evaluator links, or **`version_uuid`**) produce **422**. Each **`evaluators[]`** item is **`{ evaluator_uuid, variable_values? }`**; **`_resolve_simulation_evaluator_ref()`** pins **`evaluator_versions.uuid`** from **`evaluators.live_version_id`** only — clients cannot choose a version (**400** if there is no live version). Each link must reference **`evaluators.uuid`**. Legacy **`metrics.uuid`** values are **rejected** with **400**; the response includes the migrated evaluator UUID when `evaluators.source_metric_uuid` matches (`get_evaluator_uuid_for_legacy_metric()` in `db.py`). Only evaluators with **`evaluator_type: simulation`** are allowed. `legacy_metric_uuid_exists()` covers edge cases where a legacy `metrics` row has no migrated evaluator.
- `GET /simulations/{uuid}` - Simulation detail with linked agent, personas, scenarios, and **evaluators** (from `simulation_evaluators`). An **empty `evaluators` array** means no pivot rows — typically the client omitted `evaluators` on create, sent nonstandard property names (before strict `forbid`), or never ran a successful PUT with evaluator links.
- `PUT /simulations/{uuid}` - Update simulation; when `evaluators` is provided it **replaces** all evaluator links. Same validation as `POST`.
- `POST /simulations/{uuid}/run` - Start simulation (chat or voice). Creates `simulation_jobs` with **`details.evaluators`**: an ordered list `[{ "uuid": <evaluator_uuid>, "name": <name> }, …]` matching **`get_evaluators_for_simulation`** / the calibrate config (written by **`_snapshot_evaluators_for_job_details()`**). Used later to pair run output with evaluator rows and stable links; descriptions are resolved live from DB at response time. Unrelated to STT/TTS job `details.evaluators` shape.
- `GET /simulations/run/{task_id}` - Get simulation run status (includes timeout detection, partial results for voice)
- `POST /simulations/run/{job_uuid}/abort` - Abort a running simulation (kills process, saves partial results with abort marker, triggers next queued job); returns full `SimulationRunStatusResponse`
- `DELETE /simulations/run/{job_uuid}` - Delete a simulation job (kills process, triggers next queued job)
- `GET /simulations/{uuid}/runs` - List all runs for a simulation

#### Public Sharing

- `GET /public/evaluators/defaults?share_token={token}&types=stt,tts` - Unauthenticated, public-safe metadata for seeded default evaluators. The `share_token` must resolve to any currently public STT/TTS, agent-test, benchmark, or simulation job; invalid or private tokens return **404**. `types` is an optional comma-separated filter over `stt`, `tts`, `llm`, and `simulation`; when omitted, all seeded defaults are returned. Responses are intentionally limited to evaluator `uuid`, `name`, `description`, `evaluator_type`, `output_type`, and `live_version.output_config` so public pages can render fallback labels/rubrics without exposing prompts, judge models, owner metadata, or custom/private evaluators.

**Status API Response Fields:**

- `total_simulations` - Expected number of simulations (personas × scenarios)
- `completed_simulations` - Number of completed simulations (for in_progress text/voice simulations)
- `simulation_results` - Array of simulation results (partial for in_progress; includes both complete and in-progress simulations). Each result includes an `aborted` field (`true`/`false`/`null`) set by the abort endpoint for incomplete/complete simulations respectively; `null` for non-aborted runs.
- `metrics` - Aggregated evaluation metrics (only populated when all simulations complete)
- `evaluators` - Optional list of `SimulationEvaluatorRef`: `{ evaluator_uuid, name, description }` (**`name`** and **`description`** are resolved from the current DB row at fetch time — via **`get_evaluator()`** behind **`apply_simulation_job_evaluator_enrichment()`**’s **`current_by_uuid`** cache, one SQLite read per distinct UUID per response — so edits appear without changing stored results). Requires a job-level snapshot (`simulation_jobs.details.evaluators`). **`apply_simulation_job_evaluator_enrichment()`** (`routers/simulations.py`) attaches **`evaluator_uuid`** and current DB **`description`** to each **`evaluation_results`** row from calibrate's echoed **`evaluator_id`**.
  - **Gotcha**: CSV **`name`** strings are not rewritten when an evaluator is renamed; use **`evaluator_uuid`** for stable navigation.

#### API Documentation (HTTP Basic Auth Protected)

- `GET /docs` - Swagger UI (requires HTTP Basic Auth via `DOCS_USERNAME`/`DOCS_PASSWORD`)
- `GET /redoc` - ReDoc (requires HTTP Basic Auth)
- `GET /openapi.json` - OpenAPI schema (requires HTTP Basic Auth)

FastAPI's default docs routes are disabled (`docs_url=None`, `redoc_url=None`, `openapi_url=None`). Custom routes re-serve the same pages behind HTTP Basic Auth. Credentials are read from `DOCS_USERNAME` and `DOCS_PASSWORD` env vars (defaults: `admin`/`changeme`). Uses `secrets.compare_digest` for timing-safe comparison.

#### Utilities

- `GET /` or `HEAD /` - Health check (HEAD supported for uptime monitors)
- `POST /presigned-url` - Generate S3 presigned URL for uploads
- `GET` or `HEAD /provider-status` - Check status of all configured providers (runs `calibrate status` CLI, non-blocking via `asyncio.create_subprocess_exec`). Returns `{"success": true}` if all pass, 503 with failed provider details if any fail. HEAD supported for uptime monitors.

---

## Background Job System

### Job Types

| Job Type        | Table             | Description               |
| --------------- | ----------------- | ------------------------- |
| `stt-eval`      | `jobs`            | STT provider evaluation   |
| `tts-eval`      | `jobs`            | TTS provider evaluation   |
| `llm-unit-test` | `agent_test_jobs` | Agent unit test execution |
| `llm-benchmark` | `agent_test_jobs` | Multi-model benchmarking  |
| `chat`          | `simulation_jobs` | Chat-based simulation     |
| `voice`         | `simulation_jobs` | Voice-based simulation    |

### Job Lifecycle

1. **Request**: New job request received via API
2. **Capacity Check**: Check if `running_jobs < MAX_CONCURRENT_JOBS`
3. **Creation**: Job created with status based on capacity:
   - `in_progress` if capacity available (job starts immediately)
   - `queued` if capacity full (job waits in queue)
4. **Execution**: Background thread runs the task
5. **Process Tracking**: PIDs stored in job details for cleanup (voice simulations, STT/TTS evals only; agent tests/benchmarks use blocking `process.wait()` without PID tracking)
6. **Incremental Updates**: Results updated in DB during execution (text/voice simulations, agent unit tests, benchmarks)
7. **Timeout Detection**: Status API checks if job hasn't updated in 5+ minutes; if so, marks as failed
8. **Completion**: Job updated with `done` status and `results` JSON
9. **Queue Processing**: On completion/timeout/deletion, `try_start_queued_*_job()` starts next queued job if capacity allows
10. **Recovery**: On app startup, `job_recovery.py` kills orphaned processes, restarts `in_progress` jobs, and starts queued jobs

### Job Status Values

- `queued` - Job is waiting for capacity (FIFO order)
- `in_progress` - Job is running (may have partial results for simulations, agent unit tests, and benchmarks)
- `done` - Job completed successfully
- `failed` - Job failed (timed out or error occurred; check `results.error` for details, partial results preserved)
- `cancelled` - Job was cancelled (not currently used)

### Job Queueing System

The queueing mechanism limits concurrent jobs to prevent resource exhaustion. Two levels of limits are enforced:

1. **Global limit**: `MAX_CONCURRENT_JOBS` env var (default from docker-compose: 1)
2. **Per-user limit**: `MAX_CONCURRENT_JOBS_PER_USER` env var (default: 1, set to 0 to disable)

#### Queue Architecture

Three separate queues exist, each with its own concurrency limits:

| Queue            | Job Types                        | Global Limit          | Per-User Limit                 |
| ---------------- | -------------------------------- | --------------------- | ------------------------------ |
| Eval Queue       | `stt-eval`, `tts-eval`           | `MAX_CONCURRENT_JOBS` | `MAX_CONCURRENT_JOBS_PER_USER` |
| Agent Test Queue | `llm-unit-test`, `llm-benchmark` | `MAX_CONCURRENT_JOBS` | `MAX_CONCURRENT_JOBS_PER_USER` |
| Simulation Queue | `text`, `voice`                  | `MAX_CONCURRENT_JOBS` | `MAX_CONCURRENT_JOBS_PER_USER` |

#### Key Components

| Component                    | Location   | Purpose                                     |
| ---------------------------- | ---------- | ------------------------------------------- |
| `TaskStatus.QUEUED`          | `utils.py` | Status for queued jobs                      |
| `_job_starters` registry     | `utils.py` | Maps job types to starter callbacks         |
| `register_job_starter()`     | `utils.py` | Registers callback for starting jobs        |
| `can_start_*_job()`          | `utils.py` | Checks global + per-user capacity           |
| `try_start_queued_*_job()`   | `utils.py` | Starts next eligible queued job             |
| `get_queued_*()`             | `db.py`    | Gets queued jobs with user_id (FIFO order)  |
| `count_running_*()`          | `db.py`    | Counts in-progress jobs (global)            |
| `count_running_*_for_user()` | `db.py`    | Counts in-progress jobs for a specific user |

#### User ID Resolution for Jobs

Different job tables store user ownership differently:

| Table             | User ID Source                                     |
| ----------------- | -------------------------------------------------- |
| `jobs`            | Direct `user_id` column                            |
| `agent_test_jobs` | Via `agent_id` → `agents.user_id` (JOIN)           |
| `simulation_jobs` | Via `simulation_id` → `simulations.user_id` (JOIN) |

The `get_queued_*_jobs()` functions include JOINs to return `user_id` with each queued job.

#### Job Starter Registration

Each router registers its job starters at module load time:

```python
# In router module (e.g., stt.py)
def _start_stt_job_from_queue(job: dict) -> bool:
    # Reconstruct request from job details
    # Start background thread
    ...

register_job_starter("stt-eval", _start_stt_job_from_queue)
```

#### Queue Flow

```
New Job Request (user_id known)
      │
      ▼
Check: running_count < MAX_CONCURRENT_JOBS?
      │
      ├─── NO ───→ Create job (status=queued) ──→ Wait in queue
      │
      └─── YES
            │
            ▼
      Check: user_running_count < MAX_CONCURRENT_JOBS_PER_USER?
            │
            ├─── YES ──→ Create job (status=in_progress) ──→ Start immediately
            │
            └─── NO ───→ Create job (status=queued) ──→ Wait in queue

Job Completion
      │
      ▼
try_start_queued_*_job()
      │
      ▼
For each queued job (FIFO):
      │
      ├─── Global capacity full? ──→ Stop (no more jobs can start)
      │
      └─── User at their limit? ──→ Skip this job, try next one
            │
            └─── User has capacity ──→ Start this job
```

**Fair scheduling**: When processing the queue, jobs from users who are at their per-user limit are skipped, allowing jobs from other users to start. This prevents one user from monopolizing the queue.

### Process Management for Long-Running Jobs

Voice simulation and STT/TTS evaluation jobs spawn subprocesses. To handle server restarts and cleanup gracefully:

- **Process isolation**: All subprocesses started with `start_new_session=True` (creates new process group)
- **PID tracking**: Process PIDs stored in job `details` as `pid` and `pgid` fields (STT/TTS evals and voice simulations only)
- **Orphan cleanup**: On recovery, `job_recovery.py` kills process groups using `os.killpg()` before restarting
- **Graceful termination**: Sends SIGTERM first, waits 0.5s, then SIGKILL if still running. The SIGKILL step catches both `ProcessLookupError` and `PermissionError` (on macOS, dead/zombie processes can raise `PermissionError` instead of `ProcessLookupError`)

**Note**: Agent test jobs (`llm-unit-test`, `llm-benchmark`) do **not** store subprocess `pid`/`pgid` in `details` (unlike STT/TTS/voice simulations). Workers block on `process.wait()`, so there is no separate orphan PID to reap via `killpg`; recovery restarts by spawning `run_llm_test_task` / `run_benchmark_task` again with data loaded from `agent_test_jobs.details` (including the frozen `calibrate_config` JSON — see **Agent Test Job Results Format**).

This prevents orphaned processes from accumulating across server restarts.

### Job Deletion

Jobs can be deleted via DELETE endpoints:

| Endpoint                             | Table             | Notes                                          |
| ------------------------------------ | ----------------- | ---------------------------------------------- |
| `DELETE /jobs/{job_uuid}`            | `jobs`            | STT/TTS eval jobs; kills processes             |
| `DELETE /agent-tests/job/{job_uuid}` | `agent_test_jobs` | Agent test jobs; no process cleanup (blocking) |
| `DELETE /simulations/run/{job_uuid}` | `simulation_jobs` | Simulation jobs; kills process                 |

When deleting a running job:

1. Kill running processes (if applicable)
2. Delete job from database
3. Trigger next queued job in the same queue

### Job Abort (Simulations Only)

Simulation jobs can be aborted via `POST /simulations/run/{job_uuid}/abort`. Unlike deletion (which removes the job from the DB), abort preserves results:

1. Kill running process
2. Read current intermediate results from DB (already stored by the polling loop)
3. Add `"aborted": true` to each incomplete simulation result (where `evaluation_results` is `None`), and `"aborted": false` to completed ones. Transcripts are left untouched.
4. Save with `status=done` and `aborted: true` in job details

**Race condition handling**: The `_is_job_aborted(task_id)` helper checks for the `aborted` flag in job details. It is checked at multiple layers to prevent the background monitoring thread from overwriting abort state:

- **Inside the polling loops** (primary defense): Both `_run_calibrate_text_simulation` and `_run_calibrate_voice_simulation` check `_is_job_aborted()` at the start of each loop iteration. If set, they call `kill_process_group(process.pid, task_id)` followed by `process.wait(timeout=5)` to terminate the subprocess and all its children, then `break` out of the monitoring loop. The 5-second timeout prevents the monitoring thread from blocking indefinitely if the process somehow survives SIGKILL (e.g., stuck in uninterruptible I/O); on timeout it logs a warning and moves on. This prevents orphan `calibrate` CLI processes from continuing to run (consuming resources and making LLM API calls) after the user aborts.
- **Inside `_update_text_simulation_intermediate_results`**: Checks before writing to DB, closing the race window between the loop-level check and the actual DB write.
- **After the polling loop exits**: `_run_calibrate_text_simulation` and `_run_calibrate_voice_simulation` check again before final processing. If set, they return early.
- **In `run_simulation_task`**: Checks before the final `update_simulation_job` call and in all exception handlers. If set, it returns early (the `finally` block still triggers the queue).

**Why all layers matter**: Without the in-loop checks, a race occurs when the monitoring thread has already entered the loop body (passed `process.poll()`) before the abort handler kills the process and sets `status=DONE`. The thread's `update_simulation_job(status=IN_PROGRESS)` would overwrite the abort state, leaving the job stuck as `IN_PROGRESS` with no running process.

### Job Timeout Detection

Jobs that haven't updated their `updated_at` timestamp in 5+ minutes are considered timed out:

- **Timeout threshold**: `JOB_TIMEOUT_MINUTES = 5` (configured in `utils.py`)
- **Detection**: Status API checks `updated_at` for `in_progress` jobs
- **Timeout handling**:
  1. Kill running processes (if applicable)
  2. Mark job as `failed` with error (existing partial results are preserved)
  3. Trigger next queued job

The `is_job_timed_out(updated_at)` utility function in `utils.py` handles timestamp comparison.

**Important**: SQLite stores timestamps in UTC via `CURRENT_TIMESTAMP`. The timeout function uses `datetime.utcnow()` to match. Using `datetime.now()` would cause timezone mismatches and incorrect timeout detection (e.g., jobs marked as timed out immediately after creation if server is ahead of UTC).

| Utility Function                              | Location                 | Purpose                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   |
| --------------------------------------------- | ------------------------ | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `is_job_timed_out()`                          | `utils.py`               | Checks if job has exceeded timeout                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        |
| `kill_process_group()`                        | `utils.py`               | Kills a single process group by PID                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       |
| `kill_processes_from_dict()`                  | `utils.py`               | Kills multiple processes from a dict                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      |
| `capture_exception_to_sentry()`               | `utils.py`               | Logs exception to Sentry as unhandled error                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               |
| `build_tool_configs()`                        | `utils.py`               | Builds calibrate tool configs from agent tools (handles structured_output and webhook types)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              |
| `build_evaluator_runs_for_eval_job()`         | `utils.py`               | Pairs STT/TTS nested evaluator metrics to UUIDs for `ProviderResult.evaluator_runs` using calibrate's root `config.json.evaluators_map` (`{evaluator_id: metric_key}`)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    |
| `enrich_evaluator_runs_with_current_names()`  | `utils.py`               | Fills `name` on each existing evaluator run from `get_evaluator()`; used by STT/TTS/auth + `public.py` responses                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          |
| `apply_simulation_job_evaluator_enrichment()` | `routers/simulations.py` | **`simulation_jobs` only**: reads minimal UUID snapshot from `details.evaluators`; returns top-level `evaluators` with current names; mutates each `evaluation_results` dict to add `evaluator_uuid` from calibrate's echoed `evaluator_id`. Uses an in-function **`current_by_uuid`** dict so each evaluator UUID is fetched from SQLite **once per response** (snapshot UUIDs eagerly; echoed row IDs lazily on first sight). Used by **`GET /simulations/run/{task_id}`**, **`POST …/abort`**, and **`GET /public/simulation-run/{token}`** (imported from `public.py`).                                                                                                                                                                                                               |
| `_get_evaluator_cached_for_enrichment()`      | `routers/agent_tests.py` | **`agent_test_jobs` read paths**: `if uid not in cache: cache[uid] = get_evaluator(uid)` — shared by the three enrichment helpers below so benchmark status polling does not multiply SQLite round-trips across models × tests × evaluators.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              |
| `_enrich_test_results_with_evaluators()`      | `routers/agent_tests.py` | **`agent_test_jobs` only**: reads per-test snapshot from `details.evaluators_by_test_id`; converts each row's raw calibrate `judge_results` dict into `List[JudgeResult]` with `evaluator_uuid`, **current** DB `name`/`description` (via **`_get_evaluator_cached_for_enrichment`** and optional shared **`evaluator_cache`**), the snapshot-frozen `variable_values`, and (for rating evaluators) `scale_min`/`scale_max` from the pinned version's rubric. It uses calibrate's echoed `test_case_id` and per-judge `evaluator_id`. Idempotent — re-runs only refresh `name`/`description`. Used by **`GET /agent-tests/run/{task_id}`**, **`/agent-tests/agent/{uuid}/runs`**, **`/agent-tests/runs`**, and the public **`GET /public/test-run/{token}`** (imported from `public.py`). |
| `_enrich_model_results_with_evaluators()`     | `routers/agent_tests.py` | Benchmark wrapper: runs `_enrich_test_results_with_evaluators` on every `model_results[i].test_results`, then `_enrich_evaluator_summary` on each `model_results[i].evaluator_summary`, passing the **same** optional **`evaluator_cache`** through so all models and summaries dedupe within one response. Used by **`GET /agent-tests/benchmark/{task_id}`**, **`GET /agent-tests/agent/{uuid}/runs`**, **`GET /agent-tests/runs`**, and the public **`GET /public/benchmark/{token}`** (list endpoints create **one cache per job** and reuse it for top-level `_enrich_test_results_with_evaluators` + this helper).                                                                                                                                                                  |
| `_pending_test_case_result_placeholder()`     | `routers/agent_tests.py` | Returns the **`TestCaseResult`** JSON for an unfinished row: `test_case_id`, `name`, `passed`, `reasoning`, `output`, `test_case`, `judge_results` with **`null`**s (same shape as **GET `/agent-tests/run/{task_id}`** for queued / in-progress). Used by merge, benchmark queue shells, and unit-test job create / worker start.                                                                                                                                                                                                                                                                                                                                                                                                                                                        |
| `_merge_test_results_by_test_names()`         | `routers/agent_tests.py` | **Unit test + benchmark write/read shaping**: given ordered `test_names` and a partial list parsed from calibrate, returns a full-length list in suite order—completed rows keyed by `name` plus **`_pending_test_case_result_placeholder(name)`** for not-yet-finished tests. **Benchmark** intermediate updates (`_update_benchmark_intermediate_results`) and final per-model assembly use this so every model’s `test_results` mirrors **GET `/agent-tests/run/{task_id}`** semantics.                                                                                                                                                                                                                                                                                                |
| `_benchmark_queued_model_results()`           | `routers/agent_tests.py` | **Benchmark job creation + `run_benchmark_task` start**: builds `results.model_results` with one entry per requested model, `message: "Queued..."`, and **placeholder** `test_results` (one **`_pending_test_case_result_placeholder`** per `test_names` entry) so clients never poll an empty `model_results` or `null` nested `test_results` for named suites while the job is **queued** or before a model’s output folder exists.                                                                                                                                                                                                                                                                                                                                                     |
| `_build_evaluator_summary()`                  | `routers/agent_tests.py` | **`llm-benchmark` write path**: extracts `criteria` entries with `type` in `{binary, rating}` from a model’s calibrate `metrics.json` into the list shape stored as `results.model_results[].evaluator_summary` (and in intermediate DB updates once `metrics.json` exists).                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              |
| `_enrich_evaluator_summary()`                 | `routers/agent_tests.py` | Refreshes `name` / `description` on each `evaluator_summary` entry via **`_get_evaluator_cached_for_enrichment`** (optional shared **`evaluator_cache`**); leaves numeric aggregates untouched.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           |
| `upload_top_level_files_to_s3()`              | `utils.py`               | Run-root files only (STT/TTS success path)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                |
| `upload_directory_tree_to_s3()`               | `utils.py`               | Full directory tree (STT/TTS failure paths; LLM unit test/benchmark completion and failures)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              |

### Dataset Resolution for STT/TTS Evaluations

Both STT and TTS evaluation endpoints accept inputs in two ways: an existing `dataset_id` or inline data (`audio_paths`/`texts`). The shared `resolve_dataset_inputs()` function in `dataset_utils.py` handles this:

- **Existing dataset**: Fetches items from DB, returns `dataset_id`, `dataset_name`, and `item_ids`
- **Inline data with `dataset_name`**: Creates a new dataset in DB, returns the new `dataset_id`, `dataset_name`, and `item_ids`
- **Inline data without `dataset_name`**: No dataset created; `dataset_id`, `dataset_name`, and `item_ids` are all `None`

The resolved `dataset_id`, `dataset_name`, and `dataset_item_ids` are stored in the job's `details` dict and returned in both the create response (`TaskCreateResponse`) and status response (`TaskStatusResponse`). The `dataset_item_ids` are used by `inject_dataset_item_ids()` to annotate each result row with its corresponding `dataset_item_id`.

**Deleted dataset handling**: When returning evaluation data (job list, STT/TTS status), the `dataset_id` and `dataset_name` fields are set to `null` if the associated dataset has been soft-deleted. This is checked at read time via `get_active_dataset_ids()` in `db.py`, which batch-queries the `datasets` table for UUIDs that are not soft-deleted. The underlying dataset reference is preserved in job `details` but hidden from API responses.

The dataset list API (`GET /datasets`) includes an `eval_count` field on each dataset, showing how many evaluation jobs reference it. This is computed by `get_dataset_eval_counts()` in `db.py` which uses `json_extract(details, '$.dataset_id')` on the `jobs` table.

### STT/TTS Evaluation Flow

STT and TTS evaluations run a single `calibrate stt` or `calibrate tts` command with all providers specified at once. The calibrate CLI handles parallelization internally and generates the leaderboard automatically as part of the same command.

**Dataset integration**: Both STT and TTS evaluation requests accept an optional `dataset_id` to load inputs from a saved dataset, or `dataset_name` to persist inline data as a new dataset. The shared `resolve_dataset_inputs()` helper in `dataset_utils.py` handles both paths (avoiding duplication between the two routers). When `dataset_id` is provided, the helper loads items from the dataset and `dataset_name` is ignored; when inline data is provided with `dataset_name`, it creates a new dataset atomically before the evaluation starts. The `resolved_dataset_id` and `dataset_item_ids` (ordered list of item UUIDs) are stored in the job's `details` for traceability.

**Request validation**: `STTEvaluationRequest` and `TTSEvaluationRequest` both use `ConfigDict(extra="forbid")`. Unknown payload keys (for example legacy `evaluators` objects instead of `evaluator_uuids`) must fail with 422 rather than being silently ignored, because omission falls back to seeded default evaluators and can run the wrong rubric.

**Per-row `dataset_item_id` in results**: When an evaluation is linked to a dataset, each result row in the status response includes a `dataset_item_id` field. The `inject_dataset_item_ids()` helper in `dataset_utils.py` maps the CLI's `id` column (`audio_N` for STT 1-indexed, `N` for TTS 0-indexed) back to the corresponding dataset item UUID. Injection happens at the response level in the status endpoint (single injection point), so it covers all states: completed, in-progress intermediate, and failed/partial results. For older jobs or evaluations without a dataset, `dataset_item_ids` is absent from details and injection is safely skipped.

- **Single command execution**: All providers evaluated in one CLI call (e.g., `calibrate stt -p openai deepgram sarvam -l english -i input -o output`)
- **Internal parallelization**: The calibrate CLI handles concurrent provider execution internally
- **Automatic leaderboard**: The CLI generates the leaderboard in `output/leaderboard/` as part of the same command
- **Job completion**: After the command completes, the backend reads per-provider results and the leaderboard, then uploads to S3 (see **S3 artifacts for STT/TTS** below).
- **Leaderboard reading**: The backend finds any `.xlsx` file in the leaderboard directory (dynamic discovery) and reads the `summary` sheet. Calibrate may also emit leaderboard CSV/XLSX at the run root; those are uploaded as top-level files under the job prefix.

**Intermediate updates via on-demand disk reads**: The background task stores the `output_dir` path in job details. When the status API is called for an `in_progress` job, it reads each provider's `results.csv` (and `metrics.json` if available) directly from disk. This provides per-file progress as the CLI writes rows to `results.csv` incrementally. Unlike simulations which poll and update the DB with results, STT/TTS reads are on-demand from the status API. **Important**: Because intermediate results are only read from disk (never persisted to DB during `in_progress`), error and timeout handlers must explicitly read from disk via `_collect_intermediate_results()` before saving the failure to DB — otherwise all intermediate provider results would be lost.

**Heartbeat to prevent false timeouts**: The background task uses a polling loop (~2 seconds per iteration). While waiting for the CLI process to complete, it calls `update_job(task_id)` on each iteration to refresh the `updated_at` timestamp. This prevents the job from being falsely marked as timed out when the CLI takes longer than the 5-minute timeout threshold.

**Response Model (`TaskCreateResponse`):**

| Field          | Type            | Description                                                     |
| -------------- | --------------- | --------------------------------------------------------------- |
| `task_id`      | `str`           | Job UUID                                                        |
| `status`       | `str`           | Initial job status: `in_progress` or `queued`                   |
| `dataset_id`   | `Optional[str]` | Dataset UUID (from existing dataset or newly created)           |
| `dataset_name` | `Optional[str]` | Dataset name (from existing dataset or provided `dataset_name`) |

**Response Model (`TaskStatusResponse`):**

| Field                 | Type                             | Description                                                                                         |
| --------------------- | -------------------------------- | --------------------------------------------------------------------------------------------------- |
| `task_id`             | `str`                            | Job UUID                                                                                            |
| `status`              | `str`                            | Job status: `queued`, `in_progress`, `done`, `failed`                                               |
| `language`            | `Optional[str]`                  | Language from job details (e.g., "english", "hindi")                                                |
| `dataset_id`          | `Optional[str]`                  | Dataset UUID linked to this evaluation (from job details); `null` when the dataset has been deleted |
| `dataset_name`        | `Optional[str]`                  | Dataset name linked to this evaluation (from job details); `null` when the dataset has been deleted |
| `provider_results`    | `Optional[List[ProviderResult]]` | Results per provider (partial during in_progress, full on completion)                               |
| `leaderboard_summary` | `Optional[List[Dict]]`           | Summary after job completes                                                                         |
| `error`               | `Optional[str]`                  | Error message if job failed                                                                         |

**Evaluators (STT/TTS)**: Jobs optionally take `evaluator_uuids`; at creation time each UUID is resolved to a pinned evaluator **version** and stored in `job.details.evaluators` (hydrated dicts). The background task passes `--config` with `{"evaluators": [...]}` built by `build_evaluator_cli_payload()` when that list is non-empty. Default seeded evaluators apply when no UUIDs are sent (`default-stt-transcription` / `default-tts-audio-quality`).

**Response Model (`ProviderResult`):**

| Field            | Type                                | When Present                                                                                                                                                                                                                                                                                                                                                                             |
| ---------------- | ----------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `provider`       | `str`                               | Always                                                                                                                                                                                                                                                                                                                                                                                   |
| `success`        | `Optional[bool]`                    | `None` while provider is still running/queued (during `in_progress`), `True` when provider finishes all files with metrics ready (even if overall job is still `in_progress`), `True`/`False` when job completes or fails                                                                                                                                                                |
| `message`        | N/A in schema                       | In-progress status responses often use plain dicts with a human-readable `message` for progress text; the strict `ProviderResult` model may omit this field — clients should not rely on `message` being in the OpenAPI schema for completed jobs.                                                                                                                                       |
| `metrics`        | `Optional[Dict \| List[Dict]]`      | Raw `metrics.json` for the provider when present (after `normalize_metrics()` for legacy list-shaped metrics).                                                                                                                                                                                                                                                                           |
| `evaluator_runs` | `Optional[List[EvaluatorRunEntry]]` | One entry per evaluator whose aggregate appears as a **nested** object in `metrics` and whose metric key is present in the run-root `config.json.evaluators_map`. Each entry has `evaluator_uuid`, `metric_key` (CLI key for that run), `aggregate` (nested dict from calibrate), and `name` (current DB name at response time; use for display, `metric_key` for artifact correlation). |
| `results`        | `Optional[List[Dict]]`              | Partial rows from `results.csv` while running, complete when done                                                                                                                                                                                                                                                                                                                        |

**Public STT/TTS shares** (`GET /public/stt/{token}`, `GET /public/tts/{token}`): Same `provider_results` enrichment as authenticated status (including `evaluator_runs` names).

**Public simulation shares** (`GET /public/simulation-run/{token}`): Same **`evaluators`** list and per-row **`evaluator_uuid`** enrichment as authenticated **`GET /simulations/run/{task_id}`** (`apply_simulation_job_evaluator_enrichment()`), after voice presigned URL regeneration.

**Evaluator-snapshot pattern across job types** — three job families capture an evaluator snapshot in `details` at submission time, then enrich results with **current** DB names at every read. Each family uses a different snapshot shape because each calibrate output keys evaluator results differently:

| Job family                | Snapshot in `details`                                                                                                                                                                       | Mapping back to UUID                                                                                                                                                                                                                                                                                                                                                                                                                                              | Enrichment helper                                                                                                                                                          |
| ------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| STT / TTS                 | `details.evaluators` — full hydrated evaluator dicts                                                                                                                                        | Root `config.json.evaluators_map` maps emitted metric/result keys back to evaluator UUIDs.                                                                                                                                                                                                                                                                                                                                                                        | `enrich_evaluator_runs_with_current_names()`                                                                                                                               |
| Simulation                | `details.evaluators` — minimal `[{ "uuid" }]` list                                                                                                                                          | Each `evaluation_results.csv` row's echoed `evaluator_id`.                                                                                                                                                                                                                                                                                                                                                                                                        | `apply_simulation_job_evaluator_enrichment()`                                                                                                                              |
| LLM unit test / benchmark | `details.calibrate_config` (frozen calibrate CLI JSON) + `details.evaluators_by_test_id` — per-test UUID → `[{uuid, calibrate_name, output_type, variable_values, scale_min?, scale_max?}]` | The CLI consumes `calibrate_config`; each test case is sent with `id=<test_uuid>` and each evaluator definition with `id=<evaluator_uuid>`. Enrichment uses echoed `test_case_id` + per-judge `evaluator_id`; the snapshot freezes `variable_values` and rating bounds. Benchmarks additionally persist per-model **`results.model_results[].evaluator_summary`** from each model’s `metrics.json.criteria` at completion (see **Benchmark Job Results Format**). | `_enrich_test_results_with_evaluators()`; benchmarks also `_enrich_model_results_with_evaluators()` (nested `test_results` + `evaluator_summary` name/description refresh) |

For these three families, **display** enrichment (evaluator `name` / `description` from the current DB) happens at read time on the response payload so renames flow through without a re-run. **SQLite load**: simulation and agent-test helpers both memoize **`get_evaluator(uuid)`** inside a per-response dict keyed by UUID — **not** one query per `judge_results` row — so repeated UUIDs across models, tests, or `evaluator_summary` entries are cheap dict hits (critical when the UI polls **`/agent-tests/benchmark/{task_id}`** every few seconds). **Exception**: benchmark `evaluator_summary` **aggregate numbers** (`pass_rate`, `mean`, counts, etc.) are **persisted** from each model’s `metrics.json` when the worker finalizes that model; read paths only refresh labels on those stored rows, not recompute aggregates from `test_results`.

**TTS success determination**: A TTS provider is marked `success: true` only if at least one text was successfully synthesized (has an `audio_path` in results). If the calibrate CLI completes but no audio files were generated (e.g., voice not found, API errors), `success: false` is returned with an error message. This prevents false positives where the process exits normally but all synthesis attempts failed.

**Provider status while running**: While the job is `in_progress`, the status API reads intermediate results from disk and determines per-provider completion by comparing the number of result rows against the expected total (from `len(audio_paths)` for STT, `len(texts)` for TTS):

- **Provider fully processed** (result count >= expected total AND `metrics.json` exists): `success: true`, `message: "Done (N files processed)"`, full `results` and `metrics`. This allows individual providers to show as done while the overall job is still `in_progress` (other providers may still be running).
- **Provider with partial results** (some rows but not all, or no metrics yet): `success: null`, `message: "Running... (N files processed)"`, `results` populated from `results.csv`, `metrics` from `metrics.json` if available.
- **Provider with no results yet**: `success: null`, `message: "Queued..."`, `metrics`/`results` as `null`. **Important**: During `in_progress`, providers with no results are always shown as "Queued..." regardless of whether the output directory exists — the CLI creates the directory before writing results, so checking directory existence would cause false "Failed" states for providers that are still starting up. Failed provider detection only happens after the run completes (via `_collect_intermediate_results` / `_collect_tts_intermediate_results`).
- **TTS audio paths during in-progress**: Local audio files are uploaded to S3 on-the-fly (using the same key convention as the final upload: `tts/evals/{task_id}/outputs/{provider}/...`) and presigned download URLs are returned. Falls back to `null` if the file doesn't exist or upload fails. The uploads are idempotent — the final background task upload overwrites the same keys.
- **Fallback**: If `output_dir` is not in job details (e.g., process hasn't started yet), all providers shown with `success: null` and `metrics`/`results` as `null`

**Preserving results on failure (STT & TTS)**: When an STT or TTS job fails (subprocess error, timeout, or unexpected exception), the error handlers use `_collect_intermediate_results()` (STT) or `_collect_tts_intermediate_results()` (TTS) to read whatever partial results exist on disk before saving to DB. This ensures that providers that completed successfully retain their `metrics` and `results` even when other providers caused the failure. Each helper reads each provider's `results.csv` and `metrics.json`, marks providers with data as `success: true` (or false as appropriate), and attaches `evaluator_runs` only for metric keys present in run-root `config.json.evaluators_map`. The TTS helper additionally uploads audio files to S3 and replaces local paths with S3 keys (matching the success path pattern), so that presigned URLs can be generated when the status is fetched. **S3**: After collecting intermediates, the handlers call `upload_directory_tree_to_s3()` on the entire calibrate `output_dir` under `stt/evals/{task_id}/outputs` or `tts/evals/{task_id}/outputs` so logs and partial artifacts (including top-level `logs`) are retained for debugging.

**Critical: Merging results on timeout** (STT & TTS): When the status API detects a timeout, it must **merge** intermediate results from disk with any existing successful results already stored in the database — NOT replace them. This is critical because:

1. The temp directory may already be cleaned up (background thread exited), making disk reads return empty results
2. Some providers may have already been marked `success: true` in the database before the timeout
3. Unconditionally overwriting `provider_results` would lose these successful results

The timeout handler builds a map of existing successful providers (`success: true`) from the database, then merges with disk results: existing successes are preserved, disk results fill in the rest, and missing providers are marked as failed.

**Presigned URLs for failed jobs (TTS)**: Presigned URL generation for audio files runs for both `done` and `failed` jobs. This ensures that providers which completed successfully within a failed job still serve accessible audio URLs, not raw S3 keys.

**Metrics normalization**: The status API normalizes old list-of-dicts metrics format to the new dict format before returning. The `normalize_metrics()` helper in `utils.py` handles two old formats:

- Simple metrics: `[{"wer": 2.4}, {"string_similarity": 0.15}, ...]` → merged into result dict
- Latency metrics: `[{"metric_name": "ttfb", "mean": 0.1, ...}, ...]` → uses `metric_name` as key, rest as value

This ensures clients always receive metrics in dict format (e.g., `{"wer": 2.4, "ttfb": {"mean": 0.1, ...}}`) regardless of when the job was created.

**Calibrate STT/TTS nested evaluator metrics (current CLI)**: For each configured evaluator, calibrate emits a **single nested object** under a stable key (derived from evaluator naming), e.g. `semantic_match: {"type": "binary", "mean": 0.85}`. The run-root `config.json` includes `evaluators_map` (`{evaluator_uuid: metric_key}`); `_collect_intermediate_results()` / `_collect_tts_intermediate_results()` and success handlers read it with `read_evaluators_map_from_config()` so `evaluator_runs` can map metric keys back to evaluator UUIDs without relying on list order. Legacy flat keys `<name>_score` / `<name>_info` are obsolete. Identify evaluator-shaped entries as dict values that include `"type"` (scalar metrics like `wer` remain numbers; `ttfb` may be a dict without `"type"` depending on CLI version — see `utils.is_evaluator_metric_aggregate()` used for API pairing). Per-row CSV columns drop the `_score` suffix (column name equals the evaluator key; reasoning stays `<name>_reasoning`).

**STT Metrics** (returned by `calibrate stt`):

- `wer` - Word Error Rate (float)
- `string_similarity` - String similarity score (float), when enabled
- Plus **evaluator-named keys** (nested objects with `type` / `mean` / scale fields) from linked evaluators — **not** limited to legacy `llm_judge_score` scalar; prefer `evaluator_runs` on the API for stable UUID + display name
- `processing_time` - Processing time stats: `{mean, std, values}` (may be absent/null)
- `ttfb` - Time to First Byte stats: `{mean, std, values}` (may be absent/null)

**TTS Metrics** (returned by `calibrate tts`):

- Evaluator nested objects as above; legacy `llm_judge_score` may appear on older runs
- `ttfb` - Time to First Byte stats: `{mean, std, values}` (may be absent/null)

Note: TTS does not return `processing_time`. Both `ttfb` values may be absent if the provider doesn't support streaming or the measurement failed — clients should handle null/missing values.

**Pairing gotcha**: STT/TTS evaluator UUID enrichment requires the run-root `config.json.evaluators_map`; without it, evaluator metric aggregates are left without `evaluator_runs` rather than guessed by order.

**TTS `results` row `audio_path` field:**

- **Stored**: S3 key (e.g., `tts/evals/{job_id}/outputs/{provider}/audios/0.wav`)
- **On status fetch**: Presigned URL generated on-the-fly for completed jobs

This allows secure, time-limited access to audio files without storing expirable URLs in the database.

### S3 artifacts for STT/TTS eval jobs

Calibrate writes a **run-level** `logs` file (and optionally CSV/XLSX leaderboard files) **directly under** the CLI `output_dir`, while per-provider folders contain that provider's `metrics.json`, `results.csv`, `logs`, etc. Because per-provider uploads only walked each provider subdirectory, **top-level files were historically missing from S3** unless handled explicitly.

**Current behavior** (`utils.py`):

| Helper                           | Purpose                                                                                                                                                                                                                               |
| -------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `upload_top_level_files_to_s3()` | Uploads **only regular files whose parent is** `output_dir` (e.g. `logs`, root `leaderboard.csv`, backend's `stdout.log` / `stderr.log`). Keys: `stt/evals/{task_id}/outputs/<filename>` or `tts/evals/{task_id}/outputs/<filename>`. |
| `upload_directory_tree_to_s3()`  | Recursive upload of an entire directory tree (used on STT/TTS **failure** paths after intermediate collection, so partial runs still retain **all** log files and outputs under the same prefixes).                                   |

**Success path**: Per-provider subtree upload (unchanged relative layout under `…/outputs/{provider}/`), then **`upload_top_level_files_to_s3`** for root files, then leaderboard directory walk (`…/leaderboard/`). **Failure path**: After preserving DB intermediates, **`upload_directory_tree_to_s3`** on full `output_dir` (may overlap keys with earlier partial uploads — overwrites are acceptable).

### Simulation Incremental Updates (Text and Voice)

Both text simulations (`calibrate llm simulations run`) and voice simulations (`calibrate agent simulation`) run multiple persona-scenario combinations. Each combination creates a folder named `simulation_persona_<n>_scenario_<m>`. The backend monitors for these folders during execution and updates the database incrementally:

- **Polling interval**: Every 2 seconds while the subprocess is running
- **Completion marker**: A simulation folder is considered complete when `evaluation_results.csv` exists (created after the LLM judge evaluation step finishes)
- **In-progress detection**: A simulation is considered in-progress when `transcript.json` or `config.json` exists but `evaluation_results.csv` does not
- **Persona/scenario data resolution** (both text and voice):
  1. First try reading from `config.json` in the simulation directory
  2. Fallback: parse directory name `simulation_persona_N_scenario_M` to get 1-based indices, then look up `personas_list[N-1]` and `scenarios_list[M-1]` from the original calibrate config
- **During execution**: Status API returns partial `simulation_results` including:
  - **For completed simulations**:
    - `persona` and `scenario` data (always populated via config.json or fallback)
    - `transcript` from `transcript.json`
    - `evaluation_results` with per-criterion metrics from `evaluation_results.csv` (name, value, reasoning). When returned through the **run status** or **public simulation share** APIs, each row may also include **`evaluator_uuid`** (added at read time via **`apply_simulation_job_evaluator_enrichment()`**, not stored in the CSV)
  - **For in-progress simulations** (both text and voice):
    - `persona` and `scenario` data (always populated via config.json or fallback)
    - `transcript` from `transcript.json` (partial conversation so far)
    - `evaluation_results` is `null`
    - **Voice simulations**: No audio URLs returned until evaluation completes (all audio fields are `null`)
  - Plus `completed_simulations` count for progress tracking (counts only fully completed simulations)
  - **`metrics` field is `null` during in-progress**: The `calibrate` CLI creates `metrics.json` incrementally as each simulation completes, so reading it before all simulations finish would give incomplete aggregate data (e.g., `values` array with fewer entries than expected). The backend intentionally does NOT read `metrics.json` until the job completes.
- **On completion**: Final aggregated `metrics` (from `metrics.json`) are added to the response only after the subprocess exits
- **Presigned URL handling** (voice simulations):
  - **During in-progress**: No audio URLs are returned for simulations until their `evaluation_results` are available. This means:
    - In-progress simulations (no `evaluation_results.csv` yet) have all audio fields as `null`
    - Completed simulations (have `evaluation_results.csv`) include `audio_urls` and `conversation_wav_url` presigned URLs stored in DB
  - **When job status becomes done**: Presigned URLs are stripped from the stored results; only S3 paths (`audios_s3_path`, `conversation_wav_s3_key`) remain
  - **When fetching done status**: Presigned URLs are generated on-the-fly from S3 paths using `_get_audio_urls_from_s3_key()` and `generate_presigned_download_url()`, but NOT saved back to DB
  - **Audio URL ordering**: `_get_audio_urls_from_s3_key()` accepts the simulation's transcript and sorts audio files in conversation order. Files are named `N_bot.wav`/`N_user.wav` and grouped by exchange number N. For each exchange, the transcript's spoken turns (those with `content`, skipping tool_calls-only messages) determine whether bot or user audio comes first. Falls back to bot-first if no transcript is available.
  - This ensures audio is only accessible after evaluation completes and prevents stale URLs from accumulating

This allows clients to display progress and per-simulation results (including partial transcripts for in-progress simulations) without waiting for all simulations to complete.

### Agent Test Job Results Format

Agent test jobs (`llm-unit-test`) run a single `calibrate llm` command. The backend monitors the output directory during execution and provides incremental results as each test completes. The CLI shape — and the on-disk layout — depends on whether the agent has a `config.agent_url`:

- **Mode A — internal LLM** (`agent_config.agent_url` is **not** set):
  `calibrate llm -c config.json -m {model} -p {provider} -o output`
  Files land under a per-model subfolder: `<output_dir>/<provider>__<model>/{results.json, metrics.json, results.log, logs}`. Calibrate may also write a `<output_dir>/leaderboard/` (single-model unit tests typically still get one — the file is written but only contains one row).
- **Mode C — agent connection** (`agent_config.agent_url` is set):
  `calibrate llm -c config.json -o output --skip-verify` (**no `-m`, no `-p`** — the agent owns the model decision; the backend stores a placeholder `model = "agent-connection"` for internal logging only).
  Files land **directly at the top level** of `<output_dir>`: `{results.json, metrics.json, results.log, logs}`. **No `leaderboard/` is generated** in this mode (only one set of results, so no comparison row).

Both modes are read by the same code path — `_read_agent_test_results_json()` / `_read_agent_test_metrics_json()` use `os.walk` to find the first `results.json` / `metrics.json` anywhere under `output_dir`, so Mode A's subfolder and Mode C's top-level files are both handled without branching.

- **Polling interval**: Every 2 seconds while the subprocess is running
- **During execution**: The backend reads `results.json` and updates the database incrementally
- **Test completion detection**: When a test appears in `results.json`, it's shown with full results; pending tests use **`_pending_test_case_result_placeholder(name)`** (all `TestCaseResult` keys, unset fields `null`). The merge is implemented by `_merge_test_results_by_test_names()` in `routers/agent_tests.py`, which **reorders and pads** the parsed calibrate rows to match the submit-time list of test **names** (`test_names` from linked tests with a `name` field). The same helper is used for **benchmark** per-model `test_results` (see **Benchmark Job Results Format**).
- **S3 upload on completion**: The full calibrate output directory is uploaded recursively via `upload_directory_tree_to_s3()` to `agent-tests/runs/{task_id}/`, preserving whichever layout calibrate produced (Mode A subfolder vs. Mode C flat).
- **S3 on subprocess failure**: The same recursive upload runs in failure handlers when `output_dir` still exists; **`results.results_s3_prefix`** is set to `agent-tests/runs/{task_id}` when upload succeeds so clients can locate artifacts after a failed run.

**Response Model (`TestCaseResult`):**

| Field           | Type                          | When Present                                                                                                                                                                                                               |
| --------------- | ----------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `test_case_id`  | `Optional[str]`               | Echoed from calibrate when the row is read from `results.json`. **`null`** in **pending** placeholders until that test’s line exists on disk.                                                                              |
| `name`          | `Optional[str]`               | Always (pending and completed tests)                                                                                                                                                                                       |
| `passed`        | `Optional[bool]`              | When test completes (appears in `results.json`)                                                                                                                                                                            |
| `reasoning`     | `Optional[str]`               | When test completes — `"All evaluators passed"` / `"Tool calls matched expected"` for passing rows; the first failing evaluator's reasoning (or a no-response message) for failing rows. Null for passing tool-call tests. |
| `output`        | `Optional[TestOutput]`        | When test completes                                                                                                                                                                                                        |
| `test_case`     | `Optional[Dict]`              | When test completes (verbatim copy of the original config entry)                                                                                                                                                           |
| `judge_results` | `Optional[List[JudgeResult]]` | **Response-type tests only** — one entry per evaluator referenced by that test case (see below). `None` for tool-call tests and pending rows.                                                                              |

**Pending / in-progress rows (API contract)**: For **named** suites, the backend does not emit sparse objects with only `name` for unfinished tests. **`_pending_test_case_result_placeholder()`** writes every `TestCaseResult` key with **`null`** for values not yet known, so **`GET /agent-tests/run/{task_id}`** and each benchmark **`model_results[].test_results`** use the same stable key set for UIs. Completed rows still come from **`_parse_agent_test_results()`** and read-time enrichment.

**`JudgeResult`** — one evaluator's verdict for a response-type test case. Refreshed on every API read so renames flow through immediately:

| Field             | Type                       | Notes                                                                                                                                                                                                                                                                                                                          |
| ----------------- | -------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `evaluator_uuid`  | `Optional[str]`            | The evaluator UUID echoed by calibrate as `evaluator_id`.                                                                                                                                                                                                                                                                      |
| `name`            | `str`                      | **Current** evaluator name from the DB row when it exists (enrichment uses **`_get_evaluator_cached_for_enrichment`** — one `get_evaluator` per distinct UUID per HTTP response); otherwise the calibrate judge-results key.                                                                                                   |
| `reasoning`       | `Optional[str]`            | Per-evaluator reasoning string emitted by the judge.                                                                                                                                                                                                                                                                           |
| `match`           | `Optional[bool]`           | Set for **binary** evaluators (`true`/`false`).                                                                                                                                                                                                                                                                                |
| `score`           | `Optional[float]`          | Set for **rating** evaluators (number on the evaluator's scale).                                                                                                                                                                                                                                                               |
| `variable_values` | `Optional[Dict[str, Any]]` | The `{{var}}` substitutions used for **this** evaluator on **this** test case (e.g. `{"criteria": "Be very friendly"}`). Comes from `test_evaluators.variable_values` frozen at submission time, so it always reflects what the run actually used even if the link is later edited. `None` (not `{}`) when no variables apply. |
| `scale_min`       | `Optional[float]`          | **Rating only** — minimum value of the rubric the run actually used (from the pinned evaluator-version `output_config.scale`). `None` for binary evaluators.                                                                                                                                                                   |
| `scale_max`       | `Optional[float]`          | **Rating only** — maximum value of the rubric the run actually used. Pairs with `scale_min` so clients can render `score / scale_max` etc. without separately fetching the evaluator.                                                                                                                                          |

This allows clients to see which tests have completed with their results while other tests are still running.

**Per-run calibrate config (`details.calibrate_config`)** — **why**: The handler calls `_build_calibrate_config` once at submission, then the worker uses that exact stored config. A queued job could wait indefinitely; meanwhile `test_evaluators` links or live evaluator versions could change, so rebuilding later would send different evaluator/test IDs or prompts than the API snapshot expects. Persisting the exact dict passed to calibrate fixes that alignment.

**Pattern**: `run_agent_test` / `run_agent_benchmark` call `_build_calibrate_config(agent, tests)` **once** at submission and store **both** `details.calibrate_config` (full CLI JSON: `params`, `system_prompt`, `tools`, `test_cases`, top-level `evaluators`, etc.) and `details.evaluators_by_test_id`. The backend sends `test_cases[].id = tests.uuid` and top-level `evaluators[].id = evaluators.uuid`; criteria still reference evaluators by calibrate `name` because that is the CLI contract. `run_llm_test_task` and `run_benchmark_task` resolve the CLI payload via `_calibrate_config_from_agent_test_job(task_id, agent, tests)` in `routers/agent_tests.py`, which **`copy.deepcopy`s** stored JSON so worker mutations (benchmarks clearing `params` before `-m`) never alias the persisted row. Queue starters and `job_recovery.py` already pass fresh `agent`/`tests` from the DB into those tasks; they do **not** rebuild config themselves — the worker reads `calibrate_config` from the job row.

**Benchmark-only**: After loading, workers clear `calibrate_config["params"]` — models are supplied entirely via CLI `-m` flags; the persisted submit-time `params` only reflects the agent’s default model for config generation.

**Per-test evaluator snapshot (`details.evaluators_by_test_id`)**: Produced by that same `_build_calibrate_config` call and persisted under `agent_test_jobs.details.evaluators_by_test_id`. Shape:

```jsonc
{
  "<test_uuid>": [                                    // omitted for tool_call tests / tests with no evaluators
    {
      "uuid": "<evaluator_uuid>",
      "name": "<calibrate-rendered_name>",            // calibrate's judge_results key for this test
      "output_type": "binary" | "rating",
      "variable_values": { "criteria": "..." },       // {{var}} substitutions sent to calibrate; {} if none
      "scale_min": 1.0,                               // rating only — omitted for binary
      "scale_max": 5.0                                // rating only — omitted for binary
    },
    ...
  ],
  ...
}
```

The inner `name` is **not** the DB display name — it is the unique calibrate-rendered evaluator name produced by `build_test_evaluators_payload()` (with the `-{uuid8}` collision suffix when two evaluators share a base name in the same run). Calibrate outputs echo `test_case_id` and per-judge `evaluator_id`; the name remains useful because `judge_results` is still keyed by evaluator name.

`variable_values` and `scale_min`/`scale_max` are **frozen at submission time** (the same source-of-truth instant where `evaluator_version_id` is pinned via the `test_evaluators` link): `variable_values` comes straight from the `test_evaluators` row's JSON column, and the rating bounds come from that pinned version's `output_config.scale` (numeric values only — `bool` is rejected so a binary scale of `[false, true]` doesn't accidentally surface as `[0, 1]`). This way, a rename, a prompt edit, or a rubric change after submission never alters how a past run is displayed.

**Read-time enrichment (`_enrich_test_results_with_evaluators`)**: Every API endpoint that returns `test_results` (`GET /agent-tests/run/{task_id}`, `GET /agent-tests/agent/{uuid}/runs`, `GET /agent-tests/runs`, and the public `GET /public/test-run/{token}`) calls this helper before serializing. It:

1. Reads `details.evaluators_by_test_id` and builds a per-test `evaluator_id → snapshot_entry` map.
2. Walks each row's stored `judge_results` (raw dict from calibrate, keyed by calibrate name) and converts it to a `List[JudgeResult]` with `evaluator_uuid`, `match`/`score`, `reasoning`, and the snapshot-frozen `variable_values` + `scale_min` / `scale_max`. Each judge result must include echoed `evaluator_id`. Empty `variable_values` (`{}`) is normalised to `None` to keep the response slim.
3. Refreshes each entry's `name` / `description` from the **current** DB row via **`_get_evaluator_cached_for_enrichment`** (optional caller-supplied **`evaluator_cache`**; otherwise a fresh dict for that invocation only — still dedupes within that `test_results` list). Latest rename wins; deleted evaluators yield nullable `description`.

The helper is **idempotent**: if `judge_results` is already a list (already enriched in this request, or written that way by some future writer), only `name` is refreshed — the snapshot-derived fields (`variable_values`, `scale_min`, `scale_max`) are populated only on the first conversion from the raw calibrate dict.

**Why the snapshot lives on the job (not the test case)**: A test's `test_evaluators` pivots can change after the job is submitted; the snapshot freezes the exact evaluator UUIDs **and the variable values + rating bounds the run actually used** so result rows always map back correctly. Together with **`details.calibrate_config`**, both the raw calibrate keys and enrichment metadata come from the same submission-time `_build_calibrate_config` invocation. Names are _not_ frozen — they're refreshed on every read so a rename surfaces immediately without a re-run.

**Tool-call tests**: Calibrate emits no `judge_results` for tool-call tests (`metrics.passed` + `metrics.reasoning` are the only verdicts). `_parse_agent_test_results` stores `judge_results: None` and the enrichment helper short-circuits, so `JudgeResult` entries are never produced for tool-call rows.

### Benchmark Job Results Format

Benchmark jobs (`llm-benchmark`) run a single `calibrate llm` command with all models specified at once. The calibrate CLI handles parallelization internally and generates the leaderboard automatically. **Persisted job shape**: benchmark rows store incremental and final data under **`agent_test_jobs.results.model_results` only** — not a top-level `results.test_results` key (that shape is for **`llm-unit-test`** and **`GET /agent-tests/run/{task_id}`**). The create handler (`POST /agent-tests/agent/{uuid}/benchmark`) and the worker’s first DB update both seed **`model_results`** via **`_benchmark_queued_model_results()`** so every model row includes **`test_results`** as a name-ordered list of **full null `TestCaseResult` stubs** (from **`_pending_test_case_result_placeholder`**) matching **`details.test_names`**, before any calibrate output exists. As with unit tests, the CLI shape and the on-disk model-folder convention depend on whether the agent has a `config.agent_url`:

- **Mode A — internal LLM** (`agent_config.agent_url` is **not** set):
  `calibrate llm -c config.json -m m1 m2 -p {provider} -o output`
  Per-model folder: calibrate uses `<provider>__<model>` for `provider == "openai"` and the bare `<model>` (with `/` → `__`) otherwise. With OpenRouter it's effectively `<vendor>__<model>` because the model itself contains a slash (e.g., `openai/gpt-4.1` → `openai__gpt-4.1`).
- **Mode B — agent connection benchmark** (`agent_config.agent_url` is set, `request.models` non-empty):
  `calibrate llm -c config.json -m m1 m2 -o output --skip-verify` (**no `-p`** — the CLI sends each `-m` value as a `model` field in the request body so the agent routes internally).
  Per-model folder is **always** `<cli_model>` with `/` → `__` — **no provider prefix is prepended**. The `cli_model` may differ from the display model name when the prefix has been stripped (see below).

There is no "Mode C" path for benchmarks because the create endpoint requires `request.models` to be non-empty (single-model agent-connection runs go through the unit-test path instead, which is Mode C).

- **Single command execution**: All models evaluated in one CLI call.
- **Model name normalization (agent connections only)**: The frontend always sends model names in OpenRouter format (`provider/model`, e.g. `openai/gpt-4.1`). For agent connections, if `benchmark_provider` is not `openrouter`, the `provider/` prefix is stripped before passing to the CLI (e.g. `openai/gpt-4.1` → `gpt-4.1`), so the CLI folder becomes the bare model name (e.g. `gpt-4.1`) instead of `openai__gpt-4.1`. The original names are preserved for display in API responses (`model` field in `ModelResult`) and in `benchmark_models_verified` keys. This stripping does not apply to non-connection agents (`type: "agent"`).
- **Internal parallelization**: The calibrate CLI handles concurrent model execution internally.
- **Automatic leaderboard**: The CLI generates `output/leaderboard/llm_leaderboard.csv` in both Mode A and Mode B (any time `-m` is supplied, regardless of how many models).
- **Output discovery**: After command completes, `_find_all_results_in_output()` walks `output_dir` and keys per-folder `(results.json, metrics.json)` pairs by **folder basename**.
- **Model name matching**: `_match_model_to_folder()` maps a CLI model name to its output folder. It uses **exact matching** (not substring matching) across the candidate set, which deliberately covers both Mode A and Mode B folder shapes:
  - `openai/gpt-4` → `openai_gpt-4` (single underscore — historical compat)
  - `openai/gpt-4` → `openai__gpt-4` (double underscore — Mode A openrouter / Mode A openai / Mode B with `provider/model` cli value)
  - `openai/gpt-4` → `openai-gpt-4` (dash — historical compat)
  - `openai/gpt-4` → `openai/gpt-4` (literal — never produced by calibrate today, kept defensively)
  - `gpt-4` → `gpt-4` (bare — Mode B after `provider/` prefix stripping)

  **Gotcha**: Exact matching is required because substring matching caused silent cross-model contamination when one model name was a prefix of another (e.g. benchmarking `gpt-5.4` alongside `gpt-5.4-mini`, or `openai/gpt-4.1` alongside `openai/gpt-4.1-mini`). With substring matching, iterating folders in filesystem order could match `gpt-5.4`'s name as a substring of the `gpt-5.4-mini` folder, causing the backend to store the wrong model's `results.json`/`metrics.json` under `gpt-5.4`. The leaderboard CSV path was not affected (it's read directly from `leaderboard/llm_leaderboard.csv`). If calibrate ever introduces a **new** folder-name convention (e.g., a colon separator, or some new agent-mode shape), it must be added to the candidate set in `_match_model_to_folder` explicitly — substring fallbacks must not be reintroduced.

- **Polling interval**: Every 2 seconds while the subprocess is running
- **During execution**: The backend scans output directory for per-model results and updates incrementally. For each model, **`_update_benchmark_intermediate_results()`** merges on-disk `results.json` with **`_merge_test_results_by_test_names(details.test_names, ...)`** so `model_results[i].test_results` always has one row per named test in order—unfinished rows use the same **explicit-null `TestCaseResult`** shape as unit-test polling.
- **Job completion**: Final results, `leaderboard_summary` (CSV rows from `leaderboard/llm_leaderboard.csv`), and per-model `evaluator_summary` (from each model folder’s `metrics.json.criteria` via `_build_evaluator_summary()` in `routers/agent_tests.py`) are written into `agent_test_jobs.results` after the command completes. Intermediate DB updates while polling set `evaluator_summary` only when `metrics.json` exists for that model (same gate as `total_tests` / `passed` / `failed`).

**Response Model (`ModelResult`):**

| Field               | Type                              | When Present                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          |
| ------------------- | --------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `model`             | `str`                             | Always                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                |
| `success`           | `Optional[bool]`                  | `None` while running/queued, `True/False` when done                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   |
| `message`           | `str`                             | Progress strings include `"Queued..."`, `"Running... (N tests done)"`, `"Completed"` (intermediate). Final persisted rows may use longer completion text (e.g. benchmark success message including the model id).                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     |
| `total_tests`       | `Optional[int]`                   | When model has results (partial or complete)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          |
| `passed`            | `Optional[int]`                   | When model has results                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                |
| `failed`            | `Optional[int]`                   | When model has results                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                |
| `evaluator_summary` | `Optional[List[Dict[str, Any]]]>` | When that model’s `metrics.json` exists and contains a `criteria` object. Each element mirrors one calibrate criteria aggregate: `metric_key` (CLI column key), `type` (`binary` \| `rating`), `evaluator_uuid` (from calibrate’s `evaluator_id` when echoed), plus type-specific numeric fields (`passed`/`total`/`pass_rate` vs `mean`/`min`/`max`/`count`/`scale_min`/`scale_max`). Omitted or `null` while queued, before `metrics.json`, or for benchmark rows stored before this field existed.                                                                                                                                                                                                                 |
| `test_results`      | `Optional[List[Dict]]`            | For each model, the **full test suite in fixed order** once **`details.test_names`** is non-empty: **queued**, **no folder yet**, **partial**, and **complete** runs all use the same list length—pending rows are **`_pending_test_case_result_placeholder(name)`** (all `TestCaseResult` keys present, unset fields **`null`**); calibrate rows replace or augment by `name` via `_merge_test_results_by_test_names()`. `judge_results` enriched via `_enrich_model_results_with_evaluators` on completed rows. If **no** linked test has a `name` field, `test_names` is empty, merge returns `[]`, and APIs may show **`null`** / empty nested `test_results` (same edge case as an all-unnamed unit-test suite). |

**Per-model `evaluator_summary` (why)**: The leaderboard CSV flattens binary criteria to pass-rate **percentages** and rating criteria to **mean score** columns with no explicit `type` column. The API adds `evaluator_summary` so clients can build evaluator-aware leaderboards and tooltips without inferring column semantics from CSV position alone. **Frontend contract**: treat binary `pass_rate` as a 0–100 percentage; for ratings, compare `mean` to `scale_min`/`scale_max` (not a pass percentage). This matches calibrate’s `metrics.json.criteria` shape. **`leaderboard_summary` remains** the raw CSV dict rows for clients that already parse `llm_leaderboard.csv` — it is not removed.

**Gotchas**: (1) **`evaluator_summary` is not backfilled** for historical benchmark jobs completed before the backend started persisting it — those rows only have `leaderboard_summary` + nested `test_results`/`judge_results`. (2) **Tool-call-only suites** may yield empty or absent `criteria` in `metrics.json` depending on CLI version; `evaluator_summary` may then be `null` even when `metrics.json` exists. (3) **In-progress**: `evaluator_summary` is populated only once that model’s aggregate file exists; do not assume it is present whenever `test_results` is partial. (4) **Unnamed test cases** (every linked test missing `name`): `test_names` is empty, placeholder merge does not run, and per-model `test_results` may stay **`null`** — clients cannot show a per-row skeleton list without names. (5) **Older jobs** created before this **`model_results`-first** contract may have been stored with only a top-level `test_results` on the job row; treat as legacy — new benchmarks always populate **`results.model_results`**. (6) **Sparse pending rows in old DB snapshots**: in-progress or queued results written before **`_pending_test_case_result_placeholder`** may still show **`{"name": "..."}`** only for incomplete tests; new runs always persist the **full** null-shaped `TestCaseResult` for pending lines (unit and benchmark).

**Model states during execution:**

- `"Queued..."` - No output folder found for this model yet; **`test_results` still lists every named test** with the full null **`TestCaseResult`** stub when `test_names` is non-empty
- `"Running... (N tests done)"` - Has `results.json` but no `metrics.json` (partial calibrate output); `N` is **`len()` of parsed rows** from the file, not the padded list length
- `"Completed"` (intermediate) - Has `metrics.json` (all tests done for this model); `evaluator_summary` is present when `criteria` was emitted

This allows clients to see per-model progress as each model completes its tests, and to render **per-test** rows (including not-yet-run tests) the same way as **GET `/agent-tests/run/{task_id}`** for unit tests.

**S3 Upload Structure for Benchmarks:**

After the benchmark command completes, **`upload_directory_tree_to_s3()`** uploads the entire calibrate output directory under `outputs/` (same recursive approach as unit tests). That includes run-level **`logs`**, per-model **`logs`** and **`results.log`**, JSON/CSV/leaderboard folders, etc. Leaderboard files are also uploaded under `leaderboard/` (first pass) when present. Example shape (actual folder names depend on calibrate):

```
s3://bucket/agent-tests/benchmarks/{task_id}/
  benchmark_config.json
  leaderboard/
    llm_leaderboard.csv         # when CLI writes leaderboard here
  outputs/
    logs                        # whole-run tee from calibrate
    leaderboard/                # may mirror local layout
    <model-folder>/
      logs
      results.log
      results.json
      metrics.json
```

On benchmark **failure**, the recursive upload still runs when possible and **`results.results_s3_prefix`** may be set to `agent-tests/benchmarks/{task_id}` so artifacts remain discoverable.

### Config File Uploads to S3

All job types upload their config files to S3 for reproducibility and debugging. The config file is uploaded after job completion to the job's output directory:

| Job Type         | Config File Name         | S3 Path                                                  |
| ---------------- | ------------------------ | -------------------------------------------------------- |
| STT Evaluation   | `config.json`            | `stt/evals/{task_id}/config.json`                        |
| TTS Evaluation   | `config.json`            | `tts/evals/{task_id}/config.json`                        |
| Agent Test       | `test_config.json`       | `{s3_prefix}/test_config.json`                           |
| Benchmark        | `benchmark_config.json`  | `agent-tests/benchmarks/{task_id}/benchmark_config.json` |
| Text Simulation  | `simulation_config.json` | `{s3_prefix}/simulation_config.json`                     |
| Voice Simulation | `simulation_config.json` | `{s3_prefix}/simulation_config.json`                     |

**Config file contents:**

- **STT/TTS config**: Contains `providers`, `language`, and `audio_count`/`text_count`
- **STT/TTS job details** also store `dataset_id`, `dataset_name`, and `dataset_item_ids` for linking evaluations back to their source dataset
- **Agent tests**: Contains the full calibrate config (system prompt, tools, test cases, etc.)
- **Benchmarks**: Contains the calibrate config plus the `models` list being benchmarked
- **Simulations**: Contains the full calibrate simulation config (personas, scenarios, evaluators, tools, etc.)

---

## External Integrations

### Calibrate CLI

The backend orchestrates the `calibrate` CLI tool for actual evaluations:

```bash
# STT Evaluation (multiple providers, leaderboard generated automatically)
calibrate stt -p <provider1> <provider2> ... -l <language> -i <input_dir> -o <output_dir>

# TTS Evaluation (multiple providers, leaderboard generated automatically)
calibrate tts -p <provider1> <provider2> ... -l <language> -i <input_csv> -o <output_dir>

# LLM Tests / Benchmark (single or multiple models)
# Single model = unit test, multiple models = benchmark with leaderboard
calibrate llm -c <config.json> -m <model1> [model2 ...] -p <provider> -o <output_dir>

# LLM Simulations (text, runs 4 simulations in parallel)
calibrate llm simulations run -c <config.json> -o <output_dir> -m <model> -n 4

# Voice Agent Simulation (runs 4 simulations in parallel)
calibrate agent simulation -c <config.json> -o <output_dir> -n 4

```

**`--skip-verify` for agent connections**: All CLI commands for agent connections (`calibrate llm` for tests/benchmarks, `calibrate simulations --type text`) pass `--skip-verify` to skip the CLI's built-in connection verification. The backend already verifies connections via the `/agents/{uuid}/verify-connection` endpoint before running jobs, making the CLI's verification redundant.

**Agent connection verification** uses calibrate's Python API directly (not the CLI). The backend's `_verify_agent_connection()` in `agents.py` uses `TextAgentConnection(url=..., headers=...)` from `calibrate.connections`, then calls `await agent.verify(**kwargs)` where `kwargs` optionally contains `model` and/or `messages`. The `verify()` method returns `{"ok": bool, "error": str|None, "sample_output": dict|None}`:

- On success: `sample_output` contains the normalized agent response (`{"response": str|None, "tool_calls": list}`)
- On structure errors (wrong keys/types): `sample_output` contains the raw JSON the agent returned
- On connection/timeout/HTTP errors: `sample_output` is absent

When `model` is passed as a kwarg, it's included in the verification request payload so the agent can route to the right model — used for per-model benchmark verification. When `messages` is passed, those messages are sent instead of the library's default greeting (`[{"role": "user", "content": "Hello, are you there?"}]`). Both verify endpoints accept an optional `messages` field (list of message dicts) in the request body, which is forwarded as the `messages` kwarg to `agent.verify()`. For the post-save endpoint, the same model name normalization applies as for benchmarks: if `benchmark_provider` is not `openrouter`, the `provider/` prefix is stripped before sending to the agent (e.g. `openai/gpt-4.1` → `gpt-4.1`), but the original name is preserved as the key in `benchmark_models_verified`. The `sample_response` field in the API response carries this data through to the frontend. Two verify endpoints exist: `POST /agents/verify-connection` (pre-save, auth required, requires `agent_url` in the request body) and `POST /agents/{uuid}/verify-connection` (post-save, auth required, reads `agent_url`/`agent_headers` from the saved agent config). The pre-save endpoint is declared before the `/{agent_uuid}` routes to avoid FastAPI treating `verify-connection` as a UUID path parameter.

**Only successful verifications are persisted**: The post-save verify endpoint only writes to the agent config when verification succeeds. Failed verifications are returned in the API response (so the frontend can display the error) but do not update the stored config. This means a previously successful `connection_verified: true` is not overwritten by a subsequent failed attempt.

**Race condition protection (post-save verify)**: The post-save endpoint re-reads the agent's config from the database **after** the `await agent.verify()` call completes, right before persisting the result. This prevents concurrent verify calls (e.g., two different models verified simultaneously) from overwriting each other — without this, both calls would snapshot the config before the slow network call, then the second write would clobber the first's `benchmark_models_verified` entry.

**Security: URL validation and header sanitization** — `_verify_agent_connection()` applies two safeguards before making outbound requests:

1. **URL validation** (`_validate_agent_url()`): Rejects non-HTTP(S) schemes, missing hostnames, localhost, and `.local` domains by string check, then **resolves the hostname via `socket.getaddrinfo`** and rejects any resolved IP that is loopback, private (RFC 1918), link-local (`169.254.x.x`, `fe80::`), reserved, multicast, or unspecified — using Python's `ipaddress` module (`is_loopback`, `is_private`, `is_reserved`, `is_link_local`, `is_multicast`, `is_unspecified`). This prevents SSRF via DNS rebinding, numeric IP encoding tricks, and cloud metadata endpoints (`169.254.169.254`).
2. **Header sanitization** (`_sanitize_headers()`): Strips hop-by-hop and security-sensitive headers (`host`, `transfer-encoding`, `content-length`, `connection`, `upgrade`, `te`, `trailer`, `keep-alive`, `proxy-authorization`, `proxy-authenticate`, `proxy-connection`) from user-supplied `agent_headers` before forwarding.

**STT/TTS CLI Notes:**

- Multiple providers can be specified with `-p` (space-separated)
- The CLI handles parallelization internally
- Leaderboard is generated automatically (typically under `<output_dir>/leaderboard/` and/or root CSV/XLSX depending on CLI version)
- Optional **`--config path.json`** with `{"evaluators":[...]}` — backend writes this when `job.details.evaluators` is non-empty (`build_evaluator_cli_payload()`). Requires a calibrate build that recognizes `--config`.
- Run-level **`logs`** and per-provider **`logs`** are uploaded to S3 via the mechanisms in **S3 artifacts for STT/TTS eval jobs**
- No separate `eval` and `leaderboard` commands needed

### AWS S3

Used for:

- Storing input audio files (STT evaluation)
- Storing generated audio files (TTS evaluation, voice simulations)
- Storing evaluation results (JSON, CSV, Excel files)
- Presigned URLs for secure file access

Required environment variables:

- `S3_OUTPUT_BUCKET` - Target bucket for outputs
- `AWS_ACCESS_KEY_ID` / `AWS_SECRET_ACCESS_KEY` (optional, falls back to IAM role)
- `AWS_REGION` - Default: `ap-south-1`

**Uploading files to S3**: Routers use `upload_file_to_s3()` for single objects; **`upload_directory_tree_to_s3()`** and **`upload_top_level_files_to_s3()`** wrap it for directory batches (see STT/TTS and agent test sections). Never call `s3.upload_file(...)` directly for ad-hoc uploads. `upload_file_to_s3` guesses `Content-Type` from the file extension via `mimetypes.guess_type` (e.g. `.json` → `application/json`, `.wav` → `audio/wav`, `.mp3` → `audio/mpeg`). Files **without** an extension (e.g. calibrate's `logs`) get no `ContentType` unless extended later. Without `Content-Type`, boto3 defaults to `binary/octet-stream` for some clients. Existing objects uploaded before `Content-Type` guessing still behave as above.

### Supported Providers

**STT Providers**: deepgram, openai, sarvam, google, etc.

**TTS Providers**: smallest, cartesia, openai, google, elevenlabs, etc.

**LLM Providers**: OpenAI, Groq, OpenRouter, Google (via respective API keys)

---

## Configuration

### Environment Variables

```bash
# Database
DB_ROOT_DIR=/appdata/db          # Directory containing calibrate.db

# Job Queue
MAX_CONCURRENT_JOBS=2            # Max concurrent jobs per queue type (docker-compose default: 1)
MAX_CONCURRENT_JOBS_PER_USER=1   # Max concurrent jobs per user per queue type (default: 1, 0 to disable)

# User Limits
DEFAULT_MAX_ROWS_PER_EVAL=500    # Default max rows per eval run; overridden per-user via user_limits table
SUPERADMIN_EMAIL=admin@example.com  # Email for superadmin access (required for mutating /user-limits endpoints)

# CORS
CORS_ALLOWED_ORIGINS=*           # Comma-separated origins (e.g., "http://localhost:3000,https://app.example.com")

# JWT Authentication
JWT_SECRET_KEY=your-secret-key   # REQUIRED: At least 32 characters, change in production!
JWT_EXPIRATION_HOURS=168         # Token validity (default: 7 days)

# API Docs Authentication (HTTP Basic Auth)
DOCS_USERNAME=admin              # Username for /docs, /redoc, /openapi.json (default: admin)
DOCS_PASSWORD=changeme           # Password for docs access (default: changeme — change in production!)

# Sentry (Error Tracking)
SENTRY_DSN=                      # Sentry DSN (leave empty to disable)
SENTRY_ENVIRONMENT=development   # Environment name (development, staging, production)
SENTRY_TRACES_SAMPLE_RATE=1.0    # Performance monitoring sample rate (0.0-1.0)
SENTRY_PROFILES_SAMPLE_RATE=1.0  # Profiling sample rate (0.0-1.0)

# Langfuse Tracing (LLM Observability via OpenTelemetry)
ENVIRONMENT=development                              # Langfuse tracing environment
ENABLE_TRACING=true                                  # Enable/disable tracing
OTEL_EXPORTER_OTLP_ENDPOINT=https://host/api/public/otel  # OTLP endpoint (Langfuse self-hosted or cloud)
OTEL_EXPORTER_OTLP_HEADERS=Authorization=Basic%20xxx      # Base64-encoded Langfuse public:secret key
LANGFUSE_TRACING_ENVIRONMENT=                        # Langfuse tracing environment label
LANGFUSE_HOST=                                       # Langfuse host URL
LANGFUSE_PUBLIC_KEY=                                 # Langfuse public key
LANGFUSE_SECRET_KEY=                                 # Langfuse secret key

# AWS
S3_OUTPUT_BUCKET=your-bucket
AWS_ACCESS_KEY_ID=xxx
AWS_SECRET_ACCESS_KEY=xxx
AWS_REGION=ap-south-1

# Authentication
GOOGLE_CLIENT_ID=xxx.apps.googleusercontent.com

# Provider API Keys
OPENAI_API_KEY=xxx
DEEPGRAM_API_KEY=xxx
CARTESIA_API_KEY=xxx
SMALLEST_API_KEY=xxx
GROQ_API_KEY=xxx
GOOGLE_API_KEY=xxx
GOOGLE_APPLICATION_CREDENTIALS=/path/to/credentials.json
SARVAM_API_KEY=xxx
ELEVENLABS_API_KEY=xxx
OPENROUTER_API_KEY=xxx
```

### Agent Types

Agents have a `type` column in the database (and a `type` field in the API) that determines how they are evaluated:

| Type              | Description                                                                                                |
| ----------------- | ---------------------------------------------------------------------------------------------------------- |
| `agent` (default) | Platform-managed agent — system prompt, tools, LLM/STT/TTS providers are defined in the agent config       |
| `connection`      | External agent — the user's own agent running at `agent_url`; calibrate sends HTTP requests to it directly |

The `type` is set at creation time via `POST /agents` and returned in all agent responses. Existing agents default to `agent` via the `ALTER TABLE` migration. The duplicate endpoint carries over the original agent's type. All Pydantic models use `Literal["agent", "connection"]` for the `type` field — invalid values are rejected with a 422 at both input and output serialization.

### Agent Configuration Schema

**Agent** (`type: "agent"`):

```json
{
  "system_prompt": "You are a helpful assistant...",
  "llm": {
    "provider": "openai",
    "model": "gpt-4.1"
  },
  "stt": {
    "provider": "deepgram",
    "model": "nova-2"
  },
  "tts": {
    "provider": "elevenlabs",
    "voice_id": "xxx"
  },
  "settings": {
    "agent_speaks_first": true,
    "max_assistant_turns": 20
  },
  "system_tools": {
    "end_call": true
  },
  "data_extraction_fields": []
}
```

**Agent connection** (`type: "connection"`):

```json
{
  "agent_url": "https://your-agent.com/chat",
  "agent_headers": {
    "Authorization": "Bearer YOUR_API_KEY"
  },
  "connection_verified": true,
  "connection_verified_at": "2026-04-08T12:00:00+00:00",
  "connection_verified_error": null,
  "supports_benchmark": true,
  "benchmark_provider": "openrouter",
  "benchmark_models_verified": {
    "openai/gpt-4.1": { "verified": true, "verified_at": "...", "error": null }
  },
  "settings": {
    "agent_speaks_first": true,
    "max_assistant_turns": 20
  }
}
```

The `connection_verified` / `benchmark_models_verified` fields are managed by the verify endpoints, can also be set directly via `PUT /agents/{uuid}` (top-level fields on the update request body, merged into config), and are reset automatically when `agent_url` or `agent_headers` change (set to `false`/`null`/`{}` respectively, so the fields are always present in the response). They are stripped when duplicating an agent.

**Simulation Config Generation**: When running simulations, `_build_calibrate_simulation_config()` in `routers/simulations.py` loads linked rows via **`get_evaluators_for_simulation()`** and passes them through **`build_evaluator_cli_payload()`** (`llm_judge.py`) so the written `simulation_config.json` contains calibrate’s `evaluators` array (not the frozen `metrics` table). Write-time rules for linking evaluators to a simulation are under **Simulations** in **API Endpoints** above.

On **`POST /simulations/{uuid}/run`**, the same ordered evaluator list is persisted on the job as **`details.evaluators`** (minimal UUID-only snapshot) so status and share endpoints can correlate CSV output with **`evaluators.uuid`** without rereading the simulation entity.

- `tools` - Built from linked agent tools (see Tool Configuration Schema below)
- `evaluators` - CLI payload entries (`name`, `system_prompt`, `judge_model`, `type`, optional `scale_min`/`scale_max` for rating)
- `settings.agent_speaks_first` - Defaults to `true` if not specified in agent config
- `settings.max_turns` - Mapped from agent config's `settings.max_assistant_turns` (defaults to `50` if not set)

### Tool Configuration Schema (in Calibrate Config)

Tools in calibrate configs (used by both simulations and agent tests) support two types: **structured_output** (default) and **webhook**. The `build_tool_configs()` utility function in `utils.py` handles building these configs from database tool records.

**Structured Output Tool** (for tools that return structured data to the LLM):

```json
{
  "type": "structured_output",
  "name": "get_user_info",
  "description": "Retrieves user information",
  "parameters": [
    {
      "id": "user_id",
      "type": "string",
      "description": "The user ID",
      "required": true
    }
  ]
}
```

**Webhook Tool** (for tools that call external HTTP endpoints):

```json
{
  "type": "webhook",
  "name": "get_presigned_url",
  "description": "Always call this tool after receiving a user response",
  "parameters": [],
  "webhook": {
    "method": "POST",
    "url": "http://localhost:8000/presigned-url",
    "timeout": 20,
    "headers": [{ "name": "Authorization", "value": "Bearer X" }],
    "queryParameters": [
      {
        "id": "key",
        "type": "string",
        "description": "Query param description",
        "required": true
      }
    ],
    "body": {
      "description": "Request body description",
      "parameters": [
        {
          "id": "task_type",
          "type": "string",
          "description": "Type of task",
          "required": true
        }
      ]
    }
  }
}
```

The tool type is determined by the `type` field in the tool's `config` column in the database. If not specified, defaults to `"structured_output"`.

### Test Case Configuration Schema

Tests store conversation history in the `history` field (OpenAI chat message format) and evaluation instructions in the `evaluation` field. Two evaluation types are supported. **Rubric content for `response` tests** is no longer a free-form string in the API: new writes use **`test_evaluators`** (pivot) to pin `evaluator_version_id` and optional `variable_values`. At run time, `routers/agent_tests.py` (`_build_calibrate_config` + `build_test_evaluators_payload`) merges linked evaluators into the calibrate config; the stored `config` row may show `evaluation: { "type": "response" }` with no inline `criteria` string.

`_build_calibrate_config(agent, tests)` returns **`(config, evaluators_by_test_id)`**. See **Agent Test Job Results Format** (`details.calibrate_config`, `details.evaluators_by_test_id`, `_calibrate_config_from_agent_test_job`, and read-time enrichment).

**Tool call evaluation:**

```json
{
  "history": [
    { "role": "assistant", "content": "Hello, how can I help you today?" },
    { "role": "user", "content": "Hi" },
    {
      "role": "assistant",
      "tool_calls": [
        {
          "id": "...",
          "function": { "name": "some_tool", "arguments": "{}" },
          "type": "function"
        }
      ]
    },
    {
      "role": "tool",
      "content": "{\"status\": \"received\"}",
      "tool_call_id": "..."
    }
  ],
  "evaluation": {
    "type": "tool_call",
    "tool_calls": [
      {
        "tool": "get_weather",
        "arguments": { "city": "Mumbai" },
        "accept_any_arguments": false
      }
    ]
  }
}
```

**Response evaluation (typical stored `config` after evaluator migration / new creates):**

```json
{
  "history": [
    { "role": "assistant", "content": "Hello, how can I help you today?" },
    { "role": "user", "content": "What is your return policy?" }
  ],
  "settings": { "language": "english" },
  "evaluation": {
    "type": "response"
  }
}
```

**Legacy `evaluation.criteria` string** (still in some DB rows; `_backfill_test_evaluator_links()` links these rows to the seeded `default-llm-next-reply` evaluator on startup, but only when the test has no active `test_evaluators` rows at all). This avoids re-adding the default evaluator after a user has replaced legacy criteria with custom evaluator links. `_build_calibrate_config()` can still promote a legacy criteria string to a synthetic link at run time when no links exist. The final response-test handoff then overwrites `evaluation.criteria` with the structured refs from `criteria_per_test`; do not add a later "legacy string" cleanup there, because by that point the field is intentionally no longer the original stored string:

```json
{
  "evaluation": {
    "type": "response",
    "criteria": "The agent should clearly explain the return policy in a helpful and friendly tone"
  }
}
```

The `history` field uses the OpenAI chat messages format with `role` (system/user/assistant/tool), `content`, and optional `tool_calls`/`tool_call_id`/`name` fields. The `evaluation.type` is `"response"` (LLM judge via linked evaluators) or `"tool_call"` (expected tool call matching). The optional `settings` field can include `language` (and other keys accepted by calibrate). Plain `criteria` strings are legacy; **`POST /tests/bulk` does not accept `criteria`** — it requires **`evaluators`** on each row when `type` is `"response"` (see below).

### Bulk Test Upload

`POST /tests/bulk` creates multiple tests in a single request. All tests in the batch must share the same `type`.

**Request:**

```json
{
  "type": "response",
  "language": "hindi",
  "agent_uuids": ["agent-uuid-1", "agent-uuid-2"],
  "tests": [
    {
      "name": "test-greeting",
      "conversation_history": [{ "role": "user", "content": "Hello" }],
      "evaluators": [
        {
          "evaluator_uuid": "…",
          "variable_values": {
            "criteria": "Response should be a polite greeting"
          }
        }
      ]
    }
  ]
}
```

- `type`: `"response"` or `"tool_call"` — stored as `evaluation.type` in the config (same values used in both the API and the stored config)
- `language`: Optional **top-level only** (not per row) — when set, **every** test in the batch gets `config.settings.language`. Per-test language is only possible by creating/updating tests individually with `settings` inside `config`.
- `agent_uuids`: Optional list of agent UUIDs to link all created tests to. Agents are validated upfront (must exist and be owned by the user) before any tests are created. Linking failures for individual test-agent pairs are surfaced in the response `warnings` array but don't fail the request.
- Maximum batch size is 500 tests (enforced by `BulkTestUpload.MAX_BATCH_SIZE` in the Pydantic validator)
- Each test requires a unique `name` — validated both within the batch and against existing tests for the user
- `conversation_history`: Required, OpenAI chat message format — stored as `history` in the config to match the single-test format
- `evaluators`: Required when `type` is `"response"` — non-empty list of `EvaluatorRef` (`evaluator_uuid`, optional `variable_values`). **`EvaluatorRef` uses `extra='forbid'`** — unknown keys (including **`version_uuid`**) produce **422**. The stored **`test_evaluators`** row pins the evaluator's **live** version at write time (no client-supplied version id). Validated in `tests.py` (`_validate_evaluators`): evaluator must exist, be visible to the user, and have **`evaluator_type == "llm"`** (STT/TTS/simulation evaluators are rejected with 400).
- `tool_calls`: Required when `type` is `"tool_call"`, array of `{tool, arguments?, accept_any_arguments?}`

All tests are inserted in a single DB transaction — if any name conflicts with an existing test, none are created. The `bulk_create_tests()` function in `db.py` handles the atomic insert with name uniqueness validation. Agent linking happens after test creation via `add_test_to_agent()`. The response includes a `warnings` array (nullable) that reports any agent linking failures, and the `message` reflects the actual number of successfully linked agents rather than the requested count.

### Bulk Test Delete

`POST /tests/bulk-delete` deletes multiple tests in a single request. Only tests owned by the authenticated user are deleted; others are silently skipped.

**Request:**

```json
{
  "test_uuids": ["uuid-1", "uuid-2", "uuid-3"]
}
```

**Response:**

```json
{
  "deleted_count": 3,
  "message": "Successfully deleted 3 test(s)"
}
```

The `bulk_delete_tests()` function in `db.py` performs a single SQL `UPDATE` scoped to the user's tests, and also soft deletes related `agent_tests` entries in the same transaction.

### Bulk Unlink Tests from Agent

`POST /agent-tests/bulk-unlink` removes multiple test links from an agent at once.

**Request:**

```json
{
  "agent_uuid": "agent-uuid",
  "test_uuids": ["test-uuid-1", "test-uuid-2"]
}
```

**Response:**

```json
{
  "deleted_count": 2,
  "message": "Successfully unlinked 2 test(s) from agent"
}
```

The `bulk_remove_tests_from_agent()` function in `db.py` performs a single SQL `UPDATE` with an `IN` clause.

### Persona Configuration Schema

```json
{
  "gender": "female",
  "language": "english",
  "interruption_sensitivity": "medium"
}
```

---

## Key Design Decisions

### 1. SQLite over PostgreSQL

- **Rationale**: Simpler deployment, no external database service needed
- **Trade-off**: Single-writer limitation, not horizontally scalable
- **Mitigation**: Using connection context manager for proper cleanup

### 2. Threading over Async Tasks

- **Rationale**: Long-running subprocess calls to `calibrate` CLI
- **Implementation**: Python `threading.Thread` with `daemon=True`
- **Recovery**: Job details stored in DB, recovered on app restart

### 3. Soft Deletes

- **Rationale**: Data preservation, audit trail, easy recovery
- **Implementation**: `deleted_at` timestamp, queries filter `deleted_at IS NULL`

### 4. JSON Configuration Columns

- **Rationale**: Flexible schemas for agent/tool/test configs
- **Trade-off**: No database-level validation
- **Mitigation**: Pydantic validation at API layer

### 5. Presigned URLs for S3

- **Rationale**: Secure, time-limited access without exposing credentials
- **Implementation**: 1-hour expiration for generated URLs
- **Storage Pattern**: Store S3 keys (not presigned URLs) in the database; generate presigned URLs on-the-fly when fetching job status. This prevents expired URLs from being served to clients.

**Helper Functions** (in `utils.py`):

| Function                            | Purpose                                                                         |
| ----------------------------------- | ------------------------------------------------------------------------------- |
| `generate_presigned_download_url()` | Generate presigned URL for `get_object` (downloads)                             |
| `generate_presigned_upload_url()`   | Generate presigned URL for `put_object` (uploads)                               |
| `upload_file_to_s3()`               | Upload a local file to S3 with `Content-Type` auto-detected from file extension |

Presigned URL helpers return `None` on failure, allowing callers to handle errors (skip, fallback to S3 path, raise error).

**Example Pattern (TTS/STT evaluation results):**

```python
from utils import generate_presigned_download_url

# When storing results - save S3 key only
result_row["audio_path"] = audio_s3_key  # e.g., "tts/evals/{job_id}/outputs/{provider}/audios/0.wav"

# When fetching status - generate presigned URL on-the-fly
if job["status"] == TaskStatus.DONE.value:
    for result in results:
        if result.get("audio_path") and not result["audio_path"].startswith("http"):
            presigned_url = generate_presigned_download_url(result["audio_path"])
            if presigned_url:
                result["audio_path"] = presigned_url
```

**Backwards Compatibility**: When generating presigned URLs, skip entries that already start with `http` or `s3://` (older data with stored URLs).

**Dataset item `audio_path`**: Stored as full `s3://bucket/key` URIs (unlike eval results which store bare S3 keys). The `_presign_audio_path()` helper in `routers/datasets.py` parses out the bucket and key before calling `generate_presigned_download_url()`. Falls back to the raw path if presigning fails. This applies to all dataset item responses (list, detail, add, update).

### 6. Background Job Pattern

- **Rationale**: STT/TTS/LLM evaluations are long-running (minutes)
- **Pattern**: Create job → Return task_id → Poll for status
- **Recovery**: Jobs with `in_progress` status restarted on app boot

---

## Dependencies

Key Python packages:

- `fastapi>=0.115.6,<0.122.0` - Web framework (upper bound required for compatibility with calibrate-agent's pipecat-ai dependency)
- `uvicorn>=0.40.0` - ASGI server
- `boto3>=1.34.0` - AWS SDK for S3
- `pydantic>=2.0.0` - Data validation
- `python-dotenv>=1.0.0` - Environment variable loading
- `openpyxl>=3.1.5` - Excel file parsing for leaderboards
- `httpx>=0.27.0` - Async HTTP client (used in `auth.py` for Google OAuth token verification)
- `python-jose[cryptography]>=3.3.0` - JWT token encoding/decoding
- `bcrypt>=4.0.0` - Password hashing for username/password authentication
- `sentry-sdk[fastapi]>=2.0.0` - Error tracking and performance monitoring

External:

- `calibrate-agent` - Core CLI tool for evaluations and simulations (installed from PyPI via `pip install calibrate-agent`)
- `ffmpeg` - Required by calibrate CLI for audio processing (TTS, voice simulations)
- `nltk` data (`punkt_tab`) - Required by pipecat for sentence tokenization; pre-downloaded in Dockerfile to avoid runtime network issues

---

## Deployment

### Docker Build

```bash
docker build -t calibrate-backend .
```

**Package installation**: The Dockerfile installs packages via `uv sync` which reads from `pyproject.toml` and `uv.lock`. The `calibrate-agent` package (and its dependencies like pipecat, nltk, etc.) is listed as a dependency in `pyproject.toml` and installed automatically.

**Important**: `uv sync` installs packages into a `.venv` virtual environment, not system Python. Any `RUN` commands in the Dockerfile that need installed packages must use `uv run python ...` (not bare `python`), otherwise the system Python won't find them.

### Docker Compose

```bash
docker-compose up -d
```

The container:

- Exposes port 8000
- Mounts `/appdata` volume for database and credentials
- Requires environment variables via `.env` file or docker-compose

### GitHub Actions Deployments

- `.github/workflows/deploy-staging.yml` remains manually triggered via `workflow_dispatch`.
- `.github/workflows/deploy.yml` deploys production only on the GitHub Release `published` event. It does not expose a manual `workflow_dispatch` trigger.
- Production deploys are tied to publishing a GitHub Release, not to a raw `git push` of a `v*` tag. Pushing a tag alone will not deploy — a Release must be published (the Release can target an existing pushed tag or create a new one). Re-publishing the same Release retriggers the deploy.

### Development

```bash
cd src
uv run uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

---

## Common Patterns in Codebase

### Database Function Pattern

```python
def get_entity(uuid: str) -> Optional[Dict[str, Any]]:
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT * FROM entities WHERE uuid = ? AND deleted_at IS NULL",
            (uuid,)
        )
        row = cursor.fetchone()
        if row:
            return _parse_entity_row(row)
        return None
```

### Router Endpoint Pattern (with JWT Auth)

```python
from auth_utils import get_current_user_id

@router.get("/{uuid}", response_model=EntityResponse)
async def get_entity_endpoint(uuid: str, user_id: str = Depends(get_current_user_id)):
    entity = get_entity(uuid)
    if not entity:
        raise HTTPException(status_code=404, detail="Entity not found")
    # Verify user owns this entity
    if entity.get("user_id") != user_id:
        raise HTTPException(status_code=403, detail="Access denied")
    return entity
```

### Nested Resource Auth Pattern

For resources that don't have direct `user_id` (e.g., `simulation_jobs` linked via `simulation_id`), verify ownership through the parent entity:

```python
@router.get("/run/{task_id}", response_model=JobStatusResponse)
async def get_job_status(task_id: str, user_id: str = Depends(get_current_user_id)):
    job = get_job(task_id)
    if not job:
        raise HTTPException(status_code=404, detail="Task not found")

    # Verify user owns the parent entity
    parent_id = job.get("parent_id")
    if parent_id:
        parent = get_parent(parent_id)
        if not parent or parent.get("user_id") != user_id:
            raise HTTPException(status_code=404, detail="Task not found")  # 404, not 403

    return JobStatusResponse(...)
```

**Security note**: Return 404 (not 403) when access is denied to prevent information leakage about whether a resource exists.

### Background Task Pattern (with Queueing)

```python
# Job types sharing a queue
JOB_TYPES = ["type-a", "type-b"]

# Job starter callback (registered at module load)
def _start_job_from_queue(job: dict) -> bool:
    details = job.get("details", {})
    request = TaskRequest(**details)
    thread = threading.Thread(target=run_task, args=(job["uuid"], request), daemon=True)
    thread.start()
    return True

register_job_starter("type-a", _start_job_from_queue)

# Endpoint with queue check and JWT auth
@router.post("/run", response_model=TaskCreateResponse)
async def start_task(request: TaskRequest, user_id: str = Depends(get_current_user_id)):
    # Check capacity
    can_start = can_start_job(JOB_TYPES)
    initial_status = TaskStatus.IN_PROGRESS.value if can_start else TaskStatus.QUEUED.value

    job_id = create_job(
        job_type="type-a",
        user_id=user_id,
        status=initial_status,
        details={"param": value, "s3_bucket": bucket},
    )

    if can_start:
        thread = threading.Thread(target=run_task, args=(job_id, request), daemon=True)
        thread.start()

    return TaskCreateResponse(task_id=job_id, status=initial_status)

# Task function with queue trigger on completion
def run_task(task_id: str, request: TaskRequest):
    try:
        # ... do work (e.g., run CLI command) ...
        result = run_cli_command(...)  # Returns {"success": bool, "error": str|None, ...}

        # Determine status based on whether the command succeeded
        final_status = TaskStatus.DONE.value if result["success"] else TaskStatus.FAILED.value
        update_job(task_id, status=final_status, results={...})
    except Exception as e:
        traceback.print_exc()
        capture_exception_to_sentry(e)  # Log to Sentry as unhandled
        # Preserve existing results (e.g., partial test_results from intermediate updates)
        existing_job = get_job(task_id)
        existing_results = (existing_job.get("results") or {}) if existing_job else {}
        existing_results["error"] = str(e)
        update_job(task_id, status=TaskStatus.FAILED.value, results=existing_results)
    finally:
        # Start next queued job if capacity allows
        try_start_queued_job(JOB_TYPES)
```

**Preserving Results on Failure**: When a job fails due to an exception, ALL exception handlers (including `CalledProcessError`) fetch existing results from the database and preserve them (e.g., `total_tests`, `test_results`, `model_results`) while adding the error string. This ensures that partial results from intermediate updates — including `total_tests` — are still available to clients even when the job ultimately fails.

**Agent test `error` field is a boolean in the API**: The `error` field in `TestRunStatusResponse`, `BenchmarkStatusResponse`, and `AgentTestRunListItem` is `bool` (not a string). The full error string is stored internally in the DB for debugging/Sentry, but the API endpoints convert it to `true`/`false` via `bool(results.get("error"))`. This keeps the client interface clean while preserving diagnostic detail server-side.

**Sentry Error Logging**: All job failures are logged to Sentry using the `capture_exception_to_sentry()` utility function from `utils.py`. This function:

- Marks exceptions as **unhandled** so they appear as unresolved issues in Sentry (not handled/resolved)
- Calls `sentry_sdk.flush(timeout=2)` to ensure events are sent immediately (critical for background tasks that may complete before Sentry's async queue is processed)

This applies to:

- STT/TTS evaluation failures (provider-level and task-level)
- Agent test and benchmark failures (model-level and task-level)
- Simulation failures (text and voice)
- CLI command failures (non-zero exit codes OR tracebacks in stderr) - logged with full stderr output

**CLI Failure Detection**: All CLI wrapper functions (agent tests, benchmarks, text simulations, voice simulations) use a two-layer approach:

1. **Exit code**: Non-zero return code → immediate failure with stderr logged to Sentry
2. **Output file validation**: Exit code 0 but expected structured output missing → failure with descriptive error
   - Agent tests: `results.json` and `metrics.json` both absent
   - Benchmarks: `_find_all_results_in_output()` returns empty (no per-model result folders)
   - Simulations (text/voice): no `simulation_persona_*` directories with completed results

Stderr is logged for debugging but **never** used for failure detection. The calibrate CLI's subprocess may emit benign cleanup tracebacks (e.g., httpx `AsyncClient.aclose()` "Event loop is closed" errors) that are not real failures. Relying on exit code + structured output avoids false positives from noisy stderr.

### Incremental Job Processing Pattern (Simulations, Agent Tests, Benchmarks)

For long-running jobs that produce incremental outputs (text/voice simulations, agent unit tests, and benchmarks), use non-blocking subprocess execution with polling.

**Note**: This polling-loop pattern applies to simulations (`text`, `voice`), agent unit tests (`llm-unit-test`), and benchmarks (`llm-benchmark`). STT/TTS evaluations use a different approach: the background task blocks on `process.wait()`, but the status API reads intermediate results directly from disk on-demand (see "STT/TTS Evaluation Flow" section).

```python
def run_incremental_task(task_id: str, output_dir: Path):
    # Start process without blocking
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    processed_items = set()
    results = []
    prev_state = None  # Track state to avoid unnecessary DB updates

    # Monitor for new outputs while process runs
    while process.poll() is None:
        transcript_lengths = []  # For change detection

        for item in output_dir.iterdir():
            if item.name not in processed_items and is_item_complete(item):
                result = parse_item(item)
                results.append(result)
                processed_items.add(item.name)
            # Track in-progress items too (e.g., transcript length)
            transcript = get_transcript(item)
            transcript_lengths.append((item.name, len(transcript)))

        # Build state tuple for comparison
        current_state = (len(results), tuple(sorted(transcript_lengths)))

        # Only update DB if state changed
        if current_state != prev_state:
            update_job(task_id, status="in_progress", results={
                "completed": len(results),
                "items": results
            })
            prev_state = current_state

        time.sleep(2)  # Poll interval

    # Final update with all results
    update_job(task_id, status="done", results={"items": results, "metrics": ...})
```

**Critical**: Track previous state and only update DB when state changes. This:

- Avoids unnecessary DB writes every 2-second poll when nothing changed
- Preserves `updated_at` timestamp for the 5-minute timeout check (without this, every poll would reset `updated_at`, making stuck job detection unreliable)

Key aspects:

- Use `subprocess.Popen` instead of `subprocess.run` for non-blocking execution
- **Important**: Write stdout/stderr to temp files instead of pipes when polling with `process.poll()`. Using pipes with polling can cause deadlocks if the subprocess output exceeds the pipe buffer size:

```python
# Redirect to files to avoid pipe buffer deadlock during polling
stdout_path = output_dir / f"{provider}_stdout.log"
stderr_path = output_dir / f"{provider}_stderr.log"
stdout_f = open(stdout_path, "w")
stderr_f = open(stderr_path, "w")

try:
    process = subprocess.Popen(cmd, stdout=stdout_f, stderr=stderr_f, ...)
    while process.poll() is None:
        # Read intermediate results from output directory
        time.sleep(2)
finally:
    stdout_f.close()
    stderr_f.close()

# Read captured output after process completes
with open(stdout_path) as f:
    stdout = f.read()
```

- Check for completion markers (e.g., `evaluation_results.csv` for simulations, `results.json` for agent tests)
- Update DB incrementally so status API can return partial results
- Use thread-safe mechanisms (locks) when multiple threads update shared state
- Final aggregated metrics only computed after all items complete

---

## Future Considerations

1. **Scalability**: Consider PostgreSQL + Redis for multi-instance deployment
2. **Task Queue**: Consider Celery/RQ for more robust job management
3. **Caching**: Add Redis caching for frequently accessed data
4. **Rate Limiting**: Add rate limiting for API protection
5. **Pagination**: Implement pagination for list endpoints as data grows
