# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Workflow

- **Commit when the work is done — don't wait to be asked.** Once all the required changes for a task are made and the scoped tests pass, create the commit yourself. Use a focused message that explains the *why*. Do not push unless explicitly asked. The standard safety rules still apply: never amend published commits, never skip hooks, never `git add -A` blindly, never commit secrets.
- **Keep CLAUDE.md in sync with the app's mental model.** After making changes, ask yourself: does this commit shift a *high-level* understanding of the app — a load-bearing invariant, an architectural rule, a convention, a non-obvious gotcha, a new subsystem, or a contract between components? If yes, update CLAUDE.md in the **same commit** so future Claude sessions inherit the new mental model. Don't record routine fixes, file moves, renames, or anything obvious from reading the code — only changes a fresh reader would otherwise miss. When in doubt, prefer updating an existing bullet over adding a new one.

## Project Summary

Calibrate Backend is a FastAPI REST API that wraps the `calibrate` CLI tool to orchestrate long-running evaluation and simulation jobs:

1. **STT/TTS evaluation** — benchmark speech providers against datasets
2. **LLM agent testing** — unit tests and multi-model benchmarks
3. **Voice/chat simulations** — simulated conversations between AI agents and personas across scenarios

The backend persists state in SQLite, stores artifacts in S3, and spawns `calibrate` CLI subprocesses to do the actual work. Authentication is Google OAuth or email/password, gated by JWTs.

## Required Reading

Before making changes, read `.cursor/rules/architecture.md`. It is the canonical, exhaustive reference for the DB schema, API surface, job lifecycle, queueing, process management, abort/timeout semantics, and non-obvious gotchas (SQLite `ALTER TABLE ADD COLUMN` migration limits, UTC timestamp handling, exact-match model-folder mapping, race conditions in the simulation abort path, etc.). Do not reinvent those details from the code — they are load-bearing.

Other relevant rules in `.cursor/rules/`:

- `env-var.md` — **when adding/changing/removing env vars, also update `src/.env.example`, `docker-compose.yml`, and the GitHub Actions workflows**
- `design.md` — frontend design tokens (mostly not relevant to this backend repo)
- `context-first.md` — enforces reading `architecture.md` first

## Commands

Install dependencies:

```bash
uv sync --frozen
```

Run the dev server (note the `cd src` — the app is run from inside `src/`, not the repo root):

```bash
cd src
uv run uvicorn main:app --reload
```

API at http://localhost:8000, docs at http://localhost:8000/docs (HTTP Basic Auth, credentials from `DOCS_USERNAME`/`DOCS_PASSWORD`, default `admin`/`changeme`).

Docker:

```bash
docker build -t calibrate-backend .
docker-compose up -d
```

A **pytest** suite lives under **`tests/`** (dev deps: **`[dependency-groups] dev`** — `pytest`, `pytest-cov`, `pre-commit`). Run from repo root: **`uv run --group dev pytest`**. Coverage is enabled by default via `pyproject.toml` (`--cov=src --cov-report=term-missing`); scope to a single file/test with **`uv run --group dev pytest tests/test_foo.py::test_bar -q`** or **`uv run --group dev pytest -k <pattern>`**. `tests/conftest.py` sets `DB_ROOT_DIR` to a tmp dir and seeds JWT/S3 env vars before importing any `src/` module, then initializes the schema once per session — individual tests use fresh UUIDs to stay isolated.

CI runs on push/PR to `main` (`.github/workflows/tests.yml`) and uploads coverage to Codecov. Optional local hook: **`.githooks/pre-commit`** — enable with **`git config core.hooksPath .githooks`** (runs tests on **`main`** only when `.py`/`.json` are staged). There is no separate linter or formatter in `pyproject.toml`.

## Testing Discipline (load-bearing)

Treat tests as part of the feature, not an afterthought. The rules are non-negotiable:

- **New feature ⇒ new tests.** Any new function, endpoint, router, helper, or branch added to `src/` must ship with tests that exercise it. Mirror the layout: `src/foo.py` → `tests/test_foo.py`; a new router endpoint extends the matching `tests/test_routers_*.py` file. If no test file fits, create one alongside the existing ones — do not bury new behavior inside an unrelated test.
- **Modified feature ⇒ updated tests.** When you change behavior of existing code (bug fix, refactor, semantic change), update or add tests that pin the new behavior. If the existing test still passes after a real behavior change, the test is wrong — fix the test, don't silently lose the assertion.
- **Verify before claiming done.** Before reporting any code change as working, run the tests that cover the changed code and confirm they pass. Run the *scoped* subset (the specific file or `-k` selector that hits your change), not the whole suite — full-suite runs are slow and reserved for pre-PR or pre-commit. Type checks and "the diff looks right" are not substitutes for an actual passing test run.
- **No skipping or weakening tests to make them pass.** If a test fails after your change, the default assumption is the code is wrong, not the test. Only update an assertion when you have explicitly decided the old behavior was incorrect, and say so in the PR description.
- **Background-thread / subprocess code needs a test seam.** Job runners spawn `calibrate` and threads — tests stub `subprocess.Popen` / monkeypatch the runner registry rather than really launching CLI processes. Follow the patterns in `test_run_tasks.py`, `test_job_recovery.py`, and `test_routers_*.py`.

## Code Layout

All application code lives under `src/` and runs with `src/` as the working directory (see Dockerfile). Imports are flat (`from db import ...`, `from auth_utils import ...`) — there is no package namespace.

- `src/main.py` — FastAPI app, lifespan hooks (job recovery on startup), custom HTTP-Basic-auth'd `/docs` routes, Sentry/OTEL setup, router wiring
- `src/db.py` — single-file SQLite layer (~7000 lines). All schema DDL, migrations (wrapped in `try/except sqlite3.OperationalError` — see architecture.md gotcha on `DEFAULT CURRENT_TIMESTAMP`), and CRUD. Soft deletes (`deleted_at IS NULL` filter) everywhere.
- `src/utils.py` — S3 helpers, presigned URLs, `build_tool_configs()`, process-group kill helpers, job-queue registry (`register_job_starter`, `can_start_*_job`, `try_start_queued_*_job`), `TaskStatus` enum, `is_job_timed_out()`, Sentry capture helper
- `src/auth_utils.py` — JWT verification, `get_current_user_id` dependency, `require_superadmin`
- `src/dataset_utils.py` — `resolve_dataset_inputs()` / `inject_dataset_item_ids()` shared by STT and TTS routers
- `src/job_recovery.py` — on startup, kills orphaned process groups and restarts `in_progress` jobs
- `src/llm_judge.py` — `{{variable}}` template rendering, OpenRouter-based judge invocation, and `build_evaluator_cli_payload()` that shapes linked evaluators into the dict sent to the calibrate CLI (STT/TTS/LLM tests/simulations all share this).
- `src/annotation_eval_runner.py` — background worker for annotation-task evaluator runs; routes LLM-judge work through the calibrate CLI's STT `--eval-only` mode so judge logic isn't duplicated.
- `src/annotation_metrics.py` — human-vs-human and human-vs-evaluator agreement metrics for annotation tasks (pairwise mean agreement on shared rows; row-level overall has `evaluator_id=None`).
- `src/routers/` — one file per resource: `auth`, `users`, `user_limits`, `datasets`, `personas`, `scenarios`, `agents`, `tools`, `agent_tools`, `tests`, `agent_tests`, `stt`, `tts`, `simulations`, `evaluators`, `annotators`, `annotation_tasks`, `annotation_agreement`, `jobs`, `public`. Each router registers a job starter via `register_job_starter(...)` at module load so `try_start_queued_*_job` can resume queued work. `public.py` bypasses auth by design — its routes are excluded from the auth middleware so anyone with a valid `share_token` can view shared results.

## Architectural Load-Bearing Facts

These are the invariants most likely to trip up an edit. The architecture doc has the full explanation; this is the short form.

- **Three independent job queues**: eval (`stt-eval`, `tts-eval`), agent-test (`llm-unit-test`, `llm-benchmark`), simulation (`text`, `voice`). Each enforces both a global and per-user concurrency limit. Every mutation that completes/deletes/aborts a job must call `try_start_queued_*_job(...)` for that queue — otherwise queued work stalls.
- **User scoping is mandatory**: every DB call takes `user_id`, and routers return 404 (not 403) on access-denied to avoid leaking existence. `user_id` on `agent_test_jobs` / `simulation_jobs` is derived by JOIN through the parent entity.
- **SQLite stores UTC**; use `datetime.utcnow()` when comparing against `CURRENT_TIMESTAMP` columns. Never `datetime.now()`.
- **Subprocesses**: always start with `start_new_session=True`; store `pid`/`pgid` in job details; kill via `os.killpg` SIGTERM→SIGKILL (catch both `ProcessLookupError` and `PermissionError` on macOS). Agent test jobs are the exception — they block on `process.wait()` and do not track PIDs because `agent_test_jobs` has no `details` column.
- **Polling loops that spawn `calibrate`**: redirect stdout/stderr to temp files (not pipes) to avoid buffer-full deadlocks; poll every 2s; only `update_job` when state actually changes (to preserve `updated_at` for the 5-minute timeout check); read intermediate outputs from disk.
- **Simulation abort**: check `_is_job_aborted(task_id)` at every layer (inside polling loop, inside intermediate-update writer, after loop exit, inside `run_simulation_task`, and all exception handlers). Missing any one re-opens a race where the monitoring thread overwrites the abort state.
- **STT/TTS intermediate results are disk-only** during `in_progress` — error/timeout handlers must call `_collect_intermediate_results()` / `_collect_tts_intermediate_results()` before writing failure to DB, or successful providers' data is lost. Timeout handlers must **merge** (not overwrite) with existing DB `success: true` entries.
- **Presigned URLs are generated on-the-fly** from stored S3 keys — never persist presigned URLs in the DB (they expire). Dataset item audio paths are stored as full `s3://bucket/key` URIs (exception to the bare-key convention); `_presign_audio_path()` handles parsing.
- **Benchmark model-to-folder matching is exact**, not substring. Substring matching previously caused silent cross-model contamination (e.g. `gpt-5.4` absorbing `gpt-5.4-mini`'s results). If the calibrate CLI adds a new folder naming convention, extend the candidate list in `_match_model_to_folder`.
- **`MAX_CONCURRENT_JOBS` default is 1** in `docker-compose.yml`. In practice jobs serialize; this matters when reasoning about the queue.
- **FastAPI upper bound** (`<0.122.0`) in `pyproject.toml` is load-bearing — it's constrained by `calibrate-agent` → `pipecat-ai`. Don't loosen casually.
- **Dockerfile quirk**: `uv sync` installs into `.venv`, so any `RUN python ...` step must be `RUN uv run python ...` (see the `nltk.download('punkt_tab')` line).
- **Annotation-job soft-delete cascades through reads, not rows.** `soft_delete_annotation_job` only flips `annotation_jobs.deleted_at`. The job's annotation rows are NOT touched. Every annotation read path (`get_annotations_for_task`, `get_annotations_for_item`, `get_annotations_for_slots`, `get_annotations_for_user`, `get_annotations_for_annotator_overlap_slots`, `get_annotated_item_ids`) JOINs on `annotation_jobs` and filters `j.deleted_at IS NULL`, so a soft-deleted job's annotations atomically vanish from list views, inter-annotator agreement, and the human columns of evaluator-run reports. This cascade is intentional and also outlasts the `include_deleted_items=True` reproducibility carve-out used for soft-deleted *items* in eval-run detail — that carve-out preserves item soft-deletes only, never job soft-deletes. Consequence: deleting an `annotation_jobs` row retroactively shrinks the `human_agreement` block of old evaluator runs that referenced it. New annotation reads must keep the same `j.deleted_at IS NULL` filter or they'll leak deleted-job labels into agreement.
- **Evaluators replace `metrics`**: the `metrics` table is frozen — new work goes to `evaluators` + `evaluator_versions`. `evaluators.owner_user_id IS NULL` ⇒ seeded default (visible to everyone, not editable). Custom evaluators duplicate-and-edit. Pivots (`simulation_evaluators`, `test_evaluators`) pin `evaluator_version_id` at link time so the "live version" API only affects *future* links — reruns of old tests keep using the old version. STT/TTS jobs store the same shape inside `details.evaluators` (no pivot table). `init_db()` seeds defaults with stable `slug`s, migrates legacy `metrics` rows into `evaluators` (marked via `source_metric_uuid`), and backfills LLM `tests` where `config.evaluation.type='response'` into a `test_evaluators` link against `default-llm-next-reply` with `variable_values={"criteria": <old text>}`. All three passes are idempotent — safe to rerun.
- **Evaluator shape**: `output_type` ("binary" | "rating") lives on the `evaluators` row — it's identity. The **rubric** (`output_config`, including scale values/labels/descriptions/colors) lives on each `evaluator_versions` row so prompt iterations carry their own pinned rubric. Pivots (`simulation_evaluators`, `test_evaluators`) pin `evaluator_version_id` at link time, so past runs stay reproducible even after the rubric is edited.
- **Evaluator type vs data type**: an evaluator carries **two independent classifying columns** on the `evaluators` row:
  - `evaluator_type` (`tts | stt | llm | simulation`, default `llm`) — semantic category. `tts` judges TTS output; `stt` judges one transcript in isolation; `llm` judges one response with conversation history; `simulation` judges a full conversation. The frontend filters per run context (`GET /evaluators?evaluator_type=stt` on the STT page, etc.).
  - `data_type` (`text | audio`, default `text`) — the medium the judge actually consumes. **This is the only field that gates audio routing** in `invoke_evaluator()` (`data_type == "audio"` ⇒ audio bytes attached to the user message). It's also passed straight through to the calibrate CLI in `build_evaluator_cli_payload()`.
  In practice `tts ⇒ data_type=audio` and the rest are `text`, but the split is preserved so a user can in principle define, e.g., a TTS evaluator that judges a transcribed text instead of audio without reclassifying its category.
  The schema went through an intermediate phase where `data_type` was renamed to `evaluator_type`; `init_db()` now runs both `ALTER TABLE evaluators RENAME COLUMN data_type TO evaluator_type` **and** `ALTER TABLE evaluators ADD COLUMN data_type TEXT NOT NULL DEFAULT 'text'` (both try/except guarded — fresh DBs and DBs already on the new schema both no-op). Then it remaps legacy `evaluator_type` values (`audio → tts`, `text → llm`) and backfills `data_type = 'audio' WHERE evaluator_type = 'tts' AND data_type = 'text'` so partially-migrated DBs land in the right place. `_seed_default_evaluators` then snaps each seeded slug to its canonical `(evaluator_type, data_type)` pair. Default seeds: `default-tts-audio-quality` → `(tts, audio)`; `default-stt-transcription` → `(stt, text)`; everything else (`default-llm-next-reply`, `default-faithfulness`, `default-helpfulness`, `default-safety`, `default-conciseness`, `default-instruction-following`) → `(llm, text)`. There is no seeded `simulation` default — users create their own.
- **output_config shape** — same entry form for every output type: `{value, name, description?, color?}`. Binary = 2-entry scale with bool `value`; rating = N≥2-entry with numeric `value`. Adding a new `output_type` (categorical, ranking, …) means a new JSON shape inside `output_config` — no schema change. `variables` also live per-version (they're defined by the prompt text).
- **Per-level rubric injection**: `_format_scale_rubric()` in [llm_judge.py](src/llm_judge.py) appends a `Rubric:\n  value (name): description\n...` block to the judge prompt whenever any scale entry has a `description`. Applied identically for the `/invoke` endpoint and for the calibrate CLI payload (baked into the rendered `system_prompt` of each evaluator entry). Entries without a description are silently skipped — mixed-fill is fine.
- **API keys (`/api-keys`)** use bcrypt; the raw key is returned exactly once on creation and the DB keeps only a `key_prefix` (first 12 chars) for lookup plus the bcrypt hash. `get_user_from_api_key` accepts either `X-API-Key: <key>` or `Authorization: Bearer <key>` (keys are prefixed `calib_` so they can coexist with JWTs in the same header).
- **Evaluator invocation (`POST /evaluators/{uuid}/invoke`)** is the only endpoint that uses API-key auth instead of JWT. It renders `{{variable}}` placeholders against the request's `variables` dict, sends to OpenRouter, and enforces JSON output shape (`{pass, reasoning}` for binary, `{value, reasoning}` for rating, `{winner, reasoning}` for `kind=side_by_side`).
- **Calibrate CLI evaluator handoff** — the calibrate CLI accepts a minimal evaluator definition: `{name, system_prompt, judge_model, type ("binary"|"rating"), scale_min?, scale_max?}`. The backend's stored extras (`evaluator_type`, `data_type`, `kind`, `output_config.scale.{name,description,color}`) are NOT sent — they're either inferred by calibrate from the parent flow (e.g. TTS implies audio) or get baked into the prompt before sending (e.g. per-level `description`s become a "Rubric:" block via `_format_scale_rubric`). For STT/TTS the payload goes to `--config <path>` as `{"evaluators": [...]}`. For simulations it lives at `config.evaluators` (replacing the old `evaluation_criteria` key). For LLM tests/benchmarks the helper `build_test_evaluators_payload` produces a deduped top-level `evaluators` list (one entry per unique evaluator UUID — names suffixed with `-{uuid8}` on collision) and per-test-case `config.test_cases[].evaluation.criteria = [{name, arguments?}]` references; the top-level prompts keep `{{variable}}` placeholders unrendered so calibrate substitutes per test case using each criterion's `arguments`. STT/TTS/simulation payloads instead pre-render variables (no per-row arguments mechanism).
- **Default prompts API** — `GET /evaluators/default-prompt?purpose={llm|stt|tts|simulation}` returns the canonical default system prompt + suggested config for prefilling the create-evaluator form. The same prompts power the seeded LLM/STT/TTS evaluators (built via `_seed_from_purpose` in [db.py](src/db.py) so `DEFAULT_PROMPTS_BY_PURPOSE` is the single source of truth). The `simulation` purpose has no seeded evaluator — its prompt is a template with a literal `<ENTER EVALUATION CRITERIA HERE>` placeholder the user replaces directly (no `{{var}}`). Default judge models: `DEFAULT_TEXT_JUDGE_MODEL` for text/STT/simulation, `DEFAULT_AUDIO_JUDGE_MODEL` for TTS.

## Conventions

- New entities: UUID primary identifier (plus auto-increment `id`), `deleted_at` soft delete, `created_at`/`updated_at` timestamps, JSON `config` column for flexible fields, validated at the API layer via Pydantic.
- Many-to-many links use dedicated pivot tables with their own soft-delete column.
- When mutating child rows (e.g. `dataset_items`), bump the parent's `updated_at` in the same transaction.
- Schema migrations on existing tables: `ALTER TABLE ADD COLUMN` inside `init_db()`, wrapped in `try/except sqlite3.OperationalError: pass`. Use `DEFAULT NULL` — SQLite silently rejects non-constant defaults like `CURRENT_TIMESTAMP` in `ADD COLUMN`, and the `except` will swallow the failure.
- All job failures route through `capture_exception_to_sentry()` (marks as unhandled, flushes immediately — critical for background threads).
- CLI failure detection is two-layer: non-zero exit code **or** expected structured output missing. Never treat stderr content as a failure signal — the calibrate CLI emits benign cleanup tracebacks.
