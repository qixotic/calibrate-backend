# Environment variables

This document describes the environment variables used for running the app.

Canonical placeholders live in [`src/.env.example`](src/.env.example). When you add, rename, or remove variables, update that file plus [`docker-compose.yml`](docker-compose.yml), [`./github/workflows/deploy.yml`](deploy.yml) and [`./github/workflows/deploy.yml`](deploy-staging.yml).

---

## AWS and object storage

| Variable                      | Meaning                                                                                                                                                                                                                                                                                       |
| ----------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **`OBJECT_STORAGE_MODE`**     | Artifact storage backend. Set to **`local`** for development without AWS; uploads and job artifacts are stored on disk while keeping `s3://local-dev-artifacts/...` paths in API payloads. **Deployments always use `s3`** — `local` is a local, dev-only convenience.                        |
| **`LOCAL_ARTIFACT_ROOT`**     | Directory used when `OBJECT_STORAGE_MODE=local`. Defaults to `${DB_ROOT_DIR}/artifacts`. The backend serves these files through development-only `/local-artifacts/...` URLs.                                                                                                                 |
| **`LOCAL_ARTIFACT_BASE_URL`** | Public backend base URL used when `OBJECT_STORAGE_MODE=local` to make both upload and download `/local-artifacts/...` URLs absolute. Set to `http://localhost:8000` for normal local dev so frontend upload targets and audio links point back to the backend; if unset, those URLs are returned relative (only works when the frontend is same-origin as the backend).                                   |
| **`AWS_ACCESS_KEY_ID`**       | AWS access key for boto3 S3 calls (presigned URLs, uploads). Optional when using IAM/instance credentials or another auth mechanism supported by boto3. Empty values are treated as unset. Only used when `OBJECT_STORAGE_MODE=s3`.                                                           |
| **`AWS_SECRET_ACCESS_KEY`**   | Secret key paired with `AWS_ACCESS_KEY_ID`. Only used when `OBJECT_STORAGE_MODE=s3`.                                                                                                                                                                                                          |
| **`AWS_REGION`**              | AWS region for the S3 client. Defaults to **`ap-south-1`** if unset or empty ([`src/utils.py`](src/utils.py)). Only used when `OBJECT_STORAGE_MODE=s3`.                                                                                                                                       |
| **`S3_OUTPUT_BUCKET`**        | Bucket name for artifacts (uploads, job outputs). Required when `OBJECT_STORAGE_MODE=s3`; ignored in local mode, where [`get_s3_output_config()`](src/utils.py) returns the sentinel bucket `local-dev-artifacts`.                                                                            |
| **`S3_ENDPOINT_URL`**         | Optional S3-compatible endpoint (e.g. Google Cloud Storage interop `https://storage.googleapis.com`). When set, checksum behaviour is pinned for compatibility with non-AWS endpoints. Not listed in `.env.example` but supported by the code and [`docker-compose.yml`](docker-compose.yml). |

---

## Provider API keys (calibrate / jobs)

These are **not read directly** by most backend modules; they are passed through to the **`calibrate`** CLI subprocess (inherits the backend environment). Configure keys only for providers you actually use.

| Variable                             | Meaning                                                                                                                                                                                                                                                  |
| ------------------------------------ | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **`DEEPGRAM_API_KEY`**               | Deepgram STT/TTS API access.                                                                                                                                                                                                                             |
| **`OPENAI_API_KEY`**                 | OpenAI API access.                                                                                                                                                                                                                                       |
| **`CARTESIA_API_KEY`**               | Cartesia TTS.                                                                                                                                                                                                                                            |
| **`SMALLEST_API_KEY`**               | Smallest AI / speech-related integrations used by calibrate.                                                                                                                                                                                             |
| **`GROQ_API_KEY`**                   | Groq API access.                                                                                                                                                                                                                                         |
| **`GOOGLE_APPLICATION_CREDENTIALS`** | Path to a Google Cloud **service account JSON** key file on disk (standard GCP env var). Used where Google Cloud APIs require explicit credentials.                                                                                                      |
| **`GOOGLE_API_KEY`**                 | Google AI / Gemini-style API key where applicable.                                                                                                                                                                                                       |
| **`GOOGLE_CLOUD_PROJECT_ID`**        | Google Cloud project identifier when required by Google tooling.                                                                                                                                                                                         |
| **`SARVAM_API_KEY`**                 | Sarvam AI provider key.                                                                                                                                                                                                                                  |
| **`ELEVENLABS_API_KEY`**             | ElevenLabs TTS.                                                                                                                                                                                                                                          |
| **`OPENROUTER_API_KEY`**             | OpenRouter key for LLM/judge flows and [`GET /openrouter/providers`](src/main.py). If unset, that endpoint returns `null` (OpenRouter treated as disabled).                                                                                              |
| **`OPENROUTER_ALLOWED_PROVIDERS`**   | Optional comma-separated OpenRouter **provider slugs**. If empty/unset, [`GET /openrouter/providers`](src/main.py) reports all providers (`"providers": "all"`). If set, the response lists only those slugs after fetching the catalog from OpenRouter. |

---

## Database and bootstrap user

| Variable                      | Meaning                                                                                                                                                                                                      |
| ----------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| **`DB_ROOT_DIR`**             | **Required** directory for the SQLite file `pense.db` (joined at import time in [`src/db.py`](src/db.py)). Must be set before importing `db`.                                                                |
| **`DEFAULT_USER_EMAIL`**      | Email of the seeded “default” user created during [`init_db()`](src/db.py) if missing; also used to locate that user on subsequent starts. Existing rows with `user_id IS NULL` are backfilled to this user. |
| **`DEFAULT_USER_FIRST_NAME`** | First name for the default user insert.                                                                                                                                                                      |
| **`DEFAULT_USER_LAST_NAME`**  | Last name for the default user insert.                                                                                                                                                                       |

---

## HTTP API: docs, CORS, auth

| Variable                   | Meaning                                                                                                                                                                                                               |
| -------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **`DOCS_USERNAME`**        | HTTP Basic username for **`/docs`**, **`/redoc`**, and **`/openapi.json`**. Default **`admin`**.                                                                                                                      |
| **`DOCS_PASSWORD`**        | HTTP Basic password for the same routes. Default **`changeme`** — change in production.                                                                                                                               |
| **`PUBLIC_API_BASE_URL`**    | Deployment URL for this backend. Default **`http://localhost:8000`**. |
| **`CORS_ALLOWED_ORIGINS`** | Comma-separated list of allowed browser origins (e.g. `http://localhost:3000,https://app.example.com`). If unset, defaults to **`*`** (allow all). See [`CORSMiddleware`](src/main.py) setup.                         |
| **`GOOGLE_CLIENT_ID`**     | Google OAuth client ID used by the auth router (same as the one used for setting up auth on frontend)([`src/routers/auth.py`](src/routers/auth.py)).                                                                  |
| **`JWT_SECRET_KEY`**       | HMAC secret for signing JWTs. **Must be set in production** (use a long random value, e.g. `openssl rand -base64 32`). A weak development default exists in code if unset ([`src/auth_utils.py`](src/auth_utils.py)). |
| **`JWT_EXPIRATION_HOURS`** | JWT lifetime in hours. Default **`168`** (7 days).                                                                                                                                                                    |
| **`SUPERADMIN_EMAIL`**     | Email address allowed to mutate **user limit** endpoints (`POST`/`PUT`/`DELETE` on user-limits). Enforced by [`require_superadmin`](src/auth_utils.py).                                                               |

---

## Job queue and evaluation limits

| Variable                            | Meaning                                                                                                                                                                                                                                                                                |
| ----------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **`MAX_CONCURRENT_JOBS`**           | Global cap on concurrent background jobs across queues. Read as **`int(os.getenv(...))`** with **no fallback** in code ([`src/utils.py`](src/utils.py)), so it **must be set** for queue helpers to work (tests and Compose default this to **`1`**).                                  |
| **`MAX_CONCURRENT_JOBS_PER_USER`**  | Per-user concurrent job cap. Default **`1`**. Set to **`0`** to disable the per-user limit ([`src/utils.py`](src/utils.py)).                                                                                                                                                           |
| **`DEFAULT_MAX_ROWS_PER_EVAL`**     | Default maximum rows per evaluation run when no per-user override exists ([`src/routers/user_limits.py`](src/routers/user_limits.py)). Default **`20`**.                                                                                                                               |
| **`CALIBRATE_TEST_PARALLEL`**       | How many test cases `calibrate llm` evaluates in parallel per model. Read **natively by the calibrate CLI** (`-n flag > CALIBRATE_TEST_PARALLEL > default 4`); the backend doesn't pass `-n` for LLM tests, it just lets the subprocess inherit this var. Unset ⇒ CLI default **`4`**. |
| **`CALIBRATE_SIMULATION_PARALLEL`** | How many simulations `calibrate simulations` runs in parallel (text and voice). Read **natively by the calibrate CLI**; the backend doesn't pass `-n`, it just lets the subprocess inherit this var. Unset ⇒ the CLI's own default.                                                    |

---

## Sentry

| Variable                          | Meaning                                                                                                                |
| --------------------------------- | ---------------------------------------------------------------------------------------------------------------------- |
| **`SENTRY_DSN`**                  | If non-empty, initializes the Sentry SDK at import time ([`src/main.py`](src/main.py)). Leave empty to disable Sentry. |
| **`SENTRY_ENVIRONMENT`**          | Sentry release environment tag. Default **`development`**.                                                             |
| **`SENTRY_TRACES_SAMPLE_RATE`**   | Float **`0.0`–`1.0`** for performance trace sampling. Default **`1.0`**.                                               |
| **`SENTRY_PROFILES_SAMPLE_RATE`** | Float **`0.0`–`1.0`** for profiling sampling. Default **`1.0`**.                                                       |

---

## Observability / Langfuse / OpenTelemetry

These variables appear in [`src/.env.example`](src/.env.example) and deployment manifests. **This FastAPI codebase does not reference most of them in Python**; they are intended for OpenTelemetry exporters and tooling (including **`calibrate`** subprocesses that inherit the environment). Align values with your Langfuse or OTLP collector setup ([`SELF_HOSTING.md`](SELF_HOSTING.md) has examples).

| Variable                           | Meaning                                                                                                                                                                                                    |
| ---------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **`ENVIRONMENT`**                  | Logical deployment environment string used alongside tracing stacks (e.g. Langfuse “environment”).                                                                                                         |
| **`ENABLE_TRACING`**               | Typical toggle for downstream tracing (`true`/`false`; conventions vary by consumer).                                                                                                                      |
| **`OTEL_EXPORTER_OTLP_ENDPOINT`**  | OTLP gRPC/HTTP endpoint for exporting traces/metrics/logs (e.g. Langfuse ingest URL).                                                                                                                      |
| **`OTEL_EXPORTER_OTLP_HEADERS`**   | Headers for OTLP export (e.g. `Authorization=Basic%20<base64>` for Langfuse).                                                                                                                              |
| **`LANGFUSE_TRACING_ENVIRONMENT`** | Environment label sent to Langfuse when using native Langfuse instrumentation.                                                                                                                             |
| **`LANGFUSE_HOST`**                | Langfuse server base URL / host configuration for Langfuse clients.                                                                                                                                        |
| **`LANGFUSE_PUBLIC_KEY`**          | Langfuse project public API key.                                                                                                                                                                           |
| **`LANGFUSE_SECRET_KEY`**          | Langfuse project secret API key.                                                                                                                                                                           |
| **`LANGFUSE_BASE_URL`**            | Alternate base URL variable used in [**`docker-compose.yml`**](docker-compose.yml) and deploy workflows; set when your Langfuse deployment expects this name instead of or in addition to `LANGFUSE_HOST`. |

---

## Default agent template (`POST /agents`, type=`agent`)

Partial defaults applied when creating agents ([`src/routers/agents.py`](src/routers/agents.py)). Empty strings are treated as unset (`env_str` / `env_bool` / `env_int` in [`src/utils.py`](src/utils.py)); callers may still override in the request body via deep merge.

| Variable                          | Meaning                                                                                               |
| --------------------------------- | ----------------------------------------------------------------------------------------------------- |
| **`DEFAULT_AGENT_SYSTEM_PROMPT`** | Default `system_prompt`. Hardcoded fallback: “You are a helpful assistant.”                           |
| **`DEFAULT_AGENT_LLM_MODEL`**     | Default LLM model string (OpenRouter-style id). Fallback **`google/gemini-2.5-flash`**.               |
| **`DEFAULT_AGENT_STT_PROVIDER`**  | Default speech-to-text provider name. Fallback **`google`**.                                          |
| **`DEFAULT_AGENT_TTS_PROVIDER`**  | Default text-to-speech provider name. Fallback **`google`**.                                          |
| **`DEFAULT_AGENT_SPEAKS_FIRST`**  | Boolean (`1`/`true`/`yes`/`on`, etc.). Fallback in code is **`true`** (matches simulation behaviour). |
| **`DEFAULT_AGENT_MAX_TURNS`**     | Integer `max_assistant_turns` in agent settings. Fallback **`50`**.                                   |

---

## Default persona / simulation builder

Used when building simulation run configuration and a persona omits fields ([`src/routers/simulations.py`](src/routers/simulations.py)).

| Variable                                       | Meaning                                                  |
| ---------------------------------------------- | -------------------------------------------------------- |
| **`DEFAULT_PERSONA_GENDER`**                   | Fallback persona gender. Default **`female`**.           |
| **`DEFAULT_PERSONA_LANGUAGE`**                 | Fallback language. Default **`english`**.                |
| **`DEFAULT_PERSONA_INTERRUPTION_SENSITIVITY`** | Fallback interruption sensitivity. Default **`medium`**. |

---

## Docker Compose-only substitutions

The Compose file also uses variables such as **`IMAGE_NAME`**, **`IMAGE_TAG`**, **`CONTAINER_NAME`**, **`PORT`**, and **`APP_FOLDER_PATH`** for image naming, port publishing, and volume mounts — these configure Docker itself, not application logic inside the container.
