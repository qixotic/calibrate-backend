# Integration testing without real AI cost

Set `FAKE_AI_PROVIDERS=1` to run the full **run → results** pipeline (LLM tests,
benchmarks, STT, TTS, text/voice simulations, annotation evaluator runs, trace
scoring) with **no real LLM/STT/TTS provider call, no API key, and no cost**. Use
it in CI and local E2E; never set it in production.

## How it works

Every AI operation the backend performs is delegated to one external CLI
(`calibrate-agent`) via `subprocess`, and every call site funnels through a single
function — `get_calibrate_agent_cli()` in [src/utils.py](src/utils.py). That one
function is the only injection point:

- **Flag unset (production):** returns `"calibrate-agent"` — unchanged.
- **Flag set:** returns the path to an in-repo fake,
  [src/testing/fake_calibrate_agent.py](src/testing/fake_calibrate_agent.py).

The fake is a standalone script (no backend imports). For each subcommand it writes
the exact output files that worker's reader expects — deterministic PASS verdicts,
ratings at scale-max — then exits 0. No worker or router code changes; production is
untouched.

`provider_status.run_check` also short-circuits to an all-healthy set under the flag,
so status pills don't depend on the fake and the boot-time status probe never spawns
the CLI.

## Booting the backend in CI

The backend needs **no external services** under the flag — no AWS/S3, no provider
keys, no Google OAuth. Clone it, `uv sync`, then boot from `src/`:

```
FAKE_AI_PROVIDERS=1 OBJECT_STORAGE_MODE=local LOCAL_ARTIFACT_ROOT=/tmp/artifacts LOCAL_ARTIFACT_BASE_URL=http://localhost:8000 JWT_SECRET_KEY=ci-secret DB_ROOT_DIR=/tmp/pense-db uv run uvicorn main:app --host 0.0.0.0 --port 8000
```

- Storage goes to local disk (`OBJECT_STORAGE_MODE=local`), not S3.
- State is SQLite under `DB_ROOT_DIR`.
- Authenticate via `POST /auth/signup` (email + password → JWT) — no OAuth needed.

Frontend E2E specs gate on `E2E_FAKE_AI=1` and drive this backend.

## Canned output (frontend asserts these — keep stable)

Constants live at the top of [the fake](src/testing/fake_calibrate_agent.py):

| Constant | Value |
| --- | --- |
| `FAKE_RESPONSE` | `Simulated agent reply.` |
| `FAKE_REASONING` | `Simulated judge reasoning: criteria satisfied.` |
| `FAKE_LATENCY_MS` | `100` |
| `FAKE_COST` | `0.001` |
| `FAKE_TOKENS` | `42` |
| `FAKE_WER` | `0.0` |
| `FAKE_TTFB` | `0.5` |

Every evaluator verdict is a PASS; every rating is scale-max.

## Maintaining the fake

Any `calibrate-agent` change is a **three-file change** — do all three in one commit:

1. the real worker/reader in `routers/agent_tests.py`, `stt.py`, `tts.py`,
   `simulations.py`, `annotation_eval_runner.py`, or `trace_scoring.py`;
2. [src/testing/fake_calibrate_agent.py](src/testing/fake_calibrate_agent.py) so its
   canned output still matches (and `SUPPORTED_SUBCOMMANDS` if the subcommand set
   changed);
3. [tests/test_fake_calibrate_agent.py](tests/test_fake_calibrate_agent.py) — drives
   every call site end-to-end with the real fake (no `subprocess` patch), so drift
   fails here.

Two tests keep this honest, both run by CI's full `pytest`:

```
uv run --group dev pytest tests/test_fake_calibrate_agent.py tests/test_fake_matches_real_usage.py -q
```

- `test_fake_calibrate_agent.py` — end-to-end coverage of every worker path.
- `test_fake_matches_real_usage.py` — AST-scans every `get_calibrate_agent_cli()`
  call site under `src/`, auto-discovers each spawned subcommand, and fails if the
  fake doesn't handle one. A new real usage can't ship without the fake covering it.

## Known gap

The benchmark leaderboard CSV header is the one shape derived from reader behavior
rather than a captured real `calibrate-agent` output. If the leaderboard table
renders empty during a joint frontend run, capture one real `leaderboard/*.csv` and
mirror its columns in `_write_leaderboard`.
