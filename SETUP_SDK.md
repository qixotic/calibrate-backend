# SDK & CLI publish setup

How to configure and run client generation for the Calibrate **public API**:

| Client | Generator | Repo | Install |
|--------|-----------|------|---------|
| **Python SDK** (`calibrate-sdk`) | Fern | [`dalmia/calibrate-python-sdk`](https://github.com/dalmia/calibrate-python-sdk) | `pip install calibrate-sdk` |
| **Cloud CLI** (`calibrate`) | Speakeasy | [`dalmia/calibrate-cli`](https://github.com/dalmia/calibrate-cli) | `brew install dalmia/tap/calibrate` (after tap is live) |

> The offline evaluation engine is separate: PyPI **`calibrate-agent`**, command **`calibrate-agent`**.

## Architecture

```
production GitHub release published
  │
  ├─ Deploy to Production (deploy.yml)
  │
  └─ Auto-publish SDK and CLI (auto-publish-sdk.yml)
       ├─ compare public OpenAPI spec hash vs parent commit
       ├─ if changed → auto-bump patch from latest v* tag on client repos
       └─ call publish-sdk.yml
            ├─ prepare ─ fetch openapi/openapi.json (PUBLIC_API_BASE_URL → servers block)
            ├─ publish-python-sdk (parallel)
            │    fern generate --group python-sdk
            │    → push dalmia/calibrate-python-sdk (Fern GitHub App)
            │    → tag v<version> (PUSH_TO_REPO_TOKEN)
            │    → calibrate-python-sdk ci.yml → PyPI
            └─ publish-cli (parallel)
                 speakeasy run -t calibrate-cli
                 → sync-client-repo.sh → dalmia/calibrate-cli
                 → tag v<version> (PUSH_TO_REPO_TOKEN)
                 → calibrate-cli release.yaml → GoReleaser → GitHub Release + homebrew-tap
```

Workflows: [`.github/workflows/auto-publish-sdk.yml`](.github/workflows/auto-publish-sdk.yml) (auto + manual gate), [`.github/workflows/publish-sdk.yml`](.github/workflows/publish-sdk.yml) (generate + push)  
Validate on PRs: [`.github/workflows/validate-sdk.yml`](.github/workflows/validate-sdk.yml)

## One-time: backend secrets

Add these to **this repo** → Settings → Environments → **Production**:

| Secret | Used by | Notes |
|--------|---------|-------|
| `SDK_AUTO_PUBLISH_ENABLED` | `auto-publish-sdk.yml`, `publish-sdk.yml` | Set to **`true`** only on the canonical upstream repo (Production). **Do not set on forks or self-hosted copies** — workflows skip when absent. Both this and `PUSH_TO_REPO_TOKEN` must be set for publish to run. |
| `FERN_TOKEN` | Fern Python SDK generate | From [buildwithfern.com](https://buildwithfern.com); Fern GitHub App must be authorized on `dalmia` |
| `PYPI_TOKEN` | Fern generate (metadata) | Passed to `fern generate`; actual PyPI upload is in `calibrate-python-sdk` CI |
| `SPEAKEASY_API_KEY` | Speakeasy CLI generate + validate | From [speakeasy.com](https://www.speakeasy.com) |
| `PUSH_TO_REPO_TOKEN` | CLI sync + tagging both client repos | Classic PAT with **`contents:write`** and **`workflow`** on `dalmia/calibrate-python-sdk` and `dalmia/calibrate-cli`. **Required for publish workflows to start** (gate check) as well as client-repo pushes. |
| `PUBLIC_API_BASE_URL` | Fetch public OpenAPI spec | Production API URL injected into `servers` (e.g. `https://pense-backend.artpark.ai`) |

### PAT scopes (`PUSH_TO_REPO_TOKEN`)

| Scope | Why |
|-------|-----|
| `contents:write` | Push CLI output via `sync-client-repo.sh`; create tags on both client repos |
| `workflow` | Push Speakeasy-generated `.github/workflows/release.yaml` into `calibrate-cli` on each sync |

## One-time: Python SDK (`calibrate-python-sdk`)

Fern pushes generated code via its GitHub App (`fern/generators.yml` → `github: mode: push`). No backend PAT needed for the code push.

**Fern GitHub App:** authorize on the `dalmia` account so pushes to `calibrate-python-sdk` succeed.

**PyPI:** `calibrate-python-sdk` has auto-generated `ci.yml` that publishes on `v*` tags. Backend publish workflow tags `v<version>` after each generate.

**Hand-written files:** `.fernignore` in the SDK repo preserves `README.md` (PyPI long description).

## One-time: CLI (`calibrate-cli` + Homebrew)

### Client repo secrets

Add to **`dalmia/calibrate-cli`** → Settings → Secrets and variables → Actions:

| Secret | Purpose |
|--------|---------|
| `CLI_GPG_SECRET_KEY` | Armored GPG signing subkey (`gpg --armor --export-secret-keys <KEY_ID>`) |
| `CLI_GPG_PASSPHRASE` | Passphrase for that key |
| `HOMEBREW_TAP_GITHUB_TOKEN` | PAT with **Contents: Read and write** on `dalmia/homebrew-tap` (classic: `repo` / `public_repo`) |

Built-in `GITHUB_TOKEN` covers the GitHub Release on `calibrate-cli` itself.

#### Generate a GPG signing key

```bash
# Generate a passphrase first — use when gpg prompts, then store as CLI_GPG_PASSPHRASE
openssl rand -base64 32

gpg --full-generate-key
# RSA, sign only, 4096-bit, no expiry (or set expiry + rotate)
# Paste the openssl output when prompted for the key passphrase

gpg --list-secret-keys --keyid-format long
gpg --armor --export-secret-keys <KEY_ID>   # → CLI_GPG_SECRET_KEY
```

### Repos

- [ ] **`dalmia/homebrew-tap`** exists (can start empty; GoReleaser commits `Formula/calibrate.rb` on first green release)
- [ ] **`release.yaml`** in `calibrate-cli` — synced from Speakeasy output on each publish (`generateRelease: true`)
- [ ] **`README.md`** in `calibrate-cli` — hand-written; excluded from sync

### What `sync-client-repo.sh` preserves

On each publish, generated output overwrites `calibrate-cli` except:

- `README.md`
- `.speakeasyignore` (if present)

## Per-release checklist

1. **Merge** any public API route changes + update **both** overlay files:
   - [`fern/openapi-overrides.yml`](fern/openapi-overrides.yml) (Fern SDK names)
   - [`openapi/overlay.yaml`](openapi/overlay.yaml) (Speakeasy CLI names)  
   Enforced by [`tests/test_sdk_overrides.py`](tests/test_sdk_overrides.py).

2. **Ship** — publish is automatic after **Deploy to Production** when the public OpenAPI spec changed (patch version auto-bumps from the latest `v*` tag on `calibrate-python-sdk` / `calibrate-cli`). Manual options:
   - Actions → **Auto-publish SDK and CLI** → Run workflow (optional `force` / `version`)
   - Actions → **Publish SDK and CLI** → Run workflow → enter version (skips change detection)

3. **Verify backend workflow** — `auto-publish-sdk` (if used) then both `publish-python-sdk` and `publish-cli` jobs green.

4. **Verify Python SDK** — `calibrate-python-sdk` CI ran on `v*` tag; new version on PyPI.

5. **Verify CLI** — `calibrate-cli` **Release** workflow green on `v*` tag:
   - [ ] GitHub Release with binaries on `dalmia/calibrate-cli`
   - [ ] `Formula/calibrate.rb` in `dalmia/homebrew-tap`
   - [ ] `brew install dalmia/tap/calibrate` works

### Re-run a failed CLI release

If sync + tag succeeded but **Release** failed (e.g. missing GPG secrets at the time):

1. Add/fix secrets in `calibrate-cli`
2. Re-run the failed **Release** workflow (uses existing `v*` tag — no new backend publish needed)

## Local validation

```bash
# Boot app + fetch spec (mirrors CI)
cd src && uv run uvicorn main:app --port 8000 &
curl -o ../openapi/openapi.json http://localhost:8000/public-api/openapi.json

# Fern Python SDK config
npx fern-api check

# Speakeasy CLI config
speakeasy run -s calibrate-public-api -y
speakeasy lint openapi -s openapi/compiled.yaml
speakeasy lint config -d .

# Overlay tests
uv run --group dev pytest tests/test_sdk_overrides.py -q
```

## Troubleshooting

| Symptom | Likely cause | Fix |
|---------|--------------|-----|
| `gh: set the GH_TOKEN environment variable` | `PUSH_TO_REPO_TOKEN` missing/empty in Production | Add PAT to backend Production secrets |
| PAT rejected pushing `release.yaml` | Missing `workflow` scope on `PUSH_TO_REPO_TOKEN` | Add `workflow` scope to the backend PAT |
| Release fails: `gpg_private_key` not supplied | GPG secrets missing in **calibrate-cli** | Add `CLI_GPG_SECRET_KEY` + `CLI_GPG_PASSPHRASE` |
| Release fails: `field token not found in type config.Homebrew` | Speakeasy `.goreleaser.yaml` incompatible with GoReleaser >=2.17 | Merge backend patch (`patch-goreleaser-config.sh`) or move `token` under `repository` in `calibrate-cli`; re-run Release |
| Homebrew formula never appears | `HOMEBREW_TAP_GITHUB_TOKEN` missing or tap repo missing | Add secret; create `dalmia/homebrew-tap` |
| Ugly SDK method names | Overlay out of sync with Public API routes | Update both overlay files; run `test_sdk_overrides.py` |

## Related docs

- [`CLAUDE.md`](CLAUDE.md) — load-bearing invariants (public API tag gate, overlay sync rule, auth scheme pinning)
- PR #97 (`feat/speakeasy-clients`) — full Speakeasy migration (Python + CLI) when Speakeasy tier allows multiple targets
