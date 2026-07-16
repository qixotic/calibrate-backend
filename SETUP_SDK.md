# SDK & CLI publish setup

How to configure and run client generation for the Calibrate **public API**:

| Client | Generator | Repo | Install |
|--------|-----------|------|---------|
| **Python SDK** (`calibrate-sdk`) | Fern | [`dalmia/calibrate-python-sdk`](https://github.com/dalmia/calibrate-python-sdk) | `pip install calibrate-sdk` |
| **Cloud CLI** (`calibrate`) | Speakeasy | [`dalmia/calibrate-cli`](https://github.com/dalmia/calibrate-cli) | `brew install dalmia/tap/calibrate` (after tap is live) |
| **MCP server** (`@dalmia/calibrate-mcp`) | Speakeasy (`mcp-typescript`) | [`dalmia/calibrate-mcp`](https://github.com/dalmia/calibrate-mcp) | `npx @dalmia/calibrate-mcp start` (set `CALIBRATE_API_KEY`) |

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
            ├─ publish-cli (parallel)
            │    speakeasy run -t calibrate-cli
            │    → sync-client-repo.sh → dalmia/calibrate-cli
            │    → tag v<version> (PUSH_TO_REPO_TOKEN)
            │    → calibrate-cli release.yaml → GoReleaser → GitHub Release + homebrew-tap
            └─ publish-mcp (parallel)
                 speakeasy run -t calibrate-mcp
                 → inject .github/workflows/publish.yml (backend template)
                 → sync-client-repo.sh → dalmia/calibrate-mcp
                 → tag v<version> (PUSH_TO_REPO_TOKEN)
                 → calibrate-mcp publish.yml → npm (@dalmia/calibrate-mcp)
       └─ record sdk-v<version> tag on this repo (version history only)
       └─ sync-docs (after publish) → repository_dispatch on ARTPARK-SAHAI-ORG/calibrate (DOCS_SYNC_REPO_TOKEN)
       └─ sync-skills (after publish) → repository_dispatch on dalmia/calibrate-skills (PUSH_TO_REPO_TOKEN)
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
| `PUSH_TO_REPO_TOKEN` | CLI/MCP sync + tagging all client repos; skills drift-sync dispatch (`sync-skills` job) | Fine-grained PAT (resource owner `dalmia`) with **Contents: write** and **Workflows: write** on `dalmia/calibrate-python-sdk`, `dalmia/calibrate-cli`, `dalmia/calibrate-mcp`, and `dalmia/calibrate-skills`. **Required for publish workflows to start** (gate check) and client-repo pushes. `calibrate-skills` is in the repo list so `sync-skills` reuses it — `repository_dispatch` needs Contents: write, already granted. |
| `DOCS_SYNC_REPO_TOKEN` | Docs OpenAPI sync dispatch (`sync-docs` job) | Fine-grained PAT on [`ARTPARK-SAHAI-ORG/calibrate`](https://github.com/ARTPARK-SAHAI-ORG/calibrate) with **Actions: Read and write** (see below). Separate because a fine-grained PAT is scoped to one resource owner, and this repo is under `ARTPARK-SAHAI-ORG`, not `dalmia`. |
| `PUBLIC_API_BASE_URL` | Fetch public OpenAPI spec | Production API URL injected into `servers` (e.g. `https://pense-backend.artpark.ai`) |

### PAT permissions (`PUSH_TO_REPO_TOKEN`)

Fine-grained, resource owner `dalmia`, repositories: `calibrate-python-sdk`, `calibrate-cli`, `calibrate-mcp`, `calibrate-skills`.

| Permission | Why |
|------------|-----|
| Contents: write | Push CLI/MCP output via `sync-client-repo.sh`; create tags on the client repos; dispatch `sync-api-spec` to `calibrate-skills` |
| Workflows: write | Push Speakeasy-generated `.github/workflows/release.yaml` into `calibrate-cli` on each sync |

### Docs sync token (`DOCS_SYNC_REPO_TOKEN`)

Separate from `PUSH_TO_REPO_TOKEN` because client repos live under **`dalmia/`** (personal) while the docs repo is **`ARTPARK-SAHAI-ORG/calibrate`** (org).

1. GitHub → **Settings → Developer settings → Fine-grained personal access tokens → Generate**
2. **Resource owner:** `ARTPARK-SAHAI-ORG`
3. **Repository access:** Only `calibrate`
4. **Permissions:** **Actions: Read and write** (triggers `repository_dispatch` → `sync-api-spec.yml`)
5. **Expiration:** org policy caps at 366 days — GitHub emails before expiry; rotate annually and update the Production secret

Add the token to **this repo** → Settings → Environments → **Production** as `DOCS_SYNC_REPO_TOKEN`.

**Also on the calibrate repo** (not this repo): add `PUBLIC_API_BASE_URL` under Settings → Secrets and variables → Actions so the sync workflow can fetch the live spec.

Requires [calibrate#108](https://github.com/ARTPARK-SAHAI-ORG/calibrate/pull/108) merged (`sync-api-spec.yml` with `repository_dispatch` listener).

The `sync-docs` job runs **after** `publish` completes so the docs workflow sees freshly synced `calibrate-python-sdk` and `calibrate-cli` output (including CLI `docs/`) before generating pages.

### Skills sync token (reuses `PUSH_TO_REPO_TOKEN`)

`sync-skills` does **not** need its own secret. `PUSH_TO_REPO_TOKEN` is a fine-grained PAT owned by `dalmia`, and `dalmia/calibrate-skills` has been **added to its repository list** with **Contents: write** — the permission `repository_dispatch` requires (the token already grants it for pushing to the client repos). So the same token dispatches to the skills repo. (Contrast `DOCS_SYNC_REPO_TOKEN`, which must be separate: a fine-grained PAT is scoped to a single resource owner, and the docs repo is under `ARTPARK-SAHAI-ORG`, not `dalmia`.)

When rotating or regenerating `PUSH_TO_REPO_TOKEN`, keep all four repos —
`calibrate-python-sdk`, `calibrate-cli`, `calibrate-mcp`, `calibrate-skills` — in its repository list.

**Only step on the skills side** (not this repo): add the `OPENAPI_SPEC_URL` variable under `dalmia/calibrate-skills` → Settings → Secrets and variables → Actions → Variables, so its drift check knows which spec to fetch. Its [`sync-from-spec.yml`](https://github.com/dalmia/calibrate-skills/blob/main/.github/workflows/sync-from-spec.yml) listener is already merged. (The URL is kept in the variable rather than hardcoded in the workflow; the dispatch may also pass `client_payload.spec_url` to override it.)

`sync-skills` drift-checks the skills against the published spec (it does not regenerate them — the skills are prose). It runs after `publish` for parity with `sync-docs`, though it only depends on the spec being live.

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

### Post-generation CLI patches

Speakeasy output is `// DO NOT EDIT` and `sync-client-repo.sh` does `rsync --delete`, so hand-edits in `calibrate-cli` never survive. Behavior tweaks live in this repo as idempotent, guarded patch scripts run in `publish-cli` **after** `speakeasy run` and **before** sync (a missing anchor fails the job with `::warning::` rather than shipping a silent regression):

- `patch-goreleaser-config.sh` — GoReleaser >=2.17 Homebrew `token` field compat.
- `patch-cli-auth-commands.sh` — hoist `login`/`logout` to root, hide the legacy `auth` group.
- `patch-cli-server-url.sh` — make `--server-url` a persistable global param (flag > env `CALIBRATE_SERVER_URL` > config), stored by `calibrate configure` and shown in `calibrate whoami`. Speakeasy generates it flag-only, so self-hosted users would otherwise pass it on every call. Each script has a matching `tests/test_*_patch.py` that patches a copy of the sibling repo and (for server-url) compiles it with `go build`.

## One-time: MCP (`calibrate-mcp`)

Speakeasy **`mcp-typescript`** generates a standalone MCP server from the same public OpenAPI spec + overlay as the CLI. Generate + sync run in `publish-sdk.yml` (`publish-mcp` job).

**Publish workflow is injected, not Speakeasy-generated.** Unlike the CLI (`generateRelease: true` → `release.yaml`), the `mcp-typescript` target emits **no** release workflow. So `publish-mcp` copies [`.github/client-templates/calibrate-mcp-publish.yml`](.github/client-templates/calibrate-mcp-publish.yml) into the generated tree as `.github/workflows/publish.yml` before sync. It ships in the output (survives the `rsync --delete`), so the client repo publishes itself to npm on the `v*` tag — same "client repo self-publishes" pattern as the SDK/CLI. Edit the template in **this** repo; never hand-edit it in `calibrate-mcp` (overwritten every release). Pushing it requires the `workflow` scope on `PUSH_TO_REPO_TOKEN` (already needed for the CLI's `release.yaml`).

**Auth is npm Trusted Publishing (OIDC) — no `NPM_TOKEN` secret.** The injected `publish.yml` authenticates via GitHub's short-lived OIDC identity (`id-token: write`), which npm verifies against a trusted publisher configured on the package. Nothing to store or rotate; provenance attestation is automatic. Trusted publishing needs npm ≥ 11.5.1, so the workflow runs `npm install -g npm@11` before publishing (pinned to the 11 line, not `@latest`: npm 12.0.0 ships a broken bundle where `libnpmpublish` can't resolve `sigstore`, crashing provenance publishes).

**The workflow also cuts a GitHub release with the `.mcpb` bundle.** After the npm publish, on a `v*` tag it runs `npm run mcpb:build`, renames the packed output to **`mcp-server.mcpb`**, and `gh release create`s (or re-uploads to) a release named after the tag with that asset attached. This exists solely so the Speakeasy-generated landing page's Claude Desktop tab ("Download MCP Bundle") resolves — it links a fixed `…/releases/download/<tag>/mcp-server.mcpb`, which nothing else produces. Needs `contents: write` (added alongside `id-token: write`) and uses the default `GITHUB_TOKEN`. The asset filename is load-bearing: it must match what `src/landing-page.ts` links (`mcp-server.mcpb`); if the generated page ever changes that name, update the rename step. Because the deployed landing page bakes in a fixed version at build time, the self-hosted MCP server (GCP) must be **redeployed** after a release so its download link points at a tag that now has the asset — which the `deploy-mcp` job below now automates.

**The GCP redeploy is automated (from the injected `publish.yml`).** Right after the npm publish + release, the same workflow dispatches the deploy repo: `gh workflow run deploy.yml --repo dalmia/calibrate-mcp-deploy -f version=<tag without v>`. That repo builds a container **from the npm package** and redeploys Cloud Run + refreshes the landing page. Because it's in the publish workflow, **any** publish — auto after prod deploy, or a manual `workflow_dispatch` — also redeploys; it no-ops when `DEPLOY_DISPATCH_TOKEN` is unset. That token is a **PAT with `actions: write` on `dalmia/calibrate-mcp-deploy`**, stored as a secret in the **`calibrate-mcp`** repo (its one secret beyond OIDC). The deploy repo authenticates to GCP separately via Workload Identity Federation — see that repo's `README.md`.

### One-time: deploy-trigger token

Add a fine-grained PAT so `calibrate-mcp`'s `publish.yml` can dispatch the deploy:

1. Create a fine-grained PAT (github.com → Settings → Developer settings → Fine-grained tokens) with **`dalmia/calibrate-mcp-deploy`** as the only repository and **Actions: Read and write** permission.
2. `gh secret set DEPLOY_DISPATCH_TOKEN --repo dalmia/calibrate-mcp --body <pat>`.

### One-time npm setup

1. **Bootstrap the package** — trusted publishing can only be configured on a package that already exists. Do the first publish manually from a local clone of `calibrate-mcp`: `npm publish --access public` (needs `bun` for the build; will prompt for 2FA). Confirm with `npm view @dalmia/calibrate-mcp version`.
2. **Configure the trusted publisher** — npmjs.com → the `@dalmia/calibrate-mcp` package → **Settings → Trusted Publisher** → add **GitHub Actions**:
   - Organization/owner: `dalmia`
   - Repository: `calibrate-mcp`
   - Workflow filename: `publish.yml`
   - Environment: *(leave blank)*
3. **Keep the package public** — `npm access get status @dalmia/calibrate-mcp` should say `public`; the `dalmia` org must allow the `@dalmia` scope. No repo secret is needed.

### Repos

- [ ] **`dalmia/calibrate-mcp`** exists (can start empty; first sync populates generated tree)
- [ ] **`README.md`** in `calibrate-mcp` — hand-written; excluded from sync (same pattern as CLI)
- [ ] **`.github/workflows/publish.yml`** in `calibrate-mcp` — injected from the backend template on each sync; do not hand-edit
- [ ] **Trusted publisher** configured on the npm package (GitHub `dalmia/calibrate-mcp`, workflow `publish.yml`) — no `NPM_TOKEN` secret required

### Cursor local config (example)

Add to Cursor MCP settings (stdio transport):

```json
{
  "mcpServers": {
    "calibrate": {
      "command": "npx",
      "args": ["-y", "@dalmia/calibrate-mcp", "start"],
      "env": {
        "CALIBRATE_API_KEY": "sk_..."
      }
    }
  }
}
```

Generated output also supports **npm install** and optional **Cloudflare Workers** hosting for a remote HTTP MCP endpoint — see [Speakeasy standalone MCP docs](https://www.speakeasy.com/docs/standalone-mcp/build-server) when adapting beyond local stdio.

## Per-release checklist

1. **Merge** any public API route changes + update **both** overlay files:
   - [`fern/openapi-overrides.yml`](fern/openapi-overrides.yml) (Fern SDK names)
   - [`openapi/overlay.yaml`](openapi/overlay.yaml) (Speakeasy CLI names + `x-speakeasy-mcp` tool metadata)  
   Enforced by [`tests/test_sdk_overrides.py`](tests/test_sdk_overrides.py).

2. **Ship** — publish is automatic after **Deploy to Production** when the public OpenAPI spec changed (patch version auto-bumps from the latest `v*` tag on client repos). Manual options:
   - Actions → **Auto-publish SDK and CLI** → Run workflow (optional `force` / `version`)
   - Actions → **Publish SDK and CLI** → Run workflow → enter version (skips change detection)

3. **Verify backend workflow** — `auto-publish-sdk` (if used) then `publish-python-sdk`, `publish-cli`, and `publish-mcp` jobs green.

4. **Verify Python SDK** — `calibrate-python-sdk` CI ran on `v*` tag; new version on PyPI.

5. **Verify CLI** — `calibrate-cli` **Release** workflow green on `v*` tag:
   - [ ] GitHub Release with binaries on `dalmia/calibrate-cli`
   - [ ] `Formula/calibrate.rb` in `dalmia/homebrew-tap`
   - [ ] `brew install dalmia/tap/calibrate` works

6. **Verify MCP** — `calibrate-mcp` CI ran on `v*` tag:
   - [ ] `@dalmia/calibrate-mcp@<version>` on npm
   - [ ] `npx @dalmia/calibrate-mcp start` with `CALIBRATE_API_KEY` lists tools in Cursor

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

# Speakeasy CLI + MCP config
speakeasy run -s calibrate-public-api -y
speakeasy run -t calibrate-cli -y
speakeasy run -t calibrate-mcp -y
speakeasy lint openapi -s openapi/compiled.yaml
speakeasy lint config -d .

# Overlay tests
uv run --group dev pytest tests/test_sdk_overrides.py -q
```

## Troubleshooting

| Symptom | Likely cause | Fix |
|---------|--------------|-----|
| `gh: set the GH_TOKEN environment variable` | `PUSH_TO_REPO_TOKEN` missing/empty in Production | Add PAT to backend Production secrets |
| `sync-docs` fails with 401/403 | `DOCS_SYNC_REPO_TOKEN` missing, expired, or lacks Actions write on `ARTPARK-SAHAI-ORG/calibrate` | Regenerate fine-grained PAT; update Production secret |
| `sync-skills` fails with 401/403 | `PUSH_TO_REPO_TOKEN` expired, or `calibrate-skills` dropped from its repository list | Re-add `calibrate-skills` to the fine-grained PAT with **Contents: write**; refresh the Production secret |
| `sync-api-spec` never runs on calibrate | calibrate#108 not merged, or `repository_dispatch` not wired | Merge docs PR; confirm `sync-api-spec.yml` listens for `sync-api-spec` |
| `sync-from-spec` never runs on calibrate-skills | `OPENAPI_SPEC_URL` unset on calibrate-skills, or dispatch not reaching the repo | Set `OPENAPI_SPEC_URL` on calibrate-skills; confirm `calibrate-skills` is in `PUSH_TO_REPO_TOKEN`'s repo list and `sync-from-spec.yml` listens for `sync-api-spec` |
| PAT rejected pushing `release.yaml` | Missing **Workflows: write** on `PUSH_TO_REPO_TOKEN` | Add Workflows: write to the fine-grained PAT |
| PAT rejected pushing to `calibrate-mcp` | `calibrate-mcp` not in `PUSH_TO_REPO_TOKEN`'s repo list | Add `calibrate-mcp` to the fine-grained PAT with **Contents: write** |
| Release fails: `gpg_private_key` not supplied | GPG secrets missing in **calibrate-cli** | Add `CLI_GPG_SECRET_KEY` + `CLI_GPG_PASSPHRASE` |
| Release fails: `field token not found in type config.Homebrew` | Speakeasy `.goreleaser.yaml` incompatible with GoReleaser >=2.17 | Merge backend patch (`patch-goreleaser-config.sh`) or move `token` under `repository` in `calibrate-cli`; re-run Release |
| Homebrew formula never appears | `HOMEBREW_TAP_GITHUB_TOKEN` missing or tap repo missing | Add secret; create `dalmia/homebrew-tap` |
| MCP npm publish fails: OIDC / `id-token` error | Trusted publisher not configured, or repo/workflow/env mismatch | On npmjs.com set the trusted publisher to GitHub `dalmia/calibrate-mcp`, workflow `publish.yml`, blank environment; ensure the workflow has `id-token: write` |
| MCP npm publish fails: `npm ERR! Trusted publishing requires npm >= 11.5.1` | Runner's bundled npm too old for OIDC | The workflow runs `npm install -g npm@11`; confirm that step is present |
| MCP npm publish fails: `Cannot find module 'sigstore'` | Broken npm 12.0.0 bundle (sigstore not resolvable from libnpmpublish) | Ensure the npm upgrade is pinned to `npm@11`, not `@latest` |
| Ugly SDK method names | Overlay out of sync with Public API routes | Update both overlay files; run `test_sdk_overrides.py` |
| MCP tools missing or misnamed | `x-speakeasy-mcp` missing in `openapi/overlay.yaml` | Add `name` or `scopes` per route; run `test_sdk_overrides.py` |

## Related docs

- [`CLAUDE.md`](CLAUDE.md) — load-bearing invariants (public API tag gate, overlay sync rule, auth scheme pinning)
- PR #97 (`feat/speakeasy-clients`) — full Speakeasy migration (Python + CLI) when Speakeasy tier allows multiple targets
