# Self-Hosting Guide

This guide walks you through self-hosting Calibrate's Backend on your infra. The self-hosting guide for the frontend can be found [here](https://github.com/ARTPARK-SAHAI-ORG/calibrate-frontend/blob/main/SELF_HOSTING.md).

Pick your target cloud (AWS/GCP) and follow that section. Before you start, make sure to `fork` this repo to your team's Github account.

## Contents

1. [Architecture decisions](#architecture-decisions)
2. [Per-tenant isolation checklist](#per-tenant-isolation-checklist)
3. [Deploy on AWS](#deploy-on-aws)
4. [Deploy on GCP](#deploy-on-gcp)

# Deploy on AWS

End-to-end walkthrough on AWS (EC2 + S3). Substitute `<region>` with your AWS region (e.g. `ap-south-1`, `us-east-1`).

> The existing AWS production deploy is fully automated by [.github/workflows/deploy.yml](.github/workflows/deploy.yml). For a brand-new tenant on AWS, you provision the infra once (steps 1–7 below), then add a GitHub Actions environment with your secrets and trigger that workflow for subsequent deploys. The first-time provisioning is the part this section walks through.

## 1. Create the S3 bucket

An S3 bucket is used to store the results of all your evals and media files. Create one and name it `calibrate-backend-artifacts`. Ensure to block all public access for the S3 bucket as the app uses presigned URLs for client access.

Update the CORS permissions for your bucket:

```
[
    {
        "AllowedHeaders": [
            "*"
        ],
        "AllowedMethods": [
            "GET",
            "PUT",
            "POST"
        ],
        "AllowedOrigins": [
            "*"
        ],
        "ExposeHeaders": []
    }
]
```

Keep it stricter is you need to.

## 2. Create an IAM role for the EC2 instance

Go to **IAM → Roles → Create role**.

1. **Trusted entity**: AWS service → **EC2**.
2. **Permissions**: skip attaching managed policies for now, click Next, name it `calibrate-backend-ec2`, create.
3. Open the role → **Add permissions → Create inline policy** → JSON tab. Paste an S3 policy scoped to `calibrate-backend-artifacts` (Get/Put/Delete/ListBucket on the bucket and `/*`). Name it `calibrate-bucket-access`.

## 3. Create the security group

Go to **VPC → Security groups → Create security group** (or do it inline during EC2 launch).

1. **Name**: `calibrate-backend-sg`. **VPC**: your default VPC.
2. **Inbound rules** → Add rule:
   - HTTP (80), Source `0.0.0.0/0`
   - HTTPS (443), Source `0.0.0.0/0`
3. Leave outbound as default (all traffic).

## 4. Create an EBS volume

This will be attached to the instance and the SQLite DB file will be stored on it. 100 GB is generous to begin with. Increase later as needed.

## 5. Launch the EC2 instance

Attach the `calibrate-backend-ec2` role, the security group and the EBS volume created in the previous steps to the instance. Allocate and associate an Elastic IP with the instance too which will be the stable public address for your instance.

## 6. SSH into your instance

## 7. Install Docker

## 8. Clone the repo and build the image

```bash
sudo apt-get update && sudo apt-get install -y git    # or: sudo dnf install -y git
git clone https://github.com/<your-org>/calibrate-backend.git
cd calibrate-backend
docker build -t calibrate-backend:local .
```

Takes 5–15 minutes the first time.

## 9. Create the `.env` file

```bash
cd ~/calibrate-backend
cat > .env <<'EOF'
# Image
IMAGE_NAME=calibrate-backend
IMAGE_TAG=local
CONTAINER_NAME=calibrate-backend
PORT=80

# Persistence
APP_FOLDER_PATH=/appdata
DB_ROOT_DIR=/appdata

# Auth — generate fresh, do NOT reuse from any other tenant
JWT_SECRET_KEY=PASTE_OUTPUT_OF_openssl_rand_-base64_32
JWT_EXPIRATION_HOURS=168

# Object storage (AWS S3 — IAM instance role provides creds)
S3_ENDPOINT_URL=
S3_OUTPUT_BUCKET=calibrate-backend-artifacts
AWS_ACCESS_KEY_ID=
AWS_SECRET_ACCESS_KEY=
AWS_REGION=<region>

# Admin / default seeded user
SUPERADMIN_EMAIL=you@example.com
DEFAULT_USER_EMAIL=you@example.com
DEFAULT_USER_FIRST_NAME=You
DEFAULT_USER_LAST_NAME=Admin

# Docs HTTP basic auth
DOCS_USERNAME=admin
DOCS_PASSWORD=CHANGE_ME

# CORS — restrict to your frontend origin in production
CORS_ALLOWED_ORIGINS=*

# Concurrency
MAX_CONCURRENT_JOBS=1
MAX_CONCURRENT_JOBS_PER_USER=1
DEFAULT_MAX_ROWS_PER_EVAL=20

# Provider keys
OPENROUTER_API_KEY=
OPENAI_API_KEY=
DEEPGRAM_API_KEY=
CARTESIA_API_KEY=
SMALLEST_API_KEY=
GROQ_API_KEY=
SARVAM_API_KEY=
ELEVENLABS_API_KEY=
GOOGLE_API_KEY=
GOOGLE_CLIENT_ID=
GOOGLE_APPLICATION_CREDENTIALS=
GOOGLE_CLOUD_PROJECT_ID=

# Tracing
SENTRY_DSN=
SENTRY_ENVIRONMENT=production
SENTRY_TRACES_SAMPLE_RATE=1.0
SENTRY_PROFILES_SAMPLE_RATE=1.0
ENVIRONMENT=production
ENABLE_TRACING=false
OTEL_EXPORTER_OTLP_ENDPOINT=
OTEL_EXPORTER_OTLP_HEADERS=
LANGFUSE_TRACING_ENVIRONMENT=
LANGFUSE_HOST=
LANGFUSE_BASE_URL=
LANGFUSE_PUBLIC_KEY=
LANGFUSE_SECRET_KEY=
EOF
chmod 600 .env

# Generate JWT secret and paste into .env
openssl rand -base64 32
```

Update the default values given above as needed. For example, You might want to set the `SUPERADMIN_EMAIL` to an email address you own. Set the API keys for different providers (e.g. OpenRouter, OpenAI, etc.). Set the `GOOGLE_CLIENT_ID` to the same value as the one used for self-hosting the frontend.

## 10. Start the app

```bash
docker compose up -d
docker compose logs -f
```

Watch for `Uvicorn running on http://0.0.0.0:8000`. Ctrl-C the log tail.

## 11. Verify it works

Open `http://<INSTANCE_ELASTIC_IP>:8000/docs` on your browser. It should load the FastAPI docs for the server.

## 12. Moving from HTTP to HTTPS

Use nginx to route your custom domain (e.g. calibrate-backend.<yourdomain.com>) to the server. Use certbot to make the connection secure using HTTPs. 

## 13. Verify everything works

Open `https://<YOUR_DOMAIN>/docs` on your browser. It should load the FastAPI docs for the server.

If you frontend is set up, [create a new speech-to-text dataset](https://calibrate.artpark.ai/docs/core-concepts/speech-to-text#create-a-dataset) and upload one audio. If it uploads successfully, your S3 connection works.

# Deploy on GCP

End-to-end walkthrough on Google Cloud (Compute Engine + GCS). Substitute `<project-id>` with your GCP project ID.

## GCP / 0. Set gcloud defaults

```bash
gcloud config set project <project-id>
gcloud config set compute/region us-central1
gcloud config set compute/zone us-central1-a
```

> **Gotcha:** `gcloud` does **not** read `$REGION` / `$ZONE` from your shell. It reads from `gcloud config`. If `gcloud config list` shows the wrong default zone (e.g. `asia-south1-a`), every command will silently target the wrong region. Verify with `gcloud config list` first.

## GCP / 1. Reserve a static IP

```bash
gcloud compute addresses create calibrate-backend-ip --region=us-central1
```

> A reserved-but-unattached static IP costs ~$0.01/hour (~$7/month). Once attached to a running VM, it's free. So either attach promptly (step 4) or release it (`gcloud compute addresses delete calibrate-backend-ip --region=us-central1`) until you're ready.

## GCP / 2. Create the persistent disk

```bash
gcloud compute disks create calibrate-appdata \
  --size=100GB --type=pd-balanced --zone=us-central1-a
```

You'll see a warning that the disk is unformatted — **that's fine**. Formatting happens after the VM is up; don't try to format from your laptop.

> **Disk sizing:** the SQLite file alone won't grow into the GB range unless you accumulate millions of dataset rows. 20 GB is enough for the DB. We use 100 GB to leave room for future growth and operational headroom; resize down later with `gcloud compute disks resize` if you don't need it.

## GCP / 3. Verify firewall rules

GCP firewalls live on the **network**, not on instances. They attach via target tags. The default network usually has these pre-created:

```bash
gcloud compute firewall-rules list
```

You're looking for two rows:

| Need | What to check |
|---|---|
| Port 80 open | A row with `tcp:80` allowed, source `0.0.0.0/0`, target tag `http-server` (or empty target) |
| Port 443 open | A row with `tcp:443` allowed, source `0.0.0.0/0`, target tag `https-server` (or empty target) |

The default rules are named `default-allow-http` and `default-allow-https`. If both are present, **skip to step 4**. Otherwise create the missing one(s):

```bash
gcloud compute firewall-rules create allow-http  --allow tcp:80  --target-tags=http-server
gcloud compute firewall-rules create allow-https --allow tcp:443 --target-tags=https-server
```

> **Gotcha:** `--filter="targetTags:http-server"` and `--filter="targetTags=http-server"` **both error** on `firewall-rules list` due to a long-standing gcloud quirk. To filter, dump everything and grep client-side: `gcloud compute firewall-rules list --format="value(name,targetTags.list())" | grep http-server`.

## GCP / 4. Create the VM

```bash
gcloud compute instances create calibrate-backend \
  --machine-type=e2-standard-4 \
  --image-family=debian-12 --image-project=debian-cloud \
  --boot-disk-size=30GB \
  --disk=name=calibrate-appdata,device-name=appdata,mode=rw,boot=no \
  --tags=http-server,https-server \
  --metadata=enable-oslogin=TRUE \
  --address=$(gcloud compute addresses describe calibrate-backend-ip --region=us-central1 --format='value(address)')
```

What each flag does:

- `--machine-type=e2-standard-4` — 4 vCPU, 16 GB RAM. Reasonable starting size; bump to `e2-standard-8` if benchmarks saturate it.
- `--image-family=debian-12 --image-project=debian-cloud` — Debian 12. **Ubuntu** is fine if your team prefers it: swap to `--image-family=ubuntu-2404-lts-amd64 --image-project=ubuntu-os-cloud`. Both work identically with Docker.
- `--boot-disk-size=30GB` — the 10 GB default is too small once the OS, Docker, the image, and any temp files land on it.
- `--disk=name=calibrate-appdata,device-name=appdata,...` — attaches the persistent disk from step 2. **`device-name=appdata` is critical:** GCE creates a stable symlink at `/dev/disk/by-id/google-appdata` based on this name. `/etc/fstab` and the mount commands below depend on it.
- `--tags=http-server,https-server` — these tags are what tie the VM to the firewall rules from step 3.
- `--metadata=enable-oslogin=TRUE` — uses Google identity for SSH instead of static keys. Recommended for tenant deploys.
- `--address=$(...)` — attaches the static IP from step 1 in one shot. If you forget this, the VM gets an ephemeral IP that you'd swap later (see Troubleshooting at the bottom of this section).

## GCP / 5. SSH in and prepare the disk

```bash
gcloud compute ssh calibrate-backend
```

> **Gotcha:** if SSH errors with "resource not found" pointing at a different zone, it's the gcloud default-zone bug. Either pass `--zone=us-central1-a` explicitly or fix the default with `gcloud config set compute/zone us-central1-a`. Shell vars like `$ZONE` don't help — gcloud doesn't read them.

Inside the VM:

```bash
# Confirm the disk symlink exists
ls -l /dev/disk/by-id/ | grep appdata
# Expect: lrwxrwxrwx ... google-appdata -> ../../sdb

# Check whether already formatted (idempotency for re-runs of this guide)
sudo file -sL /dev/disk/by-id/google-appdata
# If output says "data" → blank, run mkfs below.
# If output mentions "ext4 filesystem" → already formatted, skip mkfs.
```

> **Gotcha:** `sudo file -s` on a symlink reports the symlink itself, not its target. Use `-L` to follow, or pass `/dev/sdb` directly. The `-sL` form is what works.

If blank, format it (**destructive — only the first time**):

```bash
sudo mkfs.ext4 -F /dev/disk/by-id/google-appdata
```

Mount and persist across reboots:

```bash
sudo mkdir -p /appdata
echo '/dev/disk/by-id/google-appdata /appdata ext4 discard,defaults 0 2' | sudo tee -a /etc/fstab
sudo mount /appdata
sudo chown -R $USER /appdata
```

What the `/etc/fstab` line does, field by field:

| Field | Value | Meaning |
|---|---|---|
| 1 | `/dev/disk/by-id/google-appdata` | What to mount (the GCE-stable symlink) |
| 2 | `/appdata` | Where to mount it |
| 3 | `ext4` | Filesystem type |
| 4 | `discard,defaults` | `discard` = SSD TRIM, `defaults` = standard rw/auto |
| 5 | `0` | Skip legacy `dump` backups |
| 6 | `2` | Run `fsck` on boot, after the root disk |

Verify:

```bash
df -h /appdata
# Expect: /dev/sdb (or similar)  ~98G  ...  /appdata
```

If it shows `/dev/root` instead, the disk didn't mount — you forgot the `sudo mount /appdata` step or the `/etc/fstab` line is malformed. **Writes to `/appdata` before you fix this go to the boot disk and disappear when you fix it.**

## GCP / 6. Install Docker

```bash
curl -fsSL https://get.docker.com | sudo sh
sudo usermod -aG docker $USER && newgrp docker
docker run --rm hello-world
```

Should print "Hello from Docker!" with no `permission denied` error. If you see permission denied, the `newgrp docker` didn't take effect — log out and back in.

## GCP / 7. Set up GCS (from your laptop)

```bash
PROJECT=$(gcloud config get-value project)

# 7a. Create the bucket
gcloud storage buckets create gs://calibrate-backend-artifacts \
  --location=us-central1 --uniform-bucket-level-access

# 7b. Enable versioning (recoverable from accidental overwrites/deletes)
gcloud storage buckets update gs://calibrate-backend-artifacts --versioning

# 7c. Service account for storage access
gcloud iam service-accounts create calibrate-backend-storage \
  --display-name="Calibrate backend storage"

# 7d. Grant object-level access on just this bucket (least privilege)
gcloud storage buckets add-iam-policy-binding gs://calibrate-backend-artifacts \
  --member="serviceAccount:calibrate-backend-storage@${PROJECT}.iam.gserviceaccount.com" \
  --role="roles/storage.objectAdmin"

# 7e. Generate HMAC keys — SAVE THE SECRET, it's shown ONCE
gcloud storage hmac create \
  calibrate-backend-storage@${PROJECT}.iam.gserviceaccount.com
```

Copy the `accessId` (looks like `GOOG1E...`) and `secret` from the output. You'll need them in the `.env` in step 9.

### 7f. Configure bucket CORS (required if browser uploads from a different origin)

The `/presigned-url` flow returns a URL the **browser** uploads to directly with `PUT`. That request lands on `storage.googleapis.com`, not your backend — so the backend's `CORS_ALLOWED_ORIGINS` doesn't apply. You need a CORS rule **on the bucket**.

You can skip this section if uploads only happen server-side (backend-to-GCS). It only matters when a browser on a different origin (e.g. `https://app.tenant.example.com`) needs to PUT to GCS directly.

```bash
cat > /tmp/gcs-cors.json <<'EOF'
[
  {
    "origin": ["https://app.tenant.example.com"],
    "method": ["GET", "PUT"],
    "responseHeader": ["Content-Type", "Authorization", "x-goog-resumable"],
    "maxAgeSeconds": 3600
  }
]
EOF

gcloud storage buckets update gs://calibrate-backend-artifacts \
  --cors-file=/tmp/gcs-cors.json
```

To allow multiple origins (prod + staging + local dev), pass them all in the `origin` array:

```json
"origin": [
  "https://app.tenant.example.com",
  "https://staging.tenant.example.com",
  "http://localhost:3000"
]
```

Verify:

```bash
gcloud storage buckets describe gs://calibrate-backend-artifacts --format="value(cors_config)"
```

To clear CORS (e.g. before disabling browser-direct uploads):

```bash
gcloud storage buckets update gs://calibrate-backend-artifacts --clear-cors
```

## GCP / 8. Clone the repo and build the image

On the VM:

```bash
sudo apt-get update && sudo apt-get install -y git
git clone https://github.com/<your-org>/calibrate-backend.git
cd calibrate-backend
docker build -t calibrate-backend:local .
```

The build takes 5–15 minutes the first time. The image lands in the VM's local Docker cache; no registry needed for the initial deploy. (You can graduate to GitHub Actions + Artifact Registry later — see CI/CD subsection.)

If the repo is private, set up a deploy key (preferred) or use a personal access token over HTTPS.

## GCP / 9. Create the `.env` file

On the VM, in the repo root:

```bash
cd ~/calibrate-backend
cat > .env <<'EOF'
# Image
IMAGE_NAME=calibrate-backend
IMAGE_TAG=local
CONTAINER_NAME=calibrate-backend
PORT=80

# Persistence
APP_FOLDER_PATH=/appdata
DB_ROOT_DIR=/appdata

# Auth — generate fresh, do NOT reuse from any other tenant
JWT_SECRET_KEY=PASTE_OUTPUT_OF_openssl_rand_-base64_32
JWT_EXPIRATION_HOURS=168

# Object storage (GCS via S3 interop)
S3_ENDPOINT_URL=https://storage.googleapis.com
S3_OUTPUT_BUCKET=calibrate-backend-artifacts
AWS_ACCESS_KEY_ID=<HMAC accessId from step 7e>
AWS_SECRET_ACCESS_KEY=<HMAC secret from step 7e>
AWS_REGION=auto

# Admin / default seeded user
SUPERADMIN_EMAIL=you@example.com
DEFAULT_USER_EMAIL=you@example.com
DEFAULT_USER_FIRST_NAME=You
DEFAULT_USER_LAST_NAME=Admin

# Docs HTTP basic auth
DOCS_USERNAME=admin
DOCS_PASSWORD=CHANGE_ME

# CORS — restrict to your frontend origin in production
CORS_ALLOWED_ORIGINS=*

# Concurrency
MAX_CONCURRENT_JOBS=1
MAX_CONCURRENT_JOBS_PER_USER=1
DEFAULT_MAX_ROWS_PER_EVAL=20

# Provider keys
OPENROUTER_API_KEY=
OPENAI_API_KEY=
DEEPGRAM_API_KEY=
CARTESIA_API_KEY=
SMALLEST_API_KEY=
GROQ_API_KEY=
SARVAM_API_KEY=
ELEVENLABS_API_KEY=
GOOGLE_API_KEY=
GOOGLE_CLIENT_ID=
GOOGLE_APPLICATION_CREDENTIALS=
GOOGLE_CLOUD_PROJECT_ID=

# Tracing
SENTRY_DSN=
SENTRY_ENVIRONMENT=production
SENTRY_TRACES_SAMPLE_RATE=1.0
SENTRY_PROFILES_SAMPLE_RATE=1.0
ENVIRONMENT=production
ENABLE_TRACING=false
OTEL_EXPORTER_OTLP_ENDPOINT=
OTEL_EXPORTER_OTLP_HEADERS=
LANGFUSE_TRACING_ENVIRONMENT=
LANGFUSE_HOST=
LANGFUSE_BASE_URL=
LANGFUSE_PUBLIC_KEY=
LANGFUSE_SECRET_KEY=
EOF
chmod 600 .env

# Generate JWT secret and paste into .env
openssl rand -base64 32
```

> `PORT=80` works because the existing `default-allow-http` firewall rule already exposes port 80. Compose maps `${PORT}:8000` so the container listens on 8000 internally and the VM exposes it externally on 80. Switch to `PORT=8000` once Caddy is in front (HTTPS subsection below).

## GCP / 10. Start the app

```bash
docker compose up -d
docker compose logs -f
```

Watch for `Uvicorn running on http://0.0.0.0:8000`. Ctrl-C the log tail (the container keeps running).

## GCP / 11. Verify from the internet

From your laptop:

```bash
IP=$(gcloud compute addresses describe calibrate-backend-ip --region=us-central1 --format='value(address)')
curl http://$IP/openapi.json | head -c 200
```

If you get JSON back, the API is live.

## GCP / 12. Verify GCS uploads work

Inside the container:

```bash
docker exec -it calibrate-backend uv run python -c "
from utils import get_s3_client
c = get_s3_client()
print('endpoint:', c.meta.endpoint_url)
"
# Expect: endpoint: https://storage.googleapis.com
```

After running any job (or hitting `POST /presigned-url`):

```bash
gcloud storage ls --recursive gs://calibrate-backend-artifacts/ | head
```

You should see object keys appearing.

## GCP / Object storage (GCS via S3 interop)

The codebase only speaks the AWS S3 protocol via boto3, but `get_s3_client()` ([src/utils.py](src/utils.py)) honors `S3_ENDPOINT_URL`:

```python
endpoint_url = os.getenv("S3_ENDPOINT_URL")
if endpoint_url:
    kwargs["endpoint_url"] = endpoint_url
```

Pointing this at `https://storage.googleapis.com` + HMAC keys = boto3 talking to GCS. What works:

- `upload_file` / `put_object`
- `get_object`
- Presigned URLs for `get_object` and `put_object` (SigV4)
- The `s3://bucket/key` URI scheme stored in the DB — `presign_audio_path()` parses it as bucket+key, the client routes to whichever endpoint is configured. Nothing branches on the literal `s3://` string.

What's caveated:

- Multipart uploads — not exercised in this codebase (file sizes are small enough for single-part PUTs).
- HMAC keys are tied to the service account, not user-managed. If the SA is deleted, the keys die. **Don't delete `calibrate-backend-storage`** without first rotating to new keys on a different SA.

## GCP / Authentication and first login

### The seeded default user has no password

`init_db()` creates a row in the `users` table from `DEFAULT_USER_EMAIL`, but **does not set `password_hash`** ([src/db.py:803](src/db.py:803)). The seeded user can only log in via Google OAuth (or have a password set later via the API).

### Pick one

**Path A — Google OAuth (recommended for human users)**

1. GCP Console → APIs & Services → Credentials → Create OAuth client ID.
2. Application type: **Web application**.
3. Authorized JavaScript origins: your frontend's URL (e.g. `https://app.tenant.example.com`). For local testing, also add `http://localhost:3000`.
4. Copy the client ID into `GOOGLE_CLIENT_ID` in `.env`. Restart the container.
5. The Google email logging in must match `DEFAULT_USER_EMAIL` (or you'll create a second user).

**Path B — email/password signup**

1. Hit the password signup endpoint (check [src/routers/auth.py](src/routers/auth.py) for the exact route — typically `POST /auth/signup`).
2. Creates a new user row distinct from the seeded one. Not a superadmin unless their email matches `SUPERADMIN_EMAIL`.

**Path C — API key (for programmatic access)**

API keys (`/api-keys` endpoints) authenticate via `X-API-Key` or `Authorization: Bearer calib_...`. Useful for CI integrations and `POST /evaluators/{uuid}/invoke`. Created by an authenticated user — so chicken-and-egg, you need Path A or B first.

## GCP / Moving from HTTP to HTTPS (Caddy)

**Don't put real users on plain HTTP.** JWTs and basic-auth credentials cross the wire in cleartext. Once you've verified the deploy on `http://<ip>` and DNS is pointing at the VM, immediately put it behind TLS.

This section assumes you already have the app running on `PORT=80` and the domain resolves to the VM's static IP.

Caddy is the simplest option on Linux: one binary, one-line config, automatic Let's Encrypt cert provisioning + renewal, automatic HTTP→HTTPS redirect.

### Order matters

The container currently holds port 80. Caddy will need to take it over to handle the cert challenge and serve TLS. Sequence:

1. Install Caddy (it'll fail to bind port 80 — expected at this stage).
2. Move the container off port 80 (`PORT=80` → `PORT=8000`).
3. Configure Caddy with your domain + reverse proxy to localhost:8000.
4. Restart Caddy → it binds 80/443, fetches a cert, starts serving HTTPS.

### Step 1 — Install Caddy

On the VM:

```bash
sudo apt-get install -y debian-keyring debian-archive-keyring apt-transport-https curl
curl -fsSL https://dl.cloudsmith.io/public/caddy/stable/gpg.key | sudo gpg --dearmor -o /usr/share/keyrings/caddy-stable-archive-keyring.gpg
curl -fsSL https://dl.cloudsmith.io/public/caddy/stable/debian.deb.txt | sudo tee /etc/apt/sources.list.d/caddy-stable.list
sudo apt-get update && sudo apt-get install -y caddy
```

The install starts Caddy via systemd. It'll fail to bind port 80 (the container has it). That's expected — `systemctl status caddy` will be unhappy until step 3.

### Step 2 — Move the container off port 80

```bash
cd ~/calibrate-backend
sed -i 's/^PORT=80$/PORT=8000/' .env
docker compose up -d
```

Verify the swap:

```bash
docker compose ps                                # should show 0.0.0.0:8000->8000/tcp
curl http://localhost:8000/openapi.json | head -c 100   # should still return JSON
```

### Step 3 — Configure Caddy

```bash
sudo tee /etc/caddy/Caddyfile <<'EOF'
api.tenant.example.com {
    reverse_proxy localhost:8000
}
EOF

sudo systemctl restart caddy
sudo systemctl status caddy
```

Status should show `active (running)`. If it's still erroring:

```bash
sudo journalctl -u caddy -n 50 --no-pager
```

### Step 4 — Verify

From your laptop:

```bash
curl -I https://api.tenant.example.com/openapi.json
```

Expect `HTTP/2 200`. The first request triggers Caddy to provision a Let's Encrypt cert (HTTP-01 challenge); takes a few seconds, then it caches.

Browser: `https://api.tenant.example.com/docs`.

### Step 5 — Update CORS and OAuth

Set `CORS_ALLOWED_ORIGINS` to your **frontend's** origin (NOT the backend URL — see note below):

```bash
sed -i 's|^CORS_ALLOWED_ORIGINS=.*|CORS_ALLOWED_ORIGINS=https://app.tenant.example.com|' .env
docker compose up -d
```

Multiple origins are comma-separated:

```
CORS_ALLOWED_ORIGINS=https://app.tenant.example.com,https://staging.tenant.example.com,http://localhost:3000
```

If using Google OAuth, add `https://api.tenant.example.com` to your OAuth client's **Authorized JavaScript origins** in GCP Console → APIs & Services → Credentials.

> **What CORS does:** controls which *browser-tab origins* can call your backend. The backend's own URL never appears as an `Origin` header on requests to itself, so listing it here is a no-op. Same-origin tooling like Swagger UI on `/docs` doesn't need a CORS entry either. `curl` and Postman never trigger CORS at all.

### Gotchas

- **Port 80 must stay open**, even after HTTPS works. Caddy uses HTTP-01 for cert renewal every 60 days. If you close port 80 in the firewall, renewal silently fails and the cert eventually expires.
- **HTTP→HTTPS redirect is automatic.** Caddy adds a 308 from `http://...` to `https://...` for free. `curl -I http://api.tenant.example.com/` should return `308`.
- **DNS propagation** — if Caddy fails the cert challenge with "no such host" or "context deadline exceeded," DNS hasn't propagated to the cert authority yet. Wait 5–10 minutes, then `sudo systemctl restart caddy`.
- **Port 443 firewall rule** — `default-allow-https` should already cover this. If `https://` times out: `gcloud compute firewall-rules list | grep 443`.
- **Cert renewal** is automatic. No cron, no manual action — Caddy renews ~30 days before expiry.

### Alternative: GCP HTTPS Load Balancer

Heavier setup but offloads TLS, gives you GCP-managed certs, and lets you front multiple backends. Worth it if you want WAF (Cloud Armor), multi-region, or a single ingress for backend + frontend. Not required for a single-VM tenant.

## GCP / Operational concerns

### Docker log rotation

Already configured in `docker-compose.yml`:

```yaml
logging:
  driver: json-file
  options:
    max-size: "10m"
    max-file: "5"
```

Caps each container at ~50 MB total. **Recreate the container after pulling the change**: `docker compose up -d`. A `restart` is not enough — the logging driver is set on creation.

### Persistent disk snapshots

```bash
gcloud compute resource-policies create snapshot-schedule daily-appdata \
  --region=us-central1 \
  --max-retention-days=14 \
  --start-time=03:00 \
  --daily-schedule \
  --on-source-disk-delete=keep-auto-snapshots

gcloud compute disks add-resource-policies calibrate-appdata \
  --zone=us-central1-a \
  --resource-policies=daily-appdata
```

What this gives you:

- Daily snapshot at 03:00 UTC, retained for 14 days.
- Snapshots are incremental & compressed — typically tens of MB per day, total cost ~pennies/month.
- `--on-source-disk-delete=keep-auto-snapshots` means snapshots survive even if the disk is deleted.
- RPO = 24 hours. For tighter, use `--hourly-schedule --hours-in-cycle=6`.

### Error monitoring (Sentry)

The architecture explicitly relies on `capture_exception_to_sentry()` for background-thread failures (CLAUDE.md: "All job failures route through `capture_exception_to_sentry()`"). Without `SENTRY_DSN`, those failures go to container stdout and nowhere else.

1. Create a Sentry project for this tenant.
2. Set `SENTRY_DSN`, `SENTRY_ENVIRONMENT=production` in `.env`.
3. Restart the container.

### Restart on VM reboot

The container has `restart: unless-stopped`. Docker starts at boot via systemd. **Test it once** before relying on it:

```bash
sudo reboot
# Wait 60 seconds, then from your laptop:
curl http://<ip>/openapi.json
```

If the API doesn't come back, check `systemctl status docker` and `docker compose ps` on the VM.

### Lock down SSH (Identity-Aware Proxy)

The default `default-allow-ssh` rule allows `tcp:22` from `0.0.0.0/0`. Switch to IAP-only ingress:

```bash
gcloud compute firewall-rules delete default-allow-ssh
gcloud compute firewall-rules create allow-iap-ssh \
  --allow=tcp:22 --source-ranges=35.235.240.0/20

# SSH from your laptop now uses --tunnel-through-iap
gcloud compute ssh calibrate-backend --tunnel-through-iap
```

Eliminates the entire bot-bruteforce SSH attack surface.

### Secret Manager (when ready)

`.env` on disk in cleartext is fine for a single-admin deploy. For tenant-grade, store each value as a Secret Manager secret and fetch on deploy:

```bash
echo -n "<value>" | gcloud secrets create calibrate-jwt-key --data-file=-

# Grant the VM's service account
gcloud secrets add-iam-policy-binding calibrate-jwt-key \
  --member="serviceAccount:<vm-sa-email>" \
  --role="roles/secretmanager.secretAccessor"

# At deploy time
gcloud secrets versions access latest --secret=calibrate-jwt-key
```

## GCP / CI/CD: replacing build-on-VM

Building on the VM works for a one-shot but doesn't scale. Switch to: build in GitHub Actions → push to Artifact Registry → SSH onto VM, `pull && up -d`.

The existing AWS workflows ([.github/workflows/deploy.yml](.github/workflows/deploy.yml), [.github/workflows/deploy-staging.yml](.github/workflows/deploy-staging.yml)) are the template. To adapt for GCP:

1. Create a new GitHub Actions environment with all the tenant's secrets.
2. Replace the EC2-targeted SSH step with one of:
   - `appleboy/ssh-action` against the GCE static IP (set up an OS Login key in GitHub secrets), **or**
   - [`google-github-actions/ssh-compute`](https://github.com/google-github-actions/ssh-compute) with Workload Identity Federation (no SSH key in secrets).
3. Swap Docker Hub for GCP Artifact Registry to keep the image close to the VM.
4. Use a distinct Compose project name (`docker compose -p calibrate-tenant-x`) so multiple tenants on one host don't collide.

## GCP / Troubleshooting

### "resource not found" on `gcloud compute ssh` pointing at a wrong zone

```
ERROR: ... 'projects/.../zones/asia-south1-a/instances/calibrate-backend' was not found
```

gcloud is using its global default zone, **not** your shell's `$ZONE`. Fix:

```bash
gcloud config list
gcloud config set compute/zone us-central1-a
gcloud config set compute/region us-central1
```

Or pass `--zone=us-central1-a` explicitly on every command.

### `df -h /appdata` shows `/dev/root` instead of `/dev/sdb`

The persistent disk isn't mounted. Most likely you ran `mkdir` and the `tee >> /etc/fstab` but not `sudo mount /appdata`. Also check `sudo file -sL /dev/disk/by-id/google-appdata` — if it says `data`, the disk needs `mkfs.ext4` first.

**Important:** anything written to `/appdata` while it was unmounted is on the boot disk. Once you mount the persistent disk over the same path, those files are shadowed (not deleted). To recover: `sudo umount /appdata && ls /appdata`.

### `file -s` reports "symbolic link to ../../sdb"

`-s` doesn't follow symlinks. Use `-sL`:

```bash
sudo file -sL /dev/disk/by-id/google-appdata
```

Or pass the resolved device: `sudo file -s /dev/sdb`.

### `gcloud compute firewall-rules list --filter="targetTags:http-server"` errors

Known gcloud quirk on this resource. Filter client-side:

```bash
gcloud compute firewall-rules list --format="value(name,targetTags.list())" | grep http-server
```

### Container exits immediately after `docker compose up -d`

Almost always a missing required env var. Check:

```bash
docker compose logs --tail=50
```

Look for "ValueError: S3_OUTPUT_BUCKET environment variable is required" or `KeyError: 'JWT_SECRET_KEY'`. Fill in, `docker compose up -d` again.

### `curl http://<ip>/...` times out

1. `docker compose ps` — STATUS should say `Up`.
2. Confirm port mapping in `docker compose ps`.
3. Confirm firewall rule covers your port (default rules cover 80/443; if you set `PORT=8000`, you need a separate rule).
4. Confirm static IP attached: `gcloud compute instances describe calibrate-backend --format="value(networkInterfaces[0].accessConfigs[0].natIP)"` should match `gcloud compute addresses describe calibrate-backend-ip --region=us-central1 --format="value(address)"`.

### Static IP shows `RESERVED` instead of `IN_USE`

Not attached to a VM. Either you forgot `--address=...` on `instances create`, or the VM was created with an ephemeral IP. To swap:

```bash
gcloud compute instances delete-access-config calibrate-backend \
  --access-config-name="external-nat"
gcloud compute instances add-access-config calibrate-backend \
  --access-config-name="external-nat" \
  --address=$(gcloud compute addresses describe calibrate-backend-ip --region=us-central1 --format='value(address)')
```

Brief network blip during the swap — SSH sessions drop. Plan around it for live deploys.

### GCS uploads hit the wrong endpoint

Inside the container:

```bash
docker exec -it calibrate-backend uv run python -c "
from utils import get_s3_client
c = get_s3_client()
print('endpoint:', c.meta.endpoint_url)
"
```

If it prints `https://s3.amazonaws.com` (or similar) instead of `https://storage.googleapis.com`, `S3_ENDPOINT_URL` didn't make it into the container. Check `.env`, then `docker compose up -d` to recreate.

### `docker run hello-world` says "permission denied" after install

`newgrp docker` didn't apply to your current shell. Log out and back in.

## GCP / Restore from snapshot

```bash
# 1. List recent snapshots
gcloud compute snapshots list --filter="sourceDisk:calibrate-appdata"

# 2. Create a new disk from the snapshot
gcloud compute disks create calibrate-appdata-restored \
  --source-snapshot=<snapshot-name> --zone=us-central1-a

# 3. Stop, swap, restart
gcloud compute instances stop calibrate-backend
gcloud compute instances detach-disk calibrate-backend --disk=calibrate-appdata
gcloud compute instances attach-disk calibrate-backend \
  --disk=calibrate-appdata-restored --device-name=appdata
gcloud compute instances start calibrate-backend
```

The `--device-name=appdata` is critical — it's what makes `/dev/disk/by-id/google-appdata` resolve, which `/etc/fstab` references. Without it the VM boots but `/appdata` stays unmounted.

After confirming the restore is good, delete the old disk: `gcloud compute disks delete calibrate-appdata --zone=us-central1-a`.
