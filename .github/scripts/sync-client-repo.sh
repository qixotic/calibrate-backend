#!/usr/bin/env bash
# Push freshly generated Speakeasy output into a client repository.
# Usage: sync-client-repo.sh <owner/repo> <generated_dir> <commit_message>
set -euo pipefail

REPO="${1:?owner/repo required}"
SRC_DIR="${2:?generated source dir required}"
MESSAGE="${3:?commit message required}"
TOKEN="${PUSH_TO_REPO_TOKEN:?PUSH_TO_REPO_TOKEN is required}"

if [ ! -d "$SRC_DIR" ]; then
  echo "::error::Generated directory not found: $SRC_DIR" >&2
  exit 1
fi

WORKDIR="$(mktemp -d)"
trap 'rm -rf "$WORKDIR"' EXIT

git clone --depth 1 "https://x-access-token:${TOKEN}@github.com/${REPO}.git" "$WORKDIR/repo"

# Preserve hand-written files in the client repo.
rsync -a --delete \
  --exclude '.git' \
  --exclude '.speakeasyignore' \
  --exclude 'README.md' \
  "$SRC_DIR/" "$WORKDIR/repo/"

cd "$WORKDIR/repo"
git config user.email "github-actions[bot]@users.noreply.github.com"
git config user.name "github-actions[bot]"

if git status --porcelain | grep -q .; then
  git add -A
  git commit -m "$MESSAGE"
  git push origin HEAD
  echo "Synced $SRC_DIR -> $REPO"
else
  echo "No changes to sync for $REPO"
fi
