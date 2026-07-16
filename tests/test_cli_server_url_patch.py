"""patch-cli-server-url.sh must make --server-url a persistable global param."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
PATCH_SCRIPT = REPO_ROOT / ".github/scripts/patch-cli-server-url.sh"
CALIBRATE_CLI = REPO_ROOT.parent / "calibrate-cli"


def _patched_cli(tmp_path: Path) -> Path:
    shutil.copytree(
        CALIBRATE_CLI,
        tmp_path / "cli",
        ignore=shutil.ignore_patterns(".git"),
    )
    cli_dir = tmp_path / "cli"
    subprocess.run([str(PATCH_SCRIPT), str(cli_dir)], check=True, cwd=REPO_ROOT)
    return cli_dir


def test_patch_cli_server_url_persists(tmp_path: Path) -> None:
    if not CALIBRATE_CLI.is_dir():
        pytest.skip("calibrate-cli sibling repo not present locally")

    cli_dir = _patched_cli(tmp_path)

    # server-url now resolves flag > env > config (was flag-only).
    client_go = (cli_dir / "internal/client/client.go").read_text()
    assert 'resolveStringFlag(cmd, "server-url")' in client_go
    assert 'flagutil.GetStringFlag(cmd, "server-url")' not in client_go

    # config schema carries the persisted value + read case.
    config_go = (cli_dir / "internal/config/config.go").read_text()
    assert "ServerURL" in config_go
    assert 'yaml:"server_url,omitempty"' in config_go
    assert 'case "server-url":' in config_go

    # configure writes it in both interactive and --no-interactive paths.
    configure_go = (cli_dir / "internal/cli/configure.go").read_text()
    assert "cfg.ServerURL = v" in configure_go  # --no-interactive flag path
    assert "cfg.ServerURL = cfgServerURL" in configure_go  # interactive form
    assert "serverFields" in configure_go

    # whoami surfaces the resolved value + source.
    whoami_go = (cli_dir / "internal/cli/whoami.go").read_text()
    assert 'config.ResolveCredential(cmd, "server-url")' in whoami_go

    # Idempotent: a second run makes no further changes.
    before = {p: p.read_text() for p in cli_dir.rglob("*.go")}
    subprocess.run([str(PATCH_SCRIPT), str(cli_dir)], check=True, cwd=REPO_ROOT)
    after = {p: p.read_text() for p in cli_dir.rglob("*.go")}
    assert before == after


def test_patched_cli_compiles(tmp_path: Path) -> None:
    if not CALIBRATE_CLI.is_dir():
        pytest.skip("calibrate-cli sibling repo not present locally")
    if shutil.which("go") is None:
        pytest.skip("go toolchain not available")

    cli_dir = _patched_cli(tmp_path)
    result = subprocess.run(
        ["go", "build", "./..."],
        cwd=cli_dir,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
