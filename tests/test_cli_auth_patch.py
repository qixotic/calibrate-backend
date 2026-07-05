"""patch-cli-auth-commands.sh must hoist login/logout to the CLI root."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
PATCH_SCRIPT = REPO_ROOT / ".github/scripts/patch-cli-auth-commands.sh"
CALIBRATE_CLI = REPO_ROOT.parent / "calibrate-cli"


def test_patch_cli_auth_commands_hoists_login_logout(tmp_path: Path) -> None:
    if not CALIBRATE_CLI.is_dir():
        pytest.skip("calibrate-cli sibling repo not present locally")

    shutil.copytree(
        CALIBRATE_CLI,
        tmp_path / "cli",
        ignore=shutil.ignore_patterns(".git"),
    )
    cli_dir = tmp_path / "cli"

    subprocess.run(
        [str(PATCH_SCRIPT), str(cli_dir)],
        check=True,
        cwd=REPO_ROOT,
    )

    root_go = (cli_dir / "internal/cli/root.go").read_text()
    assert "initLoginCmd(rootCmd)" in root_go
    assert 'fmt.Errorf("init login: %w", err)' in root_go
    assert (cli_dir / "internal/cli/login.go").is_file()
    assert (cli_dir / "internal/cli/logout.go").is_file()

    auth_go = (cli_dir / "internal/cli/auth.go").read_text()
    assert "authCmd.Hidden = true" in auth_go
    assert "loginCmd.Hidden = true" in auth_go
    assert "whoamiCmd.Hidden = true" in auth_go
    assert "logoutCmd.Hidden = true" in auth_go

    root_go = (cli_dir / "internal/cli/root.go").read_text()
    assert "HiddenDefaultCmd = true" in root_go

    # Idempotent on re-run.
    subprocess.run(
        [str(PATCH_SCRIPT), str(cli_dir)],
        check=True,
        cwd=REPO_ROOT,
    )
