#!/usr/bin/env bash
# Post-process Speakeasy CLI output: hoist login/logout, hide legacy auth group,
# hide shell-completion setup from top-level help.
set -euo pipefail

DIR="${1:-.speakeasy/out/cli}"
ROOT_GO="$DIR/internal/cli/root.go"
AUTH_GO="$DIR/internal/cli/auth.go"
LOGIN_GO="$DIR/internal/cli/login.go"
LOGOUT_GO="$DIR/internal/cli/logout.go"

if [[ ! -f "$ROOT_GO" || ! -f "$AUTH_GO" ]]; then
  echo "No generated CLI at $DIR — skipping CLI patch"
  exit 0
fi

if grep -q 'initLoginCmd(rootCmd)' "$ROOT_GO" \
  && grep -q 'authCmd.Hidden = true' "$AUTH_GO" \
  && grep -q 'HiddenDefaultCmd = true' "$ROOT_GO"; then
  echo "CLI already patched at $DIR"
  exit 0
fi

python3 - "$ROOT_GO" "$AUTH_GO" "$LOGIN_GO" "$LOGOUT_GO" <<'PY'
import re
import sys
from pathlib import Path

root_go, auth_go, login_go, logout_go = map(Path, sys.argv[1:5])

root_text = root_go.read_text()

if "HiddenDefaultCmd = true" not in root_text:
    root_text, n_completion = re.subn(
        r"(\t\}\n)(\tif err := agents\.InitAgentsRoot\(rootCmd\); err != nil \{\n)",
        r"\1\trootCmd.CompletionOptions.HiddenDefaultCmd = true\n\2",
        root_text,
        count=1,
    )
    if n_completion != 1:
        raise SystemExit(f"::warning::{root_go}: could not hide completion command")
    root_go.write_text(root_text)
    root_text = root_go.read_text()

if "initLoginCmd(rootCmd)" not in root_text:
    root_text, n = re.subn(
        r"(\tif err := initWhoamiCmd\(rootCmd\); err != nil \{\n"
        r"\t\treturn nil, fmt\.Errorf\(\"init whoami: %w\", err\)\n"
        r"\t\}\n)"
        r"(\tif err := initVersionCmd\(rootCmd\); err != nil \{\n)",
        r"\1"
        r"\tif err := initLoginCmd(rootCmd); err != nil {\n"
        r"\t\treturn nil, fmt.Errorf(" + '"init login: %w", err)\n'
        r"\t}\n"
        r"\tif err := initLogoutCmd(rootCmd); err != nil {\n"
        r"\t\treturn nil, fmt.Errorf(" + '"init logout: %w", err)\n'
        r"\t}\n"
        r"\2",
        root_text,
        count=1,
    )
    if n != 1:
        raise SystemExit(f"::warning::{root_go}: could not wire root login/logout commands")
    root_go.write_text(root_text)

auth_text = auth_go.read_text()

if "authCmd.Hidden = true" not in auth_text:
    auth_text, n_auth = re.subn(
        r"(authCmd := &cobra\.Command\{\n"
        r"\t\tUse:\s+\"auth\",\n"
        r"\t\tShort: \"Manage authentication credentials\",\n"
        r"\t\tLong: `Manage authentication credentials for calibrate\.\n\n"
        r"Subcommands:\n"
        r"  login   - Interactively configure credentials\n"
        r"  whoami  - Display current authentication status\n"
        r"  logout  - Clear all stored credentials`,\n"
        r"\t\})\n"
        r"\tparent\.AddCommand\(authCmd\)",
        r"\1\n\tauthCmd.Hidden = true\n\tparent.AddCommand(authCmd)",
        auth_text,
        count=1,
    )
    if n_auth != 1:
        raise SystemExit(f"::warning::{auth_go}: could not hide auth command group")

if "loginCmd.Hidden = true" not in auth_text:
    auth_text, n_login = re.subn(
        r"authCmd\.AddCommand\(&cobra\.Command\{\n"
        r"\t\tUse:\s+\"login\",\n"
        r"\t\tShort: \"Interactively configure authentication credentials\",\n"
        r"\t\tLong: `Interactively configure authentication credentials for calibrate\.\n"
        r"Secret credentials are stored in the OS keychain when available,\n"
        r"with a config file fallback\.\n\n"
        r"All fields are optional — press Enter to skip any field you don't need\.\n"
        r"Use the configure command for both authentication and global parameters\.`,\n"
        r"\t\tRunE: runAuthLoginCmd,\n"
        r"\t\}\)",
        """loginCmd := &cobra.Command{
\t\tUse:   "login",
\t\tShort: "Interactively configure authentication credentials",
\t\tLong: `Interactively configure authentication credentials for calibrate.
Secret credentials are stored in the OS keychain when available,
with a config file fallback.

All fields are optional — press Enter to skip any field you don't need.
Use the configure command for both authentication and global parameters.`,
\t\tRunE: runAuthLoginCmd,
\t}
\tloginCmd.Hidden = true
\tauthCmd.AddCommand(loginCmd)""",
        auth_text,
        count=1,
    )
    if n_login != 1 and "loginCmd.Hidden = true" not in auth_text:
        raise SystemExit(f"::warning::{auth_go}: could not hide nested auth login")

if "whoamiCmd.Hidden = true" not in auth_text:
    auth_text, n_whoami = re.subn(
        r"authCmd\.AddCommand\(&cobra\.Command\{\n"
        r"\t\tUse:\s+\"whoami\",\n"
        r"\t\tShort: \"Display current authentication configuration\",\n"
        r"\t\tLong: `Display the currently configured settings and their sources\.\n\n"
        r"Sources are shown as:\n"
        r"  \[flag\]    - Set via command line flag\n"
        r"  \[env\]     - Set via environment variable \(CALIBRATE_\*\)\n"
        r"  \[keyring\] - Set via OS keychain \(stored by login/configure command\)\n"
        r"  \[config\]  - Set via config file \(~/.config/calibrate/config\.yaml\)\n"
        r"  \[unset\]   - Not configured\n\n"
        r"Credential values are masked for security\.`,\n"
        r"\t\tRunE: runWhoamiCmd,\n"
        r"\t\}\)",
        """whoamiCmd := &cobra.Command{
\t\tUse:   "whoami",
\t\tShort: "Display current authentication configuration",
\t\tLong: `Display the currently configured settings and their sources.

Sources are shown as:
  [flag]    - Set via command line flag
  [env]     - Set via environment variable (CALIBRATE_*)
  [keyring] - Set via OS keychain (stored by login/configure command)
  [config]  - Set via config file (~/.config/calibrate/config.yaml)
  [unset]   - Not configured

Credential values are masked for security.`,
\t\tRunE: runWhoamiCmd,
\t}
\twhoamiCmd.Hidden = true
\tauthCmd.AddCommand(whoamiCmd)""",
        auth_text,
        count=1,
    )
    if n_whoami != 1 and "whoamiCmd.Hidden = true" not in auth_text:
        raise SystemExit(f"::warning::{auth_go}: could not hide nested auth whoami")

if "logoutCmd.Hidden = true" not in auth_text:
    auth_text, n_logout = re.subn(
        r"authCmd\.AddCommand\(&cobra\.Command\{\n"
        r"\t\tUse:\s+\"logout\",\n"
        r"\t\tShort: \"Clear all stored authentication credentials\",\n"
        r"\t\tLong: `Clear all stored authentication credentials from both the OS keychain and config file\.\n\n"
        r"This removes all credentials previously set via auth login or configure\.`,\n"
        r"\t\tRunE: runAuthLogoutCmd,\n"
        r"\t\}\)",
        """logoutCmd := &cobra.Command{
\t\tUse:   "logout",
\t\tShort: "Clear all stored authentication credentials",
\t\tLong: `Clear all stored authentication credentials from both the OS keychain and config file.

This removes all credentials previously set via login or configure.`,
\t\tRunE: runAuthLogoutCmd,
\t}
\tlogoutCmd.Hidden = true
\tauthCmd.AddCommand(logoutCmd)""",
        auth_text,
        count=1,
    )
    if n_logout != 1 and "logoutCmd.Hidden = true" not in auth_text:
        # Already patched with updated logout long text.
        auth_text, n_logout = re.subn(
            r"authCmd\.AddCommand\(&cobra\.Command\{\n"
            r"\t\tUse:\s+\"logout\",\n"
            r"\t\tShort: \"Clear all stored authentication credentials\",\n"
            r"\t\tLong: `Clear all stored authentication credentials from both the OS keychain and config file\.\n\n"
            r"This removes all credentials previously set via login or configure\.`,\n"
            r"\t\tRunE: runAuthLogoutCmd,\n"
            r"\t\}\)",
            """logoutCmd := &cobra.Command{
\t\tUse:   "logout",
\t\tShort: "Clear all stored authentication credentials",
\t\tLong: `Clear all stored authentication credentials from both the OS keychain and config file.

This removes all credentials previously set via login or configure.`,
\t\tRunE: runAuthLogoutCmd,
\t}
\tlogoutCmd.Hidden = true
\tauthCmd.AddCommand(logoutCmd)""",
            auth_text,
            count=1,
        )
    if n_logout != 1 and "logoutCmd.Hidden = true" not in auth_text:
        raise SystemExit(f"::warning::{auth_go}: could not hide nested auth logout")

auth_go.write_text(auth_text)

if not login_go.exists():
    login_go.write_text(
        """// Patched by patch-cli-auth-commands.sh — top-level login shortcut.

package cli

import (
\t"github.com/spf13/cobra"
)

func initLoginCmd(parent *cobra.Command) error {
\tcmd := &cobra.Command{
\t\tUse:   "login",
\t\tShort: "Interactively configure authentication credentials",
\t\tLong: `Interactively configure authentication credentials for calibrate.
Secret credentials are stored in the OS keychain when available,
with a config file fallback.

All fields are optional — press Enter to skip any field you don't need.
Use the configure command for both authentication and global parameters.`,
\t\tRunE: runAuthLoginCmd,
\t}
\tparent.AddCommand(cmd)
\treturn nil
}
"""
    )

if not logout_go.exists():
    logout_go.write_text(
        """// Patched by patch-cli-auth-commands.sh — top-level logout shortcut.

package cli

import (
\t"github.com/spf13/cobra"
)

func initLogoutCmd(parent *cobra.Command) error {
\tcmd := &cobra.Command{
\t\tUse:   "logout",
\t\tShort: "Clear all stored authentication credentials",
\t\tLong: `Clear all stored authentication credentials from both the OS keychain and config file.

This removes all credentials previously set via login or configure.`,
\t\tRunE: runAuthLogoutCmd,
\t}
\tparent.AddCommand(cmd)
\treturn nil
}
"""
    )

print(f"Patched CLI auth commands in {root_go.parent.parent.parent}")
PY
