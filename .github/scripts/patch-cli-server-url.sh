#!/usr/bin/env bash
# Post-process Speakeasy CLI output: make --server-url a persistable global
# parameter so self-hosted users set their backend URL once instead of passing
# the flag on every call. Speakeasy generates server-url as flag-only; this
# patch teaches it to resolve flag > env (CALIBRATE_SERVER_URL) > config file,
# stores it via `calibrate configure`, and surfaces it in `calibrate whoami`.
set -euo pipefail

DIR="${1:-.speakeasy/out/cli}"
CLIENT_GO="$DIR/internal/client/client.go"
CONFIG_GO="$DIR/internal/config/config.go"
CONFIGURE_GO="$DIR/internal/cli/configure.go"
WHOAMI_GO="$DIR/internal/cli/whoami.go"

if [[ ! -f "$CLIENT_GO" || ! -f "$CONFIG_GO" ]]; then
  echo "No generated CLI at $DIR — skipping server-url patch"
  exit 0
fi

python3 - "$CLIENT_GO" "$CONFIG_GO" "$CONFIGURE_GO" "$WHOAMI_GO" <<'PY'
import re
import sys
from pathlib import Path

client_go, config_go, configure_go, whoami_go = map(Path, sys.argv[1:5])


def replace_once(text, old, new, where):
    n = text.count(old)
    if n != 1:
        raise SystemExit(f"::warning::{where}: expected exactly one anchor, found {n}")
    return text.replace(old, new)


def sub_once(text, pattern, repl, where):
    text, n = re.subn(pattern, repl, text, count=1)
    if n != 1:
        raise SystemExit(f"::warning::{where}: could not apply patch (anchor not found)")
    return text


# --- client.go: resolve server-url flag > env > config (was flag-only) --------
client_text = client_go.read_text()
if 'resolveStringFlag(cmd, "server-url")' not in client_text:
    client_text = replace_once(
        client_text,
        '\tif serverURL, _ := flagutil.GetStringFlag(cmd, "server-url"); serverURL != "" {',
        '\tif serverURL := resolveStringFlag(cmd, "server-url"); serverURL != "" {',
        client_go,
    )
    client_go.write_text(client_text)

# --- config.go: add ServerURL field + GetConfigValue case ---------------------
config_text = config_go.read_text()
if "ServerURL" not in config_text:
    config_text = sub_once(
        config_text,
        r'(\tTimeout\s+string\s+`yaml:"timeout,omitempty"`\n)\}',
        lambda m: m.group(1) + '\tServerURL string `yaml:"server_url,omitempty"`\n}',
        config_go,
    )
if 'case "server-url":' not in config_text:
    config_text = sub_once(
        config_text,
        r'(\tcase "timeout":\n\t\treturn cfg\.Timeout\n)\t\}',
        lambda m: m.group(1) + '\tcase "server-url":\n\t\treturn cfg.ServerURL\n\t}',
        config_go,
    )
config_go.write_text(config_text)

# --- configure.go: prompt for + persist server-url ----------------------------
configure_text = configure_go.read_text()

if "cfg.ServerURL = v" not in configure_text:
    block = (
        '\t\tif f := cmd.Flags().Lookup("server-url"); f != nil && f.Changed {\n'
        '\t\t\tv, _ := cmd.Flags().GetString("server-url")\n'
        '\t\t\tcfg.ServerURL = v\n'
        '\t\t\tchanged = true\n'
        '\t\t}\n'
    )
    configure_text = sub_once(
        configure_text,
        r'(\t\t\tchanged = true\n\t\t\}\n)\n(\t\tif !changed \{)',
        lambda m, b=block: m.group(1) + '\n' + b + '\n' + m.group(2),
        configure_go,
    )

if "serverFields" not in configure_text:
    group = (
        '\t\tvar cfgServerURL string\n'
        '\t\tserverFields := []huh.Field{\n'
        '\t\t\thuh.NewInput().\n'
        '\t\t\t\tTitle("Server URL for self-hosted deployments. Leave blank to keep the current value.").\n'
        '\t\t\t\tDescription("--server-url").\n'
        '\t\t\t\tPlaceholder(cfg.ServerURL).\n'
        '\t\t\t\tValue(&cfgServerURL),\n'
        '\t\t}\n'
        '\t\tgroups = append(groups, huh.NewGroup(serverFields...).Title("Server"))\n'
    )
    configure_text = sub_once(
        configure_text,
        r'(\t\tgroups = append\(groups, huh\.NewGroup\(securityFields\.\.\.\)\.Title\("Authentication"\)\)\n)',
        lambda m, g=group: m.group(1) + g,
        configure_go,
    )

if "cfg.ServerURL = cfgServerURL" not in configure_text:
    configure_text = sub_once(
        configure_text,
        r'(\t\t\t\tcfg\.Security\.ApiKeyAuth = authApiKeyAuth // no keyring, store in config\n\t\t\t\}\n\t\t\}\n)',
        lambda m: m.group(1) + '\t\tif cfgServerURL != "" {\n\t\t\tcfg.ServerURL = cfgServerURL\n\t\t}\n',
        configure_go,
    )

configure_go.write_text(configure_text)

# --- whoami.go: report the resolved server-url and its source -----------------
whoami_text = whoami_go.read_text()
if 'ResolveCredential(cmd, "server-url")' not in whoami_text:
    server_block = (
        '\tfmt.Fprintln(out)\n'
        '\tfmt.Fprintln(out, "Server:")\n'
        '\t{\n'
        '\t\tvalue, source := config.ResolveCredential(cmd, "server-url")\n'
        '\t\tif value == "" {\n'
        '\t\t\tvalue = "(default)"\n'
        '\t\t}\n'
        '\t\tfmt.Fprintf(out, "  --%-25s [%-7s] %s\\n", "server-url", source, value)\n'
        '\t}\n'
    )
    whoami_text = sub_once(
        whoami_text,
        r'(\n)(\treturn nil\n\}\n?)\Z',
        lambda m, b=server_block: m.group(1) + b + m.group(2),
        whoami_go,
    )
    whoami_go.write_text(whoami_text)

# Best-effort gofmt so inserted lines match the surrounding alignment.
import shutil
import subprocess

gofmt = shutil.which("gofmt")
if gofmt:
    subprocess.run(
        [gofmt, "-w", str(client_go), str(config_go), str(configure_go), str(whoami_go)],
        check=False,
    )

print(f"Patched CLI server-url persistence in {client_go.parent.parent.parent}")
PY
