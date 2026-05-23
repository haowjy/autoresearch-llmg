#!/usr/bin/env bash
# Generate autoresearch-llmg.local.code-workspace (paths relative to this repo root).
#
# Harness root is always ".". Meridian path from meridian.toml [workspace.cursor],
# validated against meridian context kb. No symlinks.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

if ! command -v meridian >/dev/null 2>&1; then
  echo "meridian CLI not found; install Meridian first." >&2
  exit 1
fi

MERIDIAN_REL="$(
  python3 - "$ROOT" <<'PY'
import sys
import tomllib
from pathlib import Path

root = Path(sys.argv[1])
with (root / "meridian.toml").open("rb") as f:
    cfg = tomllib.load(f)
rel = cfg.get("workspace", {}).get("cursor", {}).get("meridian_llmg", "").strip()
if not rel:
    raise SystemExit("meridian.toml [workspace.cursor] meridian_llmg is missing")
print(rel)
PY
)"

# Must be relative (workspace paths are relative to the .code-workspace file dir).
if [[ "$MERIDIAN_REL" = /* ]]; then
  echo "meridian_llmg must be a relative path, not absolute: $MERIDIAN_REL" >&2
  exit 1
fi

if [[ ! -d "$ROOT/$MERIDIAN_REL" ]]; then
  echo "Meridian path not found from repo root: $MERIDIAN_REL" >&2
  echo "  expected: $ROOT/$MERIDIAN_REL" >&2
  exit 1
fi

KB="$(meridian context kb)"
LLMG="$(dirname "$KB")"
if [[ "$(cd "$ROOT/$MERIDIAN_REL" && pwd -P)" != "$(cd "$LLMG" && pwd -P)" ]]; then
  echo "warning: meridian.toml meridian_llmg != meridian context kb parent" >&2
  echo "  toml: $ROOT/$MERIDIAN_REL" >&2
  echo "  kb:   $LLMG" >&2
fi

LOCAL_WS="$ROOT/autoresearch-llmg.local.code-workspace"

cat >"$LOCAL_WS" <<EOF
{
	"folders": [
		{
			"name": "harness (autoresearch-llmg)",
			"path": "."
		},
		{
			"name": "meridian llmg (work + kb)",
			"path": "$MERIDIAN_REL"
		}
	],
	"settings": {}
}
EOF

echo "Wrote $LOCAL_WS"
echo "  harness:  ."
echo "  meridian: $MERIDIAN_REL"
echo "Open: cursor autoresearch-llmg.local.code-workspace"
