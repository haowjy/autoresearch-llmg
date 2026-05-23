#!/usr/bin/env bash
# Symlink meridian-llmg/ → Meridian checkout llmg/ (from meridian.toml remote).
# .code-workspace folder paths do not expand ${userHome}; use this link instead.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

if ! command -v meridian >/dev/null 2>&1; then
  echo "meridian CLI not found; install Meridian first." >&2
  exit 1
fi

KB="$(meridian context kb)"
LLMG="$(dirname "$KB")"
LINK="$ROOT/meridian-llmg"

ln -sfn "$LLMG" "$LINK"
echo "meridian-llmg -> $LLMG"
