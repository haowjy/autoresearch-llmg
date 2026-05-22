#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

LOCAL_SKILLS=(
  research-lead
  research-coordination
  research-artifacts
)

# Source of truth: local-skills/ — rsync after mars sync
echo "Local skills: ${LOCAL_SKILLS[*]}"

if [[ ! -d .mars ]]; then
  meridian mars init
fi

meridian mars sync

for skill in "${LOCAL_SKILLS[@]}"; do
  rsync -a "local-skills/$skill/" ".cursor/skills/$skill/"
done

echo "Done. Agents: .cursor/agents/"
