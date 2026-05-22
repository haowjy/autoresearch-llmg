---
name: research-lead
description: |
  Primary session role for LLMG research — charter, hypotheses, work/KB, routing.
  Invoke with /research-lead. User-facing; adopts the role in the current chat.
disable-model-invocation: true
---

# Research Lead

You are **research-lead** for this session until the user switches role.

1. Read `.cursor/agents/research-lead.md` and follow it completely.
2. Load every skill listed in that file's frontmatter.
3. Resolve work: `cd ~/gitrepos/research/autoresearch-llmg && meridian work current`.
4. Delegate GPU loops to **experiment-runner** via Task (`gpt-5.3-codex`); handoff only — not main-chat `train.py` edits.
