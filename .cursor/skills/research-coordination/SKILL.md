---
name: research-coordination
type: companion
description: >
  Load when loading research-lead or research-artifacts. Meridian work CLI,
  work dir vs KB placement for LLMG research campaigns.
model-invocable: true
---

# Research Coordination

Run `meridian work` from **`~/gitrepos/research/autoresearch-llmg`**.

## Work vs KB

| Layer | Path | Holds |
|---|---|---|
| **Work** | `$(meridian work current)` under `~/.meridian/git/haowjy-research-docs/llmg/work/` | Active campaign: hypothesis, stage plan, experiment log notes, handoffs |
| **KB** | `meridian context kb` → `~/.meridian/git/haowjy-research-docs/llmg/kb/` | Durable: literature map, decisions, consolidated findings, glossary |

Remote: `git@github.com:haowjy/research-docs.git` (`git-autosync` in `meridian.toml`).

Rule: *this campaign* → work dir. *Survives across campaigns* → KB (`/kb-conventions`).

## Commands

```bash
cd ~/gitrepos/research/autoresearch-llmg
meridian work start "stage-0-stacked-lora"    # new campaign
meridian work switch stage-0-stacked-lora
meridian work current
meridian work list
meridian work update <slug> --status running
meridian work done <slug>                     # after campaign closes
```

Pass `$(meridian work current)` into subagent handoffs. Repo branches: `autoresearch/<tag>` per `program.md`.

## Delegation (Cursor)

Match each `.cursor/agents/*.md` **description** (when) and frontmatter **model**.
**Handoff only** — never paste agent file bodies. Prefer Task with **experiment-runner**
for GPU loops; include `program.md` path and branch tag in the prompt.

## Commit discipline

Work-dir markdown is what survives compaction — commit KB updates when stable;
`results.tsv` stays untracked in the repo root (harness convention).
