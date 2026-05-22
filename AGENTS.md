# autoresearch-llmg — Layered Latent Memory Grafts

Research program + Karpathy autoresearch harness. **Separate** from `~/cursor-dev` dev
orchestration — use this repo's `.cursor/` and `/research-lead`.

## Quick start (Cursor)

```bash
cursor ~/gitrepos/research/research.code-workspace
/research-lead
cd ~/gitrepos/research/autoresearch-llmg && meridian work start "my-campaign"
./scripts/sync.sh   # first time: Mars base skills + local overlay
```

## Roles

| Invoke | Profile | Role |
|---|---|---|
| `/research-lead` | `.cursor/agents/research-lead.md` | Charter, hypotheses, work/KB, routing (product-lead analog) |
| *(Task tool)* | `.cursor/agents/experiment-runner.md` | Overnight `program.md` loop on `train.py` |

## Meridian work + KB

Git remote: `git@github.com:haowjy/research-docs.git` (see `meridian.toml`).

| Layer | Checkout | CLI |
|---|---|---|
| Work (campaigns) | `~/.meridian/git/haowjy-research-docs/llmg/work/<slug>/` | `meridian work current` |
| KB (durable) | `~/.meridian/git/haowjy-research-docs/llmg/kb/` | `meridian context kb` |
| Archive | `~/.meridian/git/haowjy-research-docs/archive/llmg/work/` | `meridian context work.archive` |

`git-autosync` hook commits work/kb changes to GitHub. Do not use repo-local `.meridian/` (gitignored).

## Harness (code)

| File | Role |
|---|---|
| `program.md` | Experiment-runner instructions |
| `train.py` | Agent-editable training code |
| `prepare.py` | Fixed data/eval |
| `results.tsv` | Experiment log (untracked) |

Charter: https://haowjy.github.io/blog/layered-latent-memory-grafts/

## Skills layout

| Path | Role |
|---|---|
| **`local-skills/`** | **Edit here** — research-lead, research-coordination, research-artifacts |
| **`.cursor/skills/`** | Cursor loads this — `mars sync` then rsync from `local-skills/` |
| **`.cursor/agents/`** | research-lead, experiment-runner (not from Mars) |

```bash
./scripts/sync.sh   # after changing local-skills/
```

Do **not** edit overlay skills under `.cursor/skills/` for research-* — sync overwrites them.
