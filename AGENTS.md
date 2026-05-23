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

## Research log (three layers)

| Layer | Location | Git | Purpose |
|---|---|---|---|
| **Program index** | **`RESEARCH-LOG.md`** (repo root) | **Tracked** in autoresearch-llmg | Headline stats, campaign links, pointers to results |
| **Campaign narrative** | `$(meridian work current)/experiment-log.md` | research-docs | Interpretation per campaign / stage |
| **Machine** | `results.tsv` + `llmg/runs/<ts>_<id>/` | runs untracked | Every `llmg.run`; grep, keep/discard |

Example campaign log path: `llmg/work/llmg-v1-first-experiment/experiment-log.md` (in `haowjy/research-docs`)

### When agents must update

| Event | `RESEARCH-LOG.md` | Campaign `experiment-log.md` |
|---|---|---|
| New **experiment_id** official best | Add/update headline stats row | Dated section (hypothesis → next) |
| Gate / **stage.md** pass or fail | Program snapshot bullets | Status + interpretation |
| New Meridian **work** campaign | New row in campaign table | Create log from template |
| Surprising metric | Headline note or snapshot | Full “why” and follow-up |
| Calibrate-only runs | **Skip** | **Skip** (unless human asks) |
| Campaign close | Mark campaign done; refresh snapshot | Lessons learned |

### Campaign `experiment-log.md` entry shape

1. Hypothesis · 2. Setup (commit, params) · 3. Result + `run_dir` · 4. Interpretation · 5. Surprises · 6. Next experiment IDs  

Template: bottom of the active campaign log.

### Conventions

- **`RESEARCH-LOG.md`** — best official score per `experiment_id`; git paths in tables; no full TSV paste.  
- **`stage.md`** — records per-experiment pass/fail; points to campaign `experiment-log.md` by path.  
- **KB** — promote durable decisions to `llmg/kb/decisions/` in research-docs.

### Link style (research logs)

Use Markdown **reference links** so the body stays readable and URLs sit at the bottom:

```markdown
Dataset: [saxenan3/temporalwiki-drift-cl-easy][tw-easy] …

## References

[tw-easy]: https://huggingface.co/datasets/saxenan3/temporalwiki-drift-cl-easy
[p0-tw-01]: https://github.com/haowjy/autoresearch-llmg/blob/main/llmg/experiments/P0-TW-01
```

- **Narrative:** `[label][ref-id]` inline.  
- **Tables:** backtick git paths only (no links in tables).  
- **Bottom:** `## References` with `[ref-id]: URL` lines (not bullet lists of duplicate links).  
- **Same-repo paths:** `[ref-id]: llmg/experiments/P0-TW-01` (relative) or full GitHub URL for cross-repo.  
- **Never** `file://`, `~/.meridian/...`, or absolute home paths.

Details: `local-skills/research-artifacts/SKILL.md` (sync via `./scripts/sync.sh`).

## Harness (code)

| File | Role |
|---|---|
| `program.md` | Experiment-runner instructions (LLMG rewrite pending) |
| `llmg/run.py` | Dispatch: `uv run python -m llmg.run --experiment <ID>` |
| `llmg/experiment.py` | Calibrate entry for autoresearch branches |
| `llmg/experiments/<ID>/` | `config.yaml` + `runner.py` per experiment |
| `llmg/runs/` | Per-run logs and metrics (gitignored except README) |
| `results.tsv` | Tab-separated run index (untracked) |
| `legacy/karpathy/` | Archived pretrain harness |

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
