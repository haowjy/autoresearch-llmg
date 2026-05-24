# Development setup — LLMG

Everything you need to work on [Layered Latent Memory Grafts][charter-blog] in this repo: **why** each piece exists and **how** to install it.

**Quick checklist** (details below):

```bash
cd ~/gitrepos/research/autoresearch-llmg
uv sync
uv tool install hf && hf auth login
hf download saxenan3/temporalwiki-drift-cl-easy --repo-type dataset
./scripts/sync.sh
./scripts/link-meridian-llmg.sh
cursor autoresearch-llmg.local.code-workspace
uv run python -m llmg.run --experiment P0-TW-01
```

---

## What you are setting up

| Piece | Purpose |
|-------|---------|
| **Harness repo** (`autoresearch-llmg`) | Code: `llmg.run`, experiments, eval — git-tracked |
| **Meridian checkout** | Campaign work + KB in `haowjy/research-docs` — separate git repo |
| **Cursor workspace** | Two roots: harness + `llmg/` (work, kb) side by side |
| **HF cache** | [TemporalWiki drift (easy)][tw-easy] + (later) [Gemma 4 E4B-it][gemma-e4b] |
| **Local run artifacts** | `llmg/runs/`, `results.tsv` — not committed |

Program status: [RESEARCH-LOG.md][research-log]. Agent rules: [AGENTS.md][agents].

---

## 1. Clone and Python (`uv`)

**Why:** The harness is a Python package (`llmg/`) with pinned deps (PyTorch, `datasets`, `rank-bm25`, …) managed by [uv][uv].

```bash
git clone git@github.com:haowjy/autoresearch-llmg.git
cd autoresearch-llmg
uv sync
```

Creates `.venv/` at repo root. Run experiments with `uv run python -m llmg.run`, not system Python.

**GPU:** Phase 0 (P0-TW-01) is CPU-only (BM25). LoRA phases need an NVIDIA GPU (target: RTX 3090 24GB, `google/gemma-4-E4B-it` QLoRA).

**Optional — ripgrep** (P0-TW-03 `harness_rg` baseline):

```bash
sudo apt install ripgrep   # or: cargo install ripgrep
```

Without `rg` on `PATH`, the harness falls back to `grep` (agent sandbox) or Python scan; recall stays ~2% on `test`.

---

## 2. Meridian (work + KB)

**Why:** Research narrative, decisions, and stage gates live in [research-docs][research-docs-repo], not in the harness repo. [meridian.toml][meridian-toml] pins the remote and paths so everyone uses the same checkout layout.

| What | Git path in research-docs | CLI |
|------|---------------------------|-----|
| Campaign (active) | `llmg/work/<slug>/` | `meridian work current` |
| Knowledge base | `llmg/kb/` | `meridian context kb` |
| Archive | `archive/llmg/work/` | `meridian context work.archive` |

Default checkout on disk:

```text
~/.meridian/git/haowjy-research-docs/
  llmg/work/...
  llmg/kb/...
```

**Setup:**

1. Install [Meridian][meridian] CLI.
2. From this repo (with `meridian.toml` present):

   ```bash
   meridian work start llmg-v1-first-experiment   # or your campaign slug
   meridian work current
   ```

3. Meridian clones/syncs `git@github.com:haowjy/research-docs.git` under `~/.meridian/git/` (autosync hook commits work/kb edits).

**Do not** use repo-local `.meridian/` (gitignored legacy path).

---

## 3. Cursor / VS Code workspace

**Why:** You edit harness code and Meridian markdown in one window.

### Paths that do *not* work in `folders[].path`

| Path | What Cursor does |
|------|------------------|
| `~/.meridian/git/.../llmg` | **Not** `$HOME`. Resolves to `autoresearch-llmg/~/.meridian/...` (missing). |
| `${userHome}/.meridian/...` | Variables are not expanded in workspace folder paths. |

### Two-root setup (use this)

```bash
./scripts/link-meridian-llmg.sh
cursor autoresearch-llmg.local.code-workspace
```

Writes [autoresearch-llmg.local.code-workspace][workspace-local] (gitignored): harness always `"."`, Meridian path from [meridian.toml][meridian-toml] `[workspace.cursor] meridian_llmg` (default `../../../.meridian/git/haowjy-research-docs/llmg`). Edit that value if your checkout layout differs, then re-run the script. [scripts/sync.sh][sync-sh] runs the generator after skill sync.

**Git-tracked fallback:** [autoresearch-llmg.code-workspace][workspace-file] — harness only (one root; always opens).

**Alternative:** parent `research.code-workspace` opens harness + all of `~/.meridian/git`.

---

## 4. Agent skills (`./scripts/sync.sh`)

**Why:** Cursor loads skills from `.cursor/skills/`. Source of truth is [local-skills/][local-skills]; Mars base skills sync via Meridian, then rsync overlay.

```bash
./scripts/sync.sh
```

Does: `meridian mars sync` → rsync `local-skills/*` → generate local workspace.

Edit research skills in `local-skills/`, not directly under `.cursor/skills/` (sync overwrites).

---

## 5. Hugging Face (datasets + models)

**Why:** TemporalWiki and Gemma weights come from the Hub. The harness uses the `datasets` library; downloads also work via the `hf` CLI.

### CLI (`uv tool install hf`)

```bash
uv tool install hf
hf auth login
```

Use **`--repo-type dataset`** for datasets (default `hf download` assumes a model):

```bash
hf download saxenan3/temporalwiki-drift-cl-easy --repo-type dataset
```

Cache: `~/.cache/huggingface/hub/` (and `datasets` may use `~/.cache/huggingface/datasets/`).

### Gemma (later — LoRA experiments)

1. Accept license on the model page for [google/gemma-4-E4B-it][gemma-e4b].
2. `hf download google/gemma-4-E4B-it` (large; 4-bit training fits 24GB VRAM).
3. Add training deps when implemented: `transformers`, `peft`, `accelerate`, `bitsandbytes`.

---

## 6. Run an experiment

**Why:** Validates harness, data, and observability before LoRA.

```bash
uv run python -m llmg.run --experiment P0-TW-01
```

| Output | Location |
|--------|----------|
| Per-run bundle | `llmg/runs/<timestamp>_P0-TW-01/` (`run.log`, `metrics.json`, `summary.md`, …) |
| Latest symlink | `llmg/runs/latest/` |
| Index row | `results.tsv` (untracked) |

List experiments: `uv run python -m llmg.run --list`.

**P0-TW-01:** BM25 on [tw-easy][tw-easy], train index, eval `test` — acquisition floor (~93% recall@5).

**P0-TW-01b:** Same dataset, **train+stable** index, eval `stable` — retention (~76% recall@5).

**All dataset links:** [llmg/DATASETS.md][datasets] (StreamingQA, PAT-Questions, books, etc.).

---

## 7. Research logs (three layers)

**Why:** Machine logs are not enough; campaigns need narrative; the program needs a single index.

| Layer | File | Git |
|-------|------|-----|
| Program index | [RESEARCH-LOG.md][research-log] | autoresearch-llmg |
| Campaign log | `llmg/work/<slug>/experiment-log.md` | research-docs |
| Machine | `results.tsv`, `llmg/runs/` | local only |

Link style: `[text][ref-id]` in prose; `[ref-id]: URL` under `## References` at file bottom. See [AGENTS.md][agents].

---

## 8. Repo layout (LLMG vs legacy)

```text
autoresearch-llmg/
  DEVELOPMENT.md          ← this file
  RESEARCH-LOG.md         ← program experiment index
  AGENTS.md               ← Cursor agent instructions
  meridian.toml           ← research-docs remote
  autoresearch-llmg.code-workspace       ← harness-only (git)
  autoresearch-llmg.local.code-workspace ← two roots (gitignored; run link script)
  llmg/                   ← active harness
    DATASETS.md           ← benchmark & corpus links
    experiments/P0-TW-01/
    experiments/P0-TW-01b/
    runs/                 ← per-run artifacts (gitignored)
  legacy/karpathy/        ← archived pretrain harness
  program.md              ← still Karpathy-oriented (rewrite pending)
```

Karpathy root `train.py` / `prepare.py` remain for upstream comparison; LLMG work uses `llmg/` only.

---

## 9. Troubleshooting

| Problem | Fix |
|---------|-----|
| Empty **meridian llmg** root in Cursor | Run `./scripts/link-meridian-llmg.sh`; open `autoresearch-llmg.local.code-workspace` (not `~` in the git-tracked workspace) |
| `hf download` “model not found” for TemporalWiki | Add `--repo-type dataset` |
| `stable` split 0% recall on P0-TW-01 | Expected with train-only index; use **P0-TW-01b** (train+stable index) for retention metrics |
| Meridian path wrong | `meridian context kb` — should be under `~/.meridian/git/haowjy-research-docs/llmg/kb` |
| Skills stale | `./scripts/sync.sh` |

---

## References

[charter-blog]: https://haowjy.github.io/blog/layered-latent-memory-grafts/
[research-log]: RESEARCH-LOG.md
[agents]: AGENTS.md
[uv]: https://docs.astral.sh/uv/
[research-docs-repo]: https://github.com/haowjy/research-docs
[meridian-toml]: meridian.toml
[meridian]: https://github.com/haowjy/meridian
[workspace-file]: autoresearch-llmg.code-workspace
[workspace-local]: autoresearch-llmg.local.code-workspace
[sync-sh]: scripts/sync.sh
[local-skills]: local-skills/
[gemma-e4b]: https://huggingface.co/google/gemma-4-E4B-it
[tw-easy]: https://huggingface.co/datasets/saxenan3/temporalwiki-drift-cl-easy
[datasets]: llmg/DATASETS.md
