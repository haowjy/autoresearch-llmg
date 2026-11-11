# autoresearch-llmg: Layered Latent Memory Grafts

A research program on Layered Latent Memory Grafts, run on a fork of Karpathy's
autoresearch harness. Charter: <https://haowjy.github.io/blog/layered-latent-memory-grafts/>.
Setup, with the reason for each step: [DEVELOPMENT.md](DEVELOPMENT.md).

## Agents

Research runs from Cursor through this repo's `.cursor/`: `/research-lead` (charter,
hypotheses, work and KB, routing) and the `experiment-runner` agent it hands tasks.

Edit the research skills in `local-skills/`, then run `./scripts/sync.sh`: it runs
`meridian mars sync` and copies them into `.cursor/skills/`, overwriting any edit made
there. `.cursor/agents/` is tracked and edited in place.

## Harness

- LLMG experiments live in `llmg/experiments/<ID>/` (`config.yaml` and `runner.py`)
  and run through `uv run python -m llmg.run --experiment <ID>`; `--list` names them.
  Each run writes `llmg/runs/<timestamp>_<ID>/` (gitignored) and a row in the
  untracked `results.tsv`.
- The root `program.md`, `train.py` and `prepare.py` are Karpathy's pretrain
  harness, identical to `legacy/karpathy/`. The `experiment-runner` agent follows
  them; LLMG work doesn't use them.
- Benchmark and corpus links: `llmg/DATASETS.md`.

## Research records

Work (campaigns) and the KB live in `haowjy/research-docs`, not in this repo: find
them with `meridian work current` and `meridian context kb`. The `git-autosync` hook
commits changes there to GitHub. Don't create a repo-local `.meridian/`.

A result is recorded in three places: `RESEARCH-LOG.md` here (headline stats per
`experiment_id`, campaign links), the campaign's `experiment-log.md` (interpretation),
and the run directory. What to update on which event, and the link style (reference
links, no `file://` or home paths), are in
`local-skills/research-artifacts/SKILL.md`. What a campaign established goes, with
its evidence, to its topic page in the KB.
