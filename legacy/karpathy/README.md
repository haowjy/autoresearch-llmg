# Karpathy autoresearch (archived)

Upstream-style **pretrain hacking** snapshot: GPT on climbmix, metric `val_bpb`, 5-minute wall clock.

| File | Role |
|------|------|
| `train.py` | Agent-editable pretrain loop |
| `prepare.py` | climbmix data + `evaluate_bpb` |
| `program.md` | Original autonomous loop instructions |

**Not used by LLMG.** Active harness: repo root `program.md`, `llmg/experiment.py`.

Reproduce (optional):

```bash
uv run prepare.py   # ~/.cache/autoresearch/
uv run legacy/karpathy/train.py   # if wired; or copy train.py to root temporarily
```
