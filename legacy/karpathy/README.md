# Karpathy autoresearch (archived)

Upstream-style **pretrain hacking** snapshot: GPT on climbmix, metric `val_bpb`, 5-minute wall clock.

| File | Role |
|------|------|
| `train.py` | Agent-editable pretrain loop |
| `prepare.py` | climbmix data + `evaluate_bpb` |
| `program.md` | Original autonomous loop instructions |

**Not used by LLMG.** Active harness: [program.md][program-md], [llmg/experiment.py][llmg-experiment].

See [0003-replace-karpathy-harness.md][dec-0003] in research-docs KB.

Reproduce (optional):

```bash
uv run prepare.py   # ~/.cache/autoresearch/
uv run legacy/karpathy/train.py   # if wired; or copy train.py to root temporarily
```

---

## References

[program-md]: ../../program.md
[llmg-experiment]: ../../llmg/experiment.py
[dec-0003]: https://github.com/haowjy/research-docs/blob/main/llmg/kb/decisions/0003-replace-karpathy-harness.md
