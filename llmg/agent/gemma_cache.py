"""Per-episode GPU KV cache for Gemma agent steps (one eval row / trace)."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch
    from transformers import PreTrainedModel


def longest_common_prefix_len(a: list[int], b: list[int]) -> int:
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return n


class EpisodeKVCache:
    """Greedy decode with DynamicCache; token sync matches full chat-template prompts."""

    def __init__(self, model: PreTrainedModel, device: torch.device) -> None:
        self._model = model
        self._device = device
        self._known: list[int] = []
        self._past = None
        self._last_logits = None

    def reset(self) -> None:
        self._known = []
        self._past = None
        self._last_logits = None

    @property
    def known_token_ids(self) -> list[int]:
        return list(self._known)

    def sync_to(self, target_ids: list[int]) -> None:
        """Forward any prompt tokens not yet in the KV cache (one token at a time)."""
        lcp = longest_common_prefix_len(self._known, target_ids)
        if self._past is not None and len(self._known) > lcp:
            self._past.crop(lcp)
            self._known = self._known[:lcp]
        for tid in target_ids[len(self._known) :]:
            token = self._token_tensor(tid)
            with self._no_grad():
                out = self._model(input_ids=token, past_key_values=self._past, use_cache=True)
            self._past = out.past_key_values
            self._known.append(tid)
            self._last_logits = out.logits

    def greedy_generate(self, max_new_tokens: int) -> list[int]:
        if self._last_logits is None:
            raise RuntimeError("sync_to() must run before greedy_generate()")
        generated: list[int] = []
        logits = self._last_logits[:, -1, :]
        for _ in range(max_new_tokens):
            next_tok = logits.argmax(dim=-1, keepdim=True)
            tid = int(next_tok.item())
            generated.append(tid)
            self._known.append(tid)
            with self._no_grad():
                out = self._model(input_ids=next_tok, past_key_values=self._past, use_cache=True)
            self._past = out.past_key_values
            logits = out.logits[:, -1, :]
        return generated

    def _token_tensor(self, token_id: int):
        import torch

        return torch.tensor([[token_id]], device=self._device, dtype=torch.long)

    @staticmethod
    def _no_grad():
        import torch

        return torch.no_grad()
