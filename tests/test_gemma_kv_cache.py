"""KV cache helpers and optional GPU parity (Gemma 4)."""

from __future__ import annotations

import os
import unittest

from llmg.agent.gemma_cache import EpisodeKVCache, longest_common_prefix_len


class TestLongestCommonPrefix(unittest.TestCase):
    def test_empty(self) -> None:
        self.assertEqual(longest_common_prefix_len([], [1, 2]), 0)
        self.assertEqual(longest_common_prefix_len([1], []), 0)

    def test_full_and_partial(self) -> None:
        self.assertEqual(longest_common_prefix_len([1, 2, 3], [1, 2, 9]), 2)
        self.assertEqual(longest_common_prefix_len([1, 2], [1, 2]), 2)


class TestPromptTokenIds(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        if os.environ.get("LLMG_SKIP_TOKENIZER_TESTS") == "1":
            cls.tokenizer = None
            return
        from transformers import AutoTokenizer

        cls.tokenizer = AutoTokenizer.from_pretrained("google/gemma-4-E4B-it")

    def test_tool_turn_extends_prefix(self) -> None:
        if self.tokenizer is None:
            self.skipTest("tokenizer tests disabled")
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "run_shell",
                    "description": "Run shell",
                    "parameters": {
                        "type": "object",
                        "properties": {"command": {"type": "string"}},
                        "required": ["command"],
                    },
                },
            }
        ]
        messages = [
            {"role": "system", "content": "You are a research assistant with tools."},
            {"role": "user", "content": "Who is X?"},
        ]
        ids0 = self.tokenizer.apply_chat_template(
            messages,
            tools=tools,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
            enable_thinking=False,
        )["input_ids"][0].tolist()
        messages2 = messages + [
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "type": "function",
                        "function": {"name": "run_shell", "arguments": '{"command": "ls"}'},
                    }
                ],
            },
            {"role": "tool", "name": "run_shell", "content": "exit_code=0"},
        ]
        ids1 = self.tokenizer.apply_chat_template(
            messages2,
            tools=tools,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
            enable_thinking=False,
        )["input_ids"][0].tolist()
        self.assertGreater(len(ids1), len(ids0))
        self.assertEqual(ids0, ids1[: len(ids0)])


@unittest.skipUnless(
    os.environ.get("LLMG_GPU_KV_TEST") == "1",
    "set LLMG_GPU_KV_TEST=1 to run Gemma KV parity on GPU",
)
class TestEpisodeKVParityGPU(unittest.TestCase):
    def test_incremental_matches_generate(self) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        if not torch.cuda.is_available():
            self.skipTest("CUDA required")

        model_name = "google/gemma-4-E4B-it"
        tok = AutoTokenizer.from_pretrained(model_name)
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "run_shell",
                    "description": "Run shell",
                    "parameters": {
                        "type": "object",
                        "properties": {"command": {"type": "string"}},
                        "required": ["command"],
                    },
                },
            }
        ]
        messages = [
            {"role": "system", "content": "You are a research assistant with tools."},
            {"role": "user", "content": "Who is X?"},
        ]

        def prompt_ids(msgs: list) -> list[int]:
            return tok.apply_chat_template(
                msgs,
                tools=tools,
                add_generation_prompt=True,
                return_tensors="pt",
                return_dict=True,
                enable_thinking=False,
            )["input_ids"][0].tolist()

        model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=torch.bfloat16, device_map="auto"
        )
        kv = EpisodeKVCache(model, model.device)
        t0 = prompt_ids(messages)
        kv.sync_to(t0)
        g0 = kv.greedy_generate(16)
        with torch.no_grad():
            hf0 = model.generate(
                torch.tensor([t0], device=model.device),
                max_new_tokens=16,
                do_sample=False,
                pad_token_id=tok.eos_token_id,
            )
        self.assertEqual(g0, hf0[0, len(t0) :].tolist())

        gen_text = tok.decode(g0, skip_special_tokens=False)
        messages2 = messages + [{"role": "assistant", "content": gen_text}]
        t1 = prompt_ids(messages2)
        kv.sync_to(t1)
        g1 = kv.greedy_generate(8)
        with torch.no_grad():
            hf1 = model.generate(
                torch.tensor([t1], device=model.device),
                max_new_tokens=8,
                do_sample=False,
                pad_token_id=tok.eos_token_id,
            )
        self.assertEqual(g1, hf1[0, len(t1) :].tolist())


if __name__ == "__main__":
    unittest.main()
