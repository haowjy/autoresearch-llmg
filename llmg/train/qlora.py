"""Naive QLoRA SFT on TemporalWiki train rows (question + article → object)."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import torch
from datasets import Dataset
from transformers import Trainer, TrainingArguments

from llmg.train.sft_data import build_sft_dataset

log = logging.getLogger(__name__)

# Gemma 4 multimodal checkpoints wrap some layers in Gemma4ClippableLinear;
# scope LoRA to the text tower only (see huggingface/peft#3129).
DEFAULT_LORA_TARGETS = (
    r".*\.language_model\..*\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)"
)


def _as_id_list(tokenizer, out: Any) -> list[int]:
    """Normalize apply_chat_template output to a flat input_ids list."""
    if isinstance(out, dict):
        ids = out["input_ids"]
    elif hasattr(out, "get") and out.get("input_ids") is not None:
        ids = out["input_ids"]
    else:
        ids = out
    if isinstance(ids, torch.Tensor):
        ids = ids.squeeze().tolist()
    if ids and isinstance(ids[0], list):
        ids = ids[0]
    return list(ids)


def _tokenize_messages(
    tokenizer,
    messages: list[dict[str, str]],
) -> dict[str, Any]:
    """Tokenize chat; mask non-assistant tokens for causal LM loss."""
    if hasattr(tokenizer, "apply_chat_template"):
        try:
            out = tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                return_dict=True,
                return_assistant_tokens_mask=True,
            )
            input_ids = _as_id_list(tokenizer, out)
            mask = out.get("assistant_masks") or out.get("assistant_tokens_mask")
            if mask is not None:
                if isinstance(mask, torch.Tensor):
                    mask = mask.squeeze().tolist()
                if mask and isinstance(mask[0], list):
                    mask = mask[0]
                if len(mask) == len(input_ids) and any(mask):
                    labels = [
                        tid if m else -100 for tid, m in zip(input_ids, mask, strict=True)
                    ]
                    return {"input_ids": input_ids, "labels": labels}
        except TypeError:
            pass

    # Fallback: supervise tokens after the user prompt + generation header.
    prompt_out = tokenizer.apply_chat_template(
        messages[:-1],
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
    )
    full_out = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=False,
        return_dict=True,
    )
    prompt_ids = _as_id_list(tokenizer, prompt_out)
    full_ids = _as_id_list(tokenizer, full_out)
    prompt_len = min(len(prompt_ids), len(full_ids))
    labels = [-100] * prompt_len + full_ids[prompt_len:]
    if len(labels) < len(full_ids):
        labels.extend([-100] * (len(full_ids) - len(labels)))
    return {"input_ids": full_ids, "labels": labels}


def train_naive_qlora(
    *,
    train_split: Dataset,
    output_dir: Path,
    base_model: str = "google/gemma-4-E4B-it",
    lora_rank: int = 16,
    lora_alpha: int = 32,
    learning_rate: float = 2e-4,
    train_batch_size: int = 1,
    gradient_accumulation_steps: int = 8,
    max_seq_len: int = 4096,
    train_epochs: float = 1.0,
    max_train_steps: int | None = None,
    max_train_rows: int | None = None,
    max_article_chars: int = 3000,
    sft_format: str = "plain",
    lora_targets: str | list[str] = DEFAULT_LORA_TARGETS,
) -> Path:
    """Train 4-bit QLoRA adapter; returns adapter directory."""
    from peft import LoraConfig, get_peft_model
    from transformers import AutoTokenizer, BitsAndBytesConfig, Gemma4ForConditionalGeneration

    output_dir = Path(output_dir)
    adapter_dir = output_dir / "lora_adapter"
    adapter_dir.mkdir(parents=True, exist_ok=True)

    raw = build_sft_dataset(
        train_split,
        max_rows=max_train_rows,
        max_article_chars=max_article_chars,
        sft_format=sft_format,
    )
    log.info("SFT rows=%d base_model=%s", len(raw), base_model)

    from llmg.util.hf_local import hf_local_files_only

    local = hf_local_files_only()
    tokenizer = AutoTokenizer.from_pretrained(base_model, local_files_only=local)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    def _map_batch(examples: dict[str, list]) -> dict[str, list]:
        input_ids_list: list[list[int]] = []
        labels_list: list[list[int]] = []
        for msgs in examples["messages"]:
            tok = _tokenize_messages(tokenizer, msgs)
            ids = tok["input_ids"][:max_seq_len]
            labs = tok["labels"][:max_seq_len]
            input_ids_list.append(ids)
            labels_list.append(labs)
        return {"input_ids": input_ids_list, "labels": labels_list}

    tokenized = raw.map(
        _map_batch,
        batched=True,
        remove_columns=raw.column_names,
        desc="tokenize",
    )

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    log.info("loading 4-bit Gemma4 %s (LoRA scoped to language_model)", base_model)
    model = Gemma4ForConditionalGeneration.from_pretrained(
        base_model,
        quantization_config=bnb_config,
        device_map="auto" if torch.cuda.is_available() else None,
        dtype=torch.bfloat16,
        local_files_only=local,
    )
    for name in ("vision_tower", "audio_tower"):
        tower = getattr(model.model, name, None)
        if tower is not None:
            tower.eval()
            for p in tower.parameters():
                p.requires_grad = False
    # Do not call prepare_model_for_kbit_training on the full multimodal model:
    # it upcasts all bf16 towers to fp32 and OOMs a 24GB GPU. LoRA-only prep instead.
    for param in model.parameters():
        param.requires_grad = False
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    lora_config = LoraConfig(
        r=lora_rank,
        lora_alpha=lora_alpha,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=lora_targets,
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    class _Collator:
        def __call__(self, features: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
            max_len = max(len(f["input_ids"]) for f in features)
            pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id
            input_ids, labels, attention_mask = [], [], []
            for f in features:
                ids = list(f["input_ids"])
                labs = list(f["labels"])
                pad_len = max_len - len(ids)
                input_ids.append(ids + [pad_id] * pad_len)
                labels.append(labs + [-100] * pad_len)
                attention_mask.append([1] * len(ids) + [0] * pad_len)
            return {
                "input_ids": torch.tensor(input_ids, dtype=torch.long),
                "labels": torch.tensor(labels, dtype=torch.long),
                "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            }

    training_args = TrainingArguments(
        output_dir=str(output_dir / "trainer_checkpoints"),
        per_device_train_batch_size=train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        num_train_epochs=train_epochs,
        max_steps=max_train_steps if max_train_steps is not None else -1,
        logging_steps=10,
        save_strategy="no",
        report_to=[],
        bf16=torch.cuda.is_available(),
        optim="paged_adamw_8bit",
        warmup_ratio=0.03,
        lr_scheduler_type="cosine",
        max_grad_norm=0.3,
        dataloader_pin_memory=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized,
        data_collator=_Collator(),
    )
    trainer.train()

    del trainer
    import gc

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    log.info("saving adapter to %s", adapter_dir)
    model.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)
    (adapter_dir / "qlora_train_config.json").write_text(
        json.dumps(
            {
                "base_model": base_model,
                "lora_rank": lora_rank,
                "lora_alpha": lora_alpha,
                "max_seq_len": max_seq_len,
                "max_article_chars": max_article_chars,
                "sft_format": sft_format,
                "train_rows": len(raw),
                "max_train_steps": max_train_steps,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return adapter_dir
