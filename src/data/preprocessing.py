"""Normalize source records and serialize prompts consistently."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any


def clean_records(rows: Iterable[Mapping[str, Any]]) -> list[dict[str, str]]:
    """Keep valid instruction/response pairs and normalize optional context."""
    cleaned: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for row in rows:
        instruction = row.get("instruction")
        response = row.get("response")
        context = row.get("context", "")
        if not isinstance(instruction, str) or not isinstance(response, str):
            continue
        instruction, response = instruction.strip(), response.strip()
        context = context.strip() if isinstance(context, str) else ""
        if not instruction or not response:
            continue
        prompt = f"{instruction}\n\nContext:\n{context}" if context else instruction
        key = (prompt, response)
        if key in seen:
            continue
        seen.add(key)
        cleaned.append({"instruction": prompt, "response": response})
    return cleaned


def format_chat_prompt(tokenizer: Any, instruction: str) -> str:
    """Format a user prompt through the model's chat template when available."""
    messages = [{"role": "user", "content": instruction}]
    if getattr(tokenizer, "chat_template", None):
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return f"User: {instruction}\nAssistant:"


def tokenize_example(example: Mapping[str, str], tokenizer: Any, max_length: int) -> dict[str, list[int]]:
    """Tokenize one example and mask prompt tokens from the LM loss."""
    prompt = format_chat_prompt(tokenizer, example["instruction"])
    if getattr(tokenizer, "chat_template", None):
        full_text = tokenizer.apply_chat_template(
            [
                {"role": "user", "content": example["instruction"]},
                {"role": "assistant", "content": example["response"]},
            ],
            tokenize=False,
            add_generation_prompt=False,
        )
    else:
        full_text = f"{prompt} {example['response']}"
    prompt_ids = tokenizer(prompt, add_special_tokens=False, truncation=True, max_length=max_length)["input_ids"]
    encoded = tokenizer(full_text, add_special_tokens=False, truncation=True, max_length=max_length)
    input_ids = encoded["input_ids"]
    labels = list(input_ids)
    # Train on the answer while keeping the prompt as context only.
    labels[: min(len(prompt_ids), len(labels))] = [-100] * min(len(prompt_ids), len(labels))
    if not any(label != -100 for label in labels):
        labels[-1:] = input_ids[-1:]
    return {"input_ids": input_ids, "attention_mask": encoded["attention_mask"], "labels": labels}
