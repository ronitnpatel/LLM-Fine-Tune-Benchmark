"""Common generation evaluation and qualitative sample capture."""

from __future__ import annotations

import time
from typing import Any

import torch
from torch.utils.data import DataLoader
from transformers import DataCollatorForSeq2Seq

from src.data.preprocessing import format_chat_prompt
from .metrics import generation_metrics

GENERALIZATION_PROMPTS = [
    "Explain compound interest in two short sentences for a high school student.",
    "Summarize the following idea in exactly three bullet points: A team should define its goal, divide the work into clear tasks, check progress regularly, and revise the plan when new information appears.",
    "Write a polite one-sentence reply declining an invitation because of a schedule conflict.",
    "What is the difference between a list and a dictionary in Python? Answer briefly.",
]


def generate_generalization_samples(model: Any, tokenizer: Any, max_new_tokens: int = 96,
                                    max_input_length: int = 512) -> list[dict[str, str]]:
    """Generate a fixed prompt-only qualitative set without claiming scores."""
    device = next(model.parameters()).device
    was_training = model.training
    model.eval()
    outputs: list[dict[str, str]] = []
    for prompt in GENERALIZATION_PROMPTS:
        formatted = format_chat_prompt(tokenizer, prompt)
        encoded = tokenizer(formatted, return_tensors="pt", truncation=True, max_length=max_input_length)
        encoded = {key: value.to(device) for key, value in encoded.items()}
        with torch.inference_mode():
            generated = model.generate(
                **encoded, max_new_tokens=max_new_tokens, do_sample=False,
                pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id,
            )
        response = tokenizer.decode(generated[0, encoded["input_ids"].shape[1]:], skip_special_tokens=True).strip()
        outputs.append({"prompt": prompt, "response": response})
    if was_training:
        model.train()
    return outputs


def evaluate_causal_loss(model: Any, tokenizer: Any, dataset: Any, batch_size: int = 1) -> float | None:
    """Return token-weighted causal LM loss over a tokenized held-out split."""
    if len(dataset) == 0:
        return None
    collator = DataCollatorForSeq2Seq(tokenizer, padding=True, label_pad_token_id=-100)
    loader = DataLoader(dataset, batch_size=batch_size, collate_fn=collator)
    device = next(model.parameters()).device
    was_training = model.training
    model.eval()
    weighted_loss, token_count = 0.0, 0
    with torch.inference_mode():
        for batch in loader:
            batch = {key: value.to(device) for key, value in batch.items()}
            labels = batch["labels"]
            count = int((labels != -100).sum().item())
            if count == 0:
                continue
            loss = model(**batch).loss
            weighted_loss += float(loss.item()) * count
            token_count += count
    if was_training:
        model.train()
    return weighted_loss / token_count if token_count else None


def evaluate_generation(model: Any, tokenizer: Any, dataset: Any,
                        max_new_tokens: int = 96, max_input_length: int = 512) -> tuple[dict[str, Any], list[dict[str, str]]]:
    """Evaluate a fixed dataset with greedy decoding and return raw examples."""
    device = next(model.parameters()).device
    was_training = model.training
    model.eval()
    predictions: list[str] = []
    references: list[str] = []
    samples: list[dict[str, str]] = []
    durations: list[float] = []
    for row in dataset:
        prompt = format_chat_prompt(tokenizer, row["instruction"])
        encoded = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=max_input_length)
        encoded = {key: value.to(device) for key, value in encoded.items()}
        started = time.perf_counter()
        with torch.inference_mode():
            output = model.generate(
                **encoded, max_new_tokens=max_new_tokens, do_sample=False,
                pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id,
            )
        durations.append(time.perf_counter() - started)
        generated = tokenizer.decode(output[0, encoded["input_ids"].shape[1]:], skip_special_tokens=True).strip()
        predictions.append(generated)
        references.append(row["response"])
        samples.append({"prompt": row["instruction"], "reference": row["response"], "response": generated})
    if was_training:
        model.train()
    metrics = generation_metrics(predictions, references)
    total_tokens = sum(len(tokenizer.encode(value)) for value in predictions)
    total_time = sum(durations)
    metrics.update({
        "inference_latency_seconds_mean": total_time / len(durations),
        "inference_throughput_tokens_per_second": total_tokens / total_time if total_time else None,
        "generation_max_new_tokens": max_new_tokens,
        "generation_do_sample": False,
    })
    return metrics, samples
