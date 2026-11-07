"""Shared Trainer construction and measurement helpers."""

from __future__ import annotations

import inspect
import logging
import time
from pathlib import Path
from typing import Any

import torch
from transformers import DataCollatorForSeq2Seq, Trainer, TrainerCallback, TrainingArguments

LOGGER = logging.getLogger(__name__)


class EpochTimer(TrainerCallback):
    """Measure wall time spent in each training epoch in this session."""

    def __init__(self) -> None:
        self.started_at: float | None = None
        self.epoch_seconds: list[float] = []

    def on_epoch_begin(self, args: Any, state: Any, control: Any, **kwargs: Any) -> Any:
        self.started_at = time.perf_counter()
        return control

    def on_epoch_end(self, args: Any, state: Any, control: Any, **kwargs: Any) -> Any:
        if self.started_at is not None:
            self.epoch_seconds.append(time.perf_counter() - self.started_at)
            self.started_at = None
        return control


def count_trainable_parameters(model: Any) -> dict[str, int | float]:
    """Count trainable and total parameter elements."""
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    return {"trainable_parameters": trainable, "total_parameters": total,
            "trainable_percent": 100.0 * trainable / total if total else 0.0}


def make_trainer(model: Any, tokenizer: Any, train_dataset: Any, eval_dataset: Any,
                 output_dir: str | Path, config: dict[str, Any]) -> Trainer:
    """Create a version-tolerant Trainer with checkpointing and safe defaults."""
    args = {
        "output_dir": str(output_dir),
        "num_train_epochs": config["epochs"],
        "per_device_train_batch_size": config["batch_size"],
        "per_device_eval_batch_size": 1,
        "gradient_accumulation_steps": config["gradient_accumulation_steps"],
        "learning_rate": config["learning_rate"],
        "weight_decay": config.get("weight_decay", 0.0),
        "warmup_ratio": config.get("warmup_ratio", 0.03),
        "max_grad_norm": config.get("max_grad_norm", 1.0),
        "logging_steps": config.get("logging_steps", 10),
        "save_strategy": "epoch",
        "save_total_limit": 2,
        "gradient_checkpointing": config.get("use_gradient_checkpointing", True),
        "fp16": torch.cuda.is_available() and not torch.cuda.is_bf16_supported(),
        "bf16": torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
        "report_to": [],
        "remove_unused_columns": False,
        "dataloader_pin_memory": True,
        "seed": config.get("seed", 42),
        "data_seed": config.get("seed", 42),
    }
    signature = inspect.signature(TrainingArguments.__init__).parameters
    if "eval_strategy" in signature:
        args["eval_strategy"] = "epoch"
    elif "evaluation_strategy" in signature:
        args["evaluation_strategy"] = "epoch"
    if "processing_class" in inspect.signature(Trainer.__init__).parameters:
        trainer = Trainer(
            model=model,
            args=TrainingArguments(**args),
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=DataCollatorForSeq2Seq(tokenizer, padding=True, label_pad_token_id=-100),
            processing_class=tokenizer,
        )
    else:
        trainer = Trainer(
            model=model,
            args=TrainingArguments(**args),
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=DataCollatorForSeq2Seq(tokenizer, padding=True, label_pad_token_id=-100),
            tokenizer=tokenizer,
        )
    return trainer


def train_experiment(method: str, model: Any, tokenizer: Any, train_dataset: Any,
                     eval_dataset: Any, output_dir: str | Path,
                     config: dict[str, Any], test_dataset: Any = None) -> dict[str, Any]:
    """Train and return wall-clock, memory, parameter, and loss measurements."""
    counts = count_trainable_parameters(model)
    if method in {"lora", "qlora"} and counts["trainable_parameters"] >= counts["total_parameters"]:
        raise RuntimeError(f"{method} did not freeze the base model")
    if method == "full" and counts["trainable_parameters"] != counts["total_parameters"]:
        raise RuntimeError("Full fine-tuning must enable gradients for every model parameter")
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    trainer = make_trainer(model, tokenizer, train_dataset, eval_dataset, output_dir, config)
    epoch_timer = EpochTimer()
    trainer.add_callback(epoch_timer)
    started = time.perf_counter()
    resume = any(Path(output_dir).glob("checkpoint-*"))
    result = trainer.train(resume_from_checkpoint=True if resume else None)
    elapsed = time.perf_counter() - started
    eval_metrics = trainer.evaluate()
    test_metrics = trainer.evaluate(eval_dataset=test_dataset) if test_dataset is not None else {}
    saved = Path(output_dir) / "model_artifact"
    trainer.save_model(str(saved))
    tokenizer.save_pretrained(saved)
    artifact_bytes = sum(path.stat().st_size for path in saved.rglob("*") if path.is_file())
    return {
        **counts,
        "train_seconds": elapsed,
        "train_seconds_per_epoch": (sum(epoch_timer.epoch_seconds) / len(epoch_timer.epoch_seconds)
                                     if epoch_timer.epoch_seconds else None),
        "epoch_seconds_current_session": epoch_timer.epoch_seconds,
        "train_seconds_scope": "active training invocation; earlier sessions are excluded after resume",
        "peak_gpu_memory_gib": (torch.cuda.max_memory_allocated() / 1024**3) if torch.cuda.is_available() else None,
        "train_loss": result.training_loss,
        "validation_loss": eval_metrics.get("eval_loss"),
        "test_loss": test_metrics.get("eval_loss"),
        "artifact_bytes": artifact_bytes,
        "checkpoint_resumed": resume,
        "trainer_state": result.metrics,
    }
