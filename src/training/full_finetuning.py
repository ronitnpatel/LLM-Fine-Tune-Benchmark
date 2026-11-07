"""Full parameter fine-tuning preparation."""

from __future__ import annotations

from typing import Any


def prepare_full_finetuning(model: Any) -> Any:
    """Enable gradients for every parameter and enable activation checkpointing."""
    if getattr(model, "is_loaded_in_4bit", False) or getattr(model, "is_loaded_in_8bit", False):
        raise ValueError("Full fine-tuning must use an unquantized model")
    for parameter in model.parameters():
        parameter.requires_grad = True
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    if not all(parameter.requires_grad for parameter in model.parameters()):
        raise RuntimeError("Full fine-tuning parameter verification failed")
    return model
