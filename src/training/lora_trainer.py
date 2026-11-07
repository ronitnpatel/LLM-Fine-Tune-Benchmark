"""LoRA adapter construction."""

from __future__ import annotations

from typing import Any

from peft import LoraConfig, get_peft_model


def add_lora(model: Any, rank: int = 8, alpha: int = 16, dropout: float = 0.05,
             target_modules: list[str] | str = "auto") -> Any:
    """Attach standard LoRA modules and assert the base weights are frozen."""
    if target_modules == "auto":
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    config = LoraConfig(
        r=rank, lora_alpha=alpha, lora_dropout=dropout,
        target_modules=target_modules, bias="none", task_type="CAUSAL_LM",
    )
    wrapped = get_peft_model(model, config)
    if hasattr(wrapped, "enable_input_require_grads"):
        wrapped.enable_input_require_grads()
    trainable_names = [name for name, parameter in wrapped.named_parameters() if parameter.requires_grad]
    if not trainable_names or not all("lora_" in name for name in trainable_names):
        raise RuntimeError("LoRA base-freezing verification failed")
    return wrapped
