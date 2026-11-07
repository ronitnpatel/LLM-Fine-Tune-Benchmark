"""QLoRA adapter construction and preparation."""

from __future__ import annotations

from typing import Any

from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training


def add_qlora(model: Any, rank: int = 8, alpha: int = 16, dropout: float = 0.05,
              target_modules: list[str] | str = "auto") -> Any:
    """Prepare a 4-bit base and attach trainable LoRA adapters."""
    if not getattr(model, "is_loaded_in_4bit", False):
        raise RuntimeError("QLoRA requires a model loaded with 4-bit quantization")
    if target_modules == "auto":
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    model = get_peft_model(model, LoraConfig(
        r=rank, lora_alpha=alpha, lora_dropout=dropout,
        target_modules=target_modules, bias="none", task_type="CAUSAL_LM",
    ))
    trainable = [(name, parameter) for name, parameter in model.named_parameters() if parameter.requires_grad]
    if not trainable or not all("lora_" in name for name, _ in trainable):
        raise RuntimeError("QLoRA base-freezing verification failed")
    return model
