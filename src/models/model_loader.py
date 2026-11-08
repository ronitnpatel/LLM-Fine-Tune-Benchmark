"""GPU diagnostics, model preflight, and reproducible model loading."""

from __future__ import annotations

import logging
import gc
import time
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

LOGGER = logging.getLogger(__name__)


def gpu_info() -> dict[str, Any]:
    """Return detected accelerator details without assuming a particular GPU."""
    if not torch.cuda.is_available():
        return {"available": False, "name": None, "total_memory_gib": 0.0, "free_memory_gib": 0.0}
    free, total = torch.cuda.mem_get_info()
    return {
        "available": True,
        "name": torch.cuda.get_device_name(0),
        "total_memory_gib": round(total / 1024**3, 2),
        "free_memory_gib": round(free / 1024**3, 2),
    }


def choose_model(primary: str, fallback: str, minimum_free_gib: float) -> tuple[str, dict[str, Any]]:
    """Select one model for all experiments before the baseline is evaluated."""
    info = gpu_info()
    selected = primary
    reason = "primary model selected"
    if not info["available"]:
        selected, reason = fallback, "no CUDA GPU detected; selecting smaller fallback"
    elif info["free_memory_gib"] < minimum_free_gib:
        selected, reason = fallback, f"free VRAM below {minimum_free_gib:g} GiB preflight threshold"
    return selected, {**info, "selection_reason": reason, "primary_model": primary, "fallback_model": fallback}


def load_tokenizer(model_id: str, revision: str = "main") -> Any:
    """Load a fast tokenizer and ensure a padding token exists."""
    tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    return tokenizer


def load_model(model_id: str, revision: str = "main", *, quantized: bool = False) -> Any:
    """Load a model with detected supported precision, optionally 4-bit."""
    if not torch.cuda.is_available():
        raise RuntimeError("Training requires a Colab CUDA GPU. No model training was started.")
    kwargs: dict[str, Any] = {"revision": revision, "low_cpu_mem_usage": True}
    if quantized:
        from .quantization import quantization_config

        kwargs["quantization_config"] = quantization_config()
        kwargs["device_map"] = {"": 0}
    else:
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        kwargs["torch_dtype"] = dtype
        kwargs["device_map"] = {"": 0}
    model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
    model.config.use_cache = False
    return model


def preflight_full_finetuning(model_id: str, revision: str = "main", sequence_length: int = 512) -> dict[str, Any]:
    """Run one scratch full-parameter optimizer step to test actual GPU fit.

    The disposable model is never saved or used in reported results. This probe
    checks weights, activations, gradients, and AdamW state before model choice
    is finalized and before any baseline/primary experiment is run.
    """
    if not torch.cuda.is_available():
        return {"fits": False, "reason": "CUDA unavailable", "peak_gpu_memory_gib": None}
    model = optimizer = input_ids = None
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    try:
        model = load_model(model_id, revision, quantized=False)
        model.train()
        if hasattr(model, "gradient_checkpointing_enable"):
            model.gradient_checkpointing_enable()
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        input_ids = torch.randint(
            low=0,
            high=int(model.config.vocab_size),
            size=(1, sequence_length),
            device="cuda",
        )
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
        loss = model(input_ids=input_ids, labels=input_ids).loss
        loss.backward()
        optimizer.step()
        peak = torch.cuda.max_memory_allocated() / 1024**3
        return {"fits": True, "reason": "scratch full-parameter optimizer step completed",
                "peak_gpu_memory_gib": peak, "probe_seconds": time.perf_counter() - started,
                "sequence_length": sequence_length}
    except torch.cuda.OutOfMemoryError as exc:
        return {"fits": False, "reason": f"CUDA OOM during full-finetuning probe: {exc}",
                "peak_gpu_memory_gib": torch.cuda.max_memory_allocated() / 1024**3,
                "probe_seconds": time.perf_counter() - started, "sequence_length": sequence_length}
    except Exception as exc:
        return {"fits": False, "reason": f"{type(exc).__name__}: {exc}",
                "peak_gpu_memory_gib": torch.cuda.max_memory_allocated() / 1024**3,
                "probe_seconds": time.perf_counter() - started, "sequence_length": sequence_length}
    finally:
        del model, optimizer, input_ids
        gc.collect()
        torch.cuda.empty_cache()


def model_metadata(model: Any, model_id: str, tokenizer: Any) -> dict[str, Any]:
    """Capture identifiers and architecture details used by this run."""
    return {
        "model_id": model_id,
        "model_revision": getattr(model.config, "_commit_hash", None),
        "tokenizer_id": getattr(tokenizer, "name_or_path", model_id),
        "tokenizer_revision": getattr(tokenizer, "init_kwargs", {}).get("revision"),
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "config_model_type": getattr(model.config, "model_type", None),
        "config_architectures": getattr(model.config, "architectures", None),
        "dtype": str(next(model.parameters()).dtype),
    }
