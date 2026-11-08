"""QLoRA configuration helpers."""

from __future__ import annotations

import torch
from transformers import BitsAndBytesConfig


def quantization_config() -> BitsAndBytesConfig:
    """Build NF4/double-quant config using the best supported compute dtype."""
    if not torch.cuda.is_available():
        raise RuntimeError("bitsandbytes QLoRA requires a CUDA GPU in this project")
    compute_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=compute_dtype,
    )
