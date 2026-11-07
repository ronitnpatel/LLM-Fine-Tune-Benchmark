"""Text-generation metrics with explicit normalization behavior."""

from __future__ import annotations

import re
import string
from collections.abc import Sequence

from rouge_score.rouge_scorer import RougeScorer

_ROUGE = RougeScorer(["rougeL"], use_stemmer=True)


def normalize_answer(text: str) -> str:
    """Lowercase, remove punctuation/articles, and collapse whitespace."""
    text = text.lower().translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


def generation_metrics(predictions: Sequence[str], references: Sequence[str]) -> dict[str, float]:
    """Compute normalized exact match and ROUGE-L F1 for paired texts."""
    if len(predictions) != len(references):
        raise ValueError("predictions and references must have the same length")
    if not references:
        raise ValueError("At least one prediction/reference pair is required")
    em = [normalize_answer(pred) == normalize_answer(ref) for pred, ref in zip(predictions, references)]
    rouge = [_ROUGE.score(ref, pred)["rougeL"].fmeasure for pred, ref in zip(predictions, references)]
    return {"normalized_exact_match": sum(em) / len(em), "rougeL_f1": sum(rouge) / len(rouge),
            "evaluation_examples": len(references)}
