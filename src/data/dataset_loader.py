"""Load and deterministically partition the public Dolly instruction dataset."""

from __future__ import annotations

from typing import Any

from datasets import Dataset, DatasetDict, load_dataset

from .preprocessing import clean_records


def load_splits(
    dataset_id: str,
    revision: str = "main",
    max_examples: int = 1200,
    seed: int = 42,
    validation_fraction: float = 0.1,
    test_fraction: float = 0.1,
) -> tuple[DatasetDict, dict[str, Any]]:
    """Load, clean, cap, and split data; test is excluded from training."""
    if max_examples < 3:
        raise ValueError("max_examples must be at least 3")
    if validation_fraction <= 0 or test_fraction <= 0 or validation_fraction + test_fraction >= 1:
        raise ValueError("validation_fraction and test_fraction must be positive and sum to less than 1")
    raw = load_dataset(dataset_id, revision=revision, split="train")
    records = clean_records(raw)
    if len(records) < 3:
        raise ValueError("Fewer than three valid instruction records were loaded")
    dataset = Dataset.from_list(records).shuffle(seed=seed).select(range(min(max_examples, len(records))))
    test_count = max(1, round(len(dataset) * test_fraction))
    val_count = max(1, round(len(dataset) * validation_fraction))
    if test_count + val_count >= len(dataset):
        raise ValueError("Dataset cap is too small for requested validation/test fractions")
    test = dataset.select(range(test_count))
    validation = dataset.select(range(test_count, test_count + val_count))
    train = dataset.select(range(test_count + val_count, len(dataset)))
    split = DatasetDict(train=train, validation=validation, test=test)
    metadata = {
        "dataset_id": dataset_id,
        "requested_revision": revision,
        "rows_valid": len(records),
        "rows_selected": len(dataset),
        "split_sizes": {key: len(value) for key, value in split.items()},
        "seed": seed,
    }
    return split, metadata
