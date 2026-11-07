"""Incremental machine-readable result storage and benchmark plots."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def save_results(records: list[dict[str, Any]], output_dir: str | Path) -> None:
    """Atomically write compact JSON and CSV snapshots."""
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    for filename in [
        "quality_comparison.png", "training_time_comparison.png", "gpu_memory_comparison.png",
        "artifact_size_comparison.png", "trainable_parameter_percentage.png", "lora_rank_performance.png",
    ]:
        stale = output / filename
        if stale.exists():
            stale.unlink()
    json_path, csv_path = output / "benchmark_results.json", output / "benchmark_results.csv"
    json_tmp, csv_tmp = json_path.with_suffix(".json.tmp"), csv_path.with_suffix(".csv.tmp")
    json_tmp.write_text(json.dumps(records, indent=2, default=str), encoding="utf-8")
    columns = [
        "run_id", "timestamp_utc", "experiment", "status", "method", "rank", "model_id",
        "model_revision", "dataset_id", "dataset_revision", "normalized_exact_match", "rougeL_f1",
        "validation_loss", "test_loss", "train_seconds", "train_seconds_per_epoch",
        "peak_gpu_memory_gib", "trainable_parameters", "total_parameters", "trainable_percent",
        "artifact_bytes", "inference_latency_seconds_mean", "inference_throughput_tokens_per_second",
        "error",
    ]
    frame = pd.json_normalize(records, sep=".") if records else pd.DataFrame(columns=columns)
    frame.to_csv(csv_tmp, index=False)
    json_tmp.replace(json_path)
    csv_tmp.replace(csv_path)


def generate_plots(records: list[dict[str, Any]], output_dir: str | Path) -> list[str]:
    """Write requested charts from completed rows with available measurements."""
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    completed = [row for row in records if row.get("status") == "completed"]
    if not completed:
        return []
    sns.set_theme(style="whitegrid", context="talk")
    frame = pd.DataFrame(completed)
    paths: list[str] = []
    charts = [
        ("quality_comparison.png", ["normalized_exact_match", "rougeL_f1"], "Held-out generation quality", "Score"),
        ("training_time_comparison.png", ["train_seconds"], "Training time", "Seconds"),
        ("gpu_memory_comparison.png", ["peak_gpu_memory_gib"], "Peak allocated GPU memory", "GiB"),
        ("artifact_size_comparison.png", ["artifact_bytes"], "Saved adapter/model artifact size", "Bytes"),
        ("trainable_parameter_percentage.png", ["trainable_percent"], "Trainable parameter percentage", "Percent"),
    ]
    for filename, fields, title, ylabel in charts:
        available = [field for field in fields if field in frame and frame[field].notna().any()]
        if not available:
            continue
        long = frame.melt(id_vars=["experiment"], value_vars=available, var_name="metric", value_name="value").dropna()
        if long.empty:
            continue
        fig, ax = plt.subplots(figsize=(9, 5.5))
        sns.barplot(data=long, x="experiment", y="value", hue="metric" if len(available) > 1 else None, ax=ax)
        ax.set(title=title, xlabel="Experiment", ylabel=ylabel)
        ax.tick_params(axis="x", rotation=20)
        fig.tight_layout()
        path = output / filename
        fig.savefig(path, dpi=220, bbox_inches="tight")
        plt.close(fig)
        paths.append(str(path))
    rank_rows = [row for row in completed if row.get("rank") is not None]
    if rank_rows:
        rank = pd.DataFrame(rank_rows).sort_values("rank")
        fig, ax = plt.subplots(figsize=(8, 5.5))
        for metric in ["normalized_exact_match", "rougeL_f1"]:
            if metric in rank and rank[metric].notna().any():
                sns.lineplot(data=rank, x="rank", y=metric, marker="o", label=metric, ax=ax)
        ax.set(title="LoRA rank and held-out performance", xlabel="LoRA rank", ylabel="Score")
        fig.tight_layout()
        path = output / "lora_rank_performance.png"
        fig.savefig(path, dpi=220, bbox_inches="tight")
        plt.close(fig)
        paths.append(str(path))
    return paths
