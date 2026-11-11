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
    """Write readable comparison charts from one already-selected run."""
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    completed = [row for row in records if row.get("status") == "completed"]
    if not completed:
        return []
    sns.set_theme(style="whitegrid", context="notebook", font_scale=0.95)
    paths: list[str] = []
    core_order = ["baseline", "lora", "qlora", "full_finetuning"]
    labels = {"baseline": "Baseline", "lora": "LoRA (r=8)", "qlora": "QLoRA (r=8)",
              "full_finetuning": "Full fine-tuning"}
    by_experiment = {row.get("experiment"): row for row in completed}
    core = [by_experiment[name] for name in core_order if name in by_experiment]

    # Separate axes preserve visibility of low exact-match values without implying
    # that exact match and lexical overlap are interchangeable scales.
    if core and any(row.get("normalized_exact_match") is not None for row in core):
        fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
        x = list(range(len(core)))
        exact = [row.get("normalized_exact_match") for row in core]
        rouge = [row.get("rougeL_f1") for row in core]
        axes[0].bar(x, [value if value is not None else 0 for value in exact], color="#5479a8")
        axes[0].set_title("Normalized exact match")
        axes[0].set_ylabel("Score")
        axes[0].set_ylim(0, max(0.08, max((value or 0) for value in exact) * 1.35))
        axes[1].bar(x, [value if value is not None else 0 for value in rouge], color="#d28a5c")
        axes[1].set_title("ROUGE-L F1")
        axes[1].set_ylabel("Score")
        axes[1].set_ylim(0, max(0.35, max((value or 0) for value in rouge) * 1.2))
        for ax in axes:
            ax.set_xticks(x, [labels[row["experiment"]] for row in core], rotation=18, ha="right")
            ax.set_xlabel("")
        fig.suptitle("Held-out generation quality")
        fig.tight_layout()
        path = output / "quality_comparison.png"
        fig.savefig(path, dpi=220, bbox_inches="tight")
        plt.close(fig)
        paths.append(str(path))

    # A resumed checkpoint provides no valid elapsed training time or training
    # peak, so omit it from these two efficiency comparisons.
    measured = [row for row in completed if not row.get("checkpoint_resumed")]
    for filename, field, title, ylabel, unit_scale in [
        ("training_time_comparison.png", "train_seconds", "Training time (completed invocation)", "Minutes", 60.0),
        ("gpu_memory_comparison.png", "peak_gpu_memory_gib", "Peak allocated GPU memory", "GiB", 1.0),
        ("artifact_size_comparison.png", "artifact_bytes", "Saved adapter/model artifact size", "MiB", 1024**2),
    ]:
        source_rows = core if filename == "artifact_size_comparison.png" else measured
        rows = [row for row in source_rows if row.get(field) is not None and row.get("experiment") in labels]
        if not rows:
            continue
        values = [float(row[field]) / unit_scale for row in rows]
        fig, ax = plt.subplots(figsize=(8, 4.5))
        bars = ax.bar(range(len(rows)), values, color="#5479a8")
        ax.bar_label(bars, fmt="%.2f", padding=3, fontsize=8)
        ax.set_xticks(range(len(rows)), [labels[row["experiment"]] for row in rows], rotation=18, ha="right")
        ax.set(title=title, xlabel="", ylabel=ylabel)
        if filename in {"training_time_comparison.png", "gpu_memory_comparison.png"} and any(
            row.get("experiment") == "full_finetuning" and row.get("checkpoint_resumed") for row in completed
        ):
            ax.text(0.01, -0.30, "Full fine-tuning omitted: its saved record resumed a completed checkpoint.",
                    transform=ax.transAxes, fontsize=8, va="top")
        fig.tight_layout()
        path = output / filename
        fig.savefig(path, dpi=220, bbox_inches="tight")
        plt.close(fig)
        paths.append(str(path))

    parameter_rows = [row for row in core if row.get("trainable_percent") is not None]
    if parameter_rows:
        fig, ax = plt.subplots(figsize=(8, 4.5))
        values = [row["trainable_percent"] for row in parameter_rows]
        bars = ax.bar(range(len(values)), values, color="#5b9a77")
        ax.bar_label(bars, fmt="%.2f%%", padding=3, fontsize=8)
        ax.set_xticks(range(len(values)), [labels[row["experiment"]] for row in parameter_rows], rotation=18, ha="right")
        ax.set(title="Trainable parameter share", xlabel="", ylabel="Percent of model parameters")
        fig.tight_layout()
        path = output / "trainable_parameter_percentage.png"
        fig.savefig(path, dpi=220, bbox_inches="tight")
        plt.close(fig)
        paths.append(str(path))

    rank_rows = [row for row in completed if row.get("rank") is not None]
    if rank_rows:
        rank_rows.sort(key=lambda row: row["rank"])
        fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
        for ax, metric, title, color in [
            (axes[0], "normalized_exact_match", "Normalized exact match", "#5479a8"),
            (axes[1], "rougeL_f1", "ROUGE-L F1", "#d28a5c"),
        ]:
            vals = [row.get(metric) for row in rank_rows]
            ax.plot([row["rank"] for row in rank_rows], vals, marker="o", color=color)
            ax.set(title=title, xlabel="LoRA rank", ylabel="Score")
        fig.suptitle("LoRA rank and held-out performance")
        fig.tight_layout()
        path = output / "lora_rank_performance.png"
        fig.savefig(path, dpi=220, bbox_inches="tight")
        plt.close(fig)
        paths.append(str(path))
    return paths
