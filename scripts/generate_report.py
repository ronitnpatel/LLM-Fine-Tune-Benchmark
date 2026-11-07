"""Generate the research report from recorded benchmark JSON."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.benchmarking.benchmark import generate_plots


def build_report(records: list[dict[str, Any]], plot_dir: Path) -> str:
    """Render a concise evidence-based report; pending runs have no claims."""
    completed = [row for row in records if row.get("status") == "completed"]
    statuses: dict[str, int] = {}
    for row in records:
        status = str(row.get("status", "unknown"))
        statuses[status] = statuses.get(status, 0) + 1
    if not completed:
        findings = "No completed experiment records are available. Results remain pending; no performance claims are made."
        table = "_No completed quantitative results._"
    else:
        fields = ["experiment", "model_id", "normalized_exact_match", "rougeL_f1", "validation_loss",
                  "train_seconds", "peak_gpu_memory_gib", "trainable_percent", "artifact_bytes"]
        header = "| " + " | ".join(fields) + " |"
        sep = "|" + "|".join(["---"] * len(fields)) + "|"
        rows = ["| " + " | ".join(str(row.get(field, "")) for field in fields) + " |" for row in completed]
        table = "\n".join([header, sep, *rows])
        findings = f"{len(completed)} experiment record(s) completed. Interpret metrics according to their distinct definitions below."
    plots = sorted(path.name for path in plot_dir.glob("*.png"))
    plot_links = "\n".join(f"- [ {name} ](../../plots/{name})" for name in plots) or "_No charts generated yet._"
    return f"""# Fine-Tuning Benchmark Report

> Generated from `results/metrics/benchmark_results.json`. This report describes recorded runs only.

## Executive summary

{findings}

## Research objective and background

This project compares full fine-tuning, LoRA, and QLoRA under a shared model and data split. LoRA freezes the base weights and trains low-rank updates. QLoRA combines adapter training with a 4-bit NF4 base representation. Full fine-tuning updates the full parameter set. The comparison focuses on measured quality and resource tradeoffs.

## Model and data

The selected model and resolved revision are captured in each result record. The default primary candidate is Qwen2.5-0.5B-Instruct; SmolLM2-135M-Instruct is the pre-experiment memory fallback. The dataset is Databricks Dolly 15k (CC BY-SA 3.0), sampled and split deterministically. Model license: Apache-2.0. Dataset attribution and ShareAlike terms apply; see the upstream dataset card.

## Experimental methodology

Methods share the selected model, preprocessing, seed, training/validation/test split, and greedy generation settings. The test set is held out from training. The smoke profile is a pipeline check, not a statistically strong benchmark. Failed or skipped runs remain identified by status and reason in the JSON.

## Environment and configurations

Hardware, package versions, model/data revisions, precision, effective training settings, seed, timestamp, and split metadata are recorded per run. See the JSON/CSV for the complete metadata. Missing measurements are left empty rather than estimated unless explicitly labeled.

## Quantitative results

{table}

Status counts: `{json.dumps(statuses, sort_keys=True)}`.

Normalized exact match is strict after lowercase, article, punctuation, and whitespace normalization; it penalizes valid paraphrases. ROUGE-L F1 measures longest-common-subsequence lexical overlap and does not establish factual correctness. Loss is comparable only under the same tokenization and masking setup. Memory and timing are Colab-run measurements and hardware dependent. Artifact bytes exclude the cached base model unless noted.

## Visualizations

{plot_links}

## Qualitative examples

See `results/sample_outputs/` for saved prompts, references, and generated outputs. Manual inspection is illustrative and is not a blinded human study.

## Tradeoffs, limitations, and recommendations

Use LoRA when a small, portable adapter is useful and base-model precision fits. Use QLoRA when reducing base-weight memory is important and the Colab runtime supports bitsandbytes. Full fine-tuning offers maximum parameter flexibility at higher memory and storage cost. Prefer no method based on a smoke run alone. This small model/subset cannot establish production-scale generalization; free Colab hardware and session duration vary, and Dolly has known source-data limitations.

## Conclusion

Conclusions should be added only after reviewing completed measurements, uncertainty, and qualitative outputs. The report generator does not infer a winner or claim an improvement from absent or incomplete results.
"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", default="results/metrics/benchmark_results.json")
    parser.add_argument("--output", default="results/reports/benchmark_report.md")
    parser.add_argument("--plots", default="results/plots")
    args = parser.parse_args()
    source, output, plots = Path(args.results), Path(args.output), Path(args.plots)
    records = json.loads(source.read_text(encoding="utf-8")) if source.exists() else []
    generate_plots(records, plots)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(build_report(records, plots), encoding="utf-8")


if __name__ == "__main__":
    main()
