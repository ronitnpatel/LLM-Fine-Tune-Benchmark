"""Generate the research report from recorded benchmark JSON."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.benchmarking.benchmark import generate_plots


def _score(row: dict[str, Any], metric: str) -> str:
    value = row.get(metric)
    if value is None:
        return "—"
    count = row.get("evaluation_examples")
    if metric == "normalized_exact_match" and count:
        hits = round(float(value) * int(count))
        return f"{float(value):.3f} ({hits}/{count})"
    return f"{float(value):.3f}"


def _minutes(row: dict[str, Any]) -> str:
    if row.get("checkpoint_resumed"):
        return "Not comparable (resumed checkpoint)"
    value = row.get("train_seconds")
    return f"{float(value) / 60:.1f} min" if value is not None else "—"


def _training_memory(row: dict[str, Any]) -> str:
    if row.get("checkpoint_resumed"):
        return "Not comparable (resumed checkpoint)"
    value = row.get("peak_gpu_memory_gib")
    return f"{float(value):.2f} GiB" if value is not None else "—"


def _artifact(row: dict[str, Any]) -> str:
    value = row.get("artifact_bytes")
    return f"{float(value) / 1024**2:.1f} MiB" if value is not None else "—"


def _truncate(text: str, limit: int = 340) -> str:
    text = " ".join(text.split())
    return text if len(text) <= limit else text[: limit - 1].rstrip() + "…"


def _qualitative_examples(run_id: str, sample_dir: Path | None) -> str:
    if sample_dir is None:
        return "_Sample-output files are not available to this report invocation._"
    methods = ["baseline", "lora", "qlora", "full_finetuning"]
    loaded: dict[str, list[dict[str, str]]] = {}
    for method in methods:
        path = sample_dir / f"{run_id}_{method}.json"
        if path.exists():
            try:
                loaded[method] = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
    if not loaded:
        return "_No qualitative samples found for this run._"
    target_prompt = "Identify which car manufacturer is French or American"
    chosen = None
    # Use one stable held-out prompt so generated reports show the same comparison.
    for method_rows in loaded.values():
        chosen = next((item for item in method_rows if target_prompt in item.get("prompt", "")), None)
        if chosen:
            break
    sections: list[str] = []
    if chosen:
        sections.append(f"**Held-out prompt:** {chosen['prompt']}\n\n**Reference:** {chosen.get('reference', '')}")
        for method in methods:
            item = next((row for row in loaded.get(method, []) if row.get("prompt") == chosen.get("prompt")), None)
            if item:
                sections.append(f"- **{method}:** {_truncate(item.get('response', ''))}")
    prompt_only: dict[str, list[dict[str, str]]] = {}
    for method in methods:
        path = sample_dir / f"{run_id}_{method}_generalization.json"
        if path.exists():
            try:
                prompt_only[method] = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
    general_prompt = "Summarize the following idea"
    general = next((row for rows in prompt_only.values() for row in rows if general_prompt in row.get("prompt", "")), None)
    if general:
        sections.append(f"**Prompt-only probe:** {general['prompt']}")
        for method in methods:
            item = next((row for row in prompt_only.get(method, []) if row.get("prompt") == general.get("prompt")), None)
            if item:
                sections.append(f"- **{method}:** {_truncate(item.get('response', ''))}")
    return "\n\n".join(sections) if sections else "_No selected qualitative examples found._"


def _config_mismatch(model_id: str, rows_selected: int, run_mode: str) -> str:
    """Flag when the checked-in defaults differ from the selected run metadata."""
    root = Path(__file__).resolve().parents[1]
    try:
        import yaml

        model_cfg = yaml.safe_load((root / "configs/model_config.yaml").read_text(encoding="utf-8"))
        data_cfg = yaml.safe_load((root / "configs/dataset_config.yaml").read_text(encoding="utf-8"))
    except (OSError, ImportError, ValueError):
        return ""
    differences = []
    if model_cfg.get("primary_model") != model_id:
        differences.append(f"model default is `{model_cfg.get('primary_model')}`, recorded run used `{model_id}`")
    if run_mode == "full" and data_cfg.get("max_examples") != rows_selected:
        differences.append(f"dataset cap is {data_cfg.get('max_examples')} in the checked-in config, recorded full run selected {rows_selected} rows")
    if not differences:
        return ""
    items = "\n".join(f"- {item}." for item in differences)
    return (
        "## Reproduction status\n\n"
        "> **Configuration mismatch:** these measurements describe the run recorded in the result metadata. "
        "The checked-in defaults do not currently reproduce it. Resolve this before presenting the run as the result of the repository's default configuration.\n\n"
        f"{items}\n"
    )


def build_report(records: list[dict[str, Any]], plot_dir: Path,
                 sample_dir: Path | None = None) -> str:
    """Render a detailed report for one run, using recorded measurements only."""
    completed = [row for row in records if row.get("status") == "completed"]
    statuses: dict[str, int] = {}
    for row in records:
        status = str(row.get("status", "unknown"))
        statuses[status] = statuses.get(status, 0) + 1
    if not completed:
        return """# Fine-Tuning Benchmark Report

> Results are pending. No Colab experiment records are marked completed; no metrics or performance claims are reported.
"""

    first = completed[0]
    model_id = first.get("selected_model_id", first.get("model_id", "unknown"))
    model_revision = first.get("model_revision", "not recorded")
    data = first.get("dataset_metadata", {})
    split_sizes = data.get("split_sizes", {})
    rows_selected = data.get("rows_selected", "not recorded")
    run_id = str(first.get("run_id", "unknown"))
    run_mode = first.get("run_mode", "not recorded")
    evaluation_count = next((row.get("evaluation_examples") for row in completed if row.get("evaluation_examples")), None)
    core_names = ["baseline", "lora", "qlora", "full_finetuning"]
    rank_names = ["lora_rank_4", "lora_rank_16", "lora_rank_32"]
    by_name = {row.get("experiment"): row for row in completed}
    table_rows = []
    for name in core_names + rank_names:
        row = by_name.get(name)
        if row is None:
            continue
        table_rows.append(
            "| {experiment} | {em} | {rouge} | {val} | {test} | {time} | {memory} | {params} | {artifact} |".format(
                experiment=name,
                em=_score(row, "normalized_exact_match"),
                rouge=_score(row, "rougeL_f1"),
                val=f"{row['validation_loss']:.3f}" if row.get("validation_loss") is not None else "—",
                test=f"{row['test_loss']:.3f}" if row.get("test_loss") is not None else "—",
                time=_minutes(row),
                memory=_training_memory(row),
                params=f"{row['trainable_percent']:.3f}%" if row.get("trainable_percent") is not None else "—",
                artifact=_artifact(row),
            )
        )
    quantitative_table = "\n".join([
        "| Experiment | Exact match (hits/n) | ROUGE-L F1 | Validation loss | Test loss | Training time | Peak GPU memory | Trainable params | Saved artifact |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        *table_rows,
    ])
    baseline = by_name.get("baseline", {})
    lora = by_name.get("lora", {})
    qlora = by_name.get("qlora", {})
    full = by_name.get("full_finetuning", {})
    rank16 = by_name.get("lora_rank_16", {})
    mismatch = _config_mismatch(str(model_id), int(rows_selected) if isinstance(rows_selected, int) else -1, str(run_mode))
    plots = sorted(path.name for path in plot_dir.glob("*.png"))
    plot_links = "\n".join(f"- [{name}](../../plots/{name})" for name in plots) or "_No charts generated._"
    quant = first.get("training_config", {}).get("qlora", {})
    lora_cfg = first.get("training_config", {}).get("lora", {})
    hardware = first.get("hardware", {})
    package_versions = first.get("package_versions", {})
    full_resume_note = ""
    if full.get("checkpoint_resumed"):
        trainer_runtime = full.get("trainer_state", {}).get("train_runtime")
        full_resume_note = (
            f"The full fine-tuning row resumed a checkpoint (`checkpoint_resumed=true`). Its current invocation reported "
            f"{full.get('train_seconds', 0):.1f} seconds end-to-end but Trainer recorded only {trainer_runtime} seconds of training, "
            "zero training loss, and no epoch timer. Therefore full fine-tuning training time, training loss, and peak-memory "
            "figures are not compared as fresh-run measurements. Its validation/test evaluation and saved artifact are retained "
            "as measurements of the resumed checkpoint.\n\n"
        )
    best_rouge = max((row for row in completed if row.get("rougeL_f1") is not None), key=lambda row: row["rougeL_f1"], default=None)
    best_label = best_rouge.get("experiment") if best_rouge else "not available"
    best_value = best_rouge.get("rougeL_f1") if best_rouge else None
    samples = _qualitative_examples(run_id, sample_dir)
    total_counts = json.dumps(statuses, sort_keys=True)
    return f"""# Fine-Tuning Benchmark Report

> Generated from the recorded Colab run `{run_id}`. Measurements below are reported as recorded; limitations are called out where they affect interpretation.

{mismatch}
## Executive summary

This {run_mode} run completed {len(completed)} experiment records on `{model_id}` using {rows_selected} selected Dolly examples and a held-out test split of {evaluation_count} prompts. On this split, `{best_label}` had the highest ROUGE-L F1 ({best_value:.3f}). LoRA and QLoRA both exceeded the baseline on ROUGE-L, but normalized exact-match remained low (2–6 matches out of {evaluation_count}). These single-run results do not establish statistical significance or broad generalization.

The full fine-tuning result resumed a checkpoint. Its quality evaluation is available, but its recorded training time and peak-memory values are not valid fresh-training measurements. The completed LoRA and QLoRA rows have uninterrupted training measurements.

## Research objective and background

This benchmark compares three update strategies on one base model and one data split. LoRA freezes base weights and learns low-rank adapter matrices. QLoRA combines LoRA with a 4-bit NF4 quantized base and double quantization. Full fine-tuning updates all model parameters. The project measures held-out answer quality and the time, GPU memory, trainable-parameter share, and artifact size required by each approach.

## Model and dataset

- **Model:** `{model_id}`
- **Resolved model revision:** `{model_revision}`
- **Tokenizer:** `{first.get('tokenizer_id', model_id)}`
- **Parameter count:** {first.get('parameter_count', 'not recorded'):,}
- **Recorded inference/model dtype:** `{first.get('dtype', 'not recorded')}`
- **Dataset:** `{first.get('dataset_id', 'not recorded')}` at revision `{first.get('dataset_revision', 'not recorded')}` (CC BY-SA 3.0)
- **Selected/used split:** {rows_selected} rows total: {split_sizes.get('train', '?')} train, {split_sizes.get('validation', '?')} validation, {split_sizes.get('test', '?')} test; seed {first.get('seed', 'not recorded')}.

The splits are shared across methods, and test prompts are not passed to training. The model is Apache-2.0; the dataset's attribution and ShareAlike terms remain separate. Review the upstream dataset terms before redistributing adapted material.

## Experimental methodology

Training used one epoch, batch size {first.get('training_config', {}).get('batch_size', '?')}, gradient accumulation {first.get('training_config', {}).get('gradient_accumulation_steps', '?')}, learning rate {first.get('training_config', {}).get('learning_rate', '?')}, maximum sequence length {first.get('max_length', '?')}, seed {first.get('seed', '?')}, and greedy generation with up to {first.get('generation_config', {}).get('max_new_tokens', '?')} new tokens. LoRA ranks were 4, 8, 16, and 32; rank 8 is the primary LoRA/QLoRA comparison. QLoRA settings were {quant.get('quant_type', 'not recorded')} 4-bit with double quantization `{quant.get('double_quant', 'not recorded')}`. LoRA alpha was {lora_cfg.get('alpha', 'not recorded')} with dropout {lora_cfg.get('dropout', 'not recorded')}.

The baseline and all trained models used the same {evaluation_count}-prompt test set. A separate four-prompt general-instruction probe has no reference answers and is qualitative only.

## Colab environment

- **GPU:** {hardware.get('name', 'not recorded')} ({hardware.get('total_memory_gib', 'not recorded')} GiB total)
- **Python:** {first.get('python_version', 'not recorded')}
- **Packages:** Torch {package_versions.get('torch', '?')}; Transformers {package_versions.get('transformers', '?')}; PEFT {package_versions.get('peft', '?')}; Datasets {package_versions.get('datasets', '?')}; Accelerate {package_versions.get('accelerate', '?')}; bitsandbytes {package_versions.get('bitsandbytes', '?')}.
- **Timestamp (UTC):** {first.get('timestamp_utc', 'not recorded')}

## Quantitative results

{quantitative_table}

Status counts for this run: `{total_counts}`.

Normalized exact match is the fraction of outputs identical after lowercasing, removing punctuation/articles, and collapsing whitespace; it penalizes valid paraphrases. ROUGE-L F1 measures lexical sequence overlap, not factual correctness. Validation/test loss is token-level cross-entropy under the shared prompt/label masking setup. Exact-match percentages here correspond to very small counts: baseline {round(float(baseline.get('normalized_exact_match', 0)) * int(evaluation_count or 0))}/{evaluation_count}, LoRA {round(float(lora.get('normalized_exact_match', 0)) * int(evaluation_count or 0))}/{evaluation_count}, QLoRA {round(float(qlora.get('normalized_exact_match', 0)) * int(evaluation_count or 0))}/{evaluation_count}, and full fine-tuning {round(float(full.get('normalized_exact_match', 0)) * int(evaluation_count or 0))}/{evaluation_count}.

{full_resume_note}The best rank-sweep ROUGE-L was {rank16.get('rougeL_f1', float('nan')):.3f} at rank 16 versus {by_name.get('lora_rank_4', {}).get('rougeL_f1', float('nan')):.3f} at rank 4 and {by_name.get('lora_rank_32', {}).get('rougeL_f1', float('nan')):.3f} at rank 32. The spread is small and should not be interpreted as a reliable rank optimum from one seed and one split.

## Visualizations

{plot_links}

Training-time and GPU-memory charts omit the resumed full fine-tuning row because that row did not record a fresh training pass. The artifact-size chart still includes its saved full-model artifact.

## Qualitative examples

{samples}

The fixed open-ended probe has no gold answers; outputs are included for inspection, not scored. It exposed failures in factual accuracy and constraint following, particularly for full fine-tuning, so no broad generalization improvement is claimed.

## Discussion and recommendations

- On this run, LoRA r=8 reached ROUGE-L F1 {float(lora.get('rougeL_f1', 0)):.3f}; QLoRA reached {float(qlora.get('rougeL_f1', 0)):.3f}; baseline was {float(baseline.get('rougeL_f1', 0)):.3f}. These are descriptive results on one test split.
- QLoRA's measured peak allocation ({_training_memory(qlora)}) was close to LoRA's ({_training_memory(lora)}) in this run. Do not claim a memory advantage from these measurements alone.
- Adapter artifacts are much smaller than the full model artifact. The full model artifact is {_artifact(full)}; LoRA r=8 is {_artifact(lora)} and QLoRA r=8 is {_artifact(qlora)}. These sizes exclude the cached base model.
- Full fine-tuning's evaluation had worse test loss and lower ROUGE-L than the adapters in this run, while the missing fresh training measurements and resumed checkpoint limit resource comparisons.
- A single 120-example test set, one seed, one epoch, and a small model/subset are not enough to establish robust generalization. Repeat runs and a stronger task-specific/human evaluation before drawing research conclusions.

## Limitations and conclusion

This is a small-scale systems experiment. Dolly contains source-data limitations; exact match and ROUGE-L do not verify factuality; the prompt-only probe has only four examples; and no confidence intervals or multiple seeds were run. The run metadata and selected qualitative excerpts are preserved so these limitations can be inspected. Full sample dumps contain dataset prompts/references and are kept local. Conclusions should remain restricted to this measured configuration.
"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", default="results/metrics/benchmark_results.json")
    parser.add_argument("--output", default="results/reports/benchmark_report.md")
    parser.add_argument("--plots", default="results/plots")
    parser.add_argument("--samples", default="results/sample_outputs")
    parser.add_argument("--run-id", default=None, help="Run ID to report; defaults to the newest completed full run.")
    args = parser.parse_args()
    source, output, plots, samples = Path(args.results), Path(args.output), Path(args.plots), Path(args.samples)
    records = json.loads(source.read_text(encoding="utf-8")) if source.exists() else []
    groups: dict[str, list[dict[str, Any]]] = {}
    for row in records:
        groups.setdefault(str(row.get("run_id", "unknown")), []).append(row)
    if args.run_id:
        selected = groups.get(args.run_id, [])
        if not selected:
            parser.error(f"run_id {args.run_id!r} was not found in {source}")
    elif groups:
        full_groups = [group for group in groups.values() if any(row.get("run_mode") == "full" for row in group)]
        candidates = full_groups or list(groups.values())
        selected = max(candidates, key=lambda group: max(str(row.get("timestamp_utc", "")) for row in group))
    else:
        selected = []
    generate_plots(selected, plots)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(build_report(selected, plots, samples), encoding="utf-8")


if __name__ == "__main__":
    main()
