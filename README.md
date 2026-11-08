# LLM Fine-Tuning Benchmark Lab

An execution-ready, Colab-first comparison of full fine-tuning, LoRA, and QLoRA on one instruction model and one held-out dataset split. The notebook selects a model before the baseline so all methods stay comparable, records resource measurements, and writes compact artifacts to Google Drive.

> **Results status: pending.** This repository contains the experiment implementation and reporting pipeline, but no Colab GPU experiment has been run. Metrics, sample outputs, plots, and conclusions will be generated from actual runs; no benchmark values are claimed here.

[![Open in Google Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/OWNER/REPOSITORY/blob/main/notebooks/llm_finetuning_benchmark.ipynb)

Replace `OWNER/REPOSITORY` in this badge after publishing. Google Colab is the required training environment. This project does not train on the local Windows machine or use paid inference/GPU services.

## Why this benchmark

LoRA trains low-rank adapter weights while freezing the base model. QLoRA adds 4-bit NF4 base-weight quantization to reduce memory needs. Full fine-tuning updates all model weights and provides a useful resource and quality reference. The comparison measures the quality/resource tradeoffs under a shared model, data split, prompt format, training budget, and generation setup.

Research questions:

1. How do the methods compare on held-out answer quality and validation loss?
2. What are their measured training-time, peak-memory, trainable-parameter, and artifact-size costs?
3. How sensitive is LoRA quality and efficiency to ranks 4, 8, 16, and 32?
4. Do the models retain reasonable behavior on the held-out set, and what qualitative changes are visible?

## Repository layout

```text
configs/                 Model, data, and training configuration
docs/                    Methodology and experiment design
notebooks/               Google Colab orchestration notebook
src/                     Reusable data, model, training, evaluation, and reporting code
results/                 Compact metrics, plots, samples, and generated report
scripts/                 Report generation entry point
```

## Models and data

The primary candidate is [`Qwen/Qwen2.5-0.5B-Instruct`](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct), Apache-2.0. The notebook first checks available memory and runs one disposable full-parameter optimizer step at the configured sequence length. If that probe fails, it tries [`HuggingFaceTB/SmolLM2-135M-Instruct`](https://huggingface.co/HuggingFaceTB/SmolLM2-135M-Instruct), also Apache-2.0, **before any primary experiment starts**. All approaches then use that same selected model. The selected model ID, resolved revision, tokenizer, parameter count, precision, and both probe outcomes are written to run metadata. This is a feasibility probe, not a guarantee against later OOM: fragmentation or other Colab processes can still cause memory errors. Runtime OOM is recorded and the experiment stops safely.

The dataset is [`databricks/databricks-dolly-15k`](https://huggingface.co/datasets/databricks/databricks-dolly-15k), a public English instruction dataset under CC BY-SA 3.0. The default run uses at most 1,200 valid records and deterministic train/validation/test partitions. The dataset license requires attribution and share-alike obligations for adapted dataset material; review its terms before redistributing outputs or derived data. Training examples are not placed in the public repository by this project.

## Run in Google Colab

1. Push this repository to GitHub and update the Colab badge above.
2. Open `notebooks/llm_finetuning_benchmark.ipynb` in Colab. Select **Runtime → Change runtime type → T4 GPU** if offered; the code detects the actual GPU and does not assume a T4.
3. Run the setup cell. It installs dependencies in the Colab session, mounts Drive with your consent, and clones your repository into the runtime. Set `REPO_URL` in that cell to your public GitHub clone URL. For a private repository, authenticate through Colab's GitHub flow; never paste a token into the notebook.
4. Run cells in order. First use the default smoke configuration (48 examples). For a longer experiment, edit `configs/training_config.yaml` in GitHub or set the notebook run mode to `full`, then rerun from the data/model setup cell.
5. Results are saved under `MyDrive/llm-finetuning-benchmark/` after each experiment. Download the compact result artifacts or commit selected results after reviewing licenses and outputs. Model checkpoints remain on Drive and are ignored by Git.

The notebook detects GPU name and memory, reports software versions, resolves model/data revisions, reuses Hugging Face's cache where possible, and records the active configuration. Restarting in a fresh session repeats setup and reuses Drive checkpoints and data artifacts when compatible.

PEFT is pinned to a commit containing the upstream fix for Colab's preinstalled older TorchAO package; this project does not use TorchAO quantization. The exact commit is recorded in `requirements.txt` for reproducibility.

## Experiments and measurements

- Baseline generation and held-out evaluation.
- LoRA, 4-bit NF4 QLoRA, and full fine-tuning using shared data and evaluation examples.
- LoRA rank sweep at 4, 8, 16, and 32.
- Normalized exact match and ROUGE-L for generated answers, plus teacher-forced loss where applicable. These metrics measure different properties and are not interchangeable.
- Wall-clock training time, epoch duration, peak allocated GPU memory, trainable parameter count/percentage, saved artifact size, inference latency, and throughput where measurable.
- Representative prompts/responses for manual review and a simple held-out general instruction check.

The general instruction check is a small fixed prompt-only qualitative probe with no reference answers or score. It supports inspection for regressions; it does not establish broad generalization.

Any unavailable measurement is marked missing with a reason; estimated and measured values are distinguished. The split and generation settings are shared. The report is generated from the metrics JSON after runs complete.

## Results

**Pending Colab execution.** See [research report](results/reports/benchmark_report.md), [metrics CSV](results/metrics/benchmark_results.csv), [metrics JSON](results/metrics/benchmark_results.json), and [plots](results/plots/). These paths will contain generated results after a successful run. Do not interpret absent files or pending rows as completed measurements.

## Reproducibility and limitations

Seed: 42. Default dataset cap: 1,200. Default maximum sequence length: 512. Exact package versions, hardware name/memory, resolved revisions, timestamp, split sizes, and effective method configuration are captured per run. Small-model instruction tuning on a small Dolly subset is a systems demonstration, not a general claim about production-scale fine-tuning. Free Colab availability and session limits vary. Full fine-tuning remains conditional on available memory; if the fallback still fails, it is recorded as failed and no result is fabricated.

The software in this repository is MIT licensed. Model and dataset licenses remain separate; the dataset's CC BY-SA 3.0 obligations may apply to redistributed adaptations. Adapter weights are optional Drive artifacts, not included here; check upstream model and data terms before redistributing them. No Git LFS is required.

## Development checks

Lightweight checks cover preprocessing, split disjointness, metrics, serialization, and report generation. They do not download a model or claim Colab GPU execution. See the notebook's smoke workflow for a small end-to-end run. For a true GPU validation, run the notebook in Colab and retain its metadata/results artifacts.

## License

Repository code and documentation: MIT. Model weights: their upstream Apache-2.0 licenses. Dataset: CC BY-SA 3.0. Keep these licenses and attribution notices distinct.
