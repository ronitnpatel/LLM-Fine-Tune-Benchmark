# LLM Fine Tuning Benchmark Lab

A comparison of full fine tuning, LoRA, and QLoRA on one instruction model and one test split.

> **Results available.** The checked in model and dataset defaults match the completed full Colab run. It used Qwen2.5-0.5B-Instruct and 1,200 Dolly rows. Results and caveats appear below and in the research report.

## Research questions

1. How do the methods compare on test answer quality and validation loss?
2. What are their measured training time, peak memory, trainable parameter share, and artifact size?
3. How sensitive is LoRA quality and efficiency to ranks 4, 8, 16, and 32?
4. Do the models retain reasonable behavior on the test set, and what qualitative changes are visible?

## Experiment design and methodology

Each method uses the same Qwen2.5-0.5B-Instruct model, pinned model and dataset revisions, seeded data split, prompt format, and evaluation prompts.

| Experiment | Data used | Parameters updated | Purpose |
|---|---|---|---|
| Baseline | Test split | None | Measure the base model before training |
| LoRA | Shared training and validation splits | Adapter weights | Compare fine tuning with fewer updated parameters |
| QLoRA | Same splits with a four bit NF4 base model | Adapter weights | Measure quantized adapter training |
| Full fine tuning | Same splits | All model weights | Compare with updating the full model |
| LoRA rank sweep | Same splits, ranks 4, 8, 16, and 32 | Adapter weights | See how adapter rank relates to quality and cost |

The full configuration caps the dataset at 1,200 valid Dolly records. After deterministic shuffling, these are split into 960 training, 120 validation, and 120 test examples. The test split is used only for evaluation. Smoke mode uses 48 training examples from a split of 60 records to check the pipeline. It is not the reported full benchmark.

Before the baseline, a GPU preflight attempts one disposable optimizer step with all model parameters. If it fails, the notebook saves a diagnostic and stops before comparing methods. This checks hardware feasibility but cannot guarantee that a later run will avoid memory errors.

Training uses causal language modeling with loss applied to answer tokens. Prompt tokens provide context but are masked from the loss. QLoRA uses NF4 four bit quantization and double quantization. Evaluation includes normalized exact match, ROUGE-L F1, and validation or test loss for individual tokens where available. Resource measurements include training time, peak allocated GPU memory, trainable parameter share, saved artifact size, and inference speed where measurable. Exact match and ROUGE-L measure text overlap and do not verify factual correctness. The general instruction probe has four prompts and no score.

## Repository layout

```text
configs/                 Model, data, and training configuration
notebooks/               Orchestration notebook
src/                     Reusable data, model, training, evaluation, and reporting code
results/                 Metrics, plots, and generated report
scripts/                 Report generation entry point
```

## Models and data

The configured comparison model is [`Qwen/Qwen2.5-0.5B-Instruct`]. All methods use this model. Before the baseline, the notebook checks available memory and runs one disposable optimizer step with all model parameters at the configured sequence length. If the probe fails, it records the hardware diagnostic and stops. The selected model ID, resolved revision, tokenizer, parameter count, and precision are written to the run metadata. This feasibility probe cannot guarantee a later run will avoid memory errors caused by fragmentation or other Colab processes.

The dataset is [`databricks/databricks-dolly-15k`]. The default full run uses at most 1,200 valid records. The train, validation, and test partitions contain 960, 120, and 120 records.

## Experiments and measurements

- Baseline generation and test evaluation.
- LoRA, four bit NF4 QLoRA, and full fine tuning using shared data and evaluation examples.
- LoRA rank sweep at 4, 8, 16, and 32.
- Normalized exact match and ROUGE-L for generated answers, plus teacher forced loss where applicable. These metrics measure different properties and are not interchangeable.
- Training time, epoch duration, peak allocated GPU memory, trainable parameter count and percentage, saved artifact size, inference latency, and throughput where measurable.
- Representative prompts and responses for manual review and a simple test set instruction check.

The general instruction check is a small fixed prompt only qualitative probe with no reference answers or score. It supports inspection for regressions. It does not establish broad generalization.

## Results

The full run used Qwen2.5-0.5B-Instruct and 1,200 Dolly records. The split contained 960 train, 120 validation, and 120 test examples. Training ran for one epoch on a Tesla T4. These metrics were observed.

| Experiment | Exact match | ROUGE-L F1 | Test loss | Training time | Peak GPU memory | Trainable parameters |
|---|---:|---:|---:|---:|---:|---:|
| Baseline | 1.7% (2/120) | 0.192 | 2.290 | Not measured | Not measured | 0% |
| LoRA (r=8) | 4.2% (5/120) | 0.316 | 1.760 | 7.8 min | 1.89 GiB | 0.883% |
| QLoRA (r=8) | 5.0% (6/120) | 0.308 | 1.812 | 9.6 min | 1.93 GiB | 1.377% |

The rank sweep produced ROUGE-L F1 scores of 0.311 at rank 4, 0.317 at rank 16, and 0.313 at rank 32. The differences are small and do not establish a best rank. Full fine tuning test evaluation and saved artifact remain in the [research report](results/reports/benchmark_report.md).

The test set of 120 examples, one seed, and one epoch.
See [research report](results/reports/benchmark_report.md), [metrics CSV](results/metrics/benchmark_results.csv), [metrics JSON](results/metrics/benchmark_results.json), and [plots](results/plots/).