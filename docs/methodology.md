# Methodology

## Comparison design

The primary comparison uses one model identifier selected before any experiments, a fixed dataset snapshot/revision, deterministic seeded split, common prompt serialization, the same effective training examples and sequence limit, and common held-out prompts and decoding settings. Baseline and adapted models receive identical generation prompts. The notebook records both the configured plan and the actual run state.

The primary model is Qwen2.5-0.5B-Instruct. A GPU preflight checks available memory against a conservative configurable full-tuning threshold. If the preflight fails, SmolLM2-135M-Instruct is selected before the baseline. This is a feasibility heuristic, not a guarantee: optimizer state, activations, fragmentation, and Colab runtime differences can still cause OOM. An OOM is recorded and the experiment stops safely.

## Data

The initial dataset is Databricks Dolly 15k (CC BY-SA 3.0). Invalid rows (empty instruction or response, non-string fields after normalization) are discarded. Context is added to the instruction when nonempty. After a seeded shuffle and cap, examples are partitioned into train, validation, and test splits. The test set is never passed to the trainer. Split IDs and dataset revision are stored in run metadata so methods share the exact partition. A small smoke subset is drawn only from training data; validation and test remain held out.

The repository does not redistribute dataset rows. Cite Databricks and observe the dataset's attribution and ShareAlike terms when sharing adaptations or derived material.

## Training

All methods use causal language modeling with the answer portion supervised. LoRA and QLoRA freeze base parameters and report trainable counts. QLoRA uses NF4, double quantization, and a compute dtype selected from detected device support. Full fine-tuning updates all parameters and enables gradient checkpointing where supported. Batch size and accumulation are chosen conservatively for small free GPUs; the effective batch and actual configuration are recorded.

The default notebook mode is a smoke run intended to validate the pipeline, not to support strong quality conclusions. Increase the data cap/epochs only after successful smoke execution. Training checkpoints, logs, and per-experiment metrics are written incrementally to Drive.

## Metrics

- **Normalized exact match**: fraction of normalized generated answers identical to references. Strict and sensitive to valid paraphrases.
- **ROUGE-L F1**: longest-common-subsequence overlap. It is lexical overlap, not factual correctness or semantic equivalence.
- **Validation/test loss**: token-level teacher-forced cross entropy under the common formatting, when computable. Compare within the same model/tokenization and masking setup.
- **Training time / peak allocated memory**: measured from the active Colab process. They are affected by hardware contention and runtime state.
- **Trainable parameters**: count and percentage of the selected model's parameters with gradients enabled.
- **Artifact size**: bytes written for adapter or model output. It does not include the separately cached base model unless explicitly stated.
- **Latency and throughput**: wall-clock generation measurements on the same fixed prompt batch and decoding configuration. They are descriptive, not hardware-independent.

Qualitative examples aid inspection but are not a blinded human evaluation. No quality improvement is claimed without measured results and suitable caveats.

The notebook also runs a fixed four-prompt, prompt-only qualitative probe (brief explanation, constrained summary, polite decline, and a simple programming distinction) on the baseline and each trained model. This probe has no reference answers and is not scored; it is saved separately to help inspect broad behavior changes.

## Reproducibility

The notebook records random seeds, package versions, hardware, model and dataset revisions, tokenizer/model configuration, split counts, training arguments, quantization settings, generation settings, timestamp, status, and failure reason. It stores intermediate artifacts on Drive. Hugging Face revisions default to `main`; each actual resolved commit is recorded. For exact reruns, use the stored commit IDs in the config.
