# Fine Tuning Benchmark Report

> Generated from the recorded Colab run `83940cef498c`. Measurements are reported as recorded. Limitations that affect interpretation are included below.


## Executive summary

This full run completed 7 experiment records on `Qwen/Qwen2.5-0.5B-Instruct` using 1200 selected Dolly examples and a test split of 120 prompts. On this split, `lora_rank_16` had the highest ROUGE-L F1 at 0.317. LoRA and QLoRA both exceeded the baseline on ROUGE-L. Normalized exact match remained low, with 2 to 6 matches out of 120. These results from one run do not establish statistical significance or broad generalization.

The full fine tuning result resumed a checkpoint. Its quality evaluation is available, but its recorded training time and peak memory are not valid measurements from a fresh training run. The completed LoRA and QLoRA rows have uninterrupted training measurements.

## Research objective and background

This benchmark compares three update strategies on one base model and one data split. LoRA freezes base weights and learns low rank adapter matrices. QLoRA combines LoRA with a four bit NF4 quantized base and double quantization. Full fine tuning updates all model parameters. The project measures test answer quality, training time, GPU memory, trainable parameter share, and saved artifact size for each approach.

## Model and dataset

- **Model** `Qwen/Qwen2.5-0.5B-Instruct`
- **Resolved model revision** `7ae557604adf67be50417f59c2c2f167def9a775`
- **Tokenizer** `Qwen/Qwen2.5-0.5B-Instruct`
- **Parameter count** 494,032,768
- **Recorded inference and model dtype** `torch.bfloat16`
- **Dataset** `databricks/databricks-dolly-15k` at revision `bdd27f4d94b9c1f951818a7da7fd7aeea5dbff1a` under CC BY-SA 3.0
- **Data split** 1200 rows total, with 960 train, 120 validation, and 120 test records. The seed was 42.

All methods use the same splits, and test prompts are not used for training. The model uses Apache 2.0. The dataset has separate attribution and ShareAlike terms. Review the upstream dataset terms before redistributing adapted material.

## Experimental methodology

Training used one epoch, batch size 1, gradient accumulation 8, learning rate 0.0002, maximum sequence length 512, seed 42, and greedy generation with up to 96 new tokens. LoRA ranks were 4, 8, 16, and 32. Rank 8 is the main LoRA and QLoRA comparison. QLoRA used NF4 four bit quantization with double quantization set to `True`. LoRA alpha was 16 with dropout 0.05.

The baseline and all trained models used the same test set of 120 prompts. A separate general instruction probe has four prompts and no reference answers. It is qualitative only.

## Colab environment

- **GPU** Tesla T4 with 14.56 GiB total
- **Python** 3.13.16
- **Packages** Torch 2.14.1+cu130, Transformers 4.57.6, PEFT 0.21.3.dev0, Datasets 4.8.5, Accelerate 1.15.0, and bitsandbytes 0.50.2
- **Timestamp in UTC** 2026-11-10T16:37:58.899185+00:00

## Quantitative results

| Experiment | Exact match (hits/n) | ROUGE-L F1 | Validation loss | Test loss | Training time | Peak GPU memory | Trainable params | Saved artifact |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline | 0.017 (2/120) | 0.192 | Not measured | 2.290 | Not measured | Not measured | 0.000% | Not measured |
| lora | 0.042 (5/120) | 0.316 | 1.760 | 1.760 | 7.8 min | 1.89 GiB | 0.883% | 32.0 MiB |
| qlora | 0.050 (6/120) | 0.308 | 1.831 | 1.812 | 9.6 min | 1.93 GiB | 1.377% | 32.0 MiB |
| full_finetuning | 0.042 (5/120) | 0.263 | 2.615 | 2.500 | Not comparable (resumed checkpoint) | Not comparable (resumed checkpoint) | 100.000% | 957.5 MiB |
| lora_rank_4 | 0.042 (5/120) | 0.311 | 1.761 | 1.759 | 7.6 min | 1.86 GiB | 0.443% | 23.6 MiB |
| lora_rank_16 | 0.042 (5/120) | 0.317 | 1.757 | 1.760 | 7.5 min | 1.96 GiB | 1.750% | 48.8 MiB |
| lora_rank_32 | 0.042 (5/120) | 0.313 | 1.755 | 1.758 | 7.5 min | 2.09 GiB | 3.439% | 82.3 MiB |

All 7 experiment records for this run completed. (`{"completed": 7}`)

Normalized exact match is the fraction of outputs that become identical after lowercasing, removing punctuation and articles, and collapsing whitespace. It can penalize valid paraphrases. ROUGE-L F1 measures word sequence overlap and does not measure factual correctness. Validation and test loss use token level cross entropy with the shared prompt and label masking setup. Exact match percentages here represent very small counts. The baseline had 2 matches out of 120, LoRA had 5, QLoRA had 6, and full fine tuning had 5.

The full fine tuning row resumed a checkpoint (`checkpoint_resumed=true`). Its current invocation took 30.9 seconds from start to finish, but Trainer recorded only 0.0065 seconds of training, zero training loss, and no epoch timer. Its training time, training loss, and peak memory are not compared with fresh runs. Its validation and test evaluation and saved artifact remain as measurements of the resumed checkpoint.

The best rank sweep ROUGE-L was 0.317 at rank 16. Rank 4 reached 0.311 and rank 32 reached 0.313. The differences are small and do not establish a reliable best rank from one seed and one split.

## Visualizations

- [artifact_size_comparison.png](../../plots/artifact_size_comparison.png)
- [gpu_memory_comparison.png](../../plots/gpu_memory_comparison.png)
- [lora_rank_performance.png](../../plots/lora_rank_performance.png)
- [quality_comparison.png](../../plots/quality_comparison.png)
- [trainable_parameter_percentage.png](../../plots/trainable_parameter_percentage.png)
- [training_time_comparison.png](../../plots/training_time_comparison.png)

The training time and GPU memory charts omit the resumed full fine tuning row because it did not record a fresh training pass. The artifact size chart includes its saved full model artifact.

## Qualitative examples

**Prompt from the test set** Identify which car manufacturer is French or American: Citroen, Chrysler

**Reference answer** Citroen is French, Chrysler is American

- **baseline** To determine whether the car manufacturers Citroen and Chrysler are French or American, I'll analyze their names: 1. **Citroën**: This brand was originally founded in France but has been incorporated into the United States since 2004. 2. **Chrysler**: This brand was also originally founded in France but has been incorporated into the Uni…

- **lora** Chrysler is American, Citroen is French.

- **qlora** Chrysler is American and Citroen is French.

- **full_finetuning** Chrysler is French, Citroen is American

**Prompt only probe** Summarize the following idea in exactly three bullet points: A team should define its goal, divide the work into clear tasks, check progress regularly, and revise the plan when new information appears.

- **baseline** A team can effectively manage their workload by defining clear goals, breaking down tasks into manageable steps, ensuring regular updates on progress, and revising plans as needed based on new information or changes to the project scope.

- **lora** A team should define their goals, break down tasks into smaller pieces, monitor progress regularly, and make changes to plans as needed.

- **qlora** A team should define its goal, divide the work into clear tasks, check progress regularly, and revise the plan when new information appears.

- **full_finetuning** 1. Define your team's goal. 2. Divide the work into clear tasks. 3. Check progress regularly. 4. Revise the plan when new information appears.

The fixed open ended probe has no reference answers. Outputs are included for inspection and are not scored. It exposed factual errors and failures to follow instructions, particularly for full fine tuning. No broad improvement in generalization is claimed.

## Discussion and recommendations

- On this run, LoRA rank 8 reached ROUGE-L F1 of 0.316. QLoRA reached 0.308 and the baseline reached 0.192. These are descriptive results from one test split.
- QLoRA's measured peak allocation (1.93 GiB) was close to LoRA's (1.89 GiB) in this run. Do not claim a memory advantage from these measurements alone.
- Adapter artifacts are much smaller than the full model artifact. The full model artifact is 957.5 MiB. LoRA rank 8 and QLoRA rank 8 are each 32.0 MiB. These sizes exclude the cached base model.
- Full fine tuning had worse test loss and lower ROUGE-L than the adapters in this run. Missing fresh training measurements and a resumed checkpoint limit comparisons of resource use.
- One test set of 120 examples, one seed, one epoch, and a small model and data subset are not enough to establish robust generalization. Repeat runs and a stronger task specific or human evaluation before drawing research conclusions.

## Limitations and conclusion

This is a small scale systems experiment. Dolly has limitations as a source dataset. Exact match and ROUGE-L do not verify factuality. The prompt only probe has four examples, and the project did not run multiple seeds or calculate confidence intervals. Run metadata and selected qualitative excerpts are preserved so these limitations can be reviewed. Full sample dumps contain dataset prompts and reference answers and remain local. Conclusions should stay limited to this measured configuration.
