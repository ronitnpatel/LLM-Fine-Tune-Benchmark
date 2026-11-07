# Experiment design

| Run | Model | Data | Trainable weights | Purpose |
|---|---|---|---|---|
| Baseline | Selected primary model | Shared held-out test | None | Starting point |
| LoRA | Same selected model | Shared train/validation | Adapter only | Parameter-efficient reference |
| QLoRA | Same selected model, 4-bit base | Shared train/validation | Adapter only | Quantization memory tradeoff |
| Full fine-tuning | Same selected model | Shared train/validation | All model weights | Full update reference |
| Rank sweep | Same selected model | Shared train/validation | LoRA r=4,8,16,32 | Adapter capacity sensitivity |

Selection happens before baseline. If full fine-tuning fails the memory preflight, all primary runs use the fallback model; mixed-model rows must never be presented as a head-to-head comparison. A runtime OOM is an explicit failed status. No model/data swap after observing quality results.

Default smoke mode uses 48 training records, one epoch, batch size one, accumulation eight, and a 512 token cap. It is a pipeline check and should not be interpreted as a definitive quality benchmark. Full mode uses the configured data cap (1,200 by default) and epoch count.

Each experiment writes a `running` record before training and updates it to `completed`, `failed`, or `skipped`. A rerun can resume from checkpoints where compatible, while completed result rows remain intact. The summary report and plots are regenerated from completed metric records only and explicitly report the number of completed runs.
