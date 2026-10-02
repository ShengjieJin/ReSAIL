# Run the main experiments

Commands in this guide run from the ReSAIL source root on the **host**. The guarded launcher runs GPU work inside the prepared container. Follow [environment setup](environment.md) and [model/task-data preparation](inputs.md) first.

The public matrix has two tasks (ALFWorld and TextCraft), two model sizes (4B and 8B), and eight methods: `react`, `rft`, `grpo`, `epd`, `sdpo`, `oel`, `resail`, and `sdpo_resail`. ReAct evaluates the base model once. Each trained method completes C1–C3, saves the final checkpoint of each cycle, and evaluates that checkpoint. The [catalog](experiment_catalog.md) lists the supported combinations and run outputs.

W&B logging is offline by default and does not require an account. If you converted a base checkpoint into the checkout, pass its container path with `--base-checkpoint` to both C1 preparation and training, as described in the [environment guide](environment.md#prepare-models-and-tasks).

## Use one shared C1 for a comparison

Collect and verify the task/model C1 corpus once. This is GPU work because collection and guidance use the model:

```bash
bash scripts/launch_experiment.sh alfworld-4b-shared-c1 -- \
  bash scripts/prepare/text_c1.sh \
    --task alfworld --model 4b \
    --c1-input /workspace/slime/data/paper/shared/alfworld/4b/c1
```

Pass that exact path to every method in the comparison. Give each method a distinct run ID:

```bash
bash scripts/launch_experiment.sh alfworld-4b-oel-001 -- \
  bash scripts/experiments/text_cycles.sh \
    --task alfworld --model 4b --method oel --mode paper \
    --c1-input /workspace/slime/data/paper/shared/alfworld/4b/c1

bash scripts/launch_experiment.sh alfworld-4b-resail-001 -- \
  bash scripts/experiments/text_cycles.sh \
    --task alfworld --model 4b --method resail --mode paper \
    --c1-input /workspace/slime/data/paper/shared/alfworld/4b/c1
```

For TextCraft, substitute `--task textcraft` and a path such as `/workspace/slime/data/paper/shared/textcraft/4b/c1`. Use `--model 8b` and the corresponding `8b` path for Qwen3-8B. Shared C1 has 960 ALFWorld or 240 TextCraft trajectories. C2 and C3 collection is separate for each method and uses its preceding verified checkpoint.

## Collect C1 independently for each method

Omit `--c1-input` to collect a fresh, method-private C1 corpus within a new run:

```bash
bash scripts/launch_experiment.sh textcraft-4b-sdpo-resail-001 -- \
  bash scripts/experiments/text_cycles.sh \
    --task textcraft --model 4b --method sdpo_resail --mode paper
```

This mode is useful for running one method from scratch. It does not give two methods identical C1 trajectories, so label the input mode when comparing results. Both C1 modes collect new trajectories from the live task environment.

## Find checkpoints and results

The `paper` mode runs 30 optimizer updates per cycle. The launcher prints the run directory, which defaults to `runs/paper/<task>_<model>_<method>_<mode>_<timestamp>_<pid>`. Use `--run-root` to supply a fresh container path. Each cycle is stored under `<run-root>/<task>/<model>/<method>/c<cycle>/`:

| Path within a cycle | Contents |
| --- | --- |
| `bundle/expanded_config.json` | Training configuration used for the run. |
| `checkpoints/iter_0000029/` | Final Megatron checkpoint in `paper` mode. |
| `eval_hf/iter_0000029/` | Converted final Hugging Face checkpoint. |
| `eval/logs/eval_result/eval_000.json` | Per-seed success counts and evaluation metrics. |
| `eval/samples/eval_0.jsonl` | Evaluation interaction records. |
| `completion.json` | Cycle completion status. |

Paths under `/workspace/slime` correspond to the same paths in the host checkout. Launcher logs are saved separately in `runs/host_guard/paper/<run-id>.jsonl`. ReAct has only `c1/` evaluation outputs and no trained checkpoint.

Evaluation uses each cycle's final checkpoint with decoding seeds 314159, 314160, and 314161. ALFWorld reports ID and OOD results; TextCraft uses its official evaluation split. For each split, report the mean and sample standard deviation of the three per-seed success rates (`success_count / episode_count`).

For a short setup check, use `--mode smoke`; it reduces updates and evaluation episodes while retaining the main batch and parallelism settings. For command options, including base-model paths and resuming a run, see the [main text configuration guide](../configs/main/text/README.md).
