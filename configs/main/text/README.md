# Text main experiments

Run inside the ReSAIL container after preparing the ALFWorld and TextCraft task data and converting the Qwen3 base model. Each trained method performs 30 updates per cycle, then evaluates the final checkpoint. ReAct evaluates the base model once.

To collect one C1 input for several methods, run:

```bash
bash scripts/prepare/text_c1.sh --task alfworld --model 4b \
  --c1-input /workspace/slime/data/paper/shared/alfworld/4b/c1
bash scripts/experiments/text_cycles.sh --task alfworld --model 4b \
  --method resail --mode paper \
  --c1-input /workspace/slime/data/paper/shared/alfworld/4b/c1
```

Use the same C1 path for `rft`, `grpo`, `epd`, `sdpo`, `oel`, and `sdpo_resail`. The shared input contains 960 ALFWorld or 240 TextCraft trajectories plus guidance from the base model. Each method reads it without rewriting its manifest; C2 and C3 collect their own trajectories from the prior cycle's final checkpoint. Omit `--c1-input` to collect a separate fresh C1 for one method. Run `--method react` for the base-model evaluation.

Change `--task` to `textcraft` and `--model` to `8b` as needed. The default base paths are `/root/checkpoints/Qwen3-{4B|8B}_torch_dist` and `/root/models/Qwen/Qwen3-{4B|8B}`; supply `--base-checkpoint` and `--base-hf` to use other matching base artifacts. `--mode smoke --smoke-updates 2` reduces updates and evaluation episodes while keeping the training batch and parallelism. `--run-root` names a fresh run; `--resume` verifies and continues only that run identity.

Both launchers accept `--wandb-mode offline|online|disabled`. The default is `offline`; `online` requires W&B credentials in the container. Logging mode does not change the training or evaluation configuration.
