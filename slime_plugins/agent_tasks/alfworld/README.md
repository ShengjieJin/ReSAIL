# ALFWorld task

The ALFWorld adapter runs text-mode episodes inside slime. The game data comes from the [official ALFWorld repository](https://github.com/alfworld/alfworld). After [container setup](../../../docs/environment.md), run these commands from the repository root on the host:

```bash
bash scripts/run_in_container.sh env ALFWORLD_DATA=/workspace/slime/.cache/alfworld \
  alfworld-download -f
```

Run the paper cycles:

```bash
bash scripts/launch_experiment.sh alfworld-4b-resail-001 -- \
  bash scripts/experiments/text_cycles.sh \
    --task alfworld --model 4b --method resail --mode paper
```

Each cycle trains for 30 updates using 960 scheduled trajectories, 32 per update, with collection seeds 42/43/44 and training seed 1234. C1 is collected fresh by default. To compare methods against one shared C1 corpus, prepare it once with `scripts/prepare/text_c1.sh` and pass the same `--c1-input` to each run. Final checkpoints are evaluated on both in-distribution and out-of-distribution task panels with seeds 314159/314160/314161; no best-checkpoint selection is used. Thinking is disabled. The main ReSAIL arm uses global response-step Top-5% selection and retention weight 0.5.

`text_cycles.sh` accepts `--model 8b` and `--method react|rft|grpo|epd|sdpo|oel|sdpo_resail`. ReAct evaluates the base checkpoint once; the other methods complete three cycles. `--mode smoke` preserves the batch and parallelism with two updates and a smaller evaluation panel.

The adapter uses `common.data_source`, `common.projection`, `common.eval`, and `common.logging`; frozen-corpus behavior is under `frozen/`.
