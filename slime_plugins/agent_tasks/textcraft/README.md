# TextCraft task

The TextCraft adapter runs episodes inside slime. It reads Minecraft recipes and the official AgentGym-RL task-ID splits from `.cache/textcraft` by default; it does not need an AgentGym server.

After [container setup](../../../docs/environment.md), run the following commands from the repository root on the host. The preparation script downloads the [official AgentGym source](https://github.com/WooooDyy/AgentGym) and [task-ID dataset](https://huggingface.co/datasets/AgentGym/AgentGym-RL-Data-ID):

```bash
bash scripts/run_in_container.sh python3 scripts/prepare/textcraft_data.py
```

The script prepares 860 recipe files, 374 training IDs, and 100 evaluation IDs under `.cache/textcraft` in this checkout. To reuse an existing AgentGym checkout, add `--source-dir` with its path inside the container.

Run the paper cycles:

```bash
bash scripts/launch_experiment.sh textcraft-4b-resail-001 -- \
  bash scripts/experiments/text_cycles.sh \
    --task textcraft --model 4b --method resail --mode paper
```

C1 is collected fresh by default. To compare methods against one shared C1 corpus, prepare it once with `scripts/prepare/text_c1.sh` and pass the same `--c1-input` to each run. Each cycle performs 30 optimizer updates from 240 trajectories, with 8 source trajectories per update, seeds 42/43/44 for collection, training seed 1234, and final-checkpoint evaluation on the official 100 tasks with seeds 314159/314160/314161. Thinking is disabled. The main TextCraft ReSAIL arm uses global response-step Top-25% selection and retention weight 1.0.

`text_cycles.sh` also accepts `--model 8b` and `--method react|rft|grpo|epd|sdpo|oel|sdpo_resail`. ReAct evaluates the base checkpoint once; the other methods complete three cycles. `--mode smoke` keeps the batch and parallelism but uses two updates and an eight-task evaluation panel.

The adapter uses `common.data_source`, `common.projection`, `common.eval`, and `common.logging`; frozen-corpus behavior is under `frozen/`.
