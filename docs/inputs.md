# Models and task data

The source package contains code and configuration templates. Download model weights and task data from their providers into directories mounted by the [container setup](environment.md). Record the model revision and data version used for each run.

## Base models

| Size | Hugging Face model | Default HF path | Default Megatron base path |
| --- | --- | --- | --- |
| 4B | [Qwen/Qwen3-4B](https://huggingface.co/Qwen/Qwen3-4B) | `/root/models/Qwen/Qwen3-4B` | `/root/checkpoints/Qwen3-4B_torch_dist` |
| 8B | [Qwen/Qwen3-8B](https://huggingface.co/Qwen/Qwen3-8B) | `/root/models/Qwen/Qwen3-8B` | `/root/checkpoints/Qwen3-8B_torch_dist` |

The HF directory needs its complete model weights, configuration, and tokenizer. Use the [conversion command](environment.md#prepare-models-and-tasks) when you have HF weights but no matching Megatron base checkpoint. The two paths may be overridden with `--base-hf` and `--base-checkpoint` in the text launcher.

## Released checkpoints

The eight trained Cycle-3 ReSAIL checkpoints are grouped in the [ReSAIL model collection](https://huggingface.co/collections/HuggingJin/resail-6ab8e06426ca99d03aabfa0f). Download a selected repository to the host model directory mounted by the container. For example:

```bash
python3 -m pip install huggingface_hub
export RESAIL_MODEL_DIR=/absolute/path/to/models
python3 - <<'PY'
import os
from pathlib import Path
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="HuggingJin/resail-qwen3-4b-alfworld-oel-c3",
    local_dir=Path(os.environ["RESAIL_MODEL_DIR"]) / "ReSAIL/alfworld-4b-oel-c3",
)
PY
```

Select another repository ID from the collection for a different task, model size, or ReSAIL integration. The downloaded HF weights and tokenizer are separate from the base model. The [conversion tool](environment.md#prepare-models-and-tasks) can make a corresponding Megatron checkpoint when needed. The public experiment runner evaluates the final checkpoints it trains; it does not currently provide a standalone evaluation command for an imported trained checkpoint.

## Task caches

- **ALFWorld:** install `alfworld==0.4.2` through the container and download game data under `.cache/alfworld`. Follow the [ALFWorld task guide](../slime_plugins/agent_tasks/alfworld/README.md).
- **TextCraft:** prepare the official AgentGym recipes and task-ID splits under `.cache/textcraft` with `python3 scripts/prepare/textcraft_data.py`. Follow the [TextCraft task guide](../slime_plugins/agent_tasks/textcraft/README.md). The interactive environment does not require an AgentGym server.

The source package contains neither cache. The C1 collection and later evaluation use the same task cache, so keep it stable for a run.

## Cycle-1 input

There are two supported ways to start trained methods:

1. **Shared C1:** collect and verify one ALFWorld 960-trajectory or TextCraft 240-trajectory corpus for a task and model size, then pass the same `--c1-input` path to every method in that comparison. The run reads the shared corpus without changing it.
2. **Fresh C1 per method:** omit `--c1-input`; each method collects its own C1 corpus into its run directory.

Both modes collect from the live task environment. C2 and C3 collection is separate for each method and uses its preceding cycle's verified checkpoint. See the [run commands](reproduction.md) for both modes.
