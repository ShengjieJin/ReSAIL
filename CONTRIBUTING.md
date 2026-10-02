<!-- Modified for ReSAIL. See NOTICE and LICENSE for attribution and terms. -->

# Contributing

Experiment configurations are in [`configs/main/text/`](configs/main/text/), and the launcher is [`exp/paper/text.py`](exp/paper/text.py). See the [code map](docs/method.md#code-map) for task adapters and ReSAIL components.

After preparing the [environment](docs/environment.md), check configuration changes with:

```bash
bash scripts/run_in_container.sh env CUDA_VISIBLE_DEVICES= \
  bash scripts/check_public_contracts.sh
```

For the full test suite, prepare the task caches and a local Qwen3 tokenizer, then run:

```bash
bash scripts/run_in_container.sh env CUDA_VISIBLE_DEVICES= HF_HUB_OFFLINE=1 \
  RESAIL_TEST_TOKENIZER=/root/models/Qwen/Qwen3-4B \
  python3 -m pytest -q tests
```

For changes to GPU execution, run a short experiment with `--mode smoke` through [`scripts/launch_experiment.sh`](scripts/launch_experiment.sh). Describe the change and the checks performed in your pull request. Preserve the [license](LICENSE) and [upstream attribution](NOTICE).
