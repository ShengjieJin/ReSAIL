#!/usr/bin/env python3
"""Construct main C1 text datasources from newly prepared shared inputs."""

from __future__ import annotations

import functools
import argparse
import tempfile
from pathlib import Path
from types import SimpleNamespace

from exp.paper.text import ROOT, build_config
from slime_plugins.agent_tasks.alfworld.frozen import data_source as alf_source
from slime_plugins.agent_tasks.common.frozen import data_source as text_source


def _arguments(config: dict) -> SimpleNamespace:
    cli = config["cli_args"]
    values: dict[str, object] = {}
    index = 0
    while index < len(cli):
        key = str(cli[index]).removeprefix("--").replace("-", "_")
        if index + 1 < len(cli) and not str(cli[index + 1]).startswith("--"):
            values[key] = cli[index + 1]
            index += 2
        else:
            values[key] = True
            index += 1
    values.update(config["custom_config"])
    return SimpleNamespace(**values)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, default=ROOT / "data/paper")
    args_cli = parser.parse_args()
    # The same immutable shards feed several methods. Cache the decoded rows for
    # this read-only CPU preflight so each corpus is loaded only once.
    alf_source.load_frozen_trajectory_shard = functools.lru_cache(maxsize=None)(
        alf_source.load_frozen_trajectory_shard)
    text_source.load_shard = functools.lru_cache(maxsize=None)(text_source.load_shard)
    total = 0
    with tempfile.TemporaryDirectory(prefix="resail_text_sources_") as temporary:
        for task in ("alfworld", "textcraft"):
            for model in ("4b", "8b"):
                shared = args_cli.input_root / "shared" / task / model / "c1"
                if not (shared / "verification.json").is_file():
                    raise FileNotFoundError(f"prepare shared C1 first: {shared}")
                methods = ("rft", "grpo", "sdpo", "oel", "resail", "sdpo_resail")
                for method in methods:
                    config = build_config(
                        task=task, model=model, method=method, cycle=1, phase="training",
                        run_root=Path(temporary), input_root=args_cli.input_root,
                        base_checkpoint=Path(f"/root/checkpoints/Qwen3-{model.upper()}_torch_dist"),
                        base_hf=Path(f"/root/models/Qwen/Qwen3-{model.upper()}"),
                        mode="paper", c1_input=shared)
                    args = _arguments(config)
                    source = (alf_source.FrozenAlfWorldDataSource(args) if task == "alfworld"
                              else text_source.FrozenDataSource(args))
                    eligible = len(source._locations if task == "alfworld" else source._eligible)
                    scheduled = getattr(source, "_shared_schedule_indices", None)
                    if eligible == 0 or scheduled is not None:
                        raise RuntimeError(f"C1 source has no eligible rows or unexpectedly uses a schedule: {task}/{model}/{method}")
                    print(f"PASS {task} {model} {method}: eligible={eligible} "
                          f"scheduled={len(scheduled) if scheduled is not None else 'none'}", flush=True)
                    total += 1
    print(f"Verified {total} C1 datasource configurations. EPD requires materialized targets.")


if __name__ == "__main__":
    main()
