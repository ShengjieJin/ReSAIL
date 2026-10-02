#!/usr/bin/env python3
"""Collect trajectories, train, and evaluate ALFWorld and TextCraft cycles.

Experiment templates are loaded from configs/main/text.
"""

from __future__ import annotations

import argparse
import ast
import json
import math
import os
import re
import shutil
import statistics
import subprocess
import time
import urllib.request
from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[2]
RUNTIME_ROOT = Path(os.environ.get("RESAIL_RUNTIME_ROOT", "/workspace/slime"))
TEMPLATES = ROOT / "configs/main/text"
TASK_SEEDS = {1: 42, 2: 43, 3: 44}
EVAL_SEEDS = (314159, 314160, 314161)
METHODS = ("react", "rft", "grpo", "epd", "sdpo", "oel", "resail", "sdpo_resail")
GUIDANCE_METHODS = {"epd", "sdpo", "oel", "resail", "sdpo_resail"}
SDPO_LOSS_METHODS = {"sdpo", "oel", "resail", "sdpo_resail"}
PAIRED_METHODS = GUIDANCE_METHODS
STRICT_ON_POLICY_METHODS = {"grpo", "oel", "resail"}
TASK_BATCH = {"alfworld": 32, "textcraft": 8}
TASK_TRAJECTORIES = {"alfworld": 960, "textcraft": 240}
TASK_RHO = {"alfworld": 0.05, "textcraft": 0.25}
def selection_settings(task: str) -> tuple[float, float]:
    return TASK_RHO[task], 1.0 if task == "textcraft" else 0.5


def runtime(path: Path) -> str:
    path = path.resolve()
    try:
        return str(RUNTIME_ROOT / path.relative_to(ROOT))
    except ValueError:
        return str(path)


def set_option(cli: list[Any], option: str, value: Any) -> None:
    while option in cli:
        index = cli.index(option)
        del cli[index : index + 2]
    cli.extend((option, str(value)))


def drop_option(cli: list[Any], option: str) -> None:
    while option in cli:
        index = cli.index(option)
        del cli[index : index + 2]


def set_flag(cli: list[Any], option: str, enabled: bool) -> None:
    while option in cli:
        cli.remove(option)
    if enabled:
        cli.append(option)


def option(cli: list[Any], name: str) -> str:
    return str(cli[cli.index(name) + 1])


def _replace(value: Any, replacements: dict[str, str]) -> Any:
    if isinstance(value, str):
        for old, new in replacements.items():
            value = value.replace(old, new)
        return value
    if isinstance(value, list):
        return [_replace(item, replacements) for item in value]
    if isinstance(value, dict):
        return {key: _replace(item, replacements) for key, item in value.items()}
    return value


def cell_root(run_root: Path, task: str, model: str, method: str, cycle: int) -> Path:
    return run_root / task / model / method / f"c{cycle}"


def c1_input_root(input_root: Path, task: str, model: str) -> Path:
    return input_root / "shared" / task / model / "c1"


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def prior_paths(run_root: Path, task: str, model: str, method: str, cycle: int,
                base_checkpoint: Path, base_hf: Path) -> tuple[Path, Path]:
    if cycle == 1:
        return base_checkpoint, base_hf
    parent = cell_root(run_root, task, model, method, cycle - 1)
    for name, status in (("completion.json", "complete"), ("verification.json", "verified")):
        path = parent / name
        if not path.is_file() or _read_json(path).get("status") != status:
            raise RuntimeError(f"parent cycle has no verified result: {path}")
    iteration = str(_read_json(parent / "completion.json").get("final_checkpoint", ""))
    if not re.fullmatch(r"iter_\d{7}", iteration):
        raise RuntimeError(f"parent final iteration is invalid: {parent}")
    checkpoint = parent / "checkpoints" / iteration
    hf = parent / "eval_hf" / iteration
    if not checkpoint.is_dir() or not (hf / "config.json").is_file():
        raise RuntimeError(f"parent cycle checkpoint is unavailable: {parent}")
    return checkpoint.parent, hf


def build_config(*, task: str, model: str, method: str, cycle: int, phase: str,
                 run_root: Path, input_root: Path, base_checkpoint: Path, base_hf: Path,
                 mode: str, c1_input: Path | None = None, smoke_updates: int = 2,
                 cell_override: Path | None = None, wandb_mode: str = "offline") -> dict[str, Any]:
    if task not in TASK_TRAJECTORIES or model not in {"4b", "8b"} or method not in METHODS:
        raise ValueError("this entry supports ALFWorld/TextCraft 4B and 8B text methods")
    selection_fraction, retention_weight = selection_settings(task)
    if cycle not in TASK_SEEDS or phase not in {"collection", "training", "eval"}:
        raise ValueError("invalid cycle or phase")
    if method == "react" and (cycle != 1 or phase != "eval"):
        raise ValueError("ReAct is a base-model evaluation with one endpoint")
    if mode not in {"paper", "smoke"}:
        raise ValueError("invalid protocol")
    if wandb_mode not in {"offline", "online", "disabled"}:
        raise ValueError("invalid W&B mode")
    if smoke_updates < 1 or smoke_updates >= 30:
        raise ValueError("smoke updates must be between 1 and 29")
    cell = cell_override or cell_root(run_root, task, model, method, cycle)
    parent = cell_root(run_root, task, model, method, cycle - 1)
    shared = c1_input if cycle == 1 and c1_input is not None else cell / "collection"
    corpus = shared / "corpus" if cycle == 1 else cell / "collection/corpus"
    guidance = shared / "guidance" if cycle == 1 else cell / "collection/guidance"
    prior, prior_hf = prior_paths(run_root, task, model, method, cycle, base_checkpoint, base_hf)
    template_method = method if phase == "training" or method == "react" else phase
    template_phase = "train" if phase == "training" else phase
    template = TEMPLATES / f"{task}_{model}_{template_method}_{template_phase}.json"
    replacements = {
        "@REPO@": str(RUNTIME_ROOT),
        "@CELL@": runtime(cell),
        "@PARENT@": runtime(parent),
        "@CORPUS@": runtime(corpus),
        "@GUIDANCE@": runtime(guidance),
        "@SHARED@": runtime(shared),
    }
    config = _replace(_read_json(template), replacements)
    cli, custom = config["cli_args"], config["custom_config"]
    set_option(cli, "--load", runtime(prior if phase != "eval" else cell / "checkpoints"))
    set_option(cli, "--ref-load", runtime(prior))
    set_option(cli, "--hf-checkpoint", runtime(prior_hf if phase != "eval" else cell / "eval_hf/iter_0000029"))
    set_option(cli, "--rollout-seed", TASK_SEEDS[cycle])
    label = f"paper-{run_root.name}-{task}-{model}-{method}-c{cycle}-{phase}-{mode}"
    set_option(cli, "--wandb-exp-name", label)
    set_option(cli, "--wandb-run-id", label)
    set_option(cli, "--wandb-dir", runtime(cell / phase / "wandb"))
    set_option(cli, "--wandb-mode", wandb_mode)
    if phase == "collection":
        owner = shared if cycle == 1 else cell / "collection"
        set_option(cli, "--rollout-seed", TASK_SEEDS[cycle])
        if task == "alfworld":
            custom["alfworld_frozen_corpus_dir"] = runtime(owner / "corpus")
            custom["alfworld_sample_log_dir"] = runtime(owner / "samples")
            custom["alfworld_frozen_behavior_model"] = (
                f"qwen3-{model}-base" if cycle == 1 else f"paper/{task}/{model}/{method}/c{cycle - 1}"
            )
        else:
            custom["agent_frozen_corpus_dir"] = runtime(owner / "corpus")
            custom["agent_frozen_expected_trajectories"] = 240
            custom["agent_frozen_sampling_seed_namespace"] = (
                f"textcraft-{model}-shared-cycle-1" if cycle == 1 and cell_override is not None
                else f"textcraft-{model}-{method}-cycle-{cycle}")
            custom["textcraft_sample_log_dir"] = runtime(owner / "samples")
            custom["agent_frozen_behavior_model"] = (
                f"qwen3-{model}-base" if cycle == 1 else f"paper/{task}/{model}/{method}/c{cycle - 1}"
            )
    elif phase == "training":
        updates = 30 if mode == "paper" else smoke_updates
        final = updates - 1
        set_option(cli, "--num-rollout", updates)
        set_option(cli, "--checkpoint-fixed-iterations", final)
        custom["checkpoint_fixed_iterations"] = [final]
        custom["alfworld_frozen_corpus_dir" if task == "alfworld" else "agent_frozen_corpus_dir"] = runtime(corpus)
        if method in GUIDANCE_METHODS:
            custom["agent_frozen_guidance_summary_dir"] = runtime(guidance)
            custom["agent_frozen_guidance_summary_model_path"] = runtime(prior_hf)
        if task == "alfworld":
            custom["alfworld_frozen_expected_trajectories"] = 960
            if method in GUIDANCE_METHODS:
                custom["agent_frozen_expected_guidance_trajectories"] = 960
            custom["alfworld_frozen_source_audit_dir"] = runtime(cell / "logs/source_audit")
            custom["alfworld_frozen_pair_audit_dir"] = runtime(cell / "logs/pair_audit") if method in PAIRED_METHODS else None
            custom["alfworld_action_match_audit_dir"] = runtime(cell / "logs/action_match_audit")
        else:
            custom["agent_frozen_expected_trajectories"] = 240
            if method in GUIDANCE_METHODS:
                custom["agent_frozen_expected_guidance_trajectories"] = 240
            # The method-specific namespace determines sampling seeds in C2/C3.
            custom["agent_frozen_sampling_seed_namespace"] = f"textcraft-{model}-{method}-cycle-{cycle}"
            custom["agent_frozen_source_audit_dir"] = runtime(cell / "logs/source_audit")
            custom["agent_frozen_pair_audit_dir"] = runtime(cell / "logs/pair_audit") if method in PAIRED_METHODS else None
            custom["textcraft_sample_log_dir"] = runtime(cell / "samples")
        custom["strict_on_policy_audit_dir"] = runtime(cell / "logs/strict_on_policy") if method in STRICT_ON_POLICY_METHODS else None
        custom["sgs_audit_dir"] = runtime(cell / "logs/sgs") if method in {"resail", "sdpo_resail"} else None
        if method == "epd":
            target = cell / "epd_targets"
            verification = target / "verification.json"
            if not verification.is_file():
                raise FileNotFoundError(f"EPD targets must be materialized before training bundle: {verification}")
            verified = _read_json(verification)
            if verified.get("status") != "verified" or verified.get("binding_mode") != "direct_v1":
                raise RuntimeError(f"EPD target verification is invalid: {verification}")
            teacher_iteration = -1 if cycle == 1 else (29 if mode == "paper" else smoke_updates - 1)
            identity = f"paper-{task}-{model}-epd-c{cycle}-teacher-{'base-release' if cycle == 1 else f'iter-{teacher_iteration}'}"
            binding = {"corpus_dir": runtime(corpus), "guidance_summary_dir": runtime(guidance),
                       "materialization_identity": identity, "model_path": runtime(prior_hf),
                       "teacher_iteration": teacher_iteration}
            if verified.get("direct_binding") != binding:
                raise RuntimeError(f"EPD target model/corpus binding differs: {verification}")
            prefix = "alfworld_epd" if task == "alfworld" else "agent_frozen_epd"
            custom[f"{prefix}_direct_binding"] = binding
            custom[f"{prefix}_materialization_identity"] = identity
            custom[f"{prefix}_teacher_iteration"] = teacher_iteration
            custom[f"{prefix}_target_count"] = int(verified["target_count"])
            custom[f"{prefix}_target_dir"] = runtime(target)
            if task == "alfworld":
                custom["alfworld_epd_subset_target_contract"] = False
        set_option(cli, "--save", runtime(cell / "checkpoints"))
        if "--eval-prompt-data" in cli:
            cli[cli.index("--eval-prompt-data") + 2] = runtime(cell / "eval_dummy.jsonl")
    else:
        final = 29 if mode == "paper" else smoke_updates - 1
        eval_checkpoint = base_checkpoint if method == "react" else cell / "checkpoints" / f"iter_{final:07d}"
        eval_hf = base_hf if method == "react" else cell / "eval_hf" / f"iter_{final:07d}"
        set_option(cli, "--load", runtime(eval_checkpoint))
        set_option(cli, "--ref-load", runtime(eval_checkpoint))
        set_option(cli, "--hf-checkpoint", runtime(eval_hf))
        set_option(cli, "--rollout-seed", 42 if task == "alfworld" else 314159)
        if task == "alfworld" and method != "react":
            custom["alfworld_frozen_corpus_dir"] = runtime(corpus)
            custom["alfworld_frozen_expected_trajectories"] = 960
        if method == "react":
            custom[f"{task}_sample_log_dir"] = runtime(cell / "eval/samples")
            custom[f"{task}_eval_result_audit_dir"] = runtime(cell / "eval/logs/eval_result")
            custom["eval_snapshot_audit_dir"] = runtime(cell / "eval/logs/eval_snapshot")
            set_option(cli, "--eval-snapshot-audit-dir", runtime(cell / "eval/logs/eval_snapshot"))
        if "--eval-prompt-data" in cli:
            cli[cli.index("--eval-prompt-data") + 2] = runtime(cell / "eval/eval_dummy.jsonl")
        if mode == "smoke":
            custom["alfworld_eval_episodes" if task == "alfworld" else "textcraft_eval_episodes"] = 8
            custom["alfworld_eval_replicate_seeds" if task == "alfworld" else "textcraft_eval_replicate_seeds"] = [EVAL_SEEDS[0]]
    config["scientific"] = {
        "task": task, "model": model, "method": method, "cycle": cycle,
        "phase": phase, "protocol": mode, "c1_input": runtime(c1_input) if c1_input else None,
        "task_seed": TASK_SEEDS[cycle], "training_seed": 1234,
        "training_source_trajectories": 0 if method == "react" else TASK_TRAJECTORIES[task],
        "optimizer_updates": 0 if method == "react" else 30 if mode == "paper" else smoke_updates,
        "source_batch_trajectories": (4 if task == "alfworld" else 1) if method == "grpo" else TASK_BATCH[task],
        "final_checkpoint": None if method == "react" else f"iter_{(29 if mode == 'paper' else smoke_updates - 1):07d}",
        "eval_seeds": list(EVAL_SEEDS if mode == "paper" else EVAL_SEEDS[:1]),
        "selection_fraction": selection_fraction if method in {"resail", "sdpo_resail"} else None,
        "retention_weight": retention_weight if method in {"resail", "sdpo_resail"} else 0.0,
    }
    serialized = json.dumps(config, sort_keys=True)
    if any(token in serialized for token in replacements):
        raise RuntimeError(f"unresolved template placeholder: {template}")
    if "/runs/stage" in serialized:
        raise RuntimeError(f"historical run dependency in {template}")
    return config


def write_bundle(path: Path, config: dict[str, Any]) -> None:
    path.mkdir(parents=True, exist_ok=True)
    files = {
        path / "expanded_config.json": json.dumps(config, indent=2, sort_keys=True) + "\n",
        path / "runtime_config.yaml": yaml.safe_dump(config["custom_config"], sort_keys=True),
    }
    for target, content in files.items():
        if target.exists() and target.read_text(encoding="utf-8") != content:
            raise FileExistsError(f"refusing to replace a different bundle: {target}")
        target.write_text(content, encoding="utf-8")


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _run(command: list[str], *, log: Path, env: dict[str, str] | None = None) -> None:
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("a", encoding="utf-8") as handle:
        handle.write("PAPER_COMMAND " + json.dumps(command) + "\n")
        handle.flush()
        subprocess.run(command, cwd=ROOT, env=env, stdout=handle, stderr=subprocess.STDOUT, check=True)


def _validate_roots(args: argparse.Namespace) -> None:
    if not args.input_root.resolve().is_relative_to((ROOT / "data/paper").resolve()):
        raise ValueError("input-root must be inside this ReSAIL copy's data/paper directory")
    if not args.run_root.resolve().is_relative_to((ROOT / "runs/paper").resolve()):
        raise ValueError("run-root must be inside this ReSAIL copy's runs/paper directory")
    if args.run_root.resolve() == (ROOT / "runs/paper").resolve():
        raise ValueError("run-root must name one unique paper run")
    if args.c1_input is not None:
        _validate_c1_input_path(args.c1_input, args.task, args.model)


def _validate_c1_input_path(path: Path, task: str, model: str) -> None:
    if not path.resolve().is_relative_to((ROOT / "data/paper/shared").resolve()):
        raise ValueError("C1 input must be inside this ReSAIL copy's data/paper/shared directory")
    if path.name != "c1" or path.parent.name != model or path.parent.parent.name != task:
        raise ValueError("C1 input must end in <task>/<model>/c1")


def _active_workloads() -> list[str]:
    output = subprocess.run(["ps", "-eo", "pid,ppid,stat,cmd"], capture_output=True,
                            text=True, check=True).stdout
    rows = []
    for line in output.splitlines():
        fields = line.strip().split(maxsplit=3)
        if len(fields) < 4 or fields[2].startswith("Z"):
            continue
        command = fields[3]
        if "ray::" in command or "sglang.launch_server" in command:
            if any(system in command for system in ("ray::IDLE", "ray::DashboardAgent", "ray::RuntimeEnvAgent")):
                continue
            rows.append(line)
    return rows


def _wait_workload_cleanup(timeout: float = 60.0) -> None:
    deadline = time.monotonic() + timeout
    while _active_workloads():
        if time.monotonic() >= deadline:
            raise RuntimeError("Ray or SGLang workloads remain after the phase")
        time.sleep(1)


def _env(custom: dict[str, Any]) -> tuple[dict[str, str], dict[str, Any]]:
    env = dict(os.environ)
    if not env.get("WANDB_API_KEY") and (ROOT / ".env").is_file():
        for raw in (ROOT / ".env").read_text(encoding="utf-8").splitlines():
            if raw.strip().startswith("WANDB_API_KEY="):
                env["WANDB_API_KEY"] = raw.split("=", 1)[1].strip().strip("'\"")
                break
    env.update({"PYTHONUNBUFFERED": "1", "RAY_enable_open_telemetry": "0",
                "WANDB_CONSOLE": "off", "WANDB_DISABLE_STATS": "true",
                "TOKENIZERS_PARALLELISM": "false",
                "SGLANG_ENABLE_HEALTH_ENDPOINT_GENERATION": "true",
                "ALFWORLD_FROZEN_PROMPT_NAME": "alfworld_ordinary_1024",
                "OMP_NUM_THREADS": "4", "MKL_NUM_THREADS": "4", "OPENBLAS_NUM_THREADS": "4",
                "NUMEXPR_NUM_THREADS": "4"})
    env.setdefault("NO_PROXY", "localhost,127.0.0.1,172.17.0.0/16")
    env.setdefault("no_proxy", env["NO_PROXY"])
    vars = {key: env[key] for key in ("NO_PROXY", "no_proxy",
                                     "WANDB_CONSOLE", "WANDB_DISABLE_STATS", "TOKENIZERS_PARALLELISM",
                                     "SGLANG_ENABLE_HEALTH_ENDPOINT_GENERATION",
                                     "ALFWORLD_FROZEN_PROMPT_NAME", "RAY_enable_open_telemetry",
                                     "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                                     "NUMEXPR_NUM_THREADS")}
    vars.update({key: env[key] for key in ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy")
                 if env.get(key)})
    vars.update({"PYTHONPATH": "/root/Megatron-LM:/workspace/slime", "CUDA_DEVICE_MAX_CONNECTIONS": "1",
                 "NCCL_SHM_DISABLE": "1",
                 "SLIME_RUNTIME_PROFILE": str(custom["runtime_profile"]),
                 "SLIME_EXECUTION_MODE": str(custom["execution_mode"]),
                 "SLIME_REQUIRES_LIVE_ENVIRONMENT": str(bool(custom["requires_live_environment"])).lower()})
    if custom.get("alfworld_cache_dir"):
        vars["ALFWORLD_DATA"] = str(custom["alfworld_cache_dir"])
    env.update(vars)
    return env, {"env_vars": vars}


def _ray_ready() -> bool:
    try:
        with urllib.request.urlopen("http://127.0.0.1:8265/api/version", timeout=2) as response:
            return response.status == 200
    except OSError:
        return False


def _ray_start(run: Path, env: dict[str, str]) -> None:
    if _ray_ready() or _active_workloads():
        raise RuntimeError("Ray or SGLang is already active; use a dedicated idle container")
    _run(["ray", "start", "--head", "--node-ip-address", "127.0.0.1", "--num-gpus", "8",
          "--num-cpus", "32", "--disable-usage-stats", "--dashboard-host=0.0.0.0",
          "--dashboard-port=8265", "--temp-dir", f"/tmp/resail_paper_{os.getpid()}"],
         log=run / "logs/ray_start.log", env=env)
    deadline = time.monotonic() + 60
    while not _ray_ready():
        if time.monotonic() >= deadline:
            raise RuntimeError("Ray jobs API did not become ready")
        time.sleep(0.5)


def _ray_stop(run: Path, env: dict[str, str]) -> None:
    _run(["ray", "stop", "--force"], log=run / "logs/ray_stop.log", env=env)
    _wait_workload_cleanup()
    deadline = time.monotonic() + 30
    while _ray_ready():
        if time.monotonic() >= deadline:
            raise RuntimeError("Ray dashboard remained active after stop")
        time.sleep(0.5)


def _submit(bundle: Path, *, log: Path) -> None:
    config = _read_json(bundle / "expanded_config.json")
    cli = [str(item) for item in config["cli_args"]]
    if "--eval-prompt-data" in cli:
        index = cli.index("--eval-prompt-data")
        dummy = Path(cli[index + 2])
        dummy.parent.mkdir(parents=True, exist_ok=True)
        dummy.write_text('{"prompt":"paper eval placeholder","label":"0"}\n', encoding="utf-8")
    env, runtime_env = _env(config["custom_config"])
    command = ["ray", "job", "submit", "--address=http://127.0.0.1:8265",
               f"--runtime-env-json={json.dumps(runtime_env, separators=(',', ':'))}", "--",
               "python3", "train.py", *cli, "--custom-config-path", runtime(bundle / "runtime_config.yaml")]
    _run(command, log=log, env=env)
    content = re.sub(r"\x1b\[[0-9;]*[A-Za-z]", "", log.read_text(encoding="utf-8", errors="replace"))
    if not re.search(r"Job '[^']+' succeeded", content) or re.search(r"Job '[^']+' failed", content):
        raise RuntimeError(f"Ray job has no clean success status: {log}")
    _wait_workload_cleanup()


def _verify_corpus(task: str, corpus: Path, expected: int, cycle: int) -> dict[str, Any]:
    if task == "alfworld":
        from slime_plugins.agent_tasks.alfworld.frozen.corpus import verify_frozen_corpus

        return verify_frozen_corpus(corpus, expected_trajectories=expected,
                                    enforce_clean=False, response_length_finish_max_rate=1.0,
                                    require_stream_bound_sampling=expected == 960)
    from slime_plugins.agent_tasks.common.frozen.corpus import verify_iterative_corpus

    return verify_iterative_corpus(corpus, expected_trajectories=expected,
                                   expected_task="textcraft", expected_cycle_seed=TASK_SEEDS[cycle])


def _verify_guidance(root: Path, corpus: Path, model: Path, count: int) -> None:
    from slime_plugins.agent_tasks.common.frozen.guidance import validate_summary_record

    manifest, verification = _read_json(root / "manifest.json"), _read_json(root / "verification.json")
    records = [json.loads(line) for line in (root / "summaries.jsonl").read_text(encoding="utf-8").splitlines() if line]
    if (verification != {"status": "verified", "manifest": manifest}
            or manifest.get("status") != "complete"
            or manifest.get("corpus_dir") != runtime(corpus)
            or manifest.get("model_path") != runtime(model)
            or int(manifest.get("trajectory_count", -1)) != count
            or len(records) != count
            or len({str(row.get("trajectory_uid")) for row in records}) != count):
        raise RuntimeError(f"guidance binding or count failed: {root}")
    for record in records:
        validate_summary_record(record)


def _corpus_uids(task: str, corpus: Path) -> list[str]:
    if task == "alfworld":
        from slime_plugins.agent_tasks.alfworld.frozen.data_source import load_frozen_trajectory_shard
        loader = load_frozen_trajectory_shard
    else:
        from slime_plugins.agent_tasks.common.frozen.contracts import load_shard
        loader = load_shard
    return [str(row["trajectory_uid"]) for path in sorted(corpus.glob("batch_*.pt"))
            for row in loader(path)]


def _verify_shared_c1(args: argparse.Namespace) -> dict[str, Any]:
    root = args.c1_input
    if root is None:
        raise ValueError("shared C1 input path is required")
    _validate_c1_input_path(root, args.task, args.model)
    expected = {"schema_name": "paper_shared_c1_v1", "status": "verified",
                "task": args.task, "model": args.model, "task_seed": TASK_SEEDS[1],
                "trajectory_count": TASK_TRAJECTORIES[args.task],
                "base_checkpoint": runtime(args.base_checkpoint), "base_hf": runtime(args.base_hf),
                "corpus_dir": runtime(root / "corpus"), "guidance_dir": runtime(root / "guidance")}
    manifest = _read_json(root / "verification.json")
    if manifest != expected:
        raise RuntimeError(f"shared C1 identity differs: {root / 'verification.json'}")
    collection = _read_json(root / "collection/verification.json")
    if (collection.get("status") != "verified"
            or collection.get("trajectory_count") != expected["trajectory_count"]
            or collection.get("task_seed") != TASK_SEEDS[1]):
        raise RuntimeError("shared C1 collection verification differs")
    stats = _verify_corpus(args.task, root / "corpus", expected["trajectory_count"], 1)
    if stats["trajectory_count"] != expected["trajectory_count"]:
        raise RuntimeError("shared C1 corpus size differs")
    _verify_guidance(root / "guidance", root / "corpus", args.base_hf,
                     expected["trajectory_count"])
    corpus_uids = _corpus_uids(args.task, root / "corpus")
    summaries = [json.loads(line) for line in (root / "guidance/summaries.jsonl").read_text(encoding="utf-8").splitlines() if line]
    summary_uids = [str(row["trajectory_uid"]) for row in summaries]
    if (len(corpus_uids) != expected["trajectory_count"] or len(set(corpus_uids)) != len(corpus_uids)
            or set(corpus_uids) != set(summary_uids)):
        raise RuntimeError("shared C1 corpus and guidance trajectory identities differ")
    return manifest


def _materialize_guidance(args: argparse.Namespace, cell: Path, prior_hf: Path,
                          source_root: Path | None = None) -> None:
    shared = source_root or cell / "collection"
    corpus = shared / "corpus" if args.cycle == 1 else cell / "collection/corpus"
    guidance = shared / "guidance" if args.cycle == 1 else cell / "collection/guidance"
    count = TASK_TRAJECTORIES[args.task]
    if (guidance / "verification.json").is_file():
        _verify_guidance(guidance, corpus, prior_hf, count)
        return
    if guidance.exists() and any(guidance.iterdir()):
        raise RuntimeError(f"unverified guidance exists; use a fresh run: {guidance}")
    env = dict(os.environ)
    env.update({"REPO_DIR": str(ROOT), "PAPER_CORPUS_DIR": str(corpus),
                "PAPER_GUIDANCE_DIR": str(guidance), "PAPER_GUIDANCE_MODEL_PATH": runtime(prior_hf),
                "PAPER_GUIDANCE_ENGINE_COUNT": "4" if args.task == "alfworld" and args.model == "4b" else "8",
                "PAPER_GUIDANCE_GLOBAL_CONCURRENCY": "256",
                "PAPER_GUIDANCE_PORT_BASE": "29600",
                "PAPER_GUIDANCE_METADATA_PROFILE": args.task, "PAPER_GUIDANCE_EXPECTED_TRAJECTORIES": str(count),
                "PAPER_GUIDANCE_RUNTIME_PROFILE": "safe-fast",
                "PAPER_GUIDANCE_EXECUTION_MODE": "model_materialization",
                "SLIME_RUNTIME_PROFILE": "safe-fast", "SLIME_EXECUTION_MODE": "model_materialization",
                "SLIME_REQUIRES_LIVE_ENVIRONMENT": "false", "OMP_NUM_THREADS": "4",
                "MKL_NUM_THREADS": "4", "OPENBLAS_NUM_THREADS": "4", "NUMEXPR_NUM_THREADS": "4"})
    command = ["bash", "scripts/experiments/text_guidance.sh"]
    log = cell / "logs/guidance.log"

    def run_once(current_env: dict[str, str]) -> int:
        log.parent.mkdir(parents=True, exist_ok=True)
        with log.open("a", encoding="utf-8") as handle:
            handle.write("PAPER_COMMAND " + json.dumps(command) + "\n")
            handle.flush()
            return subprocess.run(command, cwd=ROOT, env=current_env,
                                  stdout=handle, stderr=subprocess.STDOUT, check=False).returncode

    def archive_failure() -> Path:
        failure = guidance / "failure.json"
        failures = guidance / "failures"
        ledger = guidance / "recovery_attempts.json"
        previous_budget = 1024
        if ledger.is_file():
            previous_budget = int(_read_json(ledger)["max_new_tokens"])
        archive = guidance / f"recovery_max{previous_budget}"
        if archive.exists() or not failure.is_file() or not failures.is_dir():
            raise RuntimeError("guidance recovery evidence is incomplete")
        archive.mkdir()
        shutil.move(str(failure), str(archive / "failure.json"))
        shutil.move(str(failures), str(archive / "failures"))
        for path in (ledger, guidance / "recovery_attempt_details"):
            if path.exists():
                shutil.move(str(path), str(archive / path.name))
        return archive

    if not any(guidance.glob("recovery_max*")) and run_once(env) == 0:
        _verify_guidance(guidance, corpus, prior_hf, count)
        return
    for _ in range(6):
        if (guidance / "verification.json").is_file():
            break
        if (guidance / "failure.json").is_file():
            archive_failure()
        archives = sorted(guidance.glob("recovery_max*"),
                          key=lambda path: int(path.name.removeprefix("recovery_max")))
        if not archives:
            raise RuntimeError("guidance recovery lacks failure evidence")
        existing = int(archives[-1].name.removeprefix("recovery_max"))
        wave = existing // 1024
        recovery_env = dict(env)
        recovery_env.update({"PAPER_GUIDANCE_MAX_NEW_TOKENS": str(existing + 1024),
                             "PAPER_GUIDANCE_ATTEMPT_SEED_OFFSET": str(2880 * wave),
                             "PAPER_GUIDANCE_PRIOR_ATTEMPT_SEED_OFFSET": str(2880 * (wave - 1)),
                             "PAPER_GUIDANCE_EXISTING_MAX_NEW_TOKENS": str(existing),
                             "PAPER_GUIDANCE_PRIOR_FAILURE_DIR": str(archives[-1])})
        run_once(recovery_env)
    if not (guidance / "verification.json").is_file():
        raise RuntimeError("guidance recovery exhausted six audited waves")
    _verify_guidance(guidance, corpus, prior_hf, count)


def _train_metrics(log: Path, updates: int, task: str, method: str, cell: Path) -> None:
    selection_fraction, retention_weight = selection_settings(task)
    rows: dict[int, dict[str, Any]] = {}
    positive_retention = 0
    pattern = re.compile(r"step (\d+): (\{.*\})")
    for line in log.read_text(encoding="utf-8", errors="replace").splitlines():
        match = pattern.search(line)
        if match:
            step = int(match.group(1))
            if step in rows:
                raise RuntimeError(f"duplicate training update {step}: {log}")
            rows[step] = ast.literal_eval(match.group(2))
    if sorted(rows) != list(range(updates)):
        raise RuntimeError(f"training update rows are incomplete: {sorted(rows)}")
    for step, row in rows.items():
        required = ("train/loss", "train/grad_norm", "train/global_batch_size")
        if method in SDPO_LOSS_METHODS:
            required += ("train/sdpo_loss",)
        if method == "grpo":
            required += ("train/pg_loss",)
        if method in {"sdpo", "sdpo_resail"}:
            required += ("train/sdpo_policy_ratio_clip_fraction", "train/sdpo_deployment_tis_ratio_clip_fraction")
        if method in {"resail", "sdpo_resail"}:
            required += ("train/sgs_loss", "train/pr_loss",
                         "train/pr_weighted_loss")
        if any(key not in row or not math.isfinite(float(row[key])) for key in required):
            raise RuntimeError(f"missing or non-finite training metric at update {step}")
        if method in {"resail", "sdpo_resail"}:
            retention = float(row["train/pr_loss"])
            weighted = float(row["train/pr_weighted_loss"])
            coefficient = retention_weight
            if retention < 0 or not math.isclose(weighted, coefficient * retention, rel_tol=2e-5, abs_tol=2e-7):
                raise RuntimeError(f"retention weight drift at update {step}")
            positive_retention += int(retention > 0)
            if method == "resail" and not math.isclose(float(row["train/loss"]),
                                                        float(row["train/sgs_loss"]) + weighted,
                                                        rel_tol=2e-5, abs_tol=2e-7):
                raise RuntimeError(f"joint ReSAIL objective drift at update {step}")
        if method in {"oel", "sdpo", "sdpo_resail"} and not math.isclose(float(row["train/loss"]), float(row["train/sdpo_loss"]),
                              rel_tol=1e-7, abs_tol=1e-9):
            raise RuntimeError(f"plain distillation loss drift at update {step}")
        if method == "grpo" and not math.isclose(float(row["train/loss"]), float(row["train/pg_loss"]),
                                                    rel_tol=1e-7, abs_tol=1e-8):
            raise RuntimeError(f"GRPO policy loss drift at update {step}")
        if int(row["train/global_batch_size"]) != TASK_BATCH[task]:
            raise RuntimeError(f"global batch size drift at update {step}")
        if method in {"sdpo", "sdpo_resail"} and any(not 0 <= float(row[key]) <= 1 for key in
                                    ("train/sdpo_policy_ratio_clip_fraction", "train/sdpo_deployment_tis_ratio_clip_fraction")):
            raise RuntimeError(f"SDPO clipped-ratio audit drift at update {step}")
    source_batch = (4 if task == "alfworld" else 1) if method == "grpo" else TASK_BATCH[task]
    if method in {"resail", "sdpo_resail"} and updates == 30 and positive_retention == 0:
        raise RuntimeError("ReSAIL retention loss stayed zero for all paper updates")
    _verify_source_audits(cell, updates, source_batch, paired=method in PAIRED_METHODS)
    if method in STRICT_ON_POLICY_METHODS:
        _verify_on_policy(cell, updates)
    if method in {"resail", "sdpo_resail"}:
        _verify_selection_audits(cell, updates, task, selection_fraction)


def _materialize_epd(args: argparse.Namespace, cell: Path, prior_hf: Path, corpus: Path) -> None:
    if args.method != "epd":
        return
    from types import SimpleNamespace

    target = cell / "epd_targets"
    shared = args.c1_input if args.cycle == 1 and args.c1_input else cell / "collection"
    guidance = shared / "guidance" if args.cycle == 1 else cell / "collection/guidance"
    teacher_iteration = -1 if args.cycle == 1 else (29 if args.mode == "paper" else args.smoke_updates - 1)
    identity = (f"paper-{args.task}-{args.model}-epd-c{args.cycle}-teacher-"
                f"{'base-release' if args.cycle == 1 else f'iter-{teacher_iteration}'}")
    endpoint_file = target / "logs/endpoints.txt"
    engine_count = 4 if args.task == "alfworld" and args.model == "4b" else 8
    common = {"corpus_dir": corpus, "output_dir": target, "model_path": prior_hf,
              "guidance_summary_dir": guidance, "materialization_identity": identity,
              "teacher_iteration": teacher_iteration, "endpoint_file": endpoint_file,
              "engine_count": engine_count, "global_concurrency": 256,
              "expected_trajectories": TASK_TRAJECTORIES[args.task], "shard_size": 1024}
    env, _ = _env({"runtime_profile": "safe-fast", "execution_mode": "model_materialization",
                   "requires_live_environment": False})
    env["REPO_DIR"] = str(ROOT)
    if args.task == "alfworld":
        from exp.paper import text_epd_alfworld as materializer

        schedule = None
        source_count = 960
        options = {**common, "trajectory_schedule": schedule, "source_trajectories": source_count,
                   "runtime_profile": "safe-fast", "execution_mode": "model_materialization",
                   "requires_live_environment": False}
        env.update({"RESAIL_EPD_CORPUS_DIR": str(corpus), "RESAIL_EPD_TARGET_DIR": str(target),
                    "RESAIL_EPD_MODEL_PATH": runtime(prior_hf),
                    "RESAIL_EPD_GUIDANCE_SUMMARY_DIR": str(guidance),
                    "RESAIL_EPD_TRAJECTORY_SCHEDULE": str(schedule) if schedule else "",
                    "RESAIL_EPD_IDENTITY": identity,
                    "RESAIL_EPD_TEACHER_ITERATION": str(teacher_iteration),
                    "RESAIL_EPD_ENGINE_COUNT": str(engine_count),
                    "RESAIL_EPD_GLOBAL_CONCURRENCY": "256", "RESAIL_EPD_SOURCE_TRAJECTORIES": str(source_count),
                    "RESAIL_EPD_RUNTIME_PROFILE": "safe-fast", "RESAIL_EPD_EXECUTION_MODE": "model_materialization"})
        script = ROOT / "scripts/experiments/text_epd_alfworld.sh"
    else:
        from exp.paper import text_epd_textcraft as materializer

        options = {**common, "metadata_profile": "textcraft", "attempt_seed_offset": 0}
        env.update({"RESAIL_EPD_CORPUS_DIR": str(corpus), "RESAIL_EPD_TARGET_DIR": str(target),
                    "RESAIL_EPD_MODEL_PATH": runtime(prior_hf),
                    "RESAIL_EPD_GUIDANCE_SUMMARY_DIR": str(guidance),
                    "RESAIL_EPD_IDENTITY": identity,
                    "RESAIL_EPD_TEACHER_ITERATION": str(teacher_iteration),
                    "RESAIL_EPD_ENGINE_COUNT": str(engine_count),
                    "RESAIL_EPD_GLOBAL_CONCURRENCY": "256", "RESAIL_EPD_EXPECTED_TRAJECTORIES": "240"})
        script = ROOT / "scripts/experiments/text_epd_textcraft.sh"
    if not (target / "verification.json").is_file():
        _run(["bash", str(script)], log=target / "logs/submit.log", env=env)
    verified = materializer.verify(SimpleNamespace(**options))
    if verified.get("status") != "verified" or verified.get("binding_mode") != "direct_v1":
        raise RuntimeError(f"EPD targets could not be verified: {target}")


def _verify_source_audits(cell: Path, updates: int, batch: int, *, paired: bool) -> None:
    source, pair = [], []
    audits = (("source_audit", source), ("pair_audit", pair)) if paired else (("source_audit", source),)
    for dirname, output in audits:
        paths = sorted((cell / "logs" / dirname).glob("rollout_*.json"))
        if [p.name for p in paths] != [f"rollout_{i:03d}.json" for i in range(updates)]:
            raise RuntimeError(f"incomplete {dirname} coverage")
        for update, path in enumerate(paths):
            audit = _read_json(path)
            rows = audit.get("sources")
            if audit.get("rollout_id") != update or audit.get("source_count") != batch or not isinstance(rows, list) or len(rows) != batch:
                raise RuntimeError(f"source count drift: {path}")
            expected = list(range(update * batch, (update + 1) * batch))
            if [int(row.get("source_draw_id", -1)) for row in rows] != expected:
                raise RuntimeError(f"source order drift: {path}")
            output.extend((int(row["source_draw_id"]), str(row["source_trajectory_uid"])) for row in rows)
    if paired and source != pair:
        raise RuntimeError("source and paired teacher trajectories differ")


def _verify_on_policy(cell: Path, updates: int) -> None:
    paths = sorted((cell / "logs/strict_on_policy").glob("update_*.json"))
    if [p.name for p in paths] != [f"update_{i:03d}.json" for i in range(updates)]:
        raise RuntimeError("strict-on-policy audit coverage is incomplete")
    previous = None
    for update, path in enumerate(paths):
        row = _read_json(path)
        expected = {"update": update, "actor_pre_update_version": update,
                    "rollout_policy_version": update, "actor_post_update_version": update + 1,
                    "policy_lag": 0, "prefetched_training_batches": 0,
                    "rollout_completed_before_train": True, "rollout_gpu_count": 8,
                    "train_gpu_count": 8,
                    "role_sequence": ["rollout", "global_barrier", "train", "weight_sync"]}
        if any(row.get(key) != value for key, value in expected.items()):
            raise RuntimeError(f"OEL policy lag drift at update {update}")
        pre = str(row.get("engine_pre_update_weight_version", ""))
        post = str(row.get("engine_post_update_weight_version", ""))
        versions = row.get("post_update_weight_versions")
        if not isinstance(versions, list) or len(versions) != 8 or set(map(str, versions)) != {post}:
            raise RuntimeError(f"OEL weight sync drift at update {update}")
        if previous is not None and pre != previous:
            raise RuntimeError(f"OEL weight version chain drift at update {update}")
        previous = post


def _verify_selection_audits(cell: Path, updates: int, task: str, fraction: float) -> None:
    from slime_plugins.agent_tasks.common.algorithms.sgs import select_sgs_steps

    paths = sorted((cell / "logs/sgs").glob("rollout_*.json"))
    if [p.name for p in paths] != [f"rollout_{i:03d}.json" for i in range(updates)]:
        raise RuntimeError("ReSAIL selection audit coverage is incomplete")
    batch = TASK_BATCH[task]
    total_attempted = total_scoreable = 0
    for update, path in enumerate(paths):
        audit = _read_json(path)
        rows = audit.get("rows")
        if (audit.get("kind") != "sgs_update" or audit.get("rollout_id") != update
                or not math.isclose(float(audit.get("fraction", math.nan)), fraction)
                or audit.get("selection_scope") != "global"
                or audit.get("score_scope") != "response"
                or audit.get("selection_mode") != "sensitivity"
                or audit.get("ranking_score_field") != "teacher_js"
                or audit.get("loss_aggregation") != "trajectory_balanced"
                or audit.get("dp_size") != 8 or not isinstance(rows, list)
                or len(rows) != int(audit.get("attempted_steps", -1))
                or audit.get("retention_support") != "all"
                or audit.get("retention_kl_direction") != "reverse"
                or int(audit.get("retention_steps", -1)) != int(audit.get("attempted_steps", -2))
                or audit.get("trajectory_floor") is True):
            raise RuntimeError(f"ReSAIL selection contract drift at update {update}")
        selection = select_sgs_steps(rows, fraction=fraction, minimum_selected=8,
                                         selection_mode="sensitivity", selection_seed=42,
                                         rollout_id=update, selection_scope="global",
                                         score_field="teacher_js", ranking_order="descending")
        selected = [row for row in rows if row.get("selected") is True]
        actual = {(int(row["source_draw_id"]), int(row["turn_idx"])) for row in selected}
        if (actual != set(selection.selected_keys)
                or int(audit.get("selected_steps", -1)) != selection.selected_count
                or int(audit.get("scoreable_steps", -1)) != selection.scoreable_count
                or selection.scoreable_count / selection.attempted_count < 0.95):
            raise RuntimeError(f"ReSAIL selected step or coverage drift at update {update}")
        counts: dict[int, int] = {}
        for row in selected:
            draw = int(row["source_draw_id"])
            counts[draw] = counts.get(draw, 0) + 1
        for row in selected:
            expected_weight = len(selected) / (batch * counts[int(row["source_draw_id"])])
            if not math.isclose(float(row.get("loss_weight", math.nan)), expected_weight,
                                rel_tol=1e-6, abs_tol=1e-8):
                raise RuntimeError(f"ReSAIL denominator drift at update {update}")
        total_attempted += selection.attempted_count
        total_scoreable += selection.scoreable_count
    if total_scoreable / total_attempted < 0.98:
        raise RuntimeError("ReSAIL global score coverage is below 98%")


def _convert(cell: Path, iteration: str, base_hf: Path) -> None:
    checkpoint = cell / "checkpoints" / iteration
    output = cell / "eval_hf" / iteration
    _verify_checkpoint(cell, iteration)
    if output.is_dir():
        _verify_hf(output)
        return
    if output.exists():
        raise RuntimeError(f"incomplete HF conversion exists: {output}")
    temporary = output.with_name(f".{iteration}.tmp-{os.getpid()}")
    output.parent.mkdir(parents=True, exist_ok=True)
    _run(["python3", "tools/convert_torch_dist_to_hf.py", "--input-dir", str(checkpoint),
          "--output-dir", str(temporary), "--origin-hf-dir", runtime(base_hf),
          "--vocab-size", "151936"], log=cell / "eval_hf/conversion.log")
    _verify_hf(temporary)
    temporary.rename(output)


def _verify_checkpoint(cell: Path, iteration: str) -> None:
    import torch
    from torch.distributed.checkpoint import FileSystemReader

    if not re.fullmatch(r"iter_\d{7}", iteration):
        raise RuntimeError(f"final Megatron checkpoint iteration is invalid: {iteration}")
    checkpoint = cell / "checkpoints" / iteration
    tracker = cell / "checkpoints/latest_checkpointed_iteration.txt"
    if (not checkpoint.is_dir() or not tracker.is_file()
            or tracker.read_text(encoding="utf-8").strip() != str(int(iteration.removeprefix("iter_")))):
        raise RuntimeError(f"final Megatron checkpoint is incomplete or has wrong iteration: {checkpoint}")

    backend_file = checkpoint / "metadata.json"
    common_file = checkpoint / "common.pt"
    metadata_file = checkpoint / ".metadata"
    if any(not path.is_file() or path.stat().st_size <= 0
           for path in (backend_file, common_file, metadata_file)):
        raise RuntimeError(f"final Megatron checkpoint is missing metadata or common state: {checkpoint}")
    try:
        backend = json.loads(backend_file.read_text(encoding="utf-8"))
        common = torch.load(common_file, map_location="cpu", weights_only=False)
        metadata = FileSystemReader(str(checkpoint)).read_metadata()
    except Exception as exc:
        raise RuntimeError(f"final Megatron checkpoint metadata is unreadable: {checkpoint}") from exc
    if (backend.get("sharded_backend") != "torch_dist" or backend.get("common_backend") != "torch"
            or not isinstance(common, dict) or "args" not in common
            or not metadata.state_dict_metadata or not metadata.storage_data):
        raise RuntimeError(f"final Megatron checkpoint metadata is incomplete: {checkpoint}")

    required_ends: dict[Path, int] = {}
    for info in metadata.storage_data.values():
        relative = Path(info.relative_path)
        if (relative.is_absolute() or str(relative) != relative.name or relative.suffix != ".distcp"
                or info.offset < 0 or info.length <= 0):
            raise RuntimeError(f"final Megatron checkpoint has invalid shard metadata: {checkpoint}")
        shard = checkpoint / relative
        required_ends[shard] = max(required_ends.get(shard, 0), info.offset + info.length)
    actual_shards = set(checkpoint.glob("*.distcp"))
    if not required_ends or actual_shards != set(required_ends):
        raise RuntimeError(f"final Megatron checkpoint shard count differs from metadata: {checkpoint}")
    for shard, required_end in required_ends.items():
        if shard.is_symlink() or shard.stat().st_size < required_end:
            raise RuntimeError(f"final Megatron checkpoint shard is missing or truncated: {shard}")
        with shard.open("rb") as handle:
            handle.seek(required_end - 1)
            if len(handle.read(1)) != 1:
                raise RuntimeError(f"final Megatron checkpoint shard is unreadable: {shard}")


def _verify_hf(output: Path) -> None:
    import safetensors

    config = _read_json(output / "config.json")
    index = _read_json(output / "model.safetensors.index.json")
    weight_map = index.get("weight_map")
    if not isinstance(config, dict) or not isinstance(weight_map, dict) or not weight_map:
        raise RuntimeError(f"HF model metadata is invalid: {output}")
    shards = {str(value) for value in weight_map.values()}
    if not shards or shards != {path.name for path in output.glob("model-*.safetensors")}:
        raise RuntimeError(f"HF shard index differs from files: {output}")
    for filename in shards:
        path = output / filename
        if path.stat().st_size <= 0:
            raise RuntimeError(f"empty HF shard: {path}")
        with safetensors.safe_open(path, framework="pt", device="cpu") as handle:
            if not handle.keys():
                raise RuntimeError(f"unreadable HF shard: {path}")


def _verify_eval(cell: Path, mode: str, task: str) -> dict[str, Any]:
    if task == "textcraft":
        return _verify_textcraft_eval(cell, mode)
    path = cell / "eval/logs/eval_result/eval_000.json"
    result = _read_json(path)
    datasets = result["datasets"]
    seeds = EVAL_SEEDS if mode == "paper" else EVAL_SEEDS[:1]
    episodes = 128 if mode == "paper" else 8
    expected = ({f"{split}__replicate_{i:02d}_seed_{seed}"
                 for split in ("alfworld_eval", "eval_out_of_distribution")
                 for i, seed in enumerate(seeds)} if mode == "paper"
                else {"alfworld_eval", "eval_out_of_distribution"})
    if set(datasets) != expected or any(int(row.get("episode_count", -1)) != episodes for row in datasets.values()):
        raise RuntimeError(f"evaluation seeds, splits, or episode counts differ: {path}")
    if any(int(row.get("student_privileged_field_count", -1)) != 0 for row in datasets.values()):
        raise RuntimeError("privileged fields reached the student evaluation")
    sample_paths = sorted((cell / "eval/samples").glob("eval_*.jsonl"))
    if [p.name for p in sample_paths] != ["eval_0.jsonl"]:
        raise RuntimeError("evaluation has no single canonical episode log")
    terminal: dict[tuple[str, int], dict[str, bool]] = {}
    observed_seeds: dict[tuple[str, int], set[int]] = {}
    for line in sample_paths[0].read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        if row.get("is_terminal") is not True:
            continue
        key = (str(row.get("eval_base_dataset_name")), int(row.get("eval_replicate_index", -1)))
        uid = str(row.get("uid") or "")
        outcomes = terminal.setdefault(key, {})
        if not uid or uid in outcomes:
            raise RuntimeError("missing or duplicate terminal evaluation task UID")
        success = row.get("success")
        outcomes[uid] = bool(float(row.get("episode_reward", row.get("reward", 0.0)) or 0.0) > 0.0
                             if success is None else success)
        observed_seeds.setdefault(key, set()).add(int(row.get("eval_replicate_seed", -1)))
    bases = ("alfworld_eval", "eval_out_of_distribution")
    keys = {(base, index) for base in bases for index in range(len(seeds))}
    if set(terminal) != keys or any(len(rows) != episodes for rows in terminal.values()):
        raise RuntimeError("evaluation episode coverage is incomplete")
    if any(observed_seeds[key] != {seeds[key[1]]} for key in keys):
        raise RuntimeError("evaluation seed identities differ")
    for base in bases:
        replicates = [terminal[(base, index)] for index in range(len(seeds))]
        identities = [set(rows) for rows in replicates]
        if any(ids != identities[0] for ids in identities[1:]):
            raise RuntimeError(f"evaluation task identities are not paired for {base}")
        rates = [sum(rows.values()) / len(rows) for rows in replicates]
        prefix = "eval/alfworld_eval" if base == "alfworld_eval" else "eval_out_of_distribution/alfworld_eval"
        metrics = result.get("metrics", {})
        if mode == "smoke":
            logged = float(metrics.get(f"{prefix}/episode/success_rate", math.nan))
            dataset = datasets[base]
            if (int(dataset.get("success_count", -1)) != sum(replicates[0].values())
                    or not math.isfinite(logged)
                    or not math.isclose(logged, rates[0], rel_tol=1e-9, abs_tol=1e-12)):
                raise RuntimeError(f"single-replicate evaluation result differs: {base}")
            continue
        count = float(metrics.get(f"{prefix}/episode/success_rate_replicate_count", math.nan))
        if count != float(len(seeds)):
            raise RuntimeError(f"evaluation replicate count mismatch for {base}")
        recomputed = {"success_rate_mean": statistics.mean(rates)}
        if len(replicates) == 3:
            recomputed["success_rate_sample_std"] = statistics.stdev(rates)
            recomputed["pass_at_3"] = sum(any(rows[uid] for rows in replicates)
                                          for uid in identities[0]) / len(identities[0])
        for name, value in recomputed.items():
            logged = float(metrics.get(f"{prefix}/episode/{name}", math.nan))
            if not math.isfinite(logged) or not math.isclose(logged, value, rel_tol=1e-9, abs_tol=1e-12):
                raise RuntimeError(f"evaluation aggregate mismatch: {prefix}/{name}")
    snapshot_path = cell / "eval/logs/eval_snapshot/eval_000.json"
    snapshot = _read_json(snapshot_path)
    if (snapshot.get("iteration") != 0 or snapshot.get("optimizer_update") != 0
            or snapshot.get("data_source_cursor_before") != snapshot.get("data_source_cursor_after")
            or snapshot.get("independent_rng_namespace") is not True
            or int(snapshot.get("snapshot", {}).get("engine_count", -1)) != 8):
        raise RuntimeError("evaluation snapshot or cursor contract failed")
    return datasets


def _verify_textcraft_eval(cell: Path, mode: str) -> dict[str, Any]:
    from slime_plugins.agent_tasks.common.frozen.sampling import eval_turn_seed
    from slime_plugins.agent_tasks.textcraft.splits import data_indices_from_file

    root = cell / "eval"
    result = _read_json(root / "logs/eval_result/eval_000.json")
    seeds = EVAL_SEEDS if mode == "paper" else EVAL_SEEDS[:1]
    count = 100 if mode == "paper" else 8
    datasets = result.get("datasets", {})
    expected = ({f"textcraft_eval__replicate_{index:02d}_seed_{seed}"
                 for index, seed in enumerate(seeds)} if mode == "paper" else {"textcraft_eval"})
    if set(datasets) != expected or any(int(row.get("episode_count", -1)) != count for row in datasets.values()):
        raise RuntimeError("TextCraft evaluation replicate or episode count differs")
    sample_paths = sorted((root / "samples").glob("eval_*.jsonl"))
    if [path.name for path in sample_paths] != ["eval_0.jsonl"]:
        raise RuntimeError("TextCraft evaluation lacks one canonical sample log")
    episodes: dict[tuple[int, str], list[tuple[int, dict[str, Any]]]] = {}
    order: dict[int, list[str]] = {index: [] for index in range(len(seeds))}
    observed_seeds: dict[int, set[int]] = {index: set() for index in range(len(seeds))}
    raw_steps: dict[int, int] = {index: 0 for index in range(len(seeds))}
    for line in sample_paths[0].read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        if row.get("eval_base_dataset_name") != "textcraft_eval":
            raise RuntimeError("TextCraft evaluation sample has wrong dataset")
        index = int(row.get("eval_replicate_index", -1))
        if index not in range(len(seeds)):
            raise RuntimeError("TextCraft evaluation replicate index is invalid")
        task_id = str(row.get("task_id") or "")
        turn = int(row.get("turn_idx", -1))
        seed = int(row.get("eval_replicate_seed", -1))
        if not task_id or turn < 0 or seed != seeds[index]:
            raise RuntimeError("TextCraft evaluation task or turn identity is invalid")
        if int(row.get("sampling_seed", -1)) != eval_turn_seed(
                task="textcraft", replicate_seed=seed, task_id=task_id, turn_idx=turn):
            raise RuntimeError("TextCraft evaluation turn seed is not task-bound")
        key = (index, task_id)
        if key not in episodes:
            episodes[key] = []
            order[index].append(task_id)
        episodes[key].append((turn, row))
        observed_seeds[index].add(seed)
        raw_steps[index] += 1
    terminal: dict[int, set[str]] = {index: set() for index in range(len(seeds))}
    successful: dict[int, set[str]] = {index: set() for index in range(len(seeds))}
    for (index, task_id), items in episodes.items():
        turns = [turn for turn, _ in items]
        if turns != list(range(len(items))):
            raise RuntimeError("TextCraft evaluation episode turn order differs")
        outcome = bool(items[-1][1].get("success", False))
        if any(bool(row.get("success", False)) != outcome for _, row in items):
            raise RuntimeError("TextCraft evaluation episode outcome changed within trajectory")
        final_turn, final = items[-1]
        if outcome and final.get("is_terminal") is not True:
            raise RuntimeError("TextCraft successful episode lacks terminal closure")
        if not outcome and (final_turn != 29 or final.get("env_horizon_reached") is not True):
            raise RuntimeError("TextCraft failed episode lacks full horizon closure")
        terminal[index].add(task_id)
        if outcome:
            successful[index].add(task_id)
    expected_task_indices = data_indices_from_file(
        ROOT / ".cache/textcraft/agentgym_rl_data_id/eval/textcraft_test.json")[:count]
    expected_order = [f"textcraft_{index}" for index in expected_task_indices]
    expected_set = set(expected_order)
    for index, seed in enumerate(seeds):
        if (terminal[index] != expected_set or order[index] != expected_order
                or observed_seeds[index] != {seed}):
            raise RuntimeError("TextCraft evaluation panel differs from official task order")
        name = (f"textcraft_eval__replicate_{index:02d}_seed_{seed}"
                if mode == "paper" else "textcraft_eval")
        dataset = datasets[name]
        if (int(dataset.get("step_count", -1)) != raw_steps[index]
                or int(dataset.get("success_count", -1)) != len(successful[index])):
            raise RuntimeError("TextCraft evaluation result and samples disagree")
        rate = float(result.get("metrics", {}).get(f"eval/{name}/episode/success_rate", math.nan))
        if not math.isfinite(rate) or not math.isclose(rate, len(successful[index]) / count,
                                                       rel_tol=1e-9, abs_tol=1e-12):
            raise RuntimeError("TextCraft evaluation success metric disagrees with samples")
    snapshot = _read_json(root / "logs/eval_snapshot/eval_000.json")
    if (snapshot.get("iteration") != 0 or snapshot.get("optimizer_update") != 0
            or snapshot.get("data_source_cursor_before") != snapshot.get("data_source_cursor_after")
            or snapshot.get("independent_rng_namespace") is not True
            or int(snapshot.get("snapshot", {}).get("engine_count", -1)) != 8):
        raise RuntimeError("TextCraft evaluation snapshot or cursor differs")
    return datasets


def prepare(args: argparse.Namespace, *, only_phases: tuple[str, ...] | None = None) -> dict[str, Path]:
    _validate_roots(args)
    cell = cell_root(args.run_root, args.task, args.model, args.method, args.cycle)
    phases = (("eval",) if args.method == "react" else
              ("collection", "training", "eval") if args.cycle > 1 or args.c1_input is None else
              ("training", "eval"))
    if only_phases is not None:
        phases = tuple(phase for phase in phases if phase in only_phases)
    elif args.method == "epd" and not (cell / "epd_targets/verification.json").is_file():
        phases = tuple(phase for phase in phases if phase == "collection")
    paths = {"collection": cell / "collection/bundle",
             "training": cell / "bundle", "eval": cell / "eval/bundle"}
    for phase in phases:
        config = build_config(task=args.task, model=args.model, method=args.method, cycle=args.cycle,
                              phase=phase, run_root=args.run_root, input_root=args.input_root,
                              base_checkpoint=args.base_checkpoint, base_hf=args.base_hf,
                              mode=args.mode, c1_input=args.c1_input, smoke_updates=args.smoke_updates,
                              wandb_mode=args.wandb_mode)
        write_bundle(paths[phase], config)
    return {phase: paths[phase] for phase in phases}


def prepare_c1(args: argparse.Namespace) -> dict[str, Any]:
    if args.c1_input is None:
        raise ValueError("prepare-c1 requires --c1-input")
    _validate_c1_input_path(args.c1_input, args.task, args.model)
    root = args.c1_input
    verification = root / "verification.json"
    if verification.is_file():
        if not args.resume:
            raise RuntimeError("shared C1 input already exists; use --resume to verify it")
        return _verify_shared_c1(args)
    if root.exists() and any(root.iterdir()):
        raise RuntimeError("partial shared C1 input exists; choose a fresh C1 input directory")
    if not (args.base_checkpoint / "latest_checkpointed_iteration.txt").is_file() or not (args.base_hf / "config.json").is_file():
        raise FileNotFoundError("base model is unavailable")
    config = build_config(task=args.task, model=args.model, method="resail", cycle=1,
                          phase="collection", run_root=args.run_root, input_root=args.input_root,
                          base_checkpoint=args.base_checkpoint, base_hf=args.base_hf,
                          mode=args.mode, c1_input=root, cell_override=root, wandb_mode=args.wandb_mode)
    bundle = root / "collection/bundle"
    write_bundle(bundle, config)
    env, _ = _env(config["custom_config"])
    _ray_start(root, env)
    try:
        _submit(bundle, log=root / "collection/logs/ray_job_submit.log")
    finally:
        _ray_stop(root, env)
    stats = _verify_corpus(args.task, root / "corpus", TASK_TRAJECTORIES[args.task], 1)
    write_json(root / "collection/verification.json", {
        "status": "verified", "trajectory_count": stats["trajectory_count"], "task_seed": TASK_SEEDS[1]})
    _materialize_guidance(args, root, args.base_hf, source_root=root)
    result = {"schema_name": "paper_shared_c1_v1", "status": "verified",
              "task": args.task, "model": args.model, "task_seed": TASK_SEEDS[1],
              "trajectory_count": TASK_TRAJECTORIES[args.task],
              "base_checkpoint": runtime(args.base_checkpoint), "base_hf": runtime(args.base_hf),
              "corpus_dir": runtime(root / "corpus"), "guidance_dir": runtime(root / "guidance")}
    write_json(verification, result)
    return _verify_shared_c1(args)


def execute_react(args: argparse.Namespace) -> dict[str, Any]:
    _validate_roots(args)
    if args.cycle != 1:
        raise ValueError("ReAct has one base-model endpoint")
    cell = cell_root(args.run_root, args.task, args.model, "react", 1)
    identity = {"task": args.task, "model": args.model, "method": "react", "protocol": args.mode,
                "run_root": runtime(args.run_root),
                "base_checkpoint": runtime(args.base_checkpoint),
                "base_hf": runtime(args.base_hf)}
    identity_path = cell / "run_identity.json"
    if identity_path.is_file() and _read_json(identity_path) != identity:
        raise RuntimeError("ReAct base-evaluation identity differs from requested model/protocol")
    completion_path = cell / "completion.json"
    if completion_path.is_file():
        completed = _read_json(completion_path)
        if args.resume and completed.get("status") == "complete" and _read_json(cell / "verification.json").get("status") == "verified":
            _verify_eval(cell, args.mode, args.task)
            return completed
        raise RuntimeError("completed ReAct evaluation exists; choose a fresh run or use --resume")
    if not identity_path.is_file():
        write_json(identity_path, identity)
    if not (args.base_checkpoint / "latest_checkpointed_iteration.txt").is_file() or not (args.base_hf / "config.json").is_file():
        raise FileNotFoundError("base model is unavailable")
    bundle = prepare(args)["eval"]
    log = cell / "eval/logs/ray_job_submit.log"
    reused = False
    if log.exists():
        if not args.resume or not re.search(r"Job '[^']+' succeeded", log.read_text(encoding="utf-8", errors="replace")) or _active_workloads():
            raise RuntimeError("existing ReAct evaluation lacks a clean completed Ray job")
        reused = True
    else:
        env, _ = _env(_read_json(bundle / "expanded_config.json")["custom_config"])
        _ray_start(cell, env)
        try:
            _submit(bundle, log=log)
        finally:
            _ray_stop(cell, env)
    datasets = _verify_eval(cell, args.mode, args.task)
    write_json(cell / "eval/verification.json", {"status": "verified", "datasets": datasets})
    if reused:
        write_json(cell / "evaluation_recovery.json", {"status": "verified", "kind": "completed_eval_artifact_reconciliation",
                                                       "gpu_evaluation_relaunched": False, "source_log": runtime(log)})
    write_json(cell / "verification.json", {"status": "verified", **identity, "eval_datasets": datasets})
    result = {"status": "complete", "task": args.task, "model": args.model,
              "method": "react", "cycle": 1, "protocol": args.mode, "eval_datasets": datasets}
    write_json(completion_path, result)
    return result


def execute(args: argparse.Namespace) -> dict[str, Any]:
    if args.method == "react":
        return execute_react(args)
    _validate_roots(args)
    cell = cell_root(args.run_root, args.task, args.model, args.method, args.cycle)
    identity = {"task": args.task, "model": args.model, "method": args.method,
                "cycle": args.cycle, "protocol": args.mode,
                "c1_input": runtime(args.c1_input) if args.c1_input else None,
                "smoke_updates": args.smoke_updates if args.mode == "smoke" else None,
                "run_root": runtime(args.run_root), "input_root": runtime(args.input_root),
                "base_checkpoint": runtime(args.base_checkpoint), "base_hf": runtime(args.base_hf)}
    identity_path = cell / "run_identity.json"
    if identity_path.is_file() and {**identity, **_read_json(identity_path)} != identity:
        raise RuntimeError(f"run identity differs from requested protocol: {identity_path}")
    completion_path = cell / "completion.json"
    if completion_path.exists():
        expected = {"task": args.task, "model": args.model, "method": args.method,
                    "cycle": args.cycle, "protocol": args.mode}
        completed = _read_json(completion_path)
        verified = _read_json(cell / "verification.json")
        if (args.resume and identity_path.is_file() and completed.get("status") == "complete"
                and verified.get("status") == "verified"
                and all(completed.get(key) == value and verified.get(key) == value for key, value in expected.items())
                and verified.get("c1_input") == identity["c1_input"]
                and completed.get("final_checkpoint") == verified.get("final_checkpoint")):
            _verify_checkpoint(cell, str(completed.get("final_checkpoint", "")))
            return _read_json(completion_path)
        raise RuntimeError("completed cell exists; choose a fresh run or use --resume")
    if not identity_path.is_file():
        write_json(identity_path, identity)
    prior, prior_hf = prior_paths(args.run_root, args.task, args.model, args.method, args.cycle,
                                  args.base_checkpoint, args.base_hf)
    if args.cycle > 1:
        parent_cell = cell_root(args.run_root, args.task, args.model, args.method, args.cycle - 1)
        parent_identity_path = parent_cell / "run_identity.json"
        expected_parent_identity = {**identity, "cycle": args.cycle - 1}
        if (not parent_identity_path.is_file()
                or _read_json(parent_identity_path) != expected_parent_identity):
            raise RuntimeError(f"parent cycle identity differs from requested protocol: {parent_identity_path}")
        parent_completion = _read_json(parent_cell / "completion.json")
        parent_verification = _read_json(parent_cell / "verification.json")
        expected_parent_result = {"task": args.task, "model": args.model, "method": args.method,
                                  "cycle": args.cycle - 1, "protocol": args.mode}
        if any(parent_completion.get(key) != value or parent_verification.get(key) != value
               for key, value in expected_parent_result.items()):
            raise RuntimeError(f"parent cycle result identity differs from requested protocol: {parent_cell}")
        parent_iteration = str(parent_completion.get("final_checkpoint", ""))
        if parent_verification.get("final_checkpoint") != parent_iteration:
            raise RuntimeError(f"parent cycle final checkpoint differs from verification: {parent_cell}")
        _verify_checkpoint(parent_cell, parent_iteration)
    if not (prior / "latest_checkpointed_iteration.txt").is_file() or not (prior_hf / "config.json").is_file():
        raise FileNotFoundError("base or prior model is unavailable")
    bundles = prepare(args, only_phases=("collection",) if args.method == "epd" else None)
    shared = args.c1_input if args.cycle == 1 and args.c1_input else cell / "collection"
    corpus = shared / "corpus" if args.cycle == 1 else cell / "collection/corpus"
    collection_verification = (shared / "collection/verification.json" if args.cycle == 1 and args.c1_input
                               else cell / "collection/verification.json")
    if args.cycle == 1 and args.c1_input:
        _verify_shared_c1(args)
    elif collection_verification.is_file():
        if _read_json(collection_verification).get("status") != "verified":
            raise RuntimeError(f"collection verification is invalid: {collection_verification}")
        _verify_corpus(args.task, corpus, TASK_TRAJECTORIES[args.task], args.cycle)
    else:
        if corpus.exists() and any(corpus.iterdir()):
            raise RuntimeError(f"unverified collection exists; use a fresh run: {corpus}")
        env, _ = _env(_read_json(bundles["collection"] / "expanded_config.json")["custom_config"])
        _ray_start(cell, env)
        try:
            _submit(bundles["collection"], log=cell / "collection/logs/ray_job_submit.log")
        finally:
            _ray_stop(cell, env)
        stats = _verify_corpus(args.task, corpus, TASK_TRAJECTORIES[args.task], args.cycle)
        write_json(collection_verification, {"status": "verified", "trajectory_count": stats["trajectory_count"],
                                             "task_seed": TASK_SEEDS[args.cycle]})
    if args.method in GUIDANCE_METHODS and not (args.cycle == 1 and args.c1_input):
        _materialize_guidance(args, cell, prior_hf)
    if args.method == "epd":
        _materialize_epd(args, cell, prior_hf, corpus)
        bundles.update(prepare(args, only_phases=("training", "eval")))
    updates = 30 if args.mode == "paper" else args.smoke_updates
    iteration = f"iter_{updates - 1:07d}"
    train_verification = cell / "training_verification.json"
    checkpoint = cell / "checkpoints" / iteration
    if train_verification.is_file():
        if _read_json(train_verification) != {"status": "verified", "updates": updates, "final_checkpoint": iteration}:
            raise RuntimeError("training verification differs from requested protocol")
        _train_metrics(cell / "logs/ray_job_submit.log", updates, args.task, args.method, cell)
        _verify_checkpoint(cell, iteration)
    else:
        if (cell / "logs/ray_job_submit.log").exists() or (cell / "checkpoints").exists():
            raise RuntimeError("unverified training state exists; use a fresh run")
        env, _ = _env(_read_json(bundles["training"] / "expanded_config.json")["custom_config"])
        _ray_start(cell, env)
        try:
            _submit(bundles["training"], log=cell / "logs/ray_job_submit.log")
        finally:
            _ray_stop(cell, env)
        _train_metrics(cell / "logs/ray_job_submit.log", updates, args.task, args.method, cell)
        _verify_checkpoint(cell, iteration)
        write_json(train_verification, {"status": "verified", "updates": updates,
                                        "final_checkpoint": iteration})
    _convert(cell, iteration, args.base_hf)
    eval_verification = cell / "eval/verification.json"
    if eval_verification.is_file():
        if _read_json(eval_verification).get("status") != "verified":
            raise RuntimeError("evaluation verification is invalid")
        datasets = _verify_eval(cell, args.mode, args.task)
    else:
        reused_eval = False
        if (cell / "eval/logs/ray_job_submit.log").exists():
            if not args.resume:
                raise RuntimeError("unverified evaluation exists; use --resume to verify its completed artifacts")
            log_text = (cell / "eval/logs/ray_job_submit.log").read_text(encoding="utf-8", errors="replace")
            if not re.search(r"Job '[^']+' succeeded", log_text) or _active_workloads():
                raise RuntimeError("existing evaluation lacks clean Ray success or workloads remain")
            reused_eval = True
        else:
            env, _ = _env(_read_json(bundles["eval"] / "expanded_config.json")["custom_config"])
            _ray_start(cell, env)
            try:
                _submit(bundles["eval"], log=cell / "eval/logs/ray_job_submit.log")
            finally:
                _ray_stop(cell, env)
        datasets = _verify_eval(cell, args.mode, args.task)
        write_json(eval_verification, {"status": "verified", "datasets": datasets})
        if reused_eval:
            write_json(cell / "evaluation_recovery.json", {
                "status": "verified", "kind": "completed_eval_artifact_reconciliation",
                "reason": "single_replicate_result_uses_base_dataset_names_and_ordinary_success_rate",
                "gpu_evaluation_relaunched": False,
                "source_log": runtime(cell / "eval/logs/ray_job_submit.log"),
                "evaluation_verification": runtime(eval_verification),
            })
    verification = {"status": "verified", "task": args.task, "model": args.model,
                    "method": args.method, "cycle": args.cycle, "protocol": args.mode,
                    "c1_input": identity["c1_input"], "updates": updates, "final_checkpoint": iteration,
                    "eval_datasets": datasets}
    write_json(cell / "verification.json", verification)
    completion = {"status": "complete", "task": args.task, "model": args.model,
                  "method": args.method, "cycle": args.cycle, "protocol": args.mode,
                  "final_checkpoint": iteration, "checkpoint": runtime(checkpoint),
                  "hf_checkpoint": runtime(cell / "eval_hf" / iteration)}
    write_json(completion_path, completion)
    return completion


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("action", choices=("check", "prepare", "prepare-c1", "run"))
    p.add_argument("--task", choices=("alfworld", "textcraft"), default="alfworld")
    p.add_argument("--model", choices=("4b", "8b"), default="4b")
    p.add_argument("--method", choices=METHODS, default="resail")
    p.add_argument("--cycle", type=int, choices=(1, 2, 3), default=1)
    p.add_argument("--mode", choices=("paper", "smoke"), default="paper")
    p.add_argument("--c1-input", type=Path)
    p.add_argument("--smoke-updates", type=int, default=2)
    p.add_argument("--run-root", type=Path, default=ROOT / "runs/paper")
    p.add_argument("--input-root", type=Path, default=ROOT / "data/paper")
    p.add_argument("--base-checkpoint", type=Path)
    p.add_argument("--base-hf", type=Path)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--wandb-mode", choices=("offline", "online", "disabled"), default="offline",
                   help="experiment logging mode; offline needs no W&B account or network")
    return p


def main() -> None:
    args = parser().parse_args()
    if args.base_checkpoint is None:
        args.base_checkpoint = Path(f"/root/checkpoints/Qwen3-{args.model.upper()}_torch_dist")
    if args.base_hf is None:
        args.base_hf = Path(f"/root/models/Qwen/Qwen3-{args.model.upper()}")
    if args.action == "prepare-c1":
        print(json.dumps(prepare_c1(args), indent=2, sort_keys=True))
        return
    if args.action == "check":
        phases = (("eval",) if args.method == "react" else
                  ("collection", "training", "eval") if args.cycle > 1 or args.c1_input is None else
                  ("training", "eval"))
        for phase in phases:
            try:
                config = build_config(task=args.task, model=args.model, method=args.method, cycle=args.cycle,
                                      phase=phase, run_root=args.run_root, input_root=args.input_root,
                                      base_checkpoint=args.base_checkpoint, base_hf=args.base_hf,
                                      mode=args.mode, c1_input=args.c1_input, smoke_updates=args.smoke_updates,
                                      wandb_mode=args.wandb_mode)
            except FileNotFoundError as exc:
                if args.method != "epd" or phase != "training":
                    raise
                print(f"training pending materialization: {exc}")
                continue
            print(phase, len(config["cli_args"]), len(config["custom_config"]))
        return
    if args.action == "run":
        print(json.dumps(execute(args), indent=2, sort_keys=True))
    else:
        bundles = prepare(args)
        print(json.dumps({key: str(value) for key, value in bundles.items()}, indent=2))


if __name__ == "__main__":
    main()
