from __future__ import annotations

import json
import statistics
from pathlib import Path
from types import SimpleNamespace

import pytest

from exp.paper import text
from exp.paper.text import ROOT, _env, _verify_eval, _verify_selection_audits, _verify_textcraft_eval, build_config, option, prior_paths, runtime

BASE = Path("/root/checkpoints/Qwen3-4B_torch_dist")
HF = Path("/root/models/Qwen/Qwen3-4B")
METHODS = ("react", "rft", "grpo", "epd", "sdpo", "oel", "resail", "sdpo_resail")


def test_runtime_proxy_is_only_inherited_when_provided(monkeypatch) -> None:
    for key in ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy"):
        monkeypatch.delenv(key, raising=False)
    custom = {"runtime_profile": "safe-fast", "execution_mode": "offline_training",
              "requires_live_environment": False}
    _, runtime_env = _env(custom)
    assert not any(key in runtime_env["env_vars"] for key in
                   ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy"))
    assert runtime_env["env_vars"]["ALFWORLD_FROZEN_PROMPT_NAME"] == "alfworld_ordinary_1024"
    monkeypatch.setenv("HTTPS_PROXY", "http://proxy.example:1234")
    _, runtime_env = _env(custom)
    assert runtime_env["env_vars"]["HTTPS_PROXY"] == "http://proxy.example:1234"


@pytest.mark.parametrize("task", ("alfworld", "textcraft"))
@pytest.mark.parametrize("model", ("4b", "8b"))
@pytest.mark.parametrize("method", METHODS)
def test_main_method_config_matrix(tmp_path: Path, task: str, model: str, method: str) -> None:
    run = tmp_path / "run"
    shared = ROOT / "data/paper/shared" / task / model / "c1"
    phase = "eval" if method == "react" else "training"
    if method == "epd":
        target = run / task / model / "epd/c1/epd_targets"
        target.mkdir(parents=True)
        base_hf = Path(f"/root/models/Qwen/Qwen3-{model.upper()}")
        binding = {"corpus_dir": runtime(shared / "corpus"),
                   "guidance_summary_dir": runtime(shared / "guidance"),
                   "materialization_identity": f"paper-{task}-{model}-epd-c1-teacher-base-release",
                   "model_path": runtime(base_hf), "teacher_iteration": -1}
        (target / "verification.json").write_text(json.dumps({
            "status": "verified", "binding_mode": "direct_v1", "direct_binding": binding,
            "target_count": 240 if task == "textcraft" else 960}))
    config = build_config(task=task, model=model, method=method, cycle=1, phase=phase,
                          run_root=run, input_root=ROOT / "data/paper",
                          base_checkpoint=Path(f"/root/checkpoints/Qwen3-{model.upper()}_torch_dist"),
                          base_hf=Path(f"/root/models/Qwen/Qwen3-{model.upper()}"),
                          mode="paper", c1_input=shared)
    cli, custom = config["cli_args"], config["custom_config"]
    assert option(cli, "--wandb-mode") == "offline"
    assert option(cli, "--seed") == "1234"
    if method != "react":
        assert option(cli, "--global-batch-size") == ("32" if task == "alfworld" else "8")
    assert "--update-js-probe" not in cli
    assert "3840" not in json.dumps(config)
    assert "@SUBSET@" not in json.dumps(config)
    if method != "react":
        assert option(cli, "--num-rollout") == "30"
        assert option(cli, "--checkpoint-fixed-iterations") == "29"
        if task == "textcraft":
            assert custom["agent_frozen_sampling_seed_namespace"] == f"textcraft-{model}-{method}-cycle-1"
        assert custom["alfworld_frozen_corpus_dir" if task == "alfworld" else "agent_frozen_corpus_dir"] == runtime(shared / "corpus")
        assert config["scientific"]["c1_input"] == runtime(shared)
        if method in {"resail", "sdpo_resail"}:
            assert custom["sgs_selection_fraction"] == (0.05 if task == "alfworld" else 0.25)
            assert custom["pr_weight"] == (0.5 if task == "alfworld" else 1.0)
    if (task, model, method) == ("alfworld", "8b", "resail"):
        assert custom["alfworld_frozen_arm"] == "iterative_trajectory_distillation"
        assert "alfworld_frozen_shared_schedule" not in custom


@pytest.mark.parametrize("mode", ("offline", "online", "disabled"))
def test_prepared_phases_keep_explicit_logging_mode(tmp_path: Path, monkeypatch, mode: str) -> None:
    monkeypatch.setattr(text, "ROOT", tmp_path)
    args = text.parser().parse_args([
        "prepare", "--run-root", str(tmp_path / "runs/paper/example"), "--wandb-mode", mode,
        "--base-checkpoint", str(BASE), "--base-hf", str(HF),
    ])
    bundles = text.prepare(args)
    assert set(bundles) == {"collection", "training", "eval"}
    for bundle in bundles.values():
        config = json.loads((bundle / "expanded_config.json").read_text())
        assert option(config["cli_args"], "--wandb-mode") == mode


def test_independent_fresh_c1_owns_collection(tmp_path: Path) -> None:
    config = build_config(task="alfworld", model="4b", method="resail", cycle=1,
                          phase="collection", run_root=tmp_path / "run", input_root=tmp_path / "inputs",
                          base_checkpoint=BASE, base_hf=HF, mode="paper")
    assert config["custom_config"]["alfworld_frozen_corpus_dir"] == str(tmp_path / "run/alfworld/4b/resail/c1/collection/corpus")
    assert config["scientific"]["c1_input"] is None


def test_shared_c1_collection_has_canonical_output_and_namespace(tmp_path: Path) -> None:
    root = ROOT / "data/paper/shared/textcraft/4b/c1"
    config = build_config(task="textcraft", model="4b", method="resail", cycle=1,
                          phase="collection", run_root=tmp_path / "run", input_root=ROOT / "data/paper",
                          base_checkpoint=BASE, base_hf=HF, mode="paper", c1_input=root,
                          cell_override=root)
    custom = config["custom_config"]
    assert custom["agent_frozen_corpus_dir"] == runtime(root / "corpus")
    assert custom["agent_frozen_sampling_seed_namespace"] == "textcraft-4b-shared-cycle-1"


@pytest.mark.parametrize("field,value", (
    ("loss_aggregation", "donor"),
    ("retention_support", "selected"),
    ("selection_mode", "random"),
    ("ranking_score_field", "distillation_kl"),
    ("score_scope", "none"),
))
def test_main_selection_audit_rejects_unsupported_variants(
    tmp_path: Path, field: str, value: str
) -> None:
    audit_path = tmp_path / "logs/sgs/rollout_000.json"
    audit_path.parent.mkdir(parents=True)
    draw_ids = [0, 0, 1, 2, 3, 4, 5, 6, 7, 7]
    rows = [
        {"source_draw_id": draw, "turn_idx": index if draw in {0, 7} else 0,
         "traj_uid": f"trajectory-{draw}", "teacher_js": float(10 - index),
         "selected": index < 8, "loss_weight": 0.5 if draw == 0 else 1.0}
        for index, draw in enumerate(draw_ids)
    ]
    audit = {
        "kind": "sgs_update", "rollout_id": 0, "fraction": 0.25,
        "selection_scope": "global", "score_scope": "response",
        "selection_mode": "sensitivity", "ranking_score_field": "teacher_js",
        "loss_aggregation": "trajectory_balanced", "dp_size": 8,
        "attempted_steps": 10, "scoreable_steps": 10, "selected_steps": 8,
        "retention_support": "all", "retention_kl_direction": "reverse",
        "retention_steps": 10, "trajectory_floor": False, "rows": rows,
    }
    audit_path.write_text(json.dumps(audit))
    _verify_selection_audits(tmp_path, 1, "textcraft", 0.25)

    audit[field] = value
    audit_path.write_text(json.dumps(audit))
    with pytest.raises(RuntimeError, match="selection contract drift"):
        _verify_selection_audits(tmp_path, 1, "textcraft", 0.25)


def test_smoke_changes_updates_without_batch_or_parent_drift(tmp_path: Path) -> None:
    run = tmp_path / "one"
    config = build_config(task="alfworld", model="4b", method="resail", cycle=1,
                          phase="training", run_root=run, input_root=tmp_path / "inputs",
                          base_checkpoint=BASE, base_hf=HF, mode="smoke", smoke_updates=2)
    cli = config["cli_args"]
    assert option(cli, "--num-rollout") == "2"
    assert option(cli, "--checkpoint-fixed-iterations") == "1"
    assert option(cli, "--rollout-batch-size") == option(cli, "--global-batch-size") == "32"
    parent = run / "alfworld/4b/resail/c1"
    (parent / "checkpoints/iter_0000001").mkdir(parents=True)
    (parent / "eval_hf/iter_0000001").mkdir(parents=True)
    (parent / "eval_hf/iter_0000001/config.json").write_text("{}")
    (parent / "completion.json").write_text(json.dumps({"status": "complete", "final_checkpoint": "iter_0000001"}))
    (parent / "verification.json").write_text(json.dumps({"status": "verified"}))
    assert prior_paths(run, "alfworld", "4b", "resail", 2, BASE, HF) == (
        parent / "checkpoints", parent / "eval_hf/iter_0000001")
    config2 = build_config(task="alfworld", model="4b", method="resail", cycle=2,
                           phase="training", run_root=run, input_root=tmp_path / "inputs",
                           base_checkpoint=BASE, base_hf=HF, mode="smoke", smoke_updates=2)
    assert option(config2["cli_args"], "--load").endswith("/resail/c1/checkpoints")
    evaluation = build_config(task="alfworld", model="4b", method="resail", cycle=2,
                              phase="eval", run_root=run, input_root=tmp_path / "inputs",
                              base_checkpoint=BASE, base_hf=HF, mode="smoke", smoke_updates=2)
    assert option(evaluation["cli_args"], "--rollout-seed") == "42"
    assert option(evaluation["cli_args"], "--load") == option(evaluation["cli_args"], "--ref-load")


def test_shared_c1_verification_does_not_rewrite_inputs(tmp_path: Path, monkeypatch) -> None:
    root = tmp_path / "shared/alfworld/4b/c1"
    (root / "collection").mkdir(parents=True)
    (root / "guidance").mkdir()
    monkeypatch.setattr(text, "_validate_c1_input_path", lambda *_: None)
    monkeypatch.setattr(text, "_verify_corpus", lambda *_: {"trajectory_count": 960})
    monkeypatch.setattr(text, "_verify_guidance", lambda *_: None)
    monkeypatch.setattr(text, "_corpus_uids", lambda *_: [str(i) for i in range(960)])
    manifest = {"schema_name": "paper_shared_c1_v1", "status": "verified",
        "task": "alfworld", "model": "4b", "task_seed": 42, "trajectory_count": 960,
        "base_checkpoint": str(BASE), "base_hf": str(HF), "corpus_dir": runtime(root / "corpus"),
        "guidance_dir": runtime(root / "guidance")}
    (root / "verification.json").write_text(json.dumps(manifest))
    (root / "collection/verification.json").write_text(json.dumps({
        "status": "verified", "trajectory_count": 960, "task_seed": 42}))
    (root / "guidance/summaries.jsonl").write_text("".join(
        json.dumps({"trajectory_uid": str(i)}) + "\n" for i in range(960)))
    files = sorted(path for path in root.rglob("*") if path.is_file())
    before = {path: path.read_bytes() for path in files}
    args = SimpleNamespace(c1_input=root, task="alfworld", model="4b", base_checkpoint=BASE, base_hf=HF)
    assert text._verify_shared_c1(args) == manifest
    assert {path: path.read_bytes() for path in files} == before

def test_textcraft_smoke_uses_unreplicated_dataset_name(tmp_path: Path, monkeypatch) -> None:
    from slime_plugins.agent_tasks.common.frozen.sampling import eval_turn_seed

    monkeypatch.setattr("exp.paper.text.ROOT", tmp_path)
    split = tmp_path / ".cache/textcraft/agentgym_rl_data_id/eval/textcraft_test.json"
    split.parent.mkdir(parents=True)
    split.write_text(json.dumps([{"item_id": f"textcraft_{i}"} for i in range(8)]))
    cell = tmp_path / "cell"
    result = cell / "eval/logs/eval_result/eval_000.json"
    samples = cell / "eval/samples/eval_0.jsonl"
    snapshot = cell / "eval/logs/eval_snapshot/eval_000.json"
    for path in (result, samples, snapshot):
        path.parent.mkdir(parents=True)
    rows = [{"eval_base_dataset_name": "textcraft_eval", "eval_replicate_index": 0,
             "eval_replicate_seed": 314159, "task_id": f"textcraft_{i}", "turn_idx": 0,
             "sampling_seed": eval_turn_seed(task="textcraft", replicate_seed=314159,
                                               task_id=f"textcraft_{i}", turn_idx=0),
             "success": True, "is_terminal": True} for i in range(8)]
    samples.write_text("".join(json.dumps(row) + "\n" for row in rows))
    result.write_text(json.dumps({"datasets": {"textcraft_eval": {
        "episode_count": 8, "step_count": 8, "success_count": 8}},
        "metrics": {"eval/textcraft_eval/episode/success_rate": 1.0}}))
    snapshot.write_text(json.dumps({"iteration": 0, "optimizer_update": 0,
        "data_source_cursor_before": 0, "data_source_cursor_after": 0,
        "independent_rng_namespace": True, "snapshot": {"engine_count": 8}}))
    assert set(_verify_textcraft_eval(cell, "smoke")) == {"textcraft_eval"}


@pytest.mark.parametrize("mode,episodes,seeds", (("smoke", 8, (314159,)),
                                                ("paper", 128, (314159, 314160, 314161))))
def test_alfworld_eval_replicate_layouts(tmp_path: Path, mode: str,
                                         episodes: int, seeds: tuple[int, ...]) -> None:
    cell = tmp_path / "cell"
    result_path = cell / "eval/logs/eval_result/eval_000.json"
    sample_path = cell / "eval/samples/eval_0.jsonl"
    snapshot_path = cell / "eval/logs/eval_snapshot/eval_000.json"
    for path in (result_path, sample_path, snapshot_path):
        path.parent.mkdir(parents=True, exist_ok=True)
    datasets: dict[str, dict[str, int]] = {}
    metrics: dict[str, float] = {}
    sample_rows: list[dict[str, object]] = []
    for base in ("alfworld_eval", "eval_out_of_distribution"):
        rates = []
        outcomes = []
        for index, seed in enumerate(seeds):
            name = (f"{base}__replicate_{index:02d}_seed_{seed}" if mode == "paper" else base)
            successes = {f"{base}-{i:04d}": i < episodes // 2 + index for i in range(episodes)}
            outcomes.append(successes)
            rates.append(sum(successes.values()) / episodes)
            datasets[name] = {"episode_count": episodes, "student_privileged_field_count": 0,
                              "success_count": sum(successes.values())}
            for uid, success in successes.items():
                sample_rows.append({"is_terminal": True, "eval_base_dataset_name": base,
                                    "eval_dataset_name": name, "eval_replicate_index": index,
                                    "eval_replicate_seed": seed, "uid": uid, "success": success})
        prefix = "eval/alfworld_eval" if base == "alfworld_eval" else "eval_out_of_distribution/alfworld_eval"
        if mode == "smoke":
            metrics[f"{prefix}/episode/success_rate"] = rates[0]
        else:
            metrics[f"{prefix}/episode/success_rate_replicate_count"] = len(seeds)
            metrics[f"{prefix}/episode/success_rate_mean"] = statistics.mean(rates)
            metrics[f"{prefix}/episode/success_rate_sample_std"] = statistics.stdev(rates)
            metrics[f"{prefix}/episode/pass_at_3"] = sum(
                any(rows[uid] for rows in outcomes) for uid in outcomes[0]) / episodes
    result_path.write_text(json.dumps({"datasets": datasets, "metrics": metrics}))
    sample_path.write_text("".join(json.dumps(row) + "\n" for row in sample_rows))
    snapshot_path.write_text(json.dumps({"iteration": 0, "optimizer_update": 0,
                                          "data_source_cursor_before": 0, "data_source_cursor_after": 0,
                                          "independent_rng_namespace": True, "snapshot": {"engine_count": 8}}))
    assert _verify_eval(cell, mode, "alfworld") == datasets
