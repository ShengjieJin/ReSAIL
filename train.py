# Modified for ReSAIL. See NOTICE and LICENSE for attribution and terms.
import json
import os
from pathlib import Path

import ray

from slime.ray.placement_group import create_placement_groups, create_rollout_manager, create_training_models
from slime.utils.arguments import parse_args
from slime.utils.checkpoint_retention import CheckpointRetentionManager
from slime.utils.logging_utils import configure_logger, finish_tracking, init_tracking, update_tracking_open_metrics
from slime.utils.misc import should_run_periodic_action
from slime.utils.training_lifecycle import (
    atomic_json as _atomic_json,
    eval_replicate_audit_fields,
    write_eval_snapshot_audit as _write_eval_snapshot_audit,
    write_policy_state as _write_policy_state,
)
from slime_plugins.agent_tasks.common.algorithms.sgs import sgs_filtering_enabled, sgs_scoring_required


def _env_bool(value) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() not in {"", "0", "false", "no", "off"}


def _agent_task_rollout_entropy_enabled(args) -> bool:
    value = getattr(args, "agent_task_diversity_entropy_enabled", None)
    if value is None:
        value = os.environ.get("AGENT_TASK_DIVERSITY_ENTROPY_ENABLED", False)
    return _env_bool(value)


def _prepare_strict_on_policy_audit(args) -> Path | None:
    value = str(getattr(args, "strict_on_policy_audit_dir", "") or "").strip()
    if not value:
        return None
    root = Path(value)
    root.mkdir(parents=True, exist_ok=True)
    start = int(args.start_rollout_id)
    if start > 0:
        state_path = Path(args.load) / "rollout" / f"policy_state_{start - 1}.json"
        try:
            state = json.loads(state_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"strict resume policy state is missing or invalid: {state_path}") from exc
        if (
            int(state.get("iteration", -1)) != start - 1
            or int(state.get("actor_version", -1)) != start
            or int(state.get("datasource_next_rollout_id", -1)) != start
        ):
            raise RuntimeError(f"strict resume policy/model/datasource boundary is inconsistent: {state}")
    for path in root.glob("update_[0-9][0-9][0-9].json"):
        update = int(path.stem.removeprefix("update_"))
        if update >= start:
            path.unlink()
    action_root_value = str(getattr(args, "alfworld_action_match_audit_dir", "") or "").strip()
    if action_root_value:
        action_root = Path(action_root_value)
        for path in action_root.glob("rollout_[0-9][0-9][0-9].json"):
            update = int(path.stem.removeprefix("rollout_"))
            if update >= start:
                path.unlink()
    return root




def _prepare_offline_lifecycle_audit(args) -> Path | None:
    if not bool(getattr(args, "offline_train_eval_colocate", False)):
        return None
    eval_root_value = str(getattr(args, "eval_snapshot_audit_dir", "") or "").strip()
    if not eval_root_value:
        raise RuntimeError("offline train/eval colocation requires eval snapshot auditing")
    root = Path(eval_root_value).parent / "offline_lifecycle"
    root.mkdir(parents=True, exist_ok=True)
    start = int(args.start_rollout_id)
    for path in root.glob("update_[0-9][0-9][0-9].json"):
        update = int(path.stem.removeprefix("update_"))
        if update >= start:
            path.unlink()
    return root


def train(args):
    configure_logger()
    strict_audit_root = _prepare_strict_on_policy_audit(args)
    offline_colocate = bool(getattr(args, "offline_train_eval_colocate", False))
    offline_lifecycle_root = _prepare_offline_lifecycle_audit(args)
    # allocate the GPUs
    pgs = create_placement_groups(args)
    init_tracking(args)

    # create the rollout manager, with sglang engines inside.
    # need to initialize rollout manager first to calculate num_rollout
    rollout_manager, num_rollout_per_epoch = create_rollout_manager(args, pgs["rollout"])

    # Update primary W&B with SGLang metrics endpoint now that servers are up.
    router_addr = ray.get(rollout_manager.get_metrics_router_addr.remote())
    update_tracking_open_metrics(args, router_addr)

    # create the actor and critic models
    actor_model, critic_model = create_training_models(args, pgs, rollout_manager)
    checkpoint_retention = CheckpointRetentionManager(args)
    if args.keep_old_actor and _agent_task_rollout_entropy_enabled(args):
        raise RuntimeError(
            "agent-task rollout actor entropy is incompatible with --keep-old-actor in synchronous training; "
            "disable diversity entropy or --keep-old-actor"
        )

    if args.offload_rollout:
        ray.get(rollout_manager.onload_weights.remote())

    # Always push actor weights to rollout once weights are loaded.
    actor_model.update_weights()

    if args.check_weight_update_equal:
        ray.get(rollout_manager.check_weights.remote(action="compare"))

    if args.offload_rollout:
        ray.get(rollout_manager.onload_kv.remote())

    if offline_colocate:
        ray.get(rollout_manager.offload.remote())
        actor_model.onload()

    def run_eval(rollout_id):
        if offline_colocate:
            actor_model.offload()
            ray.get(rollout_manager.onload_weights.remote())
            actor_model.update_weights()
            ray.get(rollout_manager.onload_kv.remote())
        cursor_before = ray.get(rollout_manager.get_data_source_cursor_state.remote())
        snapshot = ray.get(rollout_manager.register_eval_snapshot.remote(rollout_id))
        if int(snapshot["engine_count"]) <= 0:
            raise RuntimeError(f"eval snapshot {rollout_id} has no rollout engines")
        try:
            eval_result = ray.get(rollout_manager.eval.remote(rollout_id))
            verified_snapshot = ray.get(rollout_manager.verify_eval_snapshot.remote(rollout_id))
            cursor_after = ray.get(rollout_manager.get_data_source_cursor_state.remote())
        finally:
            if offline_colocate:
                ray.get(rollout_manager.offload.remote())
                actor_model.onload()
        if isinstance(eval_result, dict) and eval_result.get("requires_actor_entropy"):
            if args.offload_rollout:
                ray.get(rollout_manager.offload.remote())
            try:
                entropy_payloads = ray.get(
                    actor_model.async_score_entropy(rollout_id, eval_result["rollout_data_ref"], phase="eval")
                )
                eval_result = ray.get(
                    rollout_manager.finalize_eval_entropy.remote(
                        rollout_id, entropy_payloads, eval_result.get("metrics")
                    )
                )
            finally:
                if args.offload_rollout and not offline_colocate:
                    ray.get(rollout_manager.onload_weights.remote())
                    ray.get(rollout_manager.onload_kv.remote())
        if verified_snapshot != snapshot:
            raise RuntimeError(
                f"eval snapshot changed during evaluation: before={snapshot}, after={verified_snapshot}"
            )
        if cursor_after != cursor_before:
            raise RuntimeError(
                f"evaluation advanced the training data source: before={cursor_before}, after={cursor_after}"
            )
        _write_eval_snapshot_audit(
            args,
            rollout_id,
            {
                "schema_version": 1,
                "iteration": int(rollout_id),
                "optimizer_update": int(rollout_id) + int(args.num_rollout > 0),
                "snapshot": snapshot,
                "data_source_cursor_before": cursor_before,
                "data_source_cursor_after": cursor_after,
                **eval_replicate_audit_fields(args),
                "training_rollout_seed": int(getattr(args, "rollout_seed", 42)),
                "independent_rng_namespace": True,
            },
        )
        return eval_result

    # special case for eval-only
    if args.num_rollout == 0 and args.eval_interval is not None:
        run_eval(args.start_rollout_id)

    # A checkpoint is made durable before its scheduled evaluation.  If the
    # process dies in that narrow window, finish the owed evaluation against
    # the restored checkpoint before consuming another training batch.  The
    # retention tracker makes this idempotent once after_eval has committed.
    if args.num_rollout > 0 and args.start_rollout_id > 0:
        resume_iteration = int(args.start_rollout_id) - 1
        if checkpoint_retention.needs_recovery_eval(resume_iteration, num_rollout_per_epoch):
            eval_metrics = run_eval(resume_iteration)
            checkpoint_retention.after_eval(resume_iteration, eval_metrics)

    def offload_train(actor_trains_this_step):
        # Each model auto-offloads after train() when offload_train is set,
        # so we only need clear_memory for the non-offload case.
        if not args.offload_train:
            if not args.use_critic or actor_trains_this_step:
                actor_model.clear_memory()
            else:
                critic_model.clear_memory()

    def save(rollout_id):
        actor_trains_this_step = (not args.use_critic) or rollout_id >= args.num_critic_only_steps
        force_sync_save = (
            (rollout_id == args.num_rollout - 1)
            or checkpoint_retention.requires_sync_save
            or strict_audit_root is not None
            or offline_colocate
        )
        if actor_trains_this_step:
            actor_model.save_model(
                rollout_id,
                force_sync=force_sync_save,
            )
        if args.use_critic:
            critic_model.save_model(
                rollout_id,
                force_sync=force_sync_save,
            )
        if args.rollout_global_dataset:
            ray.get(rollout_manager.save.remote(rollout_id))
        _write_policy_state(args, rollout_id)
        checkpoint_retention.after_save(rollout_id)
        return rollout_id

    # Process-local SGLang versions reset when engines are recreated on
    # resume.  Within one process they must advance exactly once per update;
    # the absolute actor version is checkpointed separately in policy_state.
    previous_engine_post_update_weight_version: str | None = None

    # train loop.
    for rollout_id in range(args.start_rollout_id, args.num_rollout):
        if args.eval_interval is not None and rollout_id == 0 and not args.skip_eval_before_train:
            run_eval(rollout_id)

        training_snapshot = None
        if strict_audit_root is not None:
            training_snapshot = ray.get(
                rollout_manager.register_training_snapshot.remote(rollout_id, rollout_id)
            )
            if int(training_snapshot["engine_count"]) != int(args.rollout_num_gpus):
                raise RuntimeError(
                    "strict rollout snapshot engine count must equal the full rollout GPU count: "
                    f"snapshot={training_snapshot}, rollout_num_gpus={args.rollout_num_gpus}"
                )
        rollout_data_ref = ray.get(rollout_manager.generate.remote(rollout_id))
        if strict_audit_root is not None:
            verified_snapshot = ray.get(
                rollout_manager.verify_training_snapshot.remote(rollout_id, rollout_id)
            )
            if verified_snapshot != training_snapshot:
                raise RuntimeError(
                    f"strict training snapshot changed during rollout: before={training_snapshot}, "
                    f"after={verified_snapshot}"
                )

        if args.offload_rollout and not offline_colocate:
            ray.get(rollout_manager.offload.remote())

        if sgs_scoring_required(args):
            sgs_results = ray.get(
                actor_model.async_score_sgs(rollout_id, rollout_data_ref)
            )
            rollout_data_ref = ray.get(
                rollout_manager.finalize_sgs.remote(rollout_id, sgs_results)
            )

        actor_trains_this_step = (not args.use_critic) or rollout_id >= args.num_critic_only_steps
        actor_train_results = None

        if args.use_critic:
            value_refs = critic_model.async_train(rollout_id, rollout_data_ref)
            if actor_trains_this_step:
                actor_train_results = ray.get(
                    actor_model.async_train(rollout_id, rollout_data_ref, external_data=value_refs)
                )
            else:
                ray.get(value_refs)
        else:
            actor_train_results = ray.get(actor_model.async_train(rollout_id, rollout_data_ref))

        save_num_rollout = None if getattr(args, "skip_final_save", False) else args.num_rollout
        saved_checkpoint_step = None
        should_save = should_run_periodic_action(
            rollout_id, args.save_interval, num_rollout_per_epoch, save_num_rollout
        )
        should_save = should_save or checkpoint_retention.should_save_for_eval(rollout_id, num_rollout_per_epoch)
        if should_save:
            saved_checkpoint_step = save(rollout_id)

        if not offline_colocate:
            offload_train(actor_trains_this_step)
            if args.offload_rollout:
                ray.get(rollout_manager.onload_weights.remote())
            actor_model.update_weights()

            if args.offload_rollout:
                ray.get(rollout_manager.onload_kv.remote())

        if strict_audit_root is not None:
            post_update_versions = ray.get(rollout_manager.get_updatable_weight_versions.remote())
            if not post_update_versions or len(set(post_update_versions)) != 1:
                raise RuntimeError(f"strict post-update rollout weights diverged: {post_update_versions}")
            engine_pre_update_weight_version = str(training_snapshot["weight_version"])
            engine_post_update_weight_version = str(post_update_versions[0])
            try:
                version_advanced_once = int(engine_post_update_weight_version) == int(
                    engine_pre_update_weight_version
                ) + 1
            except ValueError:
                version_advanced_once = engine_post_update_weight_version != engine_pre_update_weight_version
            if not version_advanced_once:
                raise RuntimeError(
                    "strict rollout engine version did not advance exactly once: "
                    f"{engine_pre_update_weight_version} -> {engine_post_update_weight_version}"
                )
            process_start = previous_engine_post_update_weight_version is None
            if not process_start and engine_pre_update_weight_version != previous_engine_post_update_weight_version:
                raise RuntimeError(
                    "strict rollout engine version chain broke within one process: "
                    f"previous_post={previous_engine_post_update_weight_version}, "
                    f"current_pre={engine_pre_update_weight_version}"
                )
            _atomic_json(
                strict_audit_root / f"update_{rollout_id:03d}.json",
                {
                    "schema_version": 1,
                    "update": rollout_id,
                    "actor_pre_update_version": rollout_id,
                    "rollout_policy_version": rollout_id,
                    "actor_post_update_version": rollout_id + 1,
                    "policy_lag": 0,
                    "prefetched_training_batches": 0,
                    "rollout_completed_before_train": True,
                    "rollout_snapshot": training_snapshot,
                    "post_update_weight_versions": post_update_versions,
                    "engine_pre_update_weight_version": engine_pre_update_weight_version,
                    "engine_post_update_weight_version": engine_post_update_weight_version,
                    "engine_weight_version_advanced_once": True,
                    "engine_weight_version_process_start": process_start,
                    "rollout_gpu_count": int(args.rollout_num_gpus),
                    "train_gpu_count": int(args.actor_num_nodes) * int(args.actor_num_gpus_per_node),
                    "role_sequence": ["rollout", "global_barrier", "train", "weight_sync"],
                },
            )
            previous_engine_post_update_weight_version = engine_post_update_weight_version

        runs_eval = should_run_periodic_action(rollout_id, args.eval_interval, num_rollout_per_epoch)
        if runs_eval:
            eval_metrics = run_eval(rollout_id)
            checkpoint_retention.after_eval(saved_checkpoint_step, eval_metrics)
        if offline_lifecycle_root is not None:
            old_actor_refresh_flags = [
                bool(result.get("old_actor_refreshed", False))
                for result in (actor_train_results or [])
                if isinstance(result, dict)
            ]
            old_actor_refreshed = bool(old_actor_refresh_flags) and all(old_actor_refresh_flags)
            if args.keep_old_actor and (
                len(old_actor_refresh_flags) != int(args.actor_num_nodes) * int(args.actor_num_gpus_per_node)
                or not old_actor_refreshed
            ):
                raise RuntimeError("offline keep-old-actor was not refreshed on every actor rank")
            role_sequence = ["frozen_cpu_replay", "train_all_8"]
            if runs_eval:
                role_sequence.extend(
                    ["actor_offload", "eval_rollout_all_8", "rollout_offload", "train_all_8_restored"]
                )
            _atomic_json(
                offline_lifecycle_root / f"update_{rollout_id:03d}.json",
                {
                    "schema_version": 1,
                    "update": int(rollout_id),
                    "train_gpu_count": int(args.actor_num_nodes) * int(args.actor_num_gpus_per_node),
                    "eval_rollout_gpu_count": int(args.rollout_num_gpus) if runs_eval else 0,
                    "training_rollout_uses_sglang": False,
                    "rollout_engines_offloaded_during_train": True,
                    "actor_onloaded_during_train": True,
                    "eval_ran": bool(runs_eval),
                    "actor_restored_after_eval": bool(runs_eval),
                    "keep_old_actor": bool(args.keep_old_actor),
                    "old_actor_refreshed": old_actor_refreshed,
                    "old_actor_refresh_rank_count": sum(old_actor_refresh_flags),
                    "role_sequence": role_sequence,
                },
            )

    ray.get(rollout_manager.dispose.remote())
    finish_tracking(args)


if __name__ == "__main__":
    args = parse_args()
    train(args)
