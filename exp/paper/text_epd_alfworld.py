#!/usr/bin/env python3
"""Materialize ALFWorld EPD targets from summary guidelines."""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from slime.utils.processing_utils import load_tokenizer
from slime_plugins.agent_tasks.alfworld.frozen.epd import (
    EPD_MANIFEST_SCHEMA_VERSION,
    canonical_steps_from_corpus,
    load_target_records,
    materialize_records,
    validate_teacher_output,
)
from slime_plugins.agent_tasks.common.frozen.epd import build_sampling_params


TRAJECTORY_COUNT = 960
DEFAULT_ENGINE_COUNT = 4
MAX_ENGINE_COUNT = 8
GLOBAL_CONCURRENCY = 256
MAX_ATTEMPTS = 3
BASE_RELEASE_ITERATION = -1
TEACHER_PROMPT_MAX_TOKENS = 10240
TEACHER_PROMPT_TRUNCATION_SIDE = "right"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--guidance-summary-dir", type=Path, required=True)
    parser.add_argument("--trajectory-schedule", type=Path)
    parser.add_argument("--materialization-identity", required=True)
    parser.add_argument("--teacher-iteration", type=int, required=True)
    parser.add_argument("--endpoint-file", type=Path, required=True)
    parser.add_argument("--engine-count", type=int, default=DEFAULT_ENGINE_COUNT)
    parser.add_argument("--global-concurrency", type=int, default=GLOBAL_CONCURRENCY)
    parser.add_argument("--expected-trajectories", type=int, default=TRAJECTORY_COUNT)
    parser.add_argument("--source-trajectories", type=int, default=960)
    parser.add_argument("--shard-size", type=int, default=1024)
    parser.add_argument("--runtime-profile")
    parser.add_argument("--execution-mode")
    parser.add_argument("--requires-live-environment", action="store_true")
    return parser


def direct_binding(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "corpus_dir": str(args.corpus_dir),
        "guidance_summary_dir": str(args.guidance_summary_dir),
        "materialization_identity": args.materialization_identity,
        "model_path": str(args.model_path),
        "teacher_iteration": int(args.teacher_iteration),
    }


def _validate_args(args: argparse.Namespace) -> None:
    if args.expected_trajectories != TRAJECTORY_COUNT:
        raise ValueError("ALFWorld EPD requires exactly 960 trajectories")
    if getattr(args, "source_trajectories", 960) < args.expected_trajectories:
        raise ValueError("ALFWorld EPD source trajectory count cannot be smaller than the selected count")
    if args.teacher_iteration != BASE_RELEASE_ITERATION and not 0 <= args.teacher_iteration <= 29:
        raise ValueError("EPD teacher iteration must be the base release or a saved update 0-29")
    expects_base = "base-release" in args.materialization_identity
    if expects_base != (args.teacher_iteration == BASE_RELEASE_ITERATION):
        raise ValueError("ALFWorld EPD base-release identity/sentinel mismatch")
    if not 1 <= args.engine_count <= MAX_ENGINE_COUNT:
        raise ValueError(f"ALFWorld EPD engine count must be in [1, {MAX_ENGINE_COUNT}]")
    if args.global_concurrency < args.engine_count:
        raise ValueError("ALFWorld EPD global concurrency must cover every engine")


async def _run(args: argparse.Namespace) -> list[dict[str, Any]]:
    endpoints = [
        line.strip().rstrip("/")
        for line in args.endpoint_file.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    if len(endpoints) != args.engine_count or len(set(endpoints)) != args.engine_count:
        raise ValueError(f"ALFWorld EPD requires exactly {args.engine_count} distinct endpoints")
    physical_count = int(args.source_trajectories)
    steps = canonical_steps_from_corpus(args.corpus_dir, expected_trajectories=physical_count)
    tokenizer = load_tokenizer(str(args.model_path), trust_remote_code=True)

    from slime_plugins.agent_tasks.common.frozen.guidance import (
        build_summary_teacher_context,
        load_summary_records,
    )

    summary_args = SimpleNamespace(
        agent_frozen_guidance_summary_dir=str(args.guidance_summary_dir),
        agent_frozen_expected_trajectories=TRAJECTORY_COUNT,
        agent_frozen_corpus_dir=str(args.corpus_dir),
        agent_frozen_empty_guideline_policy="skip_trajectory",
    )
    summaries = load_summary_records(summary_args)
    guidance_manifest = json.loads((args.guidance_summary_dir / "manifest.json").read_text(encoding="utf-8"))
    empty_uids = [str(uid) for uid in guidance_manifest.get("empty_guideline_uids", [])]
    if args.trajectory_schedule is not None:
        schedule = json.loads(args.trajectory_schedule.read_text(encoding="utf-8"))
        scheduled_uids = [str(row["trajectory_uid"]) for row in schedule.get("draws", [])]
        if (
            len(scheduled_uids) != TRAJECTORY_COUNT
            or len(set(scheduled_uids)) != TRAJECTORY_COUNT
            or set(scheduled_uids) != set(summaries) | set(empty_uids)
        ):
            raise ValueError("ALFWorld C1 EPD schedule differs from the 960 summary trajectories")
        order = {uid: index for index, uid in enumerate(scheduled_uids)}
        steps = [step for step in steps if str(step["trajectory_uid"]) in summaries]
        steps.sort(key=lambda step: (order[str(step["trajectory_uid"])], int(step["turn_idx"])))
    else:
        steps = [step for step in steps if str(step["trajectory_uid"]) in summaries]
    for index, step in enumerate(steps):
        step["canonical_index"] = index

    def teacher_context_builder(trajectory: dict[str, Any], turn: dict[str, Any]) -> dict[str, Any]:
        return build_summary_teacher_context(
            trajectory, turn, summaries[str(trajectory["trajectory_uid"])]
        )

    sampling = build_sampling_params(temperature=1.0, max_new_tokens=1024, seed=42)
    config = {
        "task": "alfworld",
        "method": "epd",
        "phase": "teacher_target_materialization",
        "trajectory_count": TRAJECTORY_COUNT,
        "source_trajectory_count": physical_count,
        "trajectory_schedule": str(args.trajectory_schedule) if args.trajectory_schedule else None,
        "effective_trajectory_count": len(summaries),
        "empty_guideline_count": len(empty_uids),
        "empty_guideline_uids": empty_uids,
        "empty_guideline_turn_count": int(guidance_manifest.get("empty_guideline_turn_count", 0)),
        "canonical_steps": len(steps),
        "engine_count": args.engine_count,
        "global_concurrency": args.global_concurrency,
        "sampling": sampling,
        "max_attempts": MAX_ATTEMPTS,
        "success_filter": False,
        "branch_packing": False,
        "student_visible_prefix": "original",
        "teacher_context_mode": "own_outcome",
        "solution_context_format": "guidance_plan",
        "metadata_profile": "alfworld",
        "teacher_prompt_max_tokens": TEACHER_PROMPT_MAX_TOKENS,
        "teacher_prompt_truncation_side": TEACHER_PROMPT_TRUNCATION_SIDE,
        "teacher_iteration": args.teacher_iteration,
        "base_release_iteration_sentinel": BASE_RELEASE_ITERATION,
    }
    if getattr(args, "runtime_profile", None) is not None:
        config.update(
            {
                "runtime_profile": args.runtime_profile,
                "execution_mode": args.execution_mode,
                "requires_live_environment": bool(args.requires_live_environment),
            }
        )

    import httpx

    limits = httpx.Limits(max_connections=args.global_concurrency, max_keepalive_connections=args.global_concurrency)
    timeout = httpx.Timeout(connect=30.0, read=300.0, write=300.0, pool=30.0)
    async with httpx.AsyncClient(limits=limits, timeout=timeout, trust_env=False) as client:

        async def generate_request(request: dict[str, Any], endpoint: str) -> dict[str, Any]:
            response = await client.post(
                f"{endpoint}/generate",
                json={
                    "input_ids": request["input_ids"],
                    "sampling_params": request["sampling_params"],
                    "return_logprob": True,
                },
            )
            response.raise_for_status()
            return response.json()

        return await materialize_records(
            steps,
            output_dir=args.output_dir,
            endpoints=endpoints,
            tokenizer=tokenizer,
            generate_request=generate_request,
            expected_count=len(steps),
            worker_count=args.engine_count,
            global_concurrency=args.global_concurrency,
            shard_size=args.shard_size,
            config=config,
            max_attempts=MAX_ATTEMPTS,
            binding_mode="direct_v1",
            direct_binding=direct_binding(args),
            metadata_profile="alfworld",
            teacher_output_validator=validate_teacher_output,
            sampling_params_overrides=sampling,
            teacher_prompt_max_tokens=TEACHER_PROMPT_MAX_TOKENS,
            teacher_prompt_truncation_side=TEACHER_PROMPT_TRUNCATION_SIDE,
            teacher_context_builder=teacher_context_builder,
        )


def verify(args: argparse.Namespace) -> dict[str, Any]:
    manifest = json.loads((args.output_dir / "manifest.json").read_text(encoding="utf-8"))
    records = load_target_records(args.output_dir)
    config = manifest.get("config", {})
    effective = int(config.get("effective_trajectory_count", -1))
    empty_uids = [str(uid) for uid in config.get("empty_guideline_uids", [])]
    physical_count = int(config.get("source_trajectory_count", -1))
    expected_steps = canonical_steps_from_corpus(
        args.corpus_dir, expected_trajectories=physical_count
    )
    if getattr(args, "trajectory_schedule", None) is not None:
        schedule = json.loads(args.trajectory_schedule.read_text(encoding="utf-8"))
        selected_uids = [str(row["trajectory_uid"]) for row in schedule.get("draws", [])]
    else:
        selected_uids = sorted({str(step["trajectory_uid"]) for step in expected_steps})
    selected_uid_set = set(selected_uids)
    expected_identities = {
        str(step["identity"])
        for step in expected_steps
        if str(step["trajectory_uid"]) in selected_uid_set - set(empty_uids)
    }
    actual_identities = {str(record.get("identity", "")) for record in records}
    forbidden = {
        key
        for value in [manifest, *manifest.get("shards", []), *records]
        for key in value
        if "hash" in key or "sha256" in key
    }
    runtime_contract_valid = True
    if getattr(args, "runtime_profile", None) is not None:
        runtime_contract_valid = (
            config.get("runtime_profile") == args.runtime_profile
            and config.get("execution_mode") == args.execution_mode
            and config.get("requires_live_environment") is bool(args.requires_live_environment)
        )
    if (
        manifest.get("schema_version") != EPD_MANIFEST_SCHEMA_VERSION
        or manifest.get("kind") != "epd_teacher_targets"
        or manifest.get("status") != "complete"
        or manifest.get("binding_mode") != "direct_v1"
        or manifest.get("direct_binding") != direct_binding(args)
        or manifest.get("trajectory_count") != effective
        or effective + len(empty_uids) != TRAJECTORY_COUNT
        or len(set(empty_uids)) != len(empty_uids)
        or manifest.get("target_count") != len(records)
        or manifest.get("canonical_turn_count") != len(records)
        or manifest.get("success_filter") is not False
        or manifest.get("branch_packing") is not False
        or manifest.get("student_visible_prefix") != "original"
        or manifest.get("teacher_context_mode") != "own_outcome"
        or config.get("solution_context_format") != "guidance_plan"
        or config.get("metadata_profile") != "alfworld"
        or config.get("sampling", {}).get("max_new_tokens") != 1024
        or len(selected_uids) != TRAJECTORY_COUNT
        or len(selected_uid_set) != TRAJECTORY_COUNT
        or {str(record["trajectory_uid"]) for record in records} != selected_uid_set - set(empty_uids)
        or actual_identities != expected_identities
        or forbidden
        or not runtime_contract_valid
    ):
        raise RuntimeError(f"ALFWorld EPD verification failed; forbidden={sorted(forbidden)}")
    payload = {
        "schema_version": 1,
        "status": "verified",
        "kind": "alfworld_epd_teacher_targets",
        "binding_mode": "direct_v1",
        "direct_binding": direct_binding(args),
        "corpus_bound": True,
        "source_trajectory_count": physical_count,
        "trajectory_count": effective,
        "empty_guideline_count": len(empty_uids),
        "canonical_turn_count": len(records),
        "target_count": len(records),
        "valid_target_count": int(manifest["valid_target_count"]),
        "invalid_target_count": int(manifest["invalid_target_count"]),
    }
    (args.output_dir / "verification.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return payload


def main() -> None:
    args = _parser().parse_args()
    _validate_args(args)
    if not (args.output_dir / "manifest.json").is_file():
        asyncio.run(_run(args))
    print(json.dumps(verify(args), sort_keys=True))


if __name__ == "__main__":
    main()
