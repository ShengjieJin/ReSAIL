#!/usr/bin/env python3
"""Materialize TextCraft EPD teacher targets with direct bindings."""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from slime.utils.processing_utils import load_tokenizer
from slime_plugins.agent_tasks.common.frozen.epd import (
    EPD_MANIFEST_SCHEMA_VERSION,
    build_sampling_params,
    canonical_steps_from_corpus,
    load_target_records,
    materialize_records,
)

TRAJECTORY_COUNT = 240
ENGINE_COUNT = 8
GLOBAL_CONCURRENCY = 512
MAX_ATTEMPTS = 3
BASE_RELEASE_ITERATION = -1
TEACHER_PROMPT_MAX_TOKENS = 10240
TEACHER_PROMPT_TRUNCATION_SIDE = "right"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--materialization-identity", required=True)
    parser.add_argument("--teacher-iteration", type=int, required=True)
    parser.add_argument("--endpoint-file", type=Path, required=True)
    parser.add_argument("--engine-count", type=int, default=ENGINE_COUNT)
    parser.add_argument("--global-concurrency", type=int, default=GLOBAL_CONCURRENCY)
    parser.add_argument("--guidance-summary-dir", type=Path)
    parser.add_argument("--metadata-profile", choices=("textcraft",), default="textcraft")
    parser.add_argument("--expected-trajectories", type=int, default=TRAJECTORY_COUNT)
    parser.add_argument("--shard-size", type=int, default=1024)
    parser.add_argument("--attempt-seed-offset", type=int, default=0)
    return parser


def direct_binding(args: argparse.Namespace) -> dict[str, Any]:
    binding = {
        "corpus_dir": str(args.corpus_dir),
        "materialization_identity": args.materialization_identity,
        "model_path": str(args.model_path),
        "teacher_iteration": int(args.teacher_iteration),
    }
    guidance_summary_dir = getattr(args, "guidance_summary_dir", None)
    if guidance_summary_dir is not None:
        binding["guidance_summary_dir"] = str(guidance_summary_dir)
    return binding


def _validate_args(args: argparse.Namespace) -> None:
    if args.expected_trajectories <= 0:
        raise ValueError("EPD expected trajectories must be positive")
    if args.teacher_iteration != BASE_RELEASE_ITERATION and not 0 <= args.teacher_iteration <= 29:
        raise ValueError("EPD teacher iteration must be the base release or a saved update 0-29")
    expects_base = "base-release" in args.materialization_identity
    if expects_base != (args.teacher_iteration == BASE_RELEASE_ITERATION):
        raise ValueError("TextCraft EPD base-release identity/sentinel mismatch")
    if not 1 <= args.engine_count <= ENGINE_COUNT:
        raise ValueError(f"TextCraft EPD engine count must be in [1, {ENGINE_COUNT}]")
    if args.global_concurrency < args.engine_count:
        raise ValueError("TextCraft EPD global concurrency must cover every engine")
    if args.attempt_seed_offset < 0:
        raise ValueError("TextCraft EPD attempt seed offset must be nonnegative")


async def _run(args: argparse.Namespace) -> list[dict[str, Any]]:
    endpoints = [
        line.strip().rstrip("/")
        for line in args.endpoint_file.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    if len(endpoints) != args.engine_count or len(set(endpoints)) != args.engine_count:
        raise ValueError(f"TextCraft EPD requires exactly {args.engine_count} distinct endpoints")
    steps = canonical_steps_from_corpus(
        args.corpus_dir,
        expected_trajectories=args.expected_trajectories,
        expected_steps=None,
    )
    tokenizer = load_tokenizer(str(args.model_path), trust_remote_code=True)
    binding = direct_binding(args)
    sampling = build_sampling_params(
        temperature=1.0,
        max_new_tokens=512,
        seed=42,
    )
    metadata_profile = str(args.metadata_profile)
    task = "textcraft"
    config = {
        "task": task,
        "method": "epd",
        "phase": "teacher_target_materialization",
        "trajectory_count": args.expected_trajectories,
        "canonical_steps": len(steps),
        "engine_count": args.engine_count,
        "global_concurrency": args.global_concurrency,
        "sampling": sampling,
        "max_attempts": MAX_ATTEMPTS,
        "attempt_seed_offset": args.attempt_seed_offset,
        "success_filter": False,
        "branch_packing": False,
        "student_visible_prefix": "original",
        "teacher_context_mode": "own_outcome",
        "solution_context_format": (
            "guidance_plan" if getattr(args, "guidance_summary_dir", None) else "trajectory_demo"
        ),
        "metadata_profile": metadata_profile,
        "teacher_prompt_max_tokens": TEACHER_PROMPT_MAX_TOKENS,
        "teacher_prompt_truncation_side": TEACHER_PROMPT_TRUNCATION_SIDE,
        "teacher_iteration": args.teacher_iteration,
        "base_release_iteration_sentinel": BASE_RELEASE_ITERATION,
    }

    import httpx

    teacher_context_builder = None
    if getattr(args, "guidance_summary_dir", None) is not None:
        from slime_plugins.agent_tasks.common.frozen.guidance import (
            build_summary_teacher_context,
            load_summary_records,
        )

        summary_args = SimpleNamespace(
            agent_frozen_guidance_summary_dir=str(args.guidance_summary_dir),
            agent_frozen_expected_trajectories=args.expected_trajectories,
            agent_frozen_corpus_dir=str(args.corpus_dir),
            agent_frozen_empty_guideline_policy="skip_trajectory",
        )
        summary_records = load_summary_records(summary_args)
        guidance_manifest = json.loads((args.guidance_summary_dir / "manifest.json").read_text(encoding="utf-8"))
        empty_uids = [str(uid) for uid in guidance_manifest.get("empty_guideline_uids", [])]
        steps = [step for step in steps if str(step["trajectory_uid"]) in summary_records]
        for canonical_index, step in enumerate(steps):
            step["canonical_index"] = canonical_index
        config["empty_guideline_count"] = len(empty_uids)
        config["empty_guideline_uids"] = empty_uids
        config["empty_guideline_turn_count"] = int(guidance_manifest.get("empty_guideline_turn_count", 0))
        config["effective_trajectory_count"] = len(summary_records)
        config["canonical_steps"] = len(steps)

        def teacher_context_builder(trajectory: dict[str, Any], turn: dict[str, Any]) -> dict[str, Any]:
            return build_summary_teacher_context(
                trajectory,
                turn,
                summary_records[str(trajectory["trajectory_uid"])],
                metadata_profile=metadata_profile,
            )

    limits = httpx.Limits(
        max_connections=args.global_concurrency,
        max_keepalive_connections=args.global_concurrency,
    )
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

        materialize_kwargs = {}
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
            direct_binding=binding,
            teacher_prompt_max_tokens=TEACHER_PROMPT_MAX_TOKENS,
            teacher_prompt_truncation_side=TEACHER_PROMPT_TRUNCATION_SIDE,
            teacher_context_builder=teacher_context_builder,
            metadata_profile=metadata_profile,
            attempt_seed_offset=args.attempt_seed_offset,
            **materialize_kwargs,
        )


def verify(args: argparse.Namespace) -> dict[str, Any]:
    manifest = json.loads((args.output_dir / "manifest.json").read_text(encoding="utf-8"))
    records = load_target_records(args.output_dir)
    binding = direct_binding(args)
    config = manifest.get("config", {})
    effective_trajectories = int(config.get("effective_trajectory_count", args.expected_trajectories))
    empty_uids = config.get("empty_guideline_uids", [])
    empty_count = int(config.get("empty_guideline_count", 0))
    forbidden = {key for key in manifest if "hash" in key or "sha256" in key}
    forbidden |= {key for shard in manifest.get("shards", []) for key in shard if "hash" in key or "sha256" in key}
    forbidden |= {key for record in records for key in record if "hash" in key or "sha256" in key}
    if (
        manifest.get("schema_version") != EPD_MANIFEST_SCHEMA_VERSION
        or manifest.get("kind") != "epd_teacher_targets"
        or manifest.get("status") != "complete"
        or manifest.get("binding_mode") != "direct_v1"
        or manifest.get("direct_binding") != binding
        or manifest.get("trajectory_count") != effective_trajectories
        or effective_trajectories + empty_count != args.expected_trajectories
        or not isinstance(empty_uids, list)
        or len(set(str(uid) for uid in empty_uids)) != empty_count
        or manifest.get("target_count") != len(records)
        or manifest.get("canonical_turn_count") != len(records)
        or manifest.get("success_filter") is not False
        or manifest.get("branch_packing") is not False
        or manifest.get("student_visible_prefix") != "original"
        or manifest.get("teacher_context_mode") != "own_outcome"
        or manifest.get("config", {}).get("sampling", {}).get("max_new_tokens") != 512
        or config.get("metadata_profile", "textcraft") != str(args.metadata_profile)
        or manifest.get("config", {}).get("teacher_prompt_max_tokens") != TEACHER_PROMPT_MAX_TOKENS
        or manifest.get("config", {}).get("teacher_prompt_truncation_side") != TEACHER_PROMPT_TRUNCATION_SIDE
        or manifest.get("config", {}).get("solution_context_format", "trajectory_demo")
        != ("guidance_plan" if getattr(args, "guidance_summary_dir", None) else "trajectory_demo")
        or forbidden
    ):
        raise RuntimeError(f"EPD direct materialization verification failed; forbidden={sorted(forbidden)}")
    metadata_profile = str(args.metadata_profile)
    payload = {
        "schema_version": 1,
        "status": "verified",
        "kind": "textcraft_epd_teacher_targets",
        "binding_mode": "direct_v1",
        "direct_binding": binding,
        "corpus_bound": True,
        "source_trajectory_count": args.expected_trajectories,
        "trajectory_count": effective_trajectories,
        "empty_guideline_count": empty_count,
        "empty_guideline_uids": [str(uid) for uid in empty_uids],
        "empty_guideline_turn_count": int(config.get("empty_guideline_turn_count", 0)),
        "canonical_turn_count": len(records),
        "target_count": len(records),
        "valid_target_count": int(manifest["valid_target_count"]),
        "invalid_target_count": int(manifest["invalid_target_count"]),
        "base_release_iteration_sentinel": BASE_RELEASE_ITERATION,
    }
    (args.output_dir / "verification.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
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
