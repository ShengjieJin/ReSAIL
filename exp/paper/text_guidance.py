#!/usr/bin/env python3
"""Materialize one strict own-outcome guidance summary per paper trajectory."""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
import re
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from slime.utils.processing_utils import load_tokenizer
from slime_plugins.agent_tasks.common.algorithms.sdpo_context import clean_guidance_summary
from slime_plugins.agent_tasks.common.frozen.epd import canonical_steps_from_corpus, parse_generation_output
from slime_plugins.agent_tasks.common.frozen.guidance import (
    SUMMARY_SCHEMA_VERSION,
    build_summary_request,
    summary_has_schema,
    validate_summary_record,
)

TRAJECTORY_COUNT = 960
ENGINE_COUNT = 8
GLOBAL_CONCURRENCY = 256
MAX_ATTEMPTS = 3
MAX_NEW_TOKENS = 1024
MAX_PROMPT_TOKENS = 40192
REQUEST_TIMEOUT_SECONDS = 120.0
ATTEMPT_EVIDENCE_SCHEMA = 3
RETRYABLE_HTTP_STATUS_CODES = frozenset({408, 425, 429, 500, 502, 503, 504})
SCHEMA_BULLET_LABELS = (
    "Minimal plan",
    "Critical actions",
    "Checks",
    "Avoid",
    "Failure diagnosis",
    "Useful evidence",
    "Corrected plan",
)


def _normalize_schema_label_punctuation(text: str) -> str:
    """Canonicalize duplicate colons only on the fixed schema bullet labels."""
    value = str(text)
    for label in SCHEMA_BULLET_LABELS:
        value = re.sub(rf"(?m)^(\s*-\s+{re.escape(label)}):{{2,}}", rf"\1:", value)
    return value


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--endpoint-file", type=Path, required=True)
    parser.add_argument("--engine-count", type=int, default=ENGINE_COUNT)
    parser.add_argument("--global-concurrency", type=int, default=GLOBAL_CONCURRENCY)
    parser.add_argument(
        "--metadata-profile",
        choices=("textcraft", "alfworld"),
        default="textcraft",
    )
    parser.add_argument("--expected-trajectories", type=int, default=TRAJECTORY_COUNT)
    parser.add_argument("--trajectory-schedule", type=Path)
    parser.add_argument("--max-new-tokens", type=int, default=MAX_NEW_TOKENS)
    parser.add_argument(
        "--summary-schema",
        choices=("own_outcome_detailed",),
        default="own_outcome_detailed",
    )
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--attempt-seed-offset", type=int, default=0)
    parser.add_argument("--prior-attempt-seed-offset", type=int, default=0)
    parser.add_argument("--existing-max-new-tokens", type=int)
    parser.add_argument("--prior-failure-dir", type=Path)
    parser.add_argument("--empty-on-recovery-exhaustion", action="store_true")
    parser.add_argument("--runtime-profile")
    parser.add_argument("--execution-mode")
    parser.add_argument("--requires-live-environment", action="store_true")
    return parser


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    os.close(fd)
    temporary = Path(name)
    try:
        temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _load_trajectories(
    corpus_dir: Path,
    *,
    metadata_profile: str,
    expected_trajectories: int,
    trajectory_schedule: Path | None,
) -> list[dict[str, Any]]:
    if metadata_profile == "alfworld":
        from slime_plugins.agent_tasks.alfworld.frozen.data_source import load_frozen_trajectory_shard

        trajectories = {
            str(trajectory["trajectory_uid"]): trajectory
            for path in sorted(corpus_dir.glob("batch_*.pt"))
            for trajectory in load_frozen_trajectory_shard(path)
        }
    else:
        steps = canonical_steps_from_corpus(corpus_dir, expected_trajectories=expected_trajectories)
        trajectories = {str(step["trajectory_uid"]): step["trajectory"] for step in steps}
    if trajectory_schedule is None:
        ordered_uids = sorted(trajectories)
    else:
        schedule = json.loads(trajectory_schedule.read_text(encoding="utf-8"))
        draws = schedule.get("draws")
        if not isinstance(draws, list):
            raise ValueError("guidance trajectory schedule must contain a draws list")
        ordered_uids = [str(draw.get("trajectory_uid", "")) for draw in draws]
    if (
        expected_trajectories <= 0
        or len(ordered_uids) != expected_trajectories
        or len(set(ordered_uids)) != expected_trajectories
        or any(uid not in trajectories for uid in ordered_uids)
    ):
        raise ValueError("guidance trajectory selection must contain the exact expected unique identities")
    return [trajectories[uid] for uid in ordered_uids]


def _load_progress(path: Path) -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    if not path.is_file():
        return records
    committed = bytearray()
    with path.open("rb") as handle:
        while True:
            raw = handle.readline()
            if not raw:
                break
            try:
                record = json.loads(raw.decode("utf-8"))
                validate_summary_record(record)
            except (UnicodeDecodeError, json.JSONDecodeError, TypeError, ValueError):
                if handle.read().strip():
                    raise ValueError("malformed middle paper guidance progress record")
                path.write_bytes(bytes(committed))
                break
            uid = str(record["trajectory_uid"])
            if uid in records and records[uid] != record:
                raise ValueError(f"conflicting duplicate guidance progress record: {uid}")
            records[uid] = record
            committed.extend(raw)
    return records


def _prompt_ids(tokenizer: Any, prompt: str) -> list[int]:
    ids = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=True,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    if isinstance(ids, Mapping):
        ids = ids["input_ids"]
    ids = ids.tolist() if hasattr(ids, "tolist") else ids
    if isinstance(ids, list) and ids and isinstance(ids[0], list):
        ids = ids[0]
    values = [int(token_id) for token_id in ids]
    if not values or len(values) > MAX_PROMPT_TOKENS:
        raise ValueError(f"guidance prompt token count outside 1..{MAX_PROMPT_TOKENS}: {len(values)}")
    return values


def _attempt_seed(index: int, attempt: int, *, offset: int = 0) -> int:
    return 42 + int(index) * MAX_ATTEMPTS + int(attempt) - 1 + int(offset)


def _validate_prior_failures(
    archive: Path,
    *,
    missing_uids: set[str],
    uid_to_index: dict[str, int],
    requests_by_uid: dict[str, dict[str, Any]],
    seed_offset: int = 0,
    require_strict_format_only: bool = False,
    allow_transient_attempt_errors: bool = False,
) -> int:
    summary_path = archive / "failure.json"
    failures_dir = archive / "failures"
    if not summary_path.is_file() or not failures_dir.is_dir():
        raise ValueError("guidance recovery requires an archived failure.json and failures directory")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary_failures = summary.get("failures")
    if summary.get("status") != "failed_closed" or not isinstance(summary_failures, list):
        raise ValueError("guidance recovery prior failure summary is malformed")
    by_uid = {str(row.get("trajectory_uid")): row for row in summary_failures}
    files = {path.stem: path for path in failures_dir.glob("*.json") if path.is_file()}
    if (
        int(summary.get("failure_count", -1)) != len(missing_uids)
        or len(summary_failures) != len(missing_uids)
        or len(by_uid) != len(missing_uids)
        or set(by_uid) != missing_uids
        or set(files) != missing_uids
    ):
        raise ValueError("guidance recovery prior failure identities differ from missing progress identities")
    attempt_count = 0
    for uid in sorted(missing_uids):
        file_row = json.loads(files[uid].read_text(encoding="utf-8"))
        if file_row != by_uid[uid]:
            raise ValueError(f"guidance recovery failure evidence mismatch: {uid}")
        if file_row.get("request") != requests_by_uid[uid]:
            raise ValueError(f"guidance recovery failure request mismatch: {uid}")
        attempts = file_row.get("attempts")
        if not isinstance(attempts, list) or len(attempts) != MAX_ATTEMPTS:
            raise ValueError(f"guidance recovery requires three prior attempts: {uid}")
        expected_seeds = [
            _attempt_seed(uid_to_index[uid], attempt, offset=seed_offset) for attempt in range(1, MAX_ATTEMPTS + 1)
        ]
        if (
            [int(row.get("attempt", -1)) for row in attempts] != list(range(1, MAX_ATTEMPTS + 1))
            or [int(row.get("seed", -1)) for row in attempts] != expected_seeds
            or any(bool(row.get("valid")) for row in attempts)
        ):
            raise ValueError(f"guidance recovery prior attempt contract mismatch: {uid}")
        if require_strict_format_only and not _recoverable_format_failure(
            file_row,
            allow_transient_attempt_errors=allow_transient_attempt_errors,
        ):
            raise ValueError(f"guidance recovery exhaustion is not a strict-format-only failure: {uid}")
        attempt_count += len(attempts)
    return attempt_count


def _strict_format_only_failure(failure: dict[str, Any]) -> bool:
    attempts = failure.get("attempts")
    return bool(attempts) and all(
        attempt.get("finish_reason") == "stop"
        and not bool(attempt.get("valid"))
        and bool(str(attempt.get("raw_output", "")).strip())
        and bool(str(attempt.get("cleaned_summary", "")).strip())
        and int(attempt.get("response_token_count", 0)) > 0
        and attempt.get("schema_valid") is False
        for attempt in attempts
    )


def _recoverable_format_failure(
    failure: dict[str, Any],
    *,
    allow_transient_attempt_errors: bool = False,
) -> bool:
    attempts = failure.get("attempts")
    if not attempts:
        return False

    def strict(attempt: dict[str, Any]) -> bool:
        return _strict_format_only_failure({"attempts": [attempt]})

    if not any(strict(attempt) for attempt in attempts):
        return False
    if not allow_transient_attempt_errors:
        return all(strict(attempt) for attempt in attempts)

    def transient(attempt: dict[str, Any]) -> bool:
        error_kind = attempt.get("error_kind")
        if error_kind not in {"transient_request", "interrupted_request"}:
            return False
        return (
            not bool(attempt.get("valid"))
            and attempt.get("finish_reason") is None
            and not str(attempt.get("raw_output", "")).strip()
            and not str(attempt.get("cleaned_summary", "")).strip()
            and attempt.get("response_token_count") in (None, 0)
            and attempt.get("schema_valid") in (None, False)
            and bool(str(attempt.get("error", "")).strip())
        )

    return all(strict(attempt) or transient(attempt) for attempt in attempts)


async def materialize(args: argparse.Namespace) -> list[dict[str, Any]]:
    endpoints = [line.strip().rstrip("/") for line in args.endpoint_file.read_text().splitlines() if line.strip()]
    if args.engine_count <= 0 or args.global_concurrency <= 0:
        raise ValueError("guidance engine count and global concurrency must be positive")
    if (
        args.max_new_tokens <= 0
        or args.attempt_seed_offset < 0
        or args.prior_attempt_seed_offset < 0
        or not math.isfinite(args.temperature)
        or args.temperature < 0
    ):
        raise ValueError("guidance token budget must be positive and seed offset non-negative")
    if len(endpoints) != args.engine_count or len(set(endpoints)) != args.engine_count:
        raise ValueError(f"guidance summaries require exactly {args.engine_count} distinct endpoints")
    metadata_profile = str(getattr(args, "metadata_profile", "textcraft"))
    expected_trajectories = int(getattr(args, "expected_trajectories", TRAJECTORY_COUNT))
    trajectory_schedule = getattr(args, "trajectory_schedule", None)
    trajectories = _load_trajectories(
        args.corpus_dir,
        metadata_profile=metadata_profile,
        expected_trajectories=expected_trajectories,
        trajectory_schedule=trajectory_schedule,
    )
    tokenizer = load_tokenizer(str(args.model_path), trust_remote_code=True)
    requests = [
        build_summary_request(
            trajectory,
            metadata_profile=metadata_profile,
            summary_schema=args.summary_schema,
        )
        for trajectory in trajectories
    ]
    root = args.output_dir
    root.mkdir(parents=True, exist_ok=True)
    progress = root / "progress.jsonl"
    existing = _load_progress(progress)
    recovery_ledger = root / "recovery_attempts.json"
    prior_failure_attempt_count = 0
    recovery_target_uids: set[str] = set()
    expected_uids = {str(request["trajectory_uid"]) for request in requests}
    uid_to_index = {str(request["trajectory_uid"]): index for index, request in enumerate(requests)}
    requests_by_uid = {str(request["trajectory_uid"]): request for request in requests}
    prior_attempt_evidence_schema = ATTEMPT_EVIDENCE_SCHEMA
    if set(existing) - expected_uids:
        raise ValueError("paper guidance progress contains unknown trajectory identities")
    if args.existing_max_new_tokens is not None:
        if not existing or args.attempt_seed_offset <= 0 or args.prior_failure_dir is None:
            raise ValueError("guidance recovery requires progress, a positive seed offset, and prior failure evidence")
        args.prior_failure_dir = args.prior_failure_dir.resolve()
        if args.existing_max_new_tokens <= 0 or args.max_new_tokens < args.existing_max_new_tokens:
            raise ValueError("guidance recovery token budget must not shrink")
        if args.max_new_tokens == args.existing_max_new_tokens:
            raise ValueError("guidance recovery token budget must exceed the existing token budget")
        if recovery_ledger.exists():
            ledger = json.loads(recovery_ledger.read_text(encoding="utf-8"))
            if ledger.get("status") != "in_progress":
                raise RuntimeError("guidance recovery ledger is terminal but verification is unavailable")
            recovery_target_uids = {str(uid) for uid in ledger.get("target_uids", [])}
            if (
                not recovery_target_uids
                or not recovery_target_uids <= expected_uids
                or int(ledger.get("target_count", -1)) != len(recovery_target_uids)
                or ledger.get("prior_failure_dir") != str(args.prior_failure_dir)
                or int(ledger.get("max_new_tokens", -1)) != args.max_new_tokens
                or int(ledger.get("attempt_seed_offset", -1)) != args.attempt_seed_offset
                or int(ledger.get("prior_attempt_seed_offset", -1)) != args.prior_attempt_seed_offset
                or int(ledger.get("max_attempts", -1)) != MAX_ATTEMPTS
                or int(ledger.get("attempt_evidence_schema", -1)) != ATTEMPT_EVIDENCE_SCHEMA
                or int(ledger.get("prior_attempt_evidence_schema", -1)) != prior_attempt_evidence_schema
            ):
                raise RuntimeError("guidance recovery ledger contract mismatch")
        else:
            recovery_target_uids = expected_uids - set(existing)
            if not recovery_target_uids:
                raise ValueError("guidance recovery has no missing trajectories")
        prior_failure_attempt_count = _validate_prior_failures(
            args.prior_failure_dir,
            missing_uids=recovery_target_uids,
            uid_to_index=uid_to_index,
            requests_by_uid=requests_by_uid,
            seed_offset=args.prior_attempt_seed_offset,
            require_strict_format_only=True,
            allow_transient_attempt_errors=True,
        )
        if not recovery_ledger.exists():
            _atomic_json(
                recovery_ledger,
                {
                    "status": "in_progress",
                    "prior_failure_dir": str(args.prior_failure_dir),
                    "target_uids": sorted(recovery_target_uids),
                    "target_count": len(recovery_target_uids),
                    "prior_failure_attempt_count": prior_failure_attempt_count,
                    "max_new_tokens": args.max_new_tokens,
                    "attempt_seed_offset": args.attempt_seed_offset,
                    "prior_attempt_seed_offset": args.prior_attempt_seed_offset,
                    "max_attempts": MAX_ATTEMPTS,
                    "attempt_evidence_schema": ATTEMPT_EVIDENCE_SCHEMA,
                    "prior_attempt_evidence_schema": prior_attempt_evidence_schema,
                    "summary_schema": args.summary_schema,
                    "temperature": args.temperature,
                },
            )
    elif args.prior_failure_dir is not None:
        raise ValueError("prior failure evidence is only valid for guidance recovery")
    if args.empty_on_recovery_exhaustion and args.existing_max_new_tokens is None:
        raise ValueError("empty-guideline fallback is only valid for an audited recovery")

    import httpx

    transient_request_errors = (
        httpx.ConnectError,
        httpx.ConnectTimeout,
        httpx.PoolTimeout,
        httpx.ReadError,
        httpx.ReadTimeout,
        httpx.RemoteProtocolError,
        httpx.WriteError,
        httpx.WriteTimeout,
    )

    semaphore = asyncio.Semaphore(args.global_concurrency)
    lock = asyncio.Lock()
    failures: dict[str, dict[str, Any]] = {}
    timeout = httpx.Timeout(REQUEST_TIMEOUT_SECONDS)
    limits = httpx.Limits(
        max_connections=args.global_concurrency,
        max_keepalive_connections=args.global_concurrency,
    )
    async with httpx.AsyncClient(timeout=timeout, limits=limits, trust_env=False) as client:

        async def one(index: int, request: dict[str, Any]) -> None:
            uid = str(request["trajectory_uid"])
            if uid in existing:
                validate_summary_record(existing[uid], request=request)
                return
            try:
                prompt_ids = _prompt_ids(tokenizer, str(request["prompt"]))
            except Exception as exc:
                failure = {
                    "trajectory_uid": uid,
                    "request": request,
                    "failure_stage": "prompt_preparation",
                    "attempts": [],
                    "error": f"{type(exc).__name__}: {exc}",
                }
                async with lock:
                    failures[uid] = failure
                    _atomic_json(root / "failures" / f"{uid}.json", failure)
                return
            detail_dir = "recovery_attempt_details" if args.existing_max_new_tokens is not None else "attempt_details"
            detail_path = root / detail_dir / f"{uid}.json"
            attempts: list[dict[str, Any]] = []
            if detail_path.is_file():
                detail = json.loads(detail_path.read_text(encoding="utf-8"))
                attempts = list(detail.get("attempts", []))
                if detail.get("trajectory_uid") != uid or len(attempts) > MAX_ATTEMPTS:
                    raise RuntimeError(f"guidance recovery attempt ledger mismatch: {uid}")
                for position, prior in enumerate(attempts, start=1):
                    if int(prior.get("attempt", -1)) != position or int(prior.get("seed", -1)) != _attempt_seed(
                        index, position, offset=args.attempt_seed_offset
                    ):
                        raise RuntimeError(f"guidance recovery consumed seed mismatch: {uid}")
                    if prior.get("status") == "started":
                        prior.update(
                            {
                                "status": "interrupted",
                                "valid": False,
                                "error_kind": "interrupted_request",
                                "error": "interrupted_after_seed_consumed",
                            }
                        )
                _atomic_json(detail_path, {"trajectory_uid": uid, "attempts": attempts})

            async def commit_success() -> None:
                final = attempts[-1]
                if (
                    final.get("status") != "complete"
                    or not bool(final.get("valid"))
                    or final.get("finish_reason") != "stop"
                    or int(final.get("response_token_count", 0)) <= 0
                    or final.get("schema_valid") is not True
                ):
                    raise RuntimeError(f"guidance successful attempt journal is invalid: {uid}")
                record = {
                    "schema_version": SUMMARY_SCHEMA_VERSION,
                    **request,
                    "summary": str(final["cleaned_summary"]),
                    "prompt_token_count": len(prompt_ids),
                    "summary_token_count": int(final["response_token_count"]),
                    "attempt_count": len(attempts),
                    "attempts": attempts,
                    "generation_max_new_tokens": args.max_new_tokens,
                    "generation_temperature": args.temperature,
                    "attempt_seed_offset": args.attempt_seed_offset,
                }
                validate_summary_record(record, request=request)
                async with lock:
                    if uid not in existing:
                        with progress.open("a", encoding="utf-8") as handle:
                            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
                            handle.flush()
                            os.fsync(handle.fileno())
                        existing[uid] = record

            if attempts and bool(attempts[-1].get("valid")):
                await commit_success()
                return
            for attempt in range(len(attempts) + 1, MAX_ATTEMPTS + 1):
                seed = _attempt_seed(index, attempt, offset=args.attempt_seed_offset)
                attempts.append({"attempt": attempt, "seed": seed, "status": "started", "valid": False})
                _atomic_json(detail_path, {"trajectory_uid": uid, "attempts": attempts})
                try:
                    async with semaphore:
                        response = await client.post(
                            f"{endpoints[index % args.engine_count]}/generate",
                            json={
                                "input_ids": prompt_ids,
                                "sampling_params": {
                                    "temperature": args.temperature,
                                    "top_p": 1.0,
                                    "top_k": -1,
                                    "max_new_tokens": args.max_new_tokens,
                                    "sampling_seed": seed,
                                },
                                "return_logprob": True,
                            },
                        )
                    response.raise_for_status()
                    text, finish, response_ids, _ = parse_generation_output(response.json())
                    summary = _normalize_schema_label_punctuation(clean_guidance_summary(text))
                    schema_valid = summary_has_schema(
                        summary,
                        str(request["kind"]),
                        summary_schema=args.summary_schema,
                    )
                    valid = finish == "stop" and bool(response_ids) and schema_valid
                except httpx.HTTPStatusError as exc:
                    status_code = exc.response.status_code
                    result = {
                        "attempt": attempt,
                        "seed": seed,
                        "status": "complete",
                        "valid": False,
                        "error_kind": (
                            "transient_request" if status_code in RETRYABLE_HTTP_STATUS_CODES else "permanent_http"
                        ),
                        "http_status_code": status_code,
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                    attempts[-1] = result
                    _atomic_json(detail_path, {"trajectory_uid": uid, "attempts": attempts})
                except transient_request_errors as exc:
                    result = {
                        "attempt": attempt,
                        "seed": seed,
                        "status": "complete",
                        "valid": False,
                        "error_kind": "transient_request",
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                    attempts[-1] = result
                    _atomic_json(detail_path, {"trajectory_uid": uid, "attempts": attempts})
                except httpx.RequestError as exc:
                    result = {
                        "attempt": attempt,
                        "seed": seed,
                        "status": "complete",
                        "valid": False,
                        "error_kind": "permanent_request",
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                    attempts[-1] = result
                    _atomic_json(detail_path, {"trajectory_uid": uid, "attempts": attempts})
                except Exception as exc:
                    result = {
                        "attempt": attempt,
                        "seed": seed,
                        "status": "complete",
                        "valid": False,
                        "error_kind": "response_processing",
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                    attempts[-1] = result
                    _atomic_json(detail_path, {"trajectory_uid": uid, "attempts": attempts})
                else:
                    attempts[-1] = {
                        "attempt": attempt,
                        "seed": seed,
                        "status": "complete",
                        "finish_reason": finish,
                        "valid": valid,
                        "response_token_count": len(response_ids),
                        "schema_valid": schema_valid,
                        "raw_output": text,
                        "cleaned_summary": summary,
                        "summary_schema": args.summary_schema,
                        "temperature": args.temperature,
                    }
                    _atomic_json(detail_path, {"trajectory_uid": uid, "attempts": attempts})
                    if valid:
                        await commit_success()
                        return
            failure = {"trajectory_uid": uid, "request": request, "attempts": attempts}
            async with lock:
                failures[uid] = failure
                _atomic_json(root / "failures" / f"{uid}.json", failure)

        await asyncio.gather(*(one(index, request) for index, request in enumerate(requests)))

    empty_guideline_uids: list[str] = []
    if failures:
        failure = {
            "status": "failed_closed",
            "failure_count": len(failures),
            "failures": [failures[uid] for uid in sorted(failures)],
        }
        _atomic_json(root / "failure.json", failure)
        if not args.empty_on_recovery_exhaustion or not all(
            _recoverable_format_failure(item, allow_transient_attempt_errors=True) for item in failures.values()
        ):
            raise RuntimeError(
                f"guidance summary failed closed for {len(failures)} trajectories after three attempts each"
            )
        empty_guideline_uids = sorted(failures)

    records = [
        existing[str(request["trajectory_uid"])]
        for request in requests
        if str(request["trajectory_uid"]) not in empty_guideline_uids
    ]
    if len(records) + len(empty_guideline_uids) != expected_trajectories:
        raise RuntimeError("paper summary count mismatch")
    (root / "summaries.jsonl").write_text(
        "".join(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n" for record in records),
        encoding="utf-8",
    )
    recovered_records = [
        record
        for record in records
        if int(record.get("generation_max_new_tokens", -1)) == args.max_new_tokens
        and int(record.get("attempt_seed_offset", -1)) == args.attempt_seed_offset
    ]
    if args.existing_max_new_tokens is not None:
        recovered_uids = {str(record["trajectory_uid"]) for record in recovered_records}
        if recovered_uids | set(empty_guideline_uids) != recovery_target_uids:
            raise RuntimeError("guidance recovery output identities differ from the audited recovery target")
        for record in records:
            if str(record["trajectory_uid"]) not in recovery_target_uids:
                prior_budget = int(record.get("generation_max_new_tokens", -1))
                prior_offset = int(record.get("attempt_seed_offset", -1))
                if (
                    prior_budget <= 0
                    or prior_budget > args.existing_max_new_tokens
                    or prior_offset < 0
                    or prior_offset >= args.attempt_seed_offset
                ):
                    raise RuntimeError("guidance recovery reused record generation metadata mismatch")
    manifest = {
        "schema_version": SUMMARY_SCHEMA_VERSION,
        "status": "complete",
        "kind": "own_outcome_guidance_summaries",
        "trajectory_count": expected_trajectories,
        "valid_guideline_count": len(records),
        "empty_guideline_count": len(empty_guideline_uids),
        "empty_guideline_uids": empty_guideline_uids,
        "empty_guideline_turn_count": sum(
            len(trajectory["turns"])
            for trajectory in trajectories
            if str(trajectory["trajectory_uid"]) in set(empty_guideline_uids)
        ),
        "empty_guideline_policy": "skip_trajectory" if empty_guideline_uids else None,
        "corpus_dir": str(args.corpus_dir),
        "model_path": str(args.model_path),
        "max_attempts": MAX_ATTEMPTS,
        "max_new_tokens": args.max_new_tokens,
        "max_prompt_tokens": MAX_PROMPT_TOKENS,
        "timeout_seconds": REQUEST_TIMEOUT_SECONDS,
        "engine_count": args.engine_count,
        "global_concurrency": args.global_concurrency,
        "metadata_profile": metadata_profile,
        "summary_schema": args.summary_schema,
        "temperature": args.temperature,
        "native_thinking_enabled": False,
        "trajectory_schedule": str(trajectory_schedule) if trajectory_schedule else None,
        "success_count": sum(bool(record["success"]) for record in records),
        "failure_count": sum(not bool(record["success"]) for record in records),
        "successful_generation_retry_count": sum(int(record["attempt_count"]) - 1 for record in records),
        "attempt_seed_offset": args.attempt_seed_offset,
        "prior_attempt_seed_offset": args.prior_attempt_seed_offset,
        "existing_max_new_tokens": args.existing_max_new_tokens,
        "prior_failure_dir": str(args.prior_failure_dir) if args.prior_failure_dir is not None else None,
        "reused_record_count": len(records) - len(recovered_records),
        "generated_record_count": len(recovered_records),
        "recovery_prior_failed_attempt_count": prior_failure_attempt_count,
        "recovery_exhausted_attempt_count": len(empty_guideline_uids) * MAX_ATTEMPTS,
        "attempt_evidence_schema": ATTEMPT_EVIDENCE_SCHEMA,
        "prior_attempt_evidence_schema": (
            prior_attempt_evidence_schema if args.existing_max_new_tokens is not None else None
        ),
    }
    if getattr(args, "runtime_profile", None) is not None:
        manifest.update(
            {
                "runtime_profile": args.runtime_profile,
                "execution_mode": args.execution_mode,
                "requires_live_environment": bool(args.requires_live_environment),
            }
        )
    _atomic_json(root / "manifest.json", manifest)
    verification = {"status": "verified", "manifest": manifest}
    _atomic_json(root / "verification.json", verification)
    if not empty_guideline_uids:
        (root / "failure.json").unlink(missing_ok=True)
    if args.existing_max_new_tokens is not None:
        ledger = json.loads(recovery_ledger.read_text(encoding="utf-8"))
        ledger.update(
            {
                "status": "complete_with_empty" if empty_guideline_uids else "complete",
                "recovered_uids": sorted(str(record["trajectory_uid"]) for record in recovered_records),
                "empty_guideline_uids": empty_guideline_uids,
            }
        )
        _atomic_json(recovery_ledger, ledger)
    return records


def main() -> None:
    args = _parser().parse_args()
    records = asyncio.run(materialize(args))
    print(json.dumps({"status": "verified", "trajectory_count": len(records)}, sort_keys=True))


if __name__ == "__main__":
    main()
