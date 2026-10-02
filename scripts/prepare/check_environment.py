#!/usr/bin/env python3
"""Report differences from the tested environment; strict checks are optional."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import platform
import subprocess
import sys
from pathlib import Path


EXPECTED = {
    "torch": "2.11.0+cu129",
    "sglang": "0.5.12.post1",
    "transformers": "5.6.0",
    "ray": "2.55.1",
    "megatron-core": "0.16.0rc0",
    "alfworld": "0.4.2",
    "textworld": "1.7.0",
    "numpy": "1.26.4",
}


def check_environment(*, check_gpus: bool = False) -> dict:
    versions = {}
    differences = []
    for name, expected in EXPECTED.items():
        try:
            actual = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            actual = None
        versions[name] = actual
        if actual != expected:
            differences.append(f"{name}: tested {expected}, found {actual}")
    python = platform.python_version()
    if python != "3.12.3":
        differences.append(f"python: tested 3.12.3, found {python}")
    result = {"python": python, "packages": versions, "differences": differences}
    if check_gpus:
        completed = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,name,memory.total", "--format=csv,noheader,nounits"],
            text=True, capture_output=True, check=True,
        )
        devices = [line.strip() for line in completed.stdout.splitlines() if line.strip()]
        result["gpus"] = devices
        if len(devices) != 8:
            differences.append(f"paper configuration requires 8 visible GPUs, found {len(devices)}")
    result["tested_stack_status"] = "verified" if not differences else "different_environment"
    try:
        dependency_check = subprocess.run(
            [sys.executable, "-m", "pip", "check"], text=True, capture_output=True, check=False
        )
    except OSError as exc:
        dependency_check = None
        dependency_error = str(exc)
    else:
        dependency_error = dependency_check.stderr.strip()
    result["dependency_check"] = {
        "status": "error" if dependency_check is None else (
            "satisfied" if dependency_check.returncode == 0 else (
                "conflicts" if dependency_check.returncode == 1 else "error"
            )
        ),
        "issues": [line for line in dependency_check.stdout.splitlines() if line.strip()]
        if dependency_check is not None and dependency_check.returncode else [],
        "error": dependency_error,
    }
    if result["tested_stack_status"] != "verified":
        result["status"] = "different_environment"
    elif result["dependency_check"]["status"] == "conflicts":
        result["status"] = "dependency_conflicts"
    elif result["dependency_check"]["status"] == "error":
        result["status"] = "dependency_check_error"
    else:
        result["status"] = "verified"
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--strict", action="store_true", help="opt in to enforcing the tested versions and, with --gpus, GPU count")
    parser.add_argument("--strict-dependencies", action="store_true", help="fail if pip check finds conflicts or cannot complete")
    parser.add_argument("--gpus", action="store_true", help="also report visible GPUs and differences from the eight-GPU configuration")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = check_environment(check_gpus=args.gpus)
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return int(
        (args.strict and result["tested_stack_status"] != "verified")
        or (args.strict_dependencies and result["dependency_check"]["status"] != "satisfied")
    )


if __name__ == "__main__":
    raise SystemExit(main())
