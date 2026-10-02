#!/usr/bin/env python3
"""Prepare the local TextCraft recipes and official train/eval task IDs."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import tempfile
import urllib.request
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DATASET = "https://huggingface.co/datasets/AgentGym/AgentGym-RL-Data-ID/resolve/main"
SOURCE = "https://github.com/WooooDyy/AgentGym.git"


def _split(path: Path, expected: int) -> None:
    rows = json.loads(path.read_text(encoding="utf-8"))
    ids = [row["item_id"] for row in rows]
    if (len(ids) != expected or len(set(ids)) != expected
            or any(not isinstance(uid, str) or not uid.startswith("textcraft_") for uid in ids)):
        raise RuntimeError(f"TextCraft task IDs differ from the {expected}-task official split: {path}")


def _download(relative: str, destination: Path, expected: int) -> None:
    if destination.is_file():
        _split(destination, expected)
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=destination.parent, delete=False) as handle:
        temporary = Path(handle.name)
    try:
        urllib.request.urlretrieve(f"{DATASET}/{relative}", temporary)
        _split(temporary, expected)
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path,
                        help="an existing AgentGym checkout; omit to fetch only TextCraft recipes locally")
    parser.add_argument("--data-dir", type=Path, default=ROOT / ".cache/textcraft")
    args = parser.parse_args()
    data = args.data_dir.resolve()
    recipes = data / "recipes"
    if not recipes.is_dir():
        source = args.source_dir or data / "source/AgentGym"
        if not source.is_dir():
            source.parent.mkdir(parents=True, exist_ok=True)
            subprocess.run(["git", "clone", "--depth", "1", "--filter=blob:none", "--sparse",
                            SOURCE, str(source)], check=True)
            subprocess.run(["git", "-C", str(source), "sparse-checkout", "set",
                            "agentenv-textcraft/agentenv_textcraft/recipes"], check=True)
        candidate = source / "agentenv-textcraft/agentenv_textcraft/recipes"
        if not candidate.is_dir():
            raise FileNotFoundError(f"AgentGym TextCraft recipes are missing: {candidate}")
        shutil.copytree(candidate, recipes)
        if (source / ".git").exists():
            revision = subprocess.run(["git", "-C", str(source), "rev-parse", "HEAD"],
                                      check=True, capture_output=True, text=True).stdout.strip()
            (data / "source_revision.json").write_text(json.dumps({
                "source": SOURCE, "git_revision": revision}, indent=2) + "\n", encoding="utf-8")
    count = len(list(recipes.glob("*.json")))
    if count != 860:
        raise RuntimeError(f"paper TextCraft requires 860 recipe JSON files; found {count} in {recipes}")
    splits = data / "agentgym_rl_data_id"
    _download("train/textcraft_train.json", splits / "train/textcraft_train.json", 374)
    _download("eval/textcraft_test.json", splits / "eval/textcraft_test.json", 100)
    print(f"TextCraft ready: {count} recipes, 374 train IDs, 100 eval IDs in {data}")


if __name__ == "__main__":
    main()
