"""Checkpoint manifest contract for public cycle verification."""

from __future__ import annotations

import json
import shutil
import warnings
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch.distributed.checkpoint import save

from exp.paper import text


@pytest.fixture
def cell(tmp_path: Path) -> Path:
    checkpoint = tmp_path / "checkpoints/iter_0000001"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        save({"model.weight": torch.ones(2)}, checkpoint_id=str(checkpoint))
    torch.save({"args": SimpleNamespace(num_layers=1)}, checkpoint / "common.pt")
    (checkpoint / "metadata.json").write_text(json.dumps({"sharded_backend": "torch_dist",
                                                         "common_backend": "torch"}), encoding="utf-8")
    (checkpoint.parent / "latest_checkpointed_iteration.txt").write_text("1", encoding="utf-8")
    return tmp_path


def test_complete_checkpoint_passes(cell: Path) -> None:
    text._verify_checkpoint(cell, "iter_0000001")


def test_deleted_shard_fails_even_when_metadata_and_common_remain(cell: Path) -> None:
    next((cell / "checkpoints/iter_0000001").glob("*.distcp")).unlink()
    with pytest.raises(RuntimeError, match="shard count"):
        text._verify_checkpoint(cell, "iter_0000001")


def test_truncated_shard_fails_against_metadata_ranges(cell: Path) -> None:
    shard = next((cell / "checkpoints/iter_0000001").glob("*.distcp"))
    with shard.open("r+b") as handle:
        handle.truncate(1)
    with pytest.raises(RuntimeError, match="truncated"):
        text._verify_checkpoint(cell, "iter_0000001")


def test_unreadable_metadata_fails(cell: Path) -> None:
    (cell / "checkpoints/iter_0000001/.metadata").write_bytes(b"incomplete")
    with pytest.raises(RuntimeError, match="metadata is unreadable"):
        text._verify_checkpoint(cell, "iter_0000001")


def test_missing_common_state_fails(cell: Path) -> None:
    (cell / "checkpoints/iter_0000001/common.pt").unlink()
    with pytest.raises(RuntimeError, match="missing metadata or common"):
        text._verify_checkpoint(cell, "iter_0000001")


def test_wrong_iteration_tracker_fails(cell: Path) -> None:
    (cell / "checkpoints/latest_checkpointed_iteration.txt").write_text("2", encoding="utf-8")
    with pytest.raises(RuntimeError, match="wrong iteration"):
        text._verify_checkpoint(cell, "iter_0000001")


def test_completed_resume_rechecks_shard_manifest(cell: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    args = SimpleNamespace(task="alfworld", model="4b", method="resail", cycle=1,
                           mode="smoke", c1_input=None, smoke_updates=2,
                           run_root=cell.parent, input_root=cell.parent / "data",
                           base_checkpoint=cell.parent / "base", base_hf=cell.parent / "model",
                           resume=True)
    monkeypatch.setattr(text, "_validate_roots", lambda _: None)
    monkeypatch.setattr(text, "cell_root", lambda *_: cell)
    monkeypatch.setattr(text, "runtime", str)
    identity = {"task": "alfworld", "model": "4b", "method": "resail", "cycle": 1,
                "protocol": "smoke", "c1_input": None, "smoke_updates": 2,
                "run_root": str(args.run_root), "input_root": str(args.input_root),
                "base_checkpoint": str(args.base_checkpoint), "base_hf": str(args.base_hf),
                }
    (cell / "run_identity.json").write_text(json.dumps(identity), encoding="utf-8")
    shared = {"task": "alfworld", "model": "4b", "method": "resail", "cycle": 1,
              "protocol": "smoke", "final_checkpoint": "iter_0000001"}
    (cell / "completion.json").write_text(json.dumps({**shared, "status": "complete"}), encoding="utf-8")
    (cell / "verification.json").write_text(json.dumps({**shared, "status": "verified",
                                                       "c1_source": "replay"}), encoding="utf-8")
    next((cell / "checkpoints/iter_0000001").glob("*.distcp")).unlink()
    with pytest.raises(RuntimeError, match="shard count"):
        text.execute(args)


def _next_cycle(cell: Path, monkeypatch: pytest.MonkeyPatch, *, task: str = "alfworld",
                c1_input: Path | None = None) -> tuple[SimpleNamespace, Path]:
    run_root = cell / "paper_runs"
    parent = text.cell_root(run_root, task, "4b", "resail", 1)
    parent.mkdir(parents=True)
    shutil.copytree(cell / "checkpoints", parent / "checkpoints")
    result_identity = {"task": task, "model": "4b", "method": "resail", "cycle": 1,
                       "protocol": "smoke", "final_checkpoint": "iter_0000001"}
    (parent / "completion.json").write_text(json.dumps({**result_identity, "status": "complete"}), encoding="utf-8")
    (parent / "verification.json").write_text(json.dumps({**result_identity, "status": "verified"}), encoding="utf-8")
    parent_hf = parent / "eval_hf/iter_0000001"
    parent_hf.mkdir(parents=True)
    (parent_hf / "config.json").write_text("{}", encoding="utf-8")
    args = SimpleNamespace(task=task, model="4b", method="resail", cycle=2,
                           mode="smoke", c1_input=c1_input, smoke_updates=2,
                           run_root=run_root, input_root=cell / "data",
                           base_checkpoint=cell / "base", base_hf=cell / "model",
                           resume=False)
    parent_identity = {"task": task, "model": "4b", "method": "resail", "cycle": 1,
                       "protocol": "smoke", "c1_input": str(c1_input) if c1_input else None, "smoke_updates": 2,
                       "run_root": str(run_root), "input_root": str(args.input_root),
                       "base_checkpoint": str(args.base_checkpoint), "base_hf": str(args.base_hf),
                       }
    (parent / "run_identity.json").write_text(json.dumps(parent_identity), encoding="utf-8")
    monkeypatch.setattr(text, "_validate_roots", lambda _: None)
    monkeypatch.setattr(text, "prepare", lambda *_args, **_kwargs: pytest.fail("prepare started before parent verification"))
    monkeypatch.setattr(text, "_submit", lambda *_args, **_kwargs: pytest.fail("job submitted before parent verification"))
    return args, parent


def test_next_cycle_rejects_missing_parent_shard_before_prepare(cell: Path,
                                                                 monkeypatch: pytest.MonkeyPatch) -> None:
    args, parent = _next_cycle(cell, monkeypatch)
    next((parent / "checkpoints/iter_0000001").glob("*.distcp")).unlink()
    with pytest.raises(RuntimeError, match="shard count"):
        text.execute(args)


def test_next_cycle_rejects_stale_parent_verification_before_prepare(cell: Path,
                                                                     monkeypatch: pytest.MonkeyPatch) -> None:
    args, parent = _next_cycle(cell, monkeypatch)
    path = parent / "verification.json"
    verification = json.loads(path.read_text(encoding="utf-8"))
    verification["final_checkpoint"] = "iter_0000002"
    path.write_text(json.dumps(verification), encoding="utf-8")
    with pytest.raises(RuntimeError, match="parent cycle final checkpoint differs"):
        text.execute(args)


def test_next_cycle_rejects_mixed_shared_c1_before_prepare(cell: Path,
                                                             monkeypatch: pytest.MonkeyPatch) -> None:
    args, parent = _next_cycle(cell, monkeypatch, task="textcraft", c1_input=Path("/tmp/shared/c1"))
    identity_path = parent / "run_identity.json"
    identity = json.loads(identity_path.read_text(encoding="utf-8"))
    identity["c1_input"] = "/tmp/other/c1"
    identity_path.write_text(json.dumps(identity), encoding="utf-8")
    with pytest.raises(RuntimeError, match="parent cycle identity differs"):
        text.execute(args)


def test_next_cycle_rejects_stale_parent_result_fields_before_prepare(cell: Path,
                                                                       monkeypatch: pytest.MonkeyPatch) -> None:
    args, parent = _next_cycle(cell, monkeypatch)
    path = parent / "verification.json"
    verification = json.loads(path.read_text(encoding="utf-8"))
    verification["protocol"] = "paper"
    path.write_text(json.dumps(verification), encoding="utf-8")
    with pytest.raises(RuntimeError, match="parent cycle result identity differs"):
        text.execute(args)


def test_next_cycle_valid_parent_reaches_prepare(cell: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    args, _parent = _next_cycle(cell, monkeypatch)

    def reached_prepare(*_args: object, **_kwargs: object) -> None:
        raise RuntimeError("prepare reached after valid parent preflight")

    monkeypatch.setattr(text, "prepare", reached_prepare)
    with pytest.raises(RuntimeError, match="prepare reached after valid parent preflight"):
        text.execute(args)
