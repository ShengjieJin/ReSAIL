from __future__ import annotations

import argparse
import json
from pathlib import Path

import pytest

from slime_plugins.agent_tasks.common.frozen.guidance import (
    SUMMARY_SCHEMA_VERSION,
    build_summary_request,
    guidance_metadata,
    load_summary_records,
)


NUM_GPUS = 0


def _value(cli: list[str], option: str) -> str:
    return str(cli[cli.index(option) + 1])


def _trajectory(success: bool = True) -> dict:
    return {
        "trajectory_uid": "alf-trajectory-0",
        "task_id": "pick_and_place/task-0",
        "task_description": "put the apple in the cabinet",
        "split": "train",
        "success": success,
        "outcome": "success" if success else "failed: cabinet was never opened",
        "turns": [
            {
                "turn_idx": 0,
                "messages": [
                    {"role": "system", "content": "ordinary rules"},
                    {"role": "user", "content": "ordinary current prompt"},
                ],
                "current_observation": "you see an apple",
                "next_observation": "CANARY_FUTURE_OBSERVATION",
                "frozen_action": "take apple",
                "errors": [],
            },
            {
                "turn_idx": 1,
                "messages": [{"role": "user", "content": "later ordinary prompt"}],
                "current_observation": "CANARY_FUTURE_OBSERVATION",
                "next_observation": "done",
                "frozen_action": "put apple in cabinet",
                "errors": [],
            },
        ],
    }


def _summary(kind: str) -> str:
    if kind == "success":
        return (
            "Guidance summary:\n- Minimal plan: locate, take, place.\n"
            "- Critical actions: open the destination first.\n- Checks: verify inventory.\n"
            "- Avoid: irrelevant exploration."
        )
    return (
        "Failure analysis:\n- Failure diagnosis: destination stayed closed.\n"
        "- Useful evidence: the item was available.\n- Corrected plan: open then place.\n"
        "- Avoid: placing before opening."
    )




@pytest.mark.unit
def test_alfworld_sgs_scoring_requires_teacher_view_metadata():
    import slime_plugins.agent_tasks.alfworld.frozen.generate as generate

    source = Path(generate.__file__).read_text(encoding="utf-8")
    assert "if sgs_filtering_enabled(args):" in source
    assert "sgs_selection_fraction(args) < 1.0" not in source








@pytest.mark.unit
@pytest.mark.parametrize("success", [True, False])
def test_alfworld_summary_binding_uses_alfworld_profile_and_reuses_one_summary(tmp_path: Path, success: bool):
    trajectory = _trajectory(success)
    request = build_summary_request(trajectory, metadata_profile="alfworld")
    assert "CANARY_FUTURE_OBSERVATION" in request["prompt"]
    record = {
        "schema_version": SUMMARY_SCHEMA_VERSION,
        **request,
        "summary": _summary(request["kind"]),
        "attempt_count": 1,
    }
    root = tmp_path / "guidance"
    root.mkdir()
    manifest = {
        "schema_version": SUMMARY_SCHEMA_VERSION,
        "status": "complete",
        "trajectory_count": 1,
        "valid_guideline_count": 1,
        "empty_guideline_count": 0,
        "empty_guideline_uids": [],
        "corpus_dir": "/corpus",
        "model_path": "/model",
    }
    (root / "manifest.json").write_text(json.dumps(manifest))
    (root / "verification.json").write_text(json.dumps({"status": "verified", "manifest": manifest}))
    (root / "summaries.jsonl").write_text(json.dumps(record) + "\n")
    args = argparse.Namespace(
        agent_frozen_guidance_summary_dir=str(root),
        agent_frozen_expected_guidance_trajectories=1,
        agent_frozen_expected_trajectories=3840,
        alfworld_frozen_corpus_dir="/corpus",
        agent_frozen_guidance_summary_model_path="/model",
        agent_task_sdpo_metadata_profile="alfworld",
    )
    assert set(load_summary_records(args)) == {trajectory["trajectory_uid"]}
    first = guidance_metadata(args, trajectory)
    second = guidance_metadata(args, trajectory)
    assert first == second
    output = next(iter(first["sdpo_guidance_summary_outputs"].values()))
    assert "CANARY_FUTURE_OBSERVATION" not in output
    assert "trajectory evidence:" not in output












