from __future__ import annotations


import pytest

from slime.utils.types import Sample
from slime_plugins.agent_tasks.alfworld.log import _summarize_action_match


NUM_GPUS = 0


def _action_sample(
    index: int,
    *,
    success: bool,
    matched: bool,
    arm: str = "iterative_oel",
) -> Sample:
    return Sample(
        tokens=[10 + index],
        response_length=1,
        metadata={
            "frozen_arm": arm,
            "source_draw_id": index,
            "source_turn_idx": 0,
            "source_trajectory_uid": f"trajectory-{index}",
            "branch_idx": 0,
            "action_match": matched,
            "success": success,
            "sampling_seed": 1000 + index,
        },
    )


@pytest.mark.unit
def test_online_action_match_splits_by_frozen_final_trajectory_outcome():
    metrics = _summarize_action_match(
        [
            _action_sample(0, success=True, matched=True),
            _action_sample(1, success=True, matched=False),
            _action_sample(2, success=False, matched=True),
        ],
        outcome_split=True,
    )
    assert metrics["counts"]["action_match_count"] == 2
    assert metrics["counts"]["action_match_total"] == 3
    assert metrics["counts"]["success_trajectory_action_match_count"] == 1
    assert metrics["counts"]["success_trajectory_action_match_total"] == 2
    assert metrics["counts"]["failure_trajectory_action_match_count"] == 1
    assert metrics["counts"]["failure_trajectory_action_match_total"] == 1
    assert metrics["rates"]["action_match_rate_success_trajectories"] == pytest.approx(0.5)
    assert metrics["rates"]["action_match_rate_failure_trajectories"] == pytest.approx(1.0)


@pytest.mark.unit
def test_success_only_action_match_omits_zero_denominator_failure_rate():
    samples = []
    for branch_idx in range(8):
        sample = _action_sample(
            branch_idx,
            success=True,
            matched=branch_idx % 2 == 0,
            arm="iterative_success_only_grpo",
        )
        sample.metadata["source_draw_id"] = 0
        sample.metadata["branch_idx"] = branch_idx
        samples.append(sample)
    metrics = _summarize_action_match(samples, outcome_split=True)
    assert metrics["counts"]["failure_trajectory_action_match_count"] == 0
    assert metrics["counts"]["failure_trajectory_action_match_total"] == 0
    assert "action_match_rate_failure_trajectories" not in metrics["rates"]
    assert metrics["rates"]["action_match_rate_success_trajectories"] == pytest.approx(0.5)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
