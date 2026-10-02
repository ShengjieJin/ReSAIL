"""Check TLB against the original trajectory-mean objective."""

from collections import defaultdict

import pytest
import torch

from slime_plugins.agent_tasks.common.algorithms.tlb import compute_tlb_weights


@pytest.mark.parametrize(
    ("source_draw_ids", "selected_indices", "losses", "source_trajectory_count"),
    [
        pytest.param(
            [0, 0, 0, 1, 1, 2, 2], [0, 1, 2, 3, 5, 6],
            [1.0, 2.0, 7.0, 11.0, 99.0, 17.0, 19.0], 3,
            id="uneven-selected-counts",
        ),
        pytest.param(
            [0, 0, 1, 1, 2, 3], [0, 1, 4],
            [2.0, 6.0, 99.0, 99.0, 12.0, 99.0], 4,
            id="zero-selected-trajectories",
        ),
        pytest.param(
            [0, 0, 1, 1, 2], [0, 1, 2, 3, 4],
            [1.0, 3.0, 5.0, 7.0, 11.0], 3,
            id="all-steps-selected",
        ),
    ],
)
def test_tlb_weighted_mean_uses_original_batch_denominator(
    source_draw_ids: list[int],
    selected_indices: list[int],
    losses: list[float],
    source_trajectory_count: int,
) -> None:
    weights = compute_tlb_weights(
        source_draw_ids, selected_indices, source_trajectory_count=source_trajectory_count
    )
    selected = set(selected_indices)
    by_trajectory: dict[int, list[int]] = defaultdict(list)
    for index in selected_indices:
        by_trajectory[source_draw_ids[index]].append(index)

    # Each represented trajectory contributes its mean; absent trajectories
    # contribute zero, while the denominator remains the original batch B.
    expected = sum(
        sum(losses[index] for index in indices) / len(indices)
        for indices in by_trajectory.values()
    ) / source_trajectory_count
    actual = sum(weights[index] * losses[index] for index in selected_indices) / len(selected_indices)
    assert actual == pytest.approx(expected)
    assert all(weights[index] == 0.0 for index in range(len(losses)) if index not in selected)

    values = torch.tensor(losses, dtype=torch.float64, requires_grad=True)
    weighted_loss = sum(weights[index] * values[index] for index in selected_indices) / len(selected_indices)
    weighted_loss.backward()
    for index, gradient in enumerate(values.grad.tolist()):
        expected_gradient = (
            1.0 / (source_trajectory_count * len(by_trajectory[source_draw_ids[index]]))
            if index in selected else 0.0
        )
        assert gradient == pytest.approx(expected_gradient)
