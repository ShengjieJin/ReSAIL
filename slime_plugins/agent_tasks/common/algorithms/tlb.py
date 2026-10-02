"""Trajectory Loss Balancing (TLB) with the original source-batch denominator."""

from collections import Counter


def compute_tlb_weights(
    source_draw_ids: list[int], selected_indices: list[int], *, source_trajectory_count: int
) -> list[float]:
    """Balance selected steps within trajectories, including empty trajectories in the denominator.

    The training reducer averages over selected rows. Multiplying each selected
    row by N / (B * n_i) gives each trajectory its mean selected loss divided by
    the original batch size B, even when some trajectories have no selected step.
    """
    if source_trajectory_count <= 0:
        raise ValueError("TLB requires a positive source trajectory count")
    if not selected_indices or len(set(selected_indices)) != len(selected_indices):
        raise ValueError("TLB requires a non-empty set of unique selected row indices")
    if any(index < 0 or index >= len(source_draw_ids) for index in selected_indices):
        raise ValueError("TLB selected row index is outside the source batch")
    selected_by_draw = Counter(source_draw_ids[index] for index in selected_indices)
    selected_count = len(selected_indices)
    weights = [0.0] * len(source_draw_ids)
    for index in selected_indices:
        weights[index] = selected_count / (source_trajectory_count * selected_by_draw[source_draw_ids[index]])
    return weights
