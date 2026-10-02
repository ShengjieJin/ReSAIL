from types import SimpleNamespace

import pytest
import torch

from slime.utils.types import Sample
from slime_plugins.agent_tasks.common.algorithms import sgs, resail


def _sample(draw: int, turn: int, *, length: int = 3) -> Sample:
    return Sample(
        index=draw * 100 + turn,
        rollout_id=draw,
        tokens=list(range(length + 2)),
        response_length=length,
        loss_mask=[1] * length,
        metadata={"source_draw_id": draw, "source_turn_idx": turn},
    )


def _record(draw: int, turn: int, score, *, traj: str | None = None) -> dict:
    return {
        "source_draw_id": draw,
        "traj_uid": traj or f"traj-{draw}",
        "turn_idx": turn,
        "teacher_js": score,
        "action_token_count": 1,
        "response_token_count": 3,
        "precomputed": {
            "sdpo_topk_indices": torch.tensor([[1, 2]]),
            "sdpo_teacher_log_probs": torch.tensor([0.1, 0.2, 0.3]),
            "sdpo_teacher_topk_log_probs": torch.tensor([[-1.0, -2.0]] * 3),
        },
    }


def test_per_update_topx_is_deterministic_and_excludes_unscoreable_rows():
    records = [
        _record(2, 0, 0.5, traj="b"),
        _record(1, 1, None, traj="a"),
        _record(1, 0, 0.5, traj="a"),
        _record(3, 0, 0.1, traj="c"),
    ]
    selected = sgs.select_sgs_steps(records, fraction=0.25, minimum_selected=1)
    assert selected.scoreable_count == 3
    assert selected.requested_count == 1
    assert selected.selected_keys == ((1, 0),)


def test_dp_floor_keeps_only_highest_scored_rows():
    records = [_record(draw, 0, float(draw)) for draw in range(10)]
    selected = sgs.select_sgs_steps(records, fraction=0.01, minimum_selected=3)
    assert selected.dp_floor_applied
    assert selected.selected_keys == ((9, 0), (8, 0), (7, 0))


def test_ascending_ranking_keeps_only_lowest_scored_rows_with_stable_ties():
    records = [
        _record(3, 0, 0.3, traj="c"),
        _record(2, 0, 0.1, traj="b"),
        _record(1, 1, 0.1, traj="a"),
        _record(1, 0, 0.1, traj="a"),
    ]
    selected = sgs.select_sgs_steps(
        records,
        fraction=0.25,
        minimum_selected=2,
        ranking_order="ascending",
    )
    assert selected.selected_keys == ((1, 0), (1, 1))


def test_random_filter_rejects_a_ranking_order():
    with pytest.raises(ValueError, match="not configurable for random"):
        sgs.select_sgs_steps(
            [_record(0, 0, 0.1)],
            fraction=1.0,
            minimum_selected=1,
            selection_mode="random",
            ranking_order="ascending",
        )


def test_global_with_trajectory_floor_only_repairs_uncovered_trajectories():
    records = [
        _record(0, 1, 0.9),
        _record(0, 0, 1.0),
        _record(1, 1, 0.7),
        _record(1, 0, 0.8),
        _record(2, 1, 0.6),
        _record(2, 0, 0.6),
    ]
    selection = sgs.select_sgs_steps(
        records,
        fraction=0.1,
        minimum_selected=1,
        selection_scope="global_with_trajectory_floor",
    )
    reordered = sgs.select_sgs_steps(
        reversed(records),
        fraction=0.1,
        minimum_selected=1,
        selection_scope="global_with_trajectory_floor",
    )
    assert selection.requested_count == 1
    assert selection.base_selected_keys == ((0, 0),)
    assert selection.trajectory_floor_added_keys == ((1, 0), (2, 0))
    assert selection.selected_keys == ((0, 0), (1, 0), (2, 0))
    assert selection.selected_keys == reordered.selected_keys
    assert selection.base_selected_trajectory_count == 1
    assert selection.selected_trajectory_count == 3


def test_global_with_trajectory_floor_does_not_duplicate_dp_floor_coverage():
    records = [
        _record(0, 0, 1.0),
        _record(1, 0, 0.9),
        _record(2, 0, 0.8),
    ]
    selection = sgs.select_sgs_steps(
        records,
        fraction=0.01,
        minimum_selected=2,
        selection_scope="global_with_trajectory_floor",
    )
    assert selection.base_selected_keys == ((0, 0), (1, 0))
    assert selection.dp_floor_added_keys == ((1, 0),)
    assert selection.trajectory_floor_added_keys == ((2, 0),)
    assert selection.selected_keys == ((0, 0), (1, 0), (2, 0))


def test_global_with_trajectory_floor_rejects_a_trajectory_without_a_finite_score():
    records = [
        _record(0, 0, 1.0),
        _record(1, 0, None),
        _record(1, 1, float("nan")),
        _record(1, 2, float("inf")),
    ]
    with pytest.raises(RuntimeError, match="zero scoreable steps"):
        sgs.select_sgs_steps(
            records,
            fraction=0.5,
            minimum_selected=1,
            selection_scope="global_with_trajectory_floor",
        )


def test_trajectory_floor_preserves_equal_outer_trajectory_weight(monkeypatch):
    samples = [
        _sample(0, 0),
        _sample(0, 1),
        _sample(1, 0),
        _sample(2, 0),
    ]
    records = [
        _record(0, 0, 1.0),
        _record(0, 1, 0.9),
        _record(1, 0, 0.8),
        _record(2, 0, 0.7),
    ]
    selection = sgs.select_sgs_steps(
        records,
        fraction=0.5,
        minimum_selected=1,
        selection_scope="global_with_trajectory_floor",
    )
    selected = sgs.bind_selection_to_samples(samples, records, selection)
    for sample in selected:
        sample.sgs_batch_metrics = {"attempted_steps": 4.0}

    monkeypatch.setattr(
        resail,
        "default_sdpo_converter",
        lambda _args, rows: {
            "tokens": [row.tokens for row in rows],
            "self_distillation_mask": [1.0] * len(rows),
            "sdpo_loss_weights": [99.0] * len(rows),
        },
    )
    train_data = resail.convert_samples_to_train_data(
        SimpleNamespace(rollout_batch_size=3, sgs_selection_fraction=0.5),
        selected,
    )
    weights = train_data["sdpo_loss_weights"]
    assert weights[:2] == pytest.approx([2 / 3, 2 / 3])
    assert weights[2:] == pytest.approx([4 / 3, 4 / 3])
    assert sum(weights) == pytest.approx(4.0)


def test_per_update_random_selection_is_seeded_order_independent_and_not_score_ranked():
    records = [_record(draw, turn, float(draw * 10 + turn)) for draw in range(5) for turn in range(2)]
    first = sgs.select_sgs_steps(
        records,
        fraction=0.3,
        minimum_selected=1,
        selection_mode="random",
        selection_seed=42,
        rollout_id=7,
    )
    reordered = sgs.select_sgs_steps(
        reversed(records),
        fraction=0.3,
        minimum_selected=1,
        selection_mode="random",
        selection_seed=42,
        rollout_id=7,
    )
    next_update = sgs.select_sgs_steps(
        records,
        fraction=0.3,
        minimum_selected=1,
        selection_mode="random",
        selection_seed=42,
        rollout_id=8,
    )
    sensitivity = sgs.select_sgs_steps(records, fraction=0.3, minimum_selected=1)
    assert first.selected_keys == reordered.selected_keys
    assert first.selected_keys != next_update.selected_keys
    assert first.selected_keys != sensitivity.selected_keys
    assert first.selected_count == 3


def test_selected_converter_preserves_fixed_outer_trajectory_denominator(monkeypatch):
    samples = [_sample(0, 0), _sample(0, 1), _sample(1, 0)]
    records = [_record(0, 0, 0.9), _record(0, 1, 0.8), _record(1, 0, 0.7)]
    selection = sgs.select_sgs_steps(records, fraction=1.0, minimum_selected=1)
    selected = sgs.bind_selection_to_samples(samples, records, selection)
    for sample in selected:
        sample.sgs_batch_metrics = {"attempted_steps": 3.0}

    def donor(_args, rows):
        return {
            "tokens": [row.tokens for row in rows],
            "self_distillation_mask": [1.0] * len(rows),
            "sdpo_loss_weights": [99.0] * len(rows),
        }

    monkeypatch.setattr(resail, "default_sdpo_converter", donor)
    args = SimpleNamespace(rollout_batch_size=2, sgs_selection_fraction=0.5)
    train_data = resail.convert_samples_to_train_data(args, selected)
    assert train_data["sdpo_loss_weights"] == [0.75, 0.75, 1.5]
    assert sum(train_data["sdpo_loss_weights"]) == 3.0
    assert len(train_data["sdpo_teacher_topk_log_probs"]) == 3
    assert train_data["sgs_sparse_single_step"] is True


def test_top100_is_exact_donor_noop_even_when_a_row_would_be_unscoreable(monkeypatch):
    samples = [_sample(0, 0), _sample(1, 0)]
    sentinel = {"tokens": [[1], [2]], "sdpo_loss_weights": [0.25, 1.75], "marker": object()}
    monkeypatch.setattr(resail, "default_sdpo_converter", lambda _args, _rows: sentinel)
    args = SimpleNamespace(rollout_batch_size=2, sgs_selection_fraction=1.0)
    # No sensitivity markers are attached: a hypothetical scoring failure
    # cannot remove or reorder rows because x=100 bypasses the prepass.
    samples[1].metadata["hypothetical_teacher_js"] = None
    assert sgs.sgs_filtering_enabled(args) is False
    assert resail.convert_samples_to_train_data(args, samples) is sentinel


def test_top100_explicit_full_selection_retention_uses_joint_converter(monkeypatch):
    samples = [_sample(0, 0), _sample(1, 0)]
    records = [_record(0, 0, 0.9), _record(1, 0, 0.8)]
    selection = sgs.select_sgs_steps(records, fraction=1.0, minimum_selected=1)
    selected = sgs.bind_selection_to_samples(samples, records, selection)
    for sample in selected:
        sample.sgs_batch_metrics = {"attempted_steps": 2.0}

    monkeypatch.setattr(
        resail,
        "default_sdpo_converter",
        lambda _args, rows: {
            "tokens": [row.tokens for row in rows],
            "sdpo_loss_weights": [1.0] * len(rows),
            "self_distillation_mask": [1.0] * len(rows),
            **{
                field: [getattr(row, "sgs_precomputed")[field] for row in rows]
                for field in sgs.PRECOMPUTED_FIELDS
            },
        },
    )
    observed = {}

    def expand(_args, train_data, **kwargs):
        observed.update(kwargs)
        train_data["joint_retention_applied"] = True

    monkeypatch.setattr(resail, "expand_pr_rows", expand)
    args = SimpleNamespace(
        rollout_batch_size=2,
        sgs_selection_fraction=1.0,
        sgs_full_selection_retention=True,
        tlb_loss_aggregation="trajectory_balanced",
        pr_weight=0.5,
        pr_support="all",
    )
    assert sgs.sgs_filtering_enabled(args) is True
    train_data = resail.convert_samples_to_train_data(args, selected)
    assert train_data["joint_retention_applied"] is True
    assert train_data["sgs_sparse_single_step"] is True
    assert observed["selected_indices"] == [0, 1]
    assert observed["retention_weight"] == 0.5
    assert observed["retention_base_weights"] == [1.0, 1.0]


def test_fraction_below_top100_enables_filtering_and_rejects_invalid_values():
    assert sgs.sgs_filtering_enabled(
        SimpleNamespace(sgs_selection_fraction=0.25)
    ) is True
    assert sgs.sgs_filtering_enabled(SimpleNamespace()) is False
    for value in (0.0, 1.01):
        try:
            sgs.sgs_filtering_enabled(
                SimpleNamespace(sgs_selection_fraction=value)
            )
        except ValueError:
            pass
        else:
            raise AssertionError(f"invalid fraction {value} was accepted")
