import json
import logging
import re
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from slime.utils.misc import should_run_periodic_action

logger = logging.getLogger(__name__)


_ITER_DIR_PATTERN = re.compile(r"iter_(\d{7})$")
_ROLLOUT_STATE_PATTERNS = (
    re.compile(r"global_dataset_state_dict_(\d+)\.pt$"),
    re.compile(r"alfworld_data_source_state_(\d+)\.pt$"),
    re.compile(r"alfworld_frozen_data_source_state_(\d+)\.pt$"),
    re.compile(r"policy_state_(\d+)\.json$"),
)


@dataclass(frozen=True)
class BestCheckpoint:
    iteration: int
    score: float
    metric: str
    rollout_id: int


class CheckpointRetentionManager:
    def __init__(self, args):
        self.args = args
        self.policy = getattr(args, "checkpoint_retention_policy", "none")
        self.enabled = self.policy == "latest_and_best_eval"
        self.save_dir = Path(args.save) if self.enabled and getattr(args, "save", None) else None
        self.best_metric = getattr(args, "best_checkpoint_metric", None)
        self.best_limit = max(int(getattr(args, "best_checkpoint_limit", 2) or 0), 0)
        self.best_mode = getattr(args, "best_checkpoint_mode", "max")
        self.fixed_iterations = {
            int(value) for value in (getattr(args, "checkpoint_fixed_iterations", None) or [])
        }
        tracker_payload = self._load_best_tracker_payload()
        self.latest_iteration: int | None = self._read_latest_iteration()
        if self.latest_iteration is None and tracker_payload is not None:
            try:
                self.latest_iteration = int(tracker_payload["latest_iteration"])
            except (KeyError, TypeError, ValueError):
                pass
        self.best_checkpoints: list[BestCheckpoint] = self._load_best_tracker(tracker_payload)
        self.evaluated_iterations: set[int] = self._load_evaluated_iterations(tracker_payload)

    @property
    def requires_sync_save(self) -> bool:
        return self.enabled

    def should_save_for_eval(
        self,
        rollout_id: int,
        num_rollout_per_epoch: int | None = None,
    ) -> bool:
        if not self.enabled or self.best_limit <= 0:
            return False
        return should_run_periodic_action(rollout_id, self.args.eval_interval, num_rollout_per_epoch)

    def after_save(self, iteration: int) -> None:
        if not self.enabled:
            return
        self.latest_iteration = iteration
        self._write_best_tracker()
        self._prune()

    def restore_latest_durable(self, iteration: int) -> None:
        """Reset retention state to a model+datasource boundary proven durable by the runner."""

        if not self.enabled or self.save_dir is None:
            return
        iteration = int(iteration)
        checkpoint_dir = self.save_dir / f"iter_{iteration:07d}"
        if not checkpoint_dir.is_dir():
            raise FileNotFoundError(f"durable model checkpoint is missing: {checkpoint_dir}")
        self.latest_iteration = iteration
        self.evaluated_iterations = {value for value in self.evaluated_iterations if value <= iteration}
        self.best_checkpoints = [entry for entry in self.best_checkpoints if entry.iteration <= iteration]
        latest_path = self.save_dir / "latest_checkpointed_iteration.txt"
        latest_tmp = latest_path.with_suffix(".tmp")
        latest_tmp.write_text(f"{iteration}\n", encoding="utf-8")
        latest_tmp.replace(latest_path)
        self._write_best_tracker()
        self._prune()

    def after_eval(self, iteration: int | None, eval_metrics: dict[str, Any] | None) -> None:
        if not self.enabled or iteration is None or eval_metrics is None:
            return
        if self.best_limit > 0:
            if self.best_metric not in eval_metrics:
                available = ", ".join(sorted(eval_metrics))
                raise KeyError(
                    f"best checkpoint metric {self.best_metric!r} not found in eval metrics. "
                    f"Available metrics: {available}"
                )
            try:
                score = float(eval_metrics[self.best_metric])
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"best checkpoint metric {self.best_metric!r} must be numeric, "
                    f"got {eval_metrics[self.best_metric]!r}"
                ) from exc

            self.best_checkpoints = [entry for entry in self.best_checkpoints if entry.iteration != iteration]
            self.best_checkpoints.append(
                BestCheckpoint(iteration=iteration, score=score, metric=self.best_metric, rollout_id=iteration)
            )
            self.best_checkpoints = self._sort_best(self.best_checkpoints)[: self.best_limit]
        self.evaluated_iterations.add(iteration)
        self._write_best_tracker()
        self._prune()

    def needs_recovery_eval(
        self,
        iteration: int,
        num_rollout_per_epoch: int | None = None,
    ) -> bool:
        """Return whether the latest durable checkpoint still owes its scheduled eval."""
        return (
            self.latest_iteration == iteration
            and self.enabled
            and should_run_periodic_action(iteration, self.args.eval_interval, num_rollout_per_epoch)
            and iteration not in self.evaluated_iterations
        )

    def _sort_best(self, entries: list[BestCheckpoint]) -> list[BestCheckpoint]:
        if self.best_mode == "max":
            return sorted(entries, key=lambda entry: (-entry.score, entry.iteration))
        return sorted(entries, key=lambda entry: (entry.score, entry.iteration))

    def _read_latest_iteration(self) -> int | None:
        if not self.enabled or self.save_dir is None:
            return None
        tracker = self.save_dir / "latest_checkpointed_iteration.txt"
        try:
            return int(tracker.read_text(encoding="utf-8").strip())
        except (FileNotFoundError, ValueError):
            return None

    @property
    def _best_tracker_path(self) -> Path:
        assert self.save_dir is not None
        return self.save_dir / "best_checkpoint_tracker.json"

    def _load_best_tracker_payload(self) -> dict[str, Any] | None:
        if not self.enabled or self.save_dir is None or not self._best_tracker_path.is_file():
            return None
        try:
            payload = json.loads(self._best_tracker_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning("Failed to read best checkpoint tracker %s: %s", self._best_tracker_path, exc)
            return None
        if payload.get("metric") != self.best_metric or payload.get("mode") != self.best_mode:
            return None
        return payload

    def _load_best_tracker(self, payload: dict[str, Any] | None) -> list[BestCheckpoint]:
        if payload is None:
            return []

        entries: list[BestCheckpoint] = []
        for item in payload.get("checkpoints", []):
            try:
                entry = BestCheckpoint(
                    iteration=int(item["iteration"]),
                    score=float(item["score"]),
                    metric=str(item.get("metric", self.best_metric)),
                    rollout_id=int(item.get("rollout_id", int(item["iteration"]))),
                )
            except (KeyError, TypeError, ValueError):
                continue
            entries.append(entry)
        return self._sort_best(entries)[: self.best_limit]

    def _load_evaluated_iterations(self, payload: dict[str, Any] | None) -> set[int]:
        if payload is None:
            return set()
        evaluated_iterations: set[int] = set()
        for value in payload.get("evaluated_iterations", []):
            try:
                evaluated_iterations.add(int(value))
            except (TypeError, ValueError):
                continue
        return evaluated_iterations

    def _write_best_tracker(self) -> None:
        if self.save_dir is None:
            return
        payload = {
            "version": 2,
            "policy": self.policy,
            "metric": self.best_metric,
            "mode": self.best_mode,
            "limit": self.best_limit,
            "latest_iteration": self.latest_iteration,
            "evaluated_iterations": sorted(self.evaluated_iterations),
            "checkpoints": [asdict(entry) for entry in self.best_checkpoints],
        }
        self.save_dir.mkdir(parents=True, exist_ok=True)
        tmp_path = self._best_tracker_path.with_suffix(".tmp")
        tmp_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        tmp_path.replace(self._best_tracker_path)

    def _kept_iterations(self) -> set[int]:
        keep: set[int] = {entry.iteration for entry in self.best_checkpoints}
        keep.update(self.fixed_iterations)
        if self.latest_iteration is not None:
            keep.add(self.latest_iteration)
        return keep


    def _prune(self) -> None:
        if self.save_dir is None:
            return
        keep = self._kept_iterations()
        if not keep:
            return
        self._prune_iter_dirs(keep)
        self._prune_rollout_states(keep)
        self._prune_sdpo_ema_sidecars(keep)

    def _prune_iter_dirs(self, keep: set[int]) -> None:
        if self.save_dir is None or not self.save_dir.exists():
            return
        for path in self.save_dir.iterdir():
            match = _ITER_DIR_PATTERN.fullmatch(path.name)
            if not match or not path.is_dir():
                continue
            iteration = int(match.group(1))
            if iteration in keep:
                continue
            logger.info("Pruning checkpoint %s", path)
            shutil.rmtree(path)

    def _prune_rollout_states(self, keep: set[int]) -> None:
        assert self.save_dir is not None
        rollout_dir = self.save_dir / "rollout"
        if not rollout_dir.is_dir():
            return
        for path in rollout_dir.iterdir():
            iteration = _extract_rollout_state_iteration(path.name)
            if iteration is None or not path.is_file():
                continue
            if iteration not in keep:
                logger.info("Pruning rollout dataset checkpoint %s", path)
                path.unlink()

    def _prune_sdpo_ema_sidecars(self, keep: set[int]) -> None:
        assert self.save_dir is not None
        sidecar_root = self.save_dir / "sdpo_ema_teacher"
        if not sidecar_root.is_dir():
            return
        for path in sidecar_root.iterdir():
            match = _ITER_DIR_PATTERN.fullmatch(path.name)
            if not match or not path.is_dir():
                continue
            iteration = int(match.group(1))
            if iteration not in keep:
                logger.info("Pruning SDPO EMA teacher sidecar %s", path)
                shutil.rmtree(path)


def _extract_rollout_state_iteration(name: str) -> int | None:
    for pattern in _ROLLOUT_STATE_PATTERNS:
        match = pattern.fullmatch(name)
        if match:
            return int(match.group(1))
    return None


def restore_durable_checkpoint_boundary(save_dir: Path, iteration: int) -> dict[str, Any]:
    """Recoverably quarantine checkpoint state newer than a committed recovery marker.

    Megatron advances ``latest_checkpointed_iteration.txt`` during model save,
    before the datasource/policy sidecars become durable.  A crash in that window must not pair the newer model with the
    older datasource cursor.  Newer state is moved aside rather than deleted.
    """

    save_dir = Path(save_dir)
    iteration = int(iteration)
    durable_dir = save_dir / f"iter_{iteration:07d}"
    if not durable_dir.is_dir():
        raise FileNotFoundError(f"durable model checkpoint is missing: {durable_dir}")
    latest_path = save_dir / "latest_checkpointed_iteration.txt"
    try:
        latest = int(latest_path.read_text(encoding="utf-8").strip())
    except (FileNotFoundError, ValueError) as exc:
        raise RuntimeError(f"Megatron latest checkpoint pointer is missing or invalid: {latest_path}") from exc
    if latest < iteration:
        raise RuntimeError(
            f"Megatron latest pointer {latest} is older than durable recovery iteration {iteration}"
        )
    quarantine_root = save_dir / "recovery_orphans"
    suffix = 0
    while True:
        quarantine = quarantine_root / f"from_{latest:07d}_to_{iteration:07d}_{suffix:03d}"
        if not quarantine.exists():
            break
        suffix += 1
    quarantined: list[str] = []

    def move(path: Path, relative: Path) -> None:
        target = quarantine / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        path.replace(target)
        quarantined.append(str(relative))

    for path in sorted(save_dir.glob("iter_[0-9][0-9][0-9][0-9][0-9][0-9][0-9]")):
        match = _ITER_DIR_PATTERN.fullmatch(path.name)
        if match and path.is_dir() and int(match.group(1)) > iteration:
            move(path, Path(path.name))
    rollout_dir = save_dir / "rollout"
    if rollout_dir.is_dir():
        for path in sorted(rollout_dir.iterdir()):
            state_iteration = _extract_rollout_state_iteration(path.name)
            if path.is_file() and state_iteration is not None and state_iteration > iteration:
                move(path, Path("rollout") / path.name)
    ema_dir = save_dir / "sdpo_ema_teacher"
    if ema_dir.is_dir():
        for path in sorted(ema_dir.glob("iter_[0-9][0-9][0-9][0-9][0-9][0-9][0-9]")):
            match = _ITER_DIR_PATTERN.fullmatch(path.name)
            if match and path.is_dir() and int(match.group(1)) > iteration:
                move(path, Path("sdpo_ema_teacher") / path.name)

    if latest == iteration and not quarantined:
        return {
            "durable_iteration": iteration,
            "previous_latest_iteration": latest,
            "rolled_back": False,
            "quarantine_dir": None,
            "quarantined_paths": [],
        }

    latest_tmp = latest_path.with_suffix(".tmp")
    latest_tmp.write_text(f"{iteration}\n", encoding="utf-8")
    latest_tmp.replace(latest_path)
    tracker_path = save_dir / "best_checkpoint_tracker.json"
    if tracker_path.is_file():
        tracker = json.loads(tracker_path.read_text(encoding="utf-8"))
        tracker["latest_iteration"] = iteration
        tracker["evaluated_iterations"] = [
            int(value) for value in tracker.get("evaluated_iterations", []) if int(value) <= iteration
        ]
        tracker["checkpoints"] = [
            value for value in tracker.get("checkpoints", []) if int(value["iteration"]) <= iteration
        ]
        tracker_tmp = tracker_path.with_suffix(".tmp")
        tracker_tmp.write_text(json.dumps(tracker, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        tracker_tmp.replace(tracker_path)
    return {
        "durable_iteration": iteration,
        "previous_latest_iteration": latest,
        "rolled_back": True,
        "quarantine_dir": str(quarantine),
        "quarantined_paths": quarantined,
    }
