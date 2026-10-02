# Modified for ReSAIL. See NOTICE and LICENSE for attribution and terms.
from abc import ABC, abstractmethod
from collections import defaultdict
from collections.abc import Callable, Iterable
from pathlib import Path

import torch

_SourceGetter = Callable[[], Iterable[tuple[str, torch.Tensor]]]


class TensorBackuper(ABC):
    @staticmethod
    def create(source_getter, single_tag):
        if single_tag is None:
            return _TensorBackuperNormal(source_getter=source_getter)
        else:
            return _TensorBackuperNoop(source_getter=source_getter, single_tag=single_tag)

    def __init__(self, source_getter: _SourceGetter):
        self._source_getter = source_getter

    @property
    @abstractmethod
    def backup_tags(self):
        raise NotImplementedError

    @abstractmethod
    def get(self, tag: str):
        raise NotImplementedError

    @abstractmethod
    def backup(self, tag: str):
        raise NotImplementedError

    def copy(self, *, src_tag: str, dst_tag: str):
        raise NotImplementedError

    def discard(self, tag: str) -> None:
        """Release a temporary backup without touching the live tensors."""
        raise NotImplementedError

    @abstractmethod
    def restore(self, tag: str):
        raise NotImplementedError

    def save(self, tag: str, path: str | Path) -> None:
        raise NotImplementedError

    def load(self, tag: str, path: str | Path) -> None:
        raise NotImplementedError


class _TensorBackuperNormal(TensorBackuper):
    def __init__(self, source_getter):
        super().__init__(source_getter=source_getter)
        self._backups: dict[str, dict[str, torch.Tensor]] = defaultdict(dict)

    @property
    def backup_tags(self):
        return list(self._backups)

    def get(self, tag: str):
        return self._backups[tag]

    @torch.no_grad()
    def backup(self, tag: str) -> None:
        backup_dict = self._backups[tag]
        for name, param in self._source_getter():
            if name not in backup_dict:
                backup_dict[name] = _empty_cpu_like(param)
            backup_dict[name].copy_(param.detach(), non_blocking=True)
        _synchronize_if_cuda()

    @torch.no_grad()
    def copy(self, *, src_tag: str, dst_tag: str):
        for name in self._backups[dst_tag]:
            self._backups[dst_tag][name].copy_(self._backups[src_tag][name])

    def discard(self, tag: str) -> None:
        if tag not in self._backups:
            raise KeyError(f"Cannot discard unknown tensor backup tag: {tag}")
        del self._backups[tag]

    @torch.no_grad()
    def restore(self, tag: str) -> None:
        backup_dict = self._backups[tag]
        for name, param in self._source_getter():
            assert name in backup_dict
            param.copy_(backup_dict[name], non_blocking=True)
        _synchronize_if_cuda()

    @torch.no_grad()
    def save(self, tag: str, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        backup_dict = self._backups[tag]
        payload = {
            "version": 1,
            "tag": tag,
            "tensors": {name: tensor.detach().cpu().clone() for name, tensor in backup_dict.items()},
        }
        tmp_path = path.with_name(f"{path.name}.tmp")
        torch.save(payload, tmp_path)
        tmp_path.replace(path)

    @torch.no_grad()
    def load(self, tag: str, path: str | Path) -> None:
        payload = torch.load(Path(path), map_location="cpu")
        if not isinstance(payload, dict) or payload.get("version") != 1 or "tensors" not in payload:
            raise ValueError(f"Invalid tensor backup file: {path}")
        tensors = payload["tensors"]
        if not isinstance(tensors, dict):
            raise ValueError(f"Invalid tensor backup tensors in: {path}")
        source = dict(self._source_getter())
        missing = set(source) ^ set(tensors)
        if missing:
            raise ValueError(f"Tensor backup key mismatch for tag {tag!r}: {sorted(missing)[:5]}")

        backup_dict = self._backups[tag]
        for name, param in source.items():
            tensor = tensors[name]
            if tensor.shape != param.shape:
                raise ValueError(
                    f"Tensor backup shape mismatch for {name}: checkpoint {tuple(tensor.shape)} != "
                    f"model {tuple(param.shape)}"
                )
            if tensor.dtype != param.dtype:
                raise ValueError(
                    f"Tensor backup dtype mismatch for {name}: checkpoint {tensor.dtype} != model {param.dtype}"
                )
            if name not in backup_dict:
                backup_dict[name] = _empty_cpu_like(param)
            backup_dict[name].copy_(tensor, non_blocking=True)


class _TensorBackuperNoop(TensorBackuper):
    def __init__(self, source_getter, single_tag):
        super().__init__(source_getter=source_getter)
        self._single_tag = single_tag
        # Sanity check for safety
        self._backup_hash_dict = None

    @property
    def backup_tags(self):
        return [self._single_tag]

    def get(self, tag: str):
        ans = dict(self._source_getter())
        ans = {k: v.detach() for k, v in ans.items()}
        assert _compute_hash_dict(ans) == self._backup_hash_dict
        return ans

    def backup(self, tag: str) -> None:
        assert tag == self._single_tag
        self._backup_hash_dict = _compute_hash_dict(dict(self._source_getter()))
        _synchronize_if_cuda()

    def restore(self, tag: str) -> None:
        assert tag == self._single_tag
        assert _compute_hash_dict(dict(self._source_getter())) == self._backup_hash_dict
        _synchronize_if_cuda()

    def discard(self, tag: str) -> None:
        raise RuntimeError("Cannot discard the single live tensor-backup tag.")

    def save(self, tag: str, path: str | Path) -> None:
        raise RuntimeError("Cannot save tensor backup when TensorBackuper is in single-tag noop mode.")

    def load(self, tag: str, path: str | Path) -> None:
        raise RuntimeError("Cannot load tensor backup when TensorBackuper is in single-tag noop mode.")


def _synchronize_if_cuda() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _empty_cpu_like(tensor: torch.Tensor) -> torch.Tensor:
    try:
        return torch.empty_like(tensor, device=torch.device("cpu"), pin_memory=True)
    except RuntimeError:
        return torch.empty_like(tensor, device=torch.device("cpu"))


def _compute_hash_dict(tensors: dict[str, torch.Tensor]):
    return {k: _compute_hash_tensor(v) for k, v in tensors.items()}


def _compute_hash_tensor(x: torch.Tensor):
    # Not a real/good hash, but pretty fast
    x = x.contiguous()
    x = x.view(-1)
    x = x.view(torch.uint32)
    x = x.sum()
    return x.item()
