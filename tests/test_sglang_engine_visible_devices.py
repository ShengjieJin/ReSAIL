from __future__ import annotations

from dataclasses import dataclass

from slime.backends.sglang_utils.sglang_engine import _server_args_for_visible_device_scope


@dataclass
class FakeServerArgs:
    base_gpu_id: int = 4
    gpu_id_step: int = 1
    tp_size: int = 1
    pp_size: int = 1
    dp_size: int = 1
    nnodes: int = 1


def test_visible_device_scope_uses_physical_gpu_when_parent_cvd_is_unset(monkeypatch):
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    monkeypatch.setenv("SGLANG_ONE_VISIBLE_DEVICE_PER_PROCESS", "true")

    original = FakeServerArgs(base_gpu_id=4)
    child, visible_devices = _server_args_for_visible_device_scope(original)

    assert visible_devices == "4"
    assert child.base_gpu_id == 0
    assert original.base_gpu_id == 4


def test_visible_device_scope_maps_local_base_gpu_through_parent_cvd(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "4,5,6,7")
    monkeypatch.setenv("SGLANG_ONE_VISIBLE_DEVICE_PER_PROCESS", "true")

    child, visible_devices = _server_args_for_visible_device_scope(FakeServerArgs(base_gpu_id=2))

    assert visible_devices == "6"
    assert child.base_gpu_id == 0


def test_visible_device_scope_keeps_engine_gpu_list_for_tensor_parallel(monkeypatch):
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    monkeypatch.setenv("SGLANG_ONE_VISIBLE_DEVICE_PER_PROCESS", "true")

    child, visible_devices = _server_args_for_visible_device_scope(FakeServerArgs(base_gpu_id=4, tp_size=2))

    assert visible_devices == "4,5"
    assert child.base_gpu_id == 0


def test_visible_device_scope_is_disabled_without_sglang_env(monkeypatch):
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    monkeypatch.delenv("SGLANG_ONE_VISIBLE_DEVICE_PER_PROCESS", raising=False)

    original = FakeServerArgs(base_gpu_id=4)
    child, visible_devices = _server_args_for_visible_device_scope(original)

    assert visible_devices is None
    assert child is original
