from types import SimpleNamespace

import pytest


class _RemoteMethod:
    def __init__(self, fn):
        self._fn = fn

    def remote(self, *args, **kwargs):
        return self._fn(*args, **kwargs)


class _RolloutManager:
    def __init__(self):
        self.eval_calls = []
        self.snapshot = {"iteration": 300, "engine_count": 1, "weight_versions": [0]}
        self.get_metrics_router_addr = _RemoteMethod(lambda: "http://metrics")
        self.get_data_source_cursor_state = _RemoteMethod(lambda: {"next_rollout_id": 0})
        self.register_eval_snapshot = _RemoteMethod(lambda _rollout_id: dict(self.snapshot))
        self.verify_eval_snapshot = _RemoteMethod(lambda _rollout_id: dict(self.snapshot))
        self.eval = _RemoteMethod(self._eval)
        self.dispose = _RemoteMethod(lambda: None)

    def _eval(self, rollout_id):
        self.eval_calls.append(rollout_id)
        return {}


class _ActorModel:
    def update_weights(self):
        return None


@pytest.mark.unit
def test_train_eval_only_uses_start_rollout_id(monkeypatch):
    import train as train_module

    manager = _RolloutManager()
    monkeypatch.setattr(train_module, "create_placement_groups", lambda _args: {"rollout": object()})
    monkeypatch.setattr(train_module, "create_rollout_manager", lambda _args, _pg: (manager, None))
    monkeypatch.setattr(train_module, "create_training_models", lambda _args, _pgs, _manager: (_ActorModel(), None))
    monkeypatch.setattr(train_module, "init_tracking", lambda _args: None)
    monkeypatch.setattr(train_module, "update_tracking_open_metrics", lambda _args, _addr: None)
    monkeypatch.setattr(train_module, "finish_tracking", lambda _args: None)
    monkeypatch.setattr(train_module, "configure_logger", lambda: None)
    monkeypatch.setattr(train_module.ray, "get", lambda value: value)

    args = SimpleNamespace(
        num_rollout=0,
        eval_interval=10,
        start_rollout_id=300,
        keep_old_actor=False,
        offload_rollout=False,
        check_weight_update_equal=False,
        use_critic=False,
        rollout_global_dataset=False,
    )

    train_module.train(args)

    assert manager.eval_calls == [300]
