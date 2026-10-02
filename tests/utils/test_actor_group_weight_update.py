from types import SimpleNamespace

from slime.ray.actor_group import RayTrainGroup


class _RemoteCall:
    def __init__(self, value=None):
        self.value = value
        self.calls = 0

    def remote(self, *args, **kwargs):
        self.calls += 1
        return self.value


def test_update_weights_clears_engine_discovery_after_all_workers(monkeypatch):
    events = []
    handlers = []
    for rank in range(3):
        update = _RemoteCall(value=f"worker-{rank}")
        handlers.append(SimpleNamespace(update_weights=update))

    clear = _RemoteCall(value="cleared")
    group = object.__new__(RayTrainGroup)
    group._actor_handlers = handlers
    group.rollout_manager = SimpleNamespace(clear_updatable_num_new_engines=clear)

    def fake_get(refs):
        events.append(refs)
        return refs

    monkeypatch.setattr("slime.ray.actor_group.ray.get", fake_get)

    assert group.update_weights() == ["worker-0", "worker-1", "worker-2"]
    assert events == [["worker-0", "worker-1", "worker-2"], "cleared"]
    assert clear.calls == 1


def test_set_rollout_manager_retains_driver_handle(monkeypatch):
    worker_set = _RemoteCall(value="worker-set")
    group = object.__new__(RayTrainGroup)
    group._actor_handlers = [SimpleNamespace(set_rollout_manager=worker_set)]
    manager = object()
    monkeypatch.setattr("slime.ray.actor_group.ray.get", lambda refs: refs)

    assert group.set_rollout_manager(manager) == ["worker-set"]
    assert group.rollout_manager is manager
