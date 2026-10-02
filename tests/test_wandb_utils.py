from types import SimpleNamespace

from slime.utils import logging_utils, wandb_utils


class DummyWandb:
    class Settings:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    class util:
        @staticmethod
        def generate_id():
            return "suffix"

    def __init__(self):
        self.init_kwargs = None
        self.run = None
        self.metrics = []
        self.logged = []

    def init(self, **kwargs):
        self.init_kwargs = kwargs
        self.run = SimpleNamespace(id=kwargs.get("id", "generated-id"))

    def define_metric(self, *args, **kwargs):
        self.metrics.append((args, kwargs))

    def login(self, **kwargs):
        raise AssertionError(f"unexpected wandb.login call: {kwargs}")

    def log(self, metrics):
        self.logged.append(metrics)


def _wandb_args(**overrides):
    values = {
        "use_wandb": True,
        "wandb_mode": "online",
        "wandb_key": None,
        "wandb_host": None,
        "wandb_team": "entity",
        "wandb_project": "project",
        "wandb_group": "group",
        "wandb_exp_name": "experiment-name",
        "wandb_run_id": "fixed-run-id",
        "wandb_random_suffix": False,
        "wandb_dir": None,
        "use_tensorboard": False,
        "rank": 0,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_init_wandb_primary_uses_exp_name_and_run_id(monkeypatch, tmp_path):
    dummy = DummyWandb()
    monkeypatch.setattr(wandb_utils, "wandb", dummy)
    args = _wandb_args(wandb_dir=str(tmp_path))

    wandb_utils.init_wandb_primary(args)

    assert dummy.init_kwargs["name"] == "experiment-name"
    assert dummy.init_kwargs["group"] == "group"
    assert dummy.init_kwargs["id"] == "fixed-run-id"
    assert dummy.init_kwargs["resume"] == "allow"
    assert dummy.init_kwargs["dir"] == str(tmp_path)
    assert args.wandb_run_id == "fixed-run-id"


def test_init_wandb_primary_keeps_generated_id_when_run_id_is_absent(monkeypatch):
    dummy = DummyWandb()
    monkeypatch.setattr(wandb_utils, "wandb", dummy)
    args = _wandb_args(wandb_run_id=None, wandb_exp_name=None)

    wandb_utils.init_wandb_primary(args)

    assert "id" not in dummy.init_kwargs
    assert "resume" not in dummy.init_kwargs
    assert dummy.init_kwargs["name"] == "group"
    assert args.wandb_run_id == "generated-id"


def test_init_wandb_primary_random_suffix_keeps_exp_name_as_run_name(monkeypatch):
    dummy = DummyWandb()
    monkeypatch.setattr(wandb_utils, "wandb", dummy)
    args = _wandb_args(wandb_random_suffix=True)

    wandb_utils.init_wandb_primary(args)

    assert dummy.init_kwargs["group"] == "group_suffix"
    assert dummy.init_kwargs["name"] == "experiment-name-RANK_0"


def test_init_wandb_primary_disabled_mode_skips_init(monkeypatch):
    dummy = DummyWandb()
    monkeypatch.setattr(wandb_utils, "wandb", dummy)
    monkeypatch.setattr(logging_utils, "wandb", dummy)
    args = _wandb_args(wandb_mode="disabled")

    wandb_utils.init_wandb_primary(args)
    logging_utils.log(args, {"train/step": 0, "train/loss": 1.0}, step_key="train/step")

    assert dummy.init_kwargs is None
    assert dummy.logged == []
    assert args.wandb_run_id is None
    assert args.use_wandb is False
