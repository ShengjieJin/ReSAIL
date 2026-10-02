from __future__ import annotations

import json
import importlib.util
import os
from pathlib import Path
import signal
import subprocess
import sys

import pytest

_guard_path = Path(__file__).resolve().parents[2] / "tools" / "experiment_guard.py"
_spec = importlib.util.spec_from_file_location("resail_experiment_guard", _guard_path)
assert _spec is not None and _spec.loader is not None
experiment_guard = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = experiment_guard
_spec.loader.exec_module(experiment_guard)


NUM_GPUS = 0


def _config(tmp_path: Path, **overrides) -> experiment_guard.GuardConfig:
    values = {
        "container": "test-container",
        "audit_log": tmp_path / "guard.jsonl",
        "lock_file": tmp_path / "guard.lock",
    }
    values.update(overrides)
    return experiment_guard.GuardConfig(**values)


def _sample(**overrides) -> experiment_guard.ResourceSample:
    values = {
        "container_running": True,
        "pids_current": 100,
        "pids_limit": 16384,
        "cpu_limit": 160.0,
        "available_memory_bytes": 1024**4,
        "load_1m": 10.0,
        "host_gpu_count": 8,
        "container_gpu_count": 8,
    }
    values.update(overrides)
    return experiment_guard.ResourceSample(**values)


@pytest.mark.unit
def test_guard_accepts_nominal_resources_without_load_strikes(tmp_path):
    violations, strikes = experiment_guard.evaluate_sample(_sample(), _config(tmp_path), 3)
    assert violations == []
    assert strikes == 0


@pytest.mark.unit
def test_audit_file_is_private(tmp_path):
    guard = experiment_guard.DockerExperimentGuard(_config(tmp_path))
    guard.append_audit("test")
    assert guard.config.audit_log.stat().st_mode & 0o777 == 0o600


@pytest.mark.unit
def test_guard_stops_immediate_pid_memory_and_limit_drift(tmp_path):
    sample = _sample(
        pids_current=15000,
        pids_limit=None,
        cpu_limit=192.0,
        available_memory_bytes=127 * 1024**3,
    )
    violations, strikes = experiment_guard.evaluate_sample(sample, _config(tmp_path), 0)
    assert strikes == 0
    assert violations == [
        "pids_limit=None,expected=16384",
        "cpu_limit=192.0,expected=160.0",
        "pids_current=15000,stop=15000",
        f"available_memory={127 * 1024**3},minimum={128 * 1024**3}",
    ]


@pytest.mark.unit
def test_guard_requires_four_consecutive_high_load_samples(tmp_path):
    config = _config(tmp_path)
    strikes = 0
    for _ in range(3):
        violations, strikes = experiment_guard.evaluate_sample(_sample(load_1m=176.0), config, strikes)
        assert violations == []
    violations, strikes = experiment_guard.evaluate_sample(_sample(load_1m=176.0), config, strikes)
    assert violations == ["load_1m=176.0,strikes=4"]
    assert strikes == 4


@pytest.mark.unit
def test_workload_detection_ignores_ray_control_and_zombies():
    audit = "\n".join(
        [
            "10 1 S ray::IDLE ray::IDLE",
            "11 1 S ray::DashboardAg ray::DashboardAgent --node-ip-address=127.0.0.1",
            "12 1 Z ray::Actor ray::MegatronTrainRayActor",
            "13 1 S ray::Actor ray::MegatronTrainRayActor",
            "14 1 S python3 python3 exp/paper/text.py run --task alfworld",
            "17 1 S python3 python3 -m exp.paper.text run --task alfworld",
            "15 1 S python3 python3 unrelated.py",
            "16 1 S python3 python3 tools/convert_torch_dist_to_hf.py --input checkpoint",
        ]
    )
    assert experiment_guard.active_workload_lines(audit) == [*audit.splitlines()[3:6], audit.splitlines()[7]]


@pytest.mark.unit
def test_workload_detection_covers_paper_model_commands_without_cpu_checks():
    commands = [
        "python3 -m exp.paper.text run --task textcraft",
        "python3 -m exp.paper.text prepare-c1 --task alfworld",
        "python3 /workspace/slime/exp/paper/text.py prepare-c1 --task textcraft",
        "python3 -m exp.paper.text_guidance --endpoint-file x",
        "python3 exp/paper/text_epd_alfworld.py --endpoint-file x",
        "python3 -m exp.paper.text_epd_textcraft --endpoint-file x",
    ]
    cpu_commands = [
        "python3 -m exp.paper.text check --task alfworld",
        "python3 exp/paper/text.py prepare --task alfworld",
    ]
    audit = "\n".join(f"{index} 1 S python3 {command}" for index, command in enumerate([*commands, *cpu_commands]))
    assert experiment_guard.active_workload_lines(audit) == audit.splitlines()[: len(commands)]


@pytest.mark.unit
def test_workload_detection_covers_gpu_shell_wrappers_between_child_processes():
    gpu_commands = [
        "bash scripts/experiments/text_cycles.sh --task alfworld",
        "bash scripts/prepare/text_c1.sh --task alfworld",
        "bash /workspace/slime/scripts/prepare/text_c1.sh --task textcraft",
        "bash scripts/experiments/text_guidance.sh",
        "bash scripts/experiments/text_epd_alfworld.sh",
        "bash scripts/experiments/text_epd_textcraft.sh",
        "bash scripts/prepare/convert_text_checkpoint.sh 4b /models/qwen3 /checkpoints/new",
        "bash /workspace/slime/scripts/prepare/convert_text_checkpoint.sh 8b /models/qwen3 /checkpoints/new",
    ]
    cpu_commands = ["bash scripts/check_public_contracts.sh"]
    audit = "\n".join(f"{index} 1 S bash {command}" for index, command in enumerate([*gpu_commands, *cpu_commands]))
    audit += "\n99 1 S rg rg scripts/experiments/text_cycles.sh"
    assert experiment_guard.active_workload_lines(audit) == audit.splitlines()[: len(gpu_commands)]


@pytest.mark.unit
def test_guard_preflight_restarts_only_dirty_idle_container(tmp_path, monkeypatch):
    config = _config(tmp_path, restart_zombies=1000, require_idle_gpus=False)
    guard = experiment_guard.DockerExperimentGuard(config)
    inspect = {"Id": "abc", "State": {"Pid": 12, "Running": True}}
    calls = []
    zombie_counts = iter((1200, 0))
    monkeypatch.setattr(guard, "_inspect", lambda: inspect)
    monkeypatch.setattr(guard, "_refresh_cgroup", lambda _: None)
    monkeypatch.setattr(guard, "_workloads", lambda: [])
    monkeypatch.setattr(guard, "_zombie_count", lambda: next(zombie_counts))
    monkeypatch.setattr(guard, "sample", lambda: _sample())
    monkeypatch.setattr(guard, "_gpu_probe_with_retry", lambda: None)
    monkeypatch.setattr(
        guard,
        "_run",
        lambda argv, **_: calls.append(argv) or subprocess.CompletedProcess(argv, 0, "", ""),
    )

    guard.preflight()

    assert calls == [["docker", "restart", "--time", "30", "test-container"]]
    rows = [json.loads(line) for line in config.audit_log.read_text().splitlines()]
    assert [row["event"] for row in rows] == ["container_restart", "preflight_passed"]


@pytest.mark.unit
def test_guard_preflight_never_restarts_with_live_workloads(tmp_path, monkeypatch):
    guard = experiment_guard.DockerExperimentGuard(_config(tmp_path))
    inspect = {"Id": "abc", "State": {"Pid": 12, "Running": True}}
    monkeypatch.setattr(guard, "_inspect", lambda: inspect)
    monkeypatch.setattr(guard, "_refresh_cgroup", lambda _: None)
    monkeypatch.setattr(guard, "_workloads", lambda: ["12 raylet"])
    with pytest.raises(RuntimeError, match="active Ray/SGLang workloads"):
        guard.preflight()


@pytest.mark.unit
def test_guard_preflight_marks_and_rejects_degraded_pid_state(tmp_path, monkeypatch):
    guard = experiment_guard.DockerExperimentGuard(_config(tmp_path))
    inspect = {"Id": "abc", "State": {"Pid": 12, "Running": True}}
    monkeypatch.setattr(guard, "_inspect", lambda: inspect)
    monkeypatch.setattr(guard, "_refresh_cgroup", lambda _: None)
    monkeypatch.setattr(guard, "_workloads", lambda: [])
    monkeypatch.setattr(guard, "_zombie_count", lambda: 0)
    monkeypatch.setattr(guard, "sample", lambda: _sample(pids_current=13_500))
    monkeypatch.setattr(guard, "_process_attribution", lambda: {"total_processes": 100})

    with pytest.raises(RuntimeError, match="PID state is degraded"):
        guard.preflight()

    marker = json.loads(guard.degraded_marker.read_text())
    assert marker["status"] == "degraded"
    assert marker["pids_current"] == 13_500


@pytest.mark.unit
def test_guard_cli_requires_command_unless_preflight_only(tmp_path):
    base = ["--container", "c", "--audit-log", str(tmp_path / "a"), "--lock-file", str(tmp_path / "lock")]
    with pytest.raises(SystemExit):
        experiment_guard._parse_args(base)
    args, command = experiment_guard._parse_args([*base, "--preflight-only"])
    assert args.preflight_only is True
    assert command == []
    with pytest.raises(SystemExit):
        experiment_guard._parse_args(
            [*base, "--stop-pid", "7000"]
        )
    with pytest.raises(SystemExit):
        experiment_guard._parse_args([*base, "--stop-load", "nan", "--preflight-only"])


@pytest.mark.unit
def test_guard_lock_rejects_overlapping_launches(tmp_path):
    lock_file = tmp_path / "guard.lock"
    with experiment_guard.exclusive_lock(lock_file):
        with pytest.raises(RuntimeError, match="another experiment guard"):
            with experiment_guard.exclusive_lock(lock_file):
                pass
    assert f"pid={os.getpid()}" in lock_file.read_text()


@pytest.mark.unit
def test_guarded_command_has_no_cleanup_on_clean_success(tmp_path, monkeypatch):
    config = _config(tmp_path, interval_seconds=0.01)
    samples = []
    cleanups = []
    monkeypatch.setattr(experiment_guard.DockerExperimentGuard, "preflight", lambda self: _sample())
    monkeypatch.setattr(
        experiment_guard.DockerExperimentGuard,
        "sample",
        lambda self: samples.append(True) or _sample(),
    )
    monkeypatch.setattr(experiment_guard.DockerExperimentGuard, "_workloads", lambda self: [])
    monkeypatch.setattr(experiment_guard.DockerExperimentGuard, "_host_gpu_compute_pids", lambda self: [])
    monkeypatch.setattr(
        experiment_guard.DockerExperimentGuard,
        "stop_workloads",
        lambda self, reason: cleanups.append(reason),
    )

    assert experiment_guard.run_guarded(config, ["/bin/sh", "-c", "exit 0"]) == 0
    assert cleanups == []
    events = [json.loads(line)["event"] for line in config.audit_log.read_text().splitlines()]
    assert events[0] == "command_started"
    assert events[-1] == "command_finished"


@pytest.mark.unit
def test_guarded_command_stops_on_monitor_failure(tmp_path, monkeypatch):
    config = _config(tmp_path, interval_seconds=0.01)
    cleanups = []
    monkeypatch.setattr(experiment_guard.DockerExperimentGuard, "preflight", lambda self: _sample())
    monkeypatch.setattr(
        experiment_guard.DockerExperimentGuard,
        "sample",
        lambda self: (_ for _ in ()).throw(RuntimeError("cgroup disappeared")),
    )
    monkeypatch.setattr(
        experiment_guard.DockerExperimentGuard,
        "stop_workloads",
        lambda self, reason: cleanups.append(reason),
    )

    result = experiment_guard.run_guarded(config, ["/bin/sh", "-c", "sleep 10"])

    assert result == experiment_guard.SAFETY_EXIT_CODE
    assert cleanups and cleanups[0].startswith("monitor_error:RuntimeError")
    events = [json.loads(line)["event"] for line in config.audit_log.read_text().splitlines()]
    assert events.count("monitor_retry") == config.monitor_error_strikes
    assert "safety_trip" in events


@pytest.mark.unit
def test_guarded_command_survives_one_transient_monitor_failure(tmp_path, monkeypatch):
    config = _config(tmp_path, interval_seconds=0.01)
    cleanups = []
    sample_calls = 0

    def transient_sample(self):
        nonlocal sample_calls
        sample_calls += 1
        if sample_calls == 1:
            raise subprocess.TimeoutExpired(["nvidia-smi"], 30)
        return _sample()

    monkeypatch.setattr(experiment_guard.DockerExperimentGuard, "preflight", lambda self: _sample())
    monkeypatch.setattr(experiment_guard.DockerExperimentGuard, "sample", transient_sample)
    monkeypatch.setattr(experiment_guard.DockerExperimentGuard, "_workloads", lambda self: [])
    monkeypatch.setattr(experiment_guard.DockerExperimentGuard, "_host_gpu_compute_pids", lambda self: [])
    monkeypatch.setattr(
        experiment_guard.DockerExperimentGuard,
        "stop_workloads",
        lambda self, reason: cleanups.append(reason),
    )

    result = experiment_guard.run_guarded(config, ["/bin/sh", "-c", "sleep 0.05"])

    assert result == 0
    assert sample_calls >= 2
    assert cleanups == []
    events = [json.loads(line)["event"] for line in config.audit_log.read_text().splitlines()]
    assert events.count("monitor_retry") == 1
    assert "safety_trip" not in events


@pytest.mark.unit
def test_guarded_command_cleans_up_if_audit_write_fails_after_launch(tmp_path, monkeypatch):
    config = _config(tmp_path)
    cleanups = []
    monkeypatch.setattr(experiment_guard.DockerExperimentGuard, "preflight", lambda self: _sample())
    monkeypatch.setattr(
        experiment_guard.DockerExperimentGuard,
        "append_audit",
        lambda self, event, **fields: (_ for _ in ()).throw(OSError("audit disk full")),
    )
    monkeypatch.setattr(
        experiment_guard.DockerExperimentGuard,
        "stop_workloads",
        lambda self, reason: cleanups.append(reason),
    )

    with pytest.raises(OSError, match="audit disk full"):
        experiment_guard.run_guarded(config, ["/bin/sh", "-c", "sleep 10"])

    assert cleanups == ["guard_exception"]


@pytest.mark.unit
def test_guarded_command_signal_terminates_and_cleans_up(tmp_path, monkeypatch):
    config = _config(tmp_path)
    cleanups = []
    handlers = {}
    monkeypatch.setattr(experiment_guard.DockerExperimentGuard, "preflight", lambda self: _sample())
    monkeypatch.setattr(
        experiment_guard.signal,
        "signal",
        lambda signum, handler: handlers.setdefault(signum, handler) or signal.SIG_DFL,
    )
    original_append = experiment_guard.DockerExperimentGuard.append_audit

    def append_and_signal(self, event, **fields):
        original_append(self, event, **fields)
        if event == "command_started":
            handlers[signal.SIGTERM](signal.SIGTERM, None)

    monkeypatch.setattr(experiment_guard.DockerExperimentGuard, "append_audit", append_and_signal)
    monkeypatch.setattr(
        experiment_guard.DockerExperimentGuard,
        "stop_workloads",
        lambda self, reason: cleanups.append(reason),
    )

    assert experiment_guard.run_guarded(config, ["/bin/sh", "-c", "sleep 10"]) == 143
    assert cleanups == ["signal:SIGTERM"]


@pytest.mark.unit
def test_guard_audit_does_not_persist_command_arguments(tmp_path, monkeypatch):
    config = _config(tmp_path)
    monkeypatch.setattr(experiment_guard.DockerExperimentGuard, "preflight", lambda self: _sample())
    monkeypatch.setattr(experiment_guard.DockerExperimentGuard, "_workloads", lambda self: [])
    monkeypatch.setattr(experiment_guard.DockerExperimentGuard, "_host_gpu_compute_pids", lambda self: [])
    monkeypatch.setattr(experiment_guard.DockerExperimentGuard, "sample", lambda self: _sample())

    assert experiment_guard.run_guarded(config, ["/bin/true", "secret-value"]) == 0
    started = next(
        json.loads(line)
        for line in config.audit_log.read_text().splitlines()
        if json.loads(line)["event"] == "command_started"
    )
    assert started["executable"] == "/bin/true"
    assert started["argument_count"] == 2
    assert "secret-value" not in config.audit_log.read_text()


@pytest.mark.unit
def test_cleanup_escalates_to_restart_and_verifies_terminal_state(tmp_path, monkeypatch):
    guard = experiment_guard.DockerExperimentGuard(_config(tmp_path))
    calls = []
    residue = iter([(["live runner"], [123])] * 10 + [([], [])])
    monkeypatch.setattr(guard, "_workloads", lambda: ["live runner"])
    monkeypatch.setattr(guard, "cleanup_residue", lambda: next(residue))
    monkeypatch.setattr(experiment_guard.time, "sleep", lambda _: None)
    monkeypatch.setattr(
        guard,
        "_run",
        lambda argv, **_: calls.append(argv) or subprocess.CompletedProcess(argv, 1, "", "failed"),
    )

    guard.stop_workloads("test")

    assert calls[0][-3:] == ["ray", "stop", "--force"]
    assert calls[1][:3] == ["docker", "restart", "--time"]
    events = [json.loads(line)["event"] for line in guard.config.audit_log.read_text().splitlines()]
    assert "cleanup_escalated" in events
    assert events[-1] == "cleanup_finished"


@pytest.mark.unit
def test_cleanup_fails_if_restart_leaves_gpu_or_workload_residue(tmp_path, monkeypatch):
    guard = experiment_guard.DockerExperimentGuard(_config(tmp_path))
    monkeypatch.setattr(guard, "_workloads", lambda: ["live converter"])
    monkeypatch.setattr(guard, "cleanup_residue", lambda: (["live converter"], [456]))
    monkeypatch.setattr(experiment_guard.time, "sleep", lambda _: None)
    monkeypatch.setattr(
        guard,
        "_run",
        lambda argv, **_: subprocess.CompletedProcess(argv, 0, "", ""),
    )

    with pytest.raises(RuntimeError, match="did not reach a safe terminal state"):
        guard.stop_workloads("test")


@pytest.mark.unit
def test_cleanup_timeout_still_escalates_to_container_restart(tmp_path, monkeypatch):
    guard = experiment_guard.DockerExperimentGuard(_config(tmp_path))
    calls = []
    residue = iter([(["live runner"], [789])] * 10 + [([], [])])
    monkeypatch.setattr(guard, "_workloads", lambda: ["live runner"])
    monkeypatch.setattr(guard, "cleanup_residue", lambda: next(residue))
    monkeypatch.setattr(experiment_guard.time, "sleep", lambda _: None)

    def run(argv, **_):
        calls.append(argv)
        if argv[-3:] == ["ray", "stop", "--force"]:
            raise subprocess.TimeoutExpired(argv, 60)
        return subprocess.CompletedProcess(argv, 0, "", "")

    monkeypatch.setattr(guard, "_run", run)

    guard.stop_workloads("test")
    assert any(argv[:3] == ["docker", "restart", "--time"] for argv in calls)


@pytest.mark.unit
def test_runtime_sample_fails_closed_when_gpu_topology_disappears(tmp_path):
    violations, _ = experiment_guard.evaluate_sample(
        _sample(host_gpu_count=None, container_gpu_count=7), _config(tmp_path), 0
    )
    assert "host_gpu_count=None,expected=8" in violations
    assert "container_gpu_count=7,expected=8" in violations


@pytest.mark.unit
def test_guard_safe_fast_pid_health_bands(tmp_path):
    config = _config(tmp_path)
    assert experiment_guard.pid_health_state(_sample(pids_current=11999), config) == "healthy"
    assert experiment_guard.pid_health_state(_sample(pids_current=12000), config) == "warning"
    assert experiment_guard.pid_health_state(_sample(pids_current=13500), config) == "degraded"
    assert experiment_guard.pid_health_state(_sample(pids_current=15000), config) == "stop"


@pytest.mark.unit
def test_guard_process_attribution_records_required_categories():
    rows = "\n".join(
        [
            "1 0 7 S raylet raylet --node-ip-address=local",
            "2 1 2 S ray::IDLE ray::IDLE",
            "3 1 9 S ray::Actor ray::MegatronTrainRayActor",
            "4 1 11 S sglang sglang::scheduler",
            "5 1 3 S wandb-core wandb-core",
            "6 3 1 S alfworld-env python3 -c spawn_main",
            "7 1 1 S python3 python3 helper.py",
        ]
    )
    audit = experiment_guard.process_attribution(rows)
    assert audit["total_processes"] == 7
    assert audit["total_threads"] == 34
    assert audit["categories"]["ray_core_dashboard"] == {"processes": 1, "threads": 7}
    assert audit["categories"]["ray_idle_workers"] == {"processes": 1, "threads": 2}
    assert audit["categories"]["megatron_actors"] == {"processes": 1, "threads": 9}
    assert audit["categories"]["sglang_engines_schedulers"] == {"processes": 1, "threads": 11}
    assert audit["categories"]["wandb_processes"] == {"processes": 1, "threads": 3}
    assert audit["categories"]["alfworld_workers"] == {"processes": 1, "threads": 1}


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
