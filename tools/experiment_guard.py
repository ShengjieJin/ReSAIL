#!/usr/bin/env python3
"""Run a command behind lightweight Docker experiment safety checks.

The guard runs on the host.  It keeps monitoring out of the training process,
so successful runs pay only a few cgroup/proc reads per interval.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import asdict, dataclass
import datetime as dt
import fcntl
import json
import math
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time
from typing import Any, Sequence


SAFETY_EXIT_CODE = 86
_ACTIVE_WORKLOAD = re.compile(
    r"(?:ray::|sglang|(?:^|\s)train(?:_async)?\.py(?:\s|$)|"
    r"tools/(?:convert|export)\S*(?:hf|checkpoint)\S*\.py)",
    re.IGNORECASE,
)
_PAPER_GPU_WORKLOAD = re.compile(
    r"(?:^|[\s/])(?:exp\.paper\.|exp/paper/)"
    r"(?:text(?:\.py)?\s+(?:run|prepare-c1)(?:\s|$)|"
    r"(?:text_guidance|text_epd_alfworld|text_epd_textcraft)(?:\.py)?(?:\s|$))",
    re.IGNORECASE,
)
_PAPER_SHELL_WORKLOAD = re.compile(
    r"(?:^|[\s/])(?:scripts/experiments/)?"
    r"(?:(?:text_cycles|text_guidance|text_epd_alfworld|text_epd_textcraft)\.sh(?:\s|$)|"
    r"(?:scripts/prepare/)?(?:convert_text_checkpoint|text_c1)\.sh(?:\s|$))",
    re.IGNORECASE,
)
_RAY_CONTROL_PROCESSES = {"ray::IDLE", "ray::DashboardAgent", "ray::RuntimeEnvAgent"}


@dataclass(frozen=True)
class GuardConfig:
    container: str
    audit_log: Path
    lock_file: Path
    expected_gpus: int = 8
    expected_cpu_limit: float = 160.0
    expected_pids_limit: int = 16384
    warn_pids: int = 12000
    degraded_pids: int = 13500
    stop_pids: int = 15000
    min_available_memory_bytes: int = 128 * 1024**3
    stop_load: float = 176.0
    load_strikes: int = 4
    monitor_error_strikes: int = 2
    interval_seconds: float = 10.0
    restart_zombies: int = 1000
    require_idle_gpus: bool = True


@dataclass(frozen=True)
class ResourceSample:
    container_running: bool
    pids_current: int
    pids_limit: int | None
    cpu_limit: float | None
    available_memory_bytes: int
    load_1m: float
    host_gpu_count: int | None
    container_gpu_count: int | None


def _timestamp() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def _read_int_or_max(path: Path) -> int | None:
    value = path.read_text(encoding="utf-8").strip()
    return None if value == "max" else int(value)


def _read_cpu_limit(path: Path) -> float | None:
    quota, period = path.read_text(encoding="utf-8").split()
    return None if quota == "max" else int(quota) / int(period)


def _available_memory_bytes(meminfo: Path = Path("/proc/meminfo")) -> int:
    for line in meminfo.read_text(encoding="utf-8").splitlines():
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) * 1024
    raise RuntimeError("MemAvailable is missing from /proc/meminfo")


def evaluate_sample(sample: ResourceSample, config: GuardConfig, load_count: int) -> tuple[list[str], int]:
    violations: list[str] = []
    if not sample.container_running:
        violations.append("container_stopped")
    if sample.pids_limit != config.expected_pids_limit:
        violations.append(f"pids_limit={sample.pids_limit},expected={config.expected_pids_limit}")
    if sample.cpu_limit is None or abs(sample.cpu_limit - config.expected_cpu_limit) > 1e-6:
        violations.append(f"cpu_limit={sample.cpu_limit},expected={config.expected_cpu_limit}")
    if sample.pids_current >= config.stop_pids:
        violations.append(f"pids_current={sample.pids_current},stop={config.stop_pids}")
    if sample.available_memory_bytes < config.min_available_memory_bytes:
        violations.append(
            f"available_memory={sample.available_memory_bytes},minimum={config.min_available_memory_bytes}"
        )
    if sample.host_gpu_count != config.expected_gpus:
        violations.append(f"host_gpu_count={sample.host_gpu_count},expected={config.expected_gpus}")
    if sample.container_gpu_count != config.expected_gpus:
        violations.append(f"container_gpu_count={sample.container_gpu_count},expected={config.expected_gpus}")
    load_count = load_count + 1 if sample.load_1m >= config.stop_load else 0
    if load_count >= config.load_strikes:
        violations.append(f"load_1m={sample.load_1m},strikes={load_count}")
    return violations, load_count


def pid_health_state(sample: ResourceSample, config: GuardConfig) -> str:
    if sample.pids_current >= config.stop_pids:
        return "stop"
    if sample.pids_current >= config.degraded_pids:
        return "degraded"
    if sample.pids_current >= config.warn_pids:
        return "warning"
    return "healthy"


def process_attribution(process_audit: str) -> dict[str, Any]:
    categories = {
        "ray_core_dashboard": {"processes": 0, "threads": 0},
        "ray_idle_workers": {"processes": 0, "threads": 0},
        "megatron_actors": {"processes": 0, "threads": 0},
        "sglang_engines_schedulers": {"processes": 0, "threads": 0},
        "wandb_processes": {"processes": 0, "threads": 0},
        "alfworld_workers": {"processes": 0, "threads": 0},
        "other": {"processes": 0, "threads": 0},
    }
    total_processes = 0
    total_threads = 0
    for line in process_audit.splitlines():
        fields = line.strip().split(maxsplit=5)
        if len(fields) < 6:
            continue
        try:
            threads = int(fields[2])
        except ValueError:
            continue
        total_processes += 1
        total_threads += threads
        command = f"{fields[4]} {fields[5]}".lower()
        comm = fields[4].lower()
        if comm == "alfworld-env" or "alfworld-env" in command:
            category = "alfworld_workers"
        elif "ray::idle" in command:
            category = "ray_idle_workers"
        elif "megatron" in command:
            category = "megatron_actors"
        elif "sglang" in command or "scheduler" in command and "launch_server" in command:
            category = "sglang_engines_schedulers"
        elif "wandb" in command:
            category = "wandb_processes"
        elif any(
            marker in command
            for marker in ("gcs_server", "raylet", "dashboard", "runtime_env_agent", "log_monitor")
        ):
            category = "ray_core_dashboard"
        else:
            category = "other"
        categories[category]["processes"] += 1
        categories[category]["threads"] += threads
    return {"total_processes": total_processes, "total_threads": total_threads, "categories": categories}


def active_workload_lines(process_audit: str) -> list[str]:
    workloads = []
    for line in process_audit.splitlines():
        fields = line.strip().split(maxsplit=4)
        if len(fields) < 5 or fields[2].startswith("Z"):
            continue
        command = f"{fields[3]} {fields[4]}"
        if fields[4].split()[0] in _RAY_CONTROL_PROCESSES:
            continue
        shell_process = fields[3].rsplit("/", 1)[-1] in {"bash", "sh", "dash"}
        if (_ACTIVE_WORKLOAD.search(command) or _PAPER_GPU_WORKLOAD.search(command)
                or (shell_process and _PAPER_SHELL_WORKLOAD.search(command))):
            workloads.append(line)
    return workloads


def _signal_process_group(process: subprocess.Popen[Any], signum: signal.Signals) -> None:
    if process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signum)
    except ProcessLookupError:
        pass


@contextmanager
def exclusive_lock(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_CREAT | os.O_RDWR | os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags, 0o600)
    try:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(f"another experiment guard holds {path}") from exc
        os.ftruncate(descriptor, 0)
        os.write(descriptor, f"pid={os.getpid()}\n".encode())
        yield
    finally:
        os.close(descriptor)


class DockerExperimentGuard:
    def __init__(self, config: GuardConfig):
        self.config = config
        self._container_id = ""
        self._cgroup = Path()

    @property
    def degraded_marker(self) -> Path:
        return self.config.audit_log.with_suffix(".degraded.json")

    def _run(
        self, argv: Sequence[str], *, check: bool = True, timeout: float = 30.0
    ) -> subprocess.CompletedProcess[str]:
        result = subprocess.run(argv, capture_output=True, text=True, timeout=timeout)
        if check and result.returncode != 0:
            detail = (result.stderr or result.stdout).strip()
            raise RuntimeError(f"command failed ({' '.join(argv[:3])}): {detail}")
        return result

    def _inspect(self) -> dict[str, Any]:
        result = self._run(["docker", "inspect", self.config.container])
        payload = json.loads(result.stdout)
        if not isinstance(payload, list) or len(payload) != 1:
            raise RuntimeError(f"unexpected docker inspect result for {self.config.container}")
        return payload[0]

    def _refresh_cgroup(self, inspect: dict[str, Any]) -> None:
        pid = int(inspect["State"]["Pid"])
        if pid <= 0:
            raise RuntimeError(f"container {self.config.container} has no init pid")
        cgroup_lines = Path(f"/proc/{pid}/cgroup").read_text(encoding="utf-8").splitlines()
        unified = [line.split("::", 1)[1] for line in cgroup_lines if "::" in line]
        if len(unified) != 1:
            raise RuntimeError(f"container {self.config.container} does not expose one cgroup-v2 path")
        root = Path("/sys/fs/cgroup").resolve()
        candidate = (root / unified[0].lstrip("/")).resolve()
        if root not in candidate.parents:
            raise RuntimeError(f"unsafe container cgroup path: {candidate}")
        self._container_id = str(inspect["Id"])
        self._cgroup = candidate

    def _workloads(self) -> list[str]:
        try:
            result = self._run(
                ["docker", "exec", self.config.container, "ps", "-eo", "pid=,ppid=,stat=,comm=,args="],
                check=False,
            )
        except (OSError, subprocess.SubprocessError) as exc:
            return [f"process audit failed: {type(exc).__name__}: {exc}"]
        if result.returncode != 0:
            return [f"process audit failed: {(result.stderr or result.stdout).strip()}"]
        return active_workload_lines(result.stdout)

    def _zombie_count(self) -> int:
        result = self._run(
            ["docker", "exec", self.config.container, "ps", "-eo", "stat="], check=False
        )
        if result.returncode != 0:
            raise RuntimeError(f"cannot count container zombies: {(result.stderr or result.stdout).strip()}")
        return sum(1 for line in result.stdout.splitlines() if line.strip().startswith("Z"))

    def _process_attribution(self) -> dict[str, Any]:
        try:
            result = self._run(
                [
                    "docker",
                    "exec",
                    self.config.container,
                    "ps",
                    "-eo",
                    "pid=,ppid=,nlwp=,stat=,comm=,args=",
                ],
                check=False,
            )
        except (OSError, subprocess.SubprocessError) as exc:
            return {"error": f"{type(exc).__name__}: {exc}"}
        if result.returncode != 0:
            return {"error": (result.stderr or result.stdout).strip()}
        return process_attribution(result.stdout)

    def _gpu_probe(self) -> None:
        script = (
            "import json,torch; "
            "print(json.dumps({'available':torch.cuda.is_available(),'count':torch.cuda.device_count()}))"
        )
        result = self._run(["docker", "exec", self.config.container, "python3", "-c", script], timeout=45.0)
        lines = [line for line in result.stdout.splitlines() if line.strip()]
        probe = json.loads(lines[-1]) if lines else {}
        if probe != {"available": True, "count": self.config.expected_gpus}:
            raise RuntimeError(f"container CUDA probe failed: {probe}")
        if self._container_gpu_count() != self.config.expected_gpus:
            raise RuntimeError("container nvidia-smi GPU count disagrees with the required topology")
        if self._host_gpu_count() != self.config.expected_gpus:
            raise RuntimeError("host nvidia-smi GPU count disagrees with the required topology")
        if self.config.require_idle_gpus:
            pids = self._host_gpu_compute_pids()
            if pids:
                raise RuntimeError(f"GPU compute processes already exist before launch: {pids[:8]}")

    def _container_gpu_count(self) -> int:
        visible = self._run(
            [
                "docker",
                "exec",
                self.config.container,
                "nvidia-smi",
                "--query-gpu=index",
                "--format=csv,noheader",
            ]
        )
        return len([line for line in visible.stdout.splitlines() if line.strip()])

    def _host_gpu_count(self) -> int:
        visible = self._run(["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"])
        return len([line for line in visible.stdout.splitlines() if line.strip()])

    def _gpu_probe_with_retry(self) -> None:
        last_error: Exception | None = None
        for attempt in range(1, 4):
            try:
                self._gpu_probe()
                return
            except (OSError, RuntimeError, ValueError, json.JSONDecodeError, subprocess.SubprocessError) as exc:
                last_error = exc
                self.append_audit("gpu_probe_retry", attempt=attempt, error=str(exc))
                if attempt < 3:
                    time.sleep(5.0)
        assert last_error is not None
        raise last_error

    def _host_gpu_compute_pids(self) -> list[int]:
        result = self._run(
            ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader,nounits"],
            check=False,
        )
        if result.returncode != 0:
            raise RuntimeError(f"cannot inspect host GPU processes: {(result.stderr or result.stdout).strip()}")
        return [int(line.strip()) for line in result.stdout.splitlines() if line.strip().isdigit()]

    def cleanup_residue(self) -> tuple[list[str], list[int]]:
        return self._workloads(), self._host_gpu_compute_pids()

    def sample(self) -> ResourceSample:
        inspect = self._inspect()
        running = bool(inspect.get("State", {}).get("Running"))
        if running and (not self._cgroup or str(inspect.get("Id")) != self._container_id):
            self._refresh_cgroup(inspect)
        if not running:
            return ResourceSample(False, -1, None, None, _available_memory_bytes(), os.getloadavg()[0], None, None)
        return ResourceSample(
            container_running=True,
            pids_current=int((self._cgroup / "pids.current").read_text(encoding="utf-8")),
            pids_limit=_read_int_or_max(self._cgroup / "pids.max"),
            cpu_limit=_read_cpu_limit(self._cgroup / "cpu.max"),
            available_memory_bytes=_available_memory_bytes(),
            load_1m=os.getloadavg()[0],
            host_gpu_count=self._host_gpu_count(),
            container_gpu_count=self._container_gpu_count(),
        )

    def append_audit(self, event: str, **fields: Any) -> None:
        self.config.audit_log.parent.mkdir(parents=True, exist_ok=True)
        self.config.audit_log.touch(mode=0o600, exist_ok=True)
        self.config.audit_log.chmod(0o600)
        payload = {"timestamp": _timestamp(), "event": event, **fields}
        with self.config.audit_log.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, sort_keys=True) + "\n")

    def preflight(self) -> ResourceSample:
        inspect = self._inspect()
        if not bool(inspect.get("State", {}).get("Running")):
            raise RuntimeError(f"container {self.config.container} is not running")
        self._refresh_cgroup(inspect)
        workloads = self._workloads()
        if workloads:
            raise RuntimeError(f"active Ray/SGLang workloads already exist: {workloads[:3]}")
        if self.degraded_marker.is_file():
            raise RuntimeError(f"unresolved degraded PID marker blocks launch: {self.degraded_marker}")
        zombies = self._zombie_count()
        if self.config.restart_zombies > 0 and zombies >= self.config.restart_zombies:
            self.append_audit("container_restart", reason="zombie_threshold", zombie_count=zombies)
            self._run(["docker", "restart", "--time", "30", self.config.container], timeout=60.0)
            inspect = self._inspect()
            self._refresh_cgroup(inspect)
            zombies = self._zombie_count()
        sample = self.sample()
        violations, _ = evaluate_sample(sample, self.config, 0)
        if violations:
            raise RuntimeError(f"experiment resource preflight failed: {violations}")
        if pid_health_state(sample, self.config) == "degraded":
            attribution = self._process_attribution()
            self.degraded_marker.write_text(
                json.dumps(
                    {
                        "status": "degraded",
                        "pids_current": sample.pids_current,
                        "process_attribution": attribution,
                        "timestamp": _timestamp(),
                    },
                    indent=2,
                    sort_keys=True,
                )
                + "\n",
                encoding="utf-8",
            )
            raise RuntimeError("PID state is degraded; refusing launch pending diagnosis")
        self._gpu_probe_with_retry()
        self.append_audit("preflight_passed", resources=asdict(sample), zombie_count=zombies)
        return sample

    def stop_workloads(self, reason: str) -> None:
        workloads = self._workloads()
        snapshot = self.config.audit_log.with_suffix(".processes.txt")
        try:
            snapshot.parent.mkdir(parents=True, exist_ok=True)
            snapshot.write_text("\n".join(workloads) + ("\n" if workloads else ""), encoding="utf-8")
            self.append_audit(
                "cleanup_started", reason=reason, workload_count=len(workloads), snapshot=str(snapshot)
            )
        except OSError:
            # Cleanup must still run if the audit filesystem becomes unavailable.
            pass
        try:
            result = self._run(
                ["docker", "exec", self.config.container, "ray", "stop", "--force"],
                check=False,
                timeout=60.0,
            )
            ray_stop_returncode: int | None = result.returncode
        except (OSError, subprocess.SubprocessError):
            ray_stop_returncode = None
        workloads_after: list[str] = []
        gpu_pids_after: list[int] = []
        for _ in range(10):
            try:
                workloads_after, gpu_pids_after = self.cleanup_residue()
            except (OSError, RuntimeError, subprocess.SubprocessError) as exc:
                workloads_after = [f"cleanup probe failed: {type(exc).__name__}: {exc}"]
                gpu_pids_after = [-1]
            if not workloads_after and not gpu_pids_after:
                break
            time.sleep(1.0)
        if workloads_after or gpu_pids_after:
            try:
                self.append_audit(
                    "cleanup_escalated",
                    ray_stop_returncode=ray_stop_returncode,
                    workload_count=len(workloads_after),
                    gpu_pids=gpu_pids_after,
                )
            except OSError:
                pass
            self._run(["docker", "restart", "--time", "30", self.config.container], timeout=60.0)
            workloads_after, gpu_pids_after = self.cleanup_residue()
        if workloads_after or gpu_pids_after:
            raise RuntimeError(
                "experiment cleanup did not reach a safe terminal state: "
                f"workloads={workloads_after[:3]},gpu_pids={gpu_pids_after[:8]}"
            )
        try:
            self.append_audit(
                "cleanup_finished",
                ray_stop_returncode=ray_stop_returncode,
                workload_count=0,
                gpu_pids=[],
            )
        except OSError:
            pass


def _run_guarded_locked(config: GuardConfig, command: Sequence[str]) -> int:
    if not command:
        raise ValueError("a command is required after --")
    guard = DockerExperimentGuard(config)
    guard.preflight()
    signal_reason: list[str] = []
    process: subprocess.Popen[Any] | None = None
    safe_terminal = False
    cleanup_reason = "guard_exception"

    def request_stop(signum: int, _frame: Any) -> None:
        signal_reason[:] = [signal.Signals(signum).name]

    old_handlers = {signum: signal.signal(signum, request_stop) for signum in (signal.SIGINT, signal.SIGTERM)}
    load_count = 0
    monitor_error_count = 0
    previous_pid_state = "healthy"
    next_sample = time.monotonic()
    try:
        process = subprocess.Popen(list(command), start_new_session=True)
        guard.append_audit("command_started", pid=process.pid, executable=command[0], argument_count=len(command))
        while True:
            returncode = process.poll()
            if returncode is not None:
                guard.append_audit("command_finished", returncode=returncode)
                remaining, gpu_pids = guard.cleanup_residue()
                if remaining or gpu_pids:
                    cleanup_reason = "postrun_workload_leak"
                    guard.append_audit(
                        "postrun_workload_leak", workload_count=len(remaining), gpu_pids=gpu_pids
                    )
                    return SAFETY_EXIT_CODE
                if returncode != 0:
                    cleanup_reason = f"command_returncode:{returncode}"
                    return returncode
                safe_terminal = True
                return returncode
            if signal_reason:
                cleanup_reason = f"signal:{signal_reason[0]}"
                guard.append_audit("stop_requested", reason=cleanup_reason)
                return 128 + signal.Signals[signal_reason[0]].value
            now = time.monotonic()
            if now >= next_sample:
                try:
                    sample = guard.sample()
                except (
                    OSError,
                    RuntimeError,
                    ValueError,
                    json.JSONDecodeError,
                    subprocess.SubprocessError,
                ) as exc:
                    monitor_error_count += 1
                    reason = f"monitor_error:{type(exc).__name__}:{exc}"
                    guard.append_audit(
                        "monitor_retry",
                        attempt=monitor_error_count,
                        limit=config.monitor_error_strikes,
                        reason=reason,
                    )
                    if monitor_error_count >= config.monitor_error_strikes:
                        cleanup_reason = reason
                        guard.append_audit("safety_trip", reason=reason)
                        return SAFETY_EXIT_CODE
                    next_sample = now + config.interval_seconds
                    continue
                monitor_error_count = 0
                violations, load_count = evaluate_sample(sample, config, load_count)
                current_pid_state = pid_health_state(sample, config)
                attribution = guard._process_attribution()
                guard.append_audit(
                    "resource_sample",
                    resources=asdict(sample),
                    load_strikes=load_count,
                    pid_health=current_pid_state,
                    process_attribution=attribution,
                    violations=violations,
                )
                if current_pid_state != previous_pid_state:
                    guard.append_audit(
                        "pid_health_transition",
                        previous=previous_pid_state,
                        current=current_pid_state,
                        pids_current=sample.pids_current,
                        process_attribution=attribution,
                    )
                    previous_pid_state = current_pid_state
                if current_pid_state == "degraded" and not guard.degraded_marker.exists():
                    guard.degraded_marker.write_text(
                        json.dumps(
                            {
                                "status": "degraded",
                                "pids_current": sample.pids_current,
                                "process_attribution": attribution,
                                "timestamp": _timestamp(),
                            },
                            indent=2,
                            sort_keys=True,
                        )
                        + "\n",
                        encoding="utf-8",
                    )
                if violations:
                    reason = ";".join(violations)
                    cleanup_reason = reason
                    guard.append_audit("safety_trip", reason=reason)
                    return SAFETY_EXIT_CODE
                next_sample = now + config.interval_seconds
            time.sleep(min(1.0, max(0.05, next_sample - time.monotonic())))
    finally:
        try:
            if process is not None and not safe_terminal:
                _signal_process_group(process, signal.SIGTERM)
                try:
                    process.wait(timeout=30.0)
                except subprocess.TimeoutExpired:
                    _signal_process_group(process, signal.SIGKILL)
                    process.wait()
                guard.stop_workloads(cleanup_reason)
        finally:
            for signum, handler in old_handlers.items():
                signal.signal(signum, handler)


def run_guarded(config: GuardConfig, command: Sequence[str]) -> int:
    with exclusive_lock(config.lock_file):
        return _run_guarded_locked(config, command)


def _parse_args(argv: Sequence[str] | None = None) -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--container", required=True)
    parser.add_argument("--audit-log", required=True, type=Path)
    parser.add_argument("--lock-file", required=True, type=Path)
    parser.add_argument("--expected-gpus", type=int, default=8)
    parser.add_argument("--expected-cpu-limit", type=float, default=160.0)
    parser.add_argument("--expected-pids-limit", type=int, default=16384)
    parser.add_argument("--warn-pids", type=int, default=12000)
    parser.add_argument("--degraded-pids", type=int, default=13500)
    parser.add_argument("--stop-pids", type=int, default=15000)
    parser.add_argument("--min-available-memory-gib", type=float, default=128.0)
    parser.add_argument("--stop-load", type=float, default=176.0)
    parser.add_argument("--load-strikes", type=int, default=4)
    parser.add_argument("--monitor-error-strikes", type=int, default=2)
    parser.add_argument("--interval-seconds", type=float, default=10.0)
    parser.add_argument("--restart-zombies", type=int, default=1000)
    parser.add_argument("--allow-active-gpus", action="store_true")
    parser.add_argument("--preflight-only", action="store_true")
    raw = list(sys.argv[1:] if argv is None else argv)
    if "--" in raw:
        delimiter = raw.index("--")
        guard_args, command = raw[:delimiter], raw[delimiter + 1 :]
    else:
        guard_args, command = raw, []
    args = parser.parse_args(guard_args)
    for name in (
        "expected_gpus",
        "expected_pids_limit",
        "warn_pids",
        "degraded_pids",
        "stop_pids",
        "load_strikes",
        "monitor_error_strikes",
    ):
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if args.interval_seconds <= 0 or args.min_available_memory_gib <= 0 or args.expected_cpu_limit <= 0:
        parser.error("CPU, memory, and interval values must be positive")
    for name in ("interval_seconds", "min_available_memory_gib", "expected_cpu_limit", "stop_load"):
        if not math.isfinite(getattr(args, name)):
            parser.error(f"--{name.replace('_', '-')} must be finite")
    if args.stop_load <= 0:
        parser.error("--stop-load must be positive")
    if args.restart_zombies < 0:
        parser.error("--restart-zombies must be non-negative")
    if not args.warn_pids < args.degraded_pids < args.stop_pids < args.expected_pids_limit:
        parser.error("PID thresholds must satisfy warn < degraded < stop < limit")
    if not args.preflight_only and not command:
        parser.error("a command is required after --")
    return args, command


def main(argv: Sequence[str] | None = None) -> int:
    args, command = _parse_args(argv)
    config = GuardConfig(
        container=args.container,
        audit_log=args.audit_log.resolve(),
        lock_file=args.lock_file.resolve(),
        expected_gpus=args.expected_gpus,
        expected_cpu_limit=args.expected_cpu_limit,
        expected_pids_limit=args.expected_pids_limit,
        warn_pids=args.warn_pids,
        degraded_pids=args.degraded_pids,
        stop_pids=args.stop_pids,
        min_available_memory_bytes=int(args.min_available_memory_gib * 1024**3),
        stop_load=args.stop_load,
        load_strikes=args.load_strikes,
        monitor_error_strikes=args.monitor_error_strikes,
        interval_seconds=args.interval_seconds,
        restart_zombies=args.restart_zombies,
        require_idle_gpus=not args.allow_active_gpus,
    )
    guard = DockerExperimentGuard(config)
    if args.preflight_only:
        with exclusive_lock(config.lock_file):
            guard.preflight()
        return 0
    return run_guarded(config, command)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, RuntimeError, ValueError, json.JSONDecodeError) as exc:
        print(f"experiment guard: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
