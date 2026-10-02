from __future__ import annotations

import asyncio
import atexit
import contextlib
import multiprocessing as mp
import os
import queue
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

DEFAULT_CACHE_DIR = Path(".cache/alfworld")
TEXTWORLD_SPLITS = {
    "train": "train",
    "eval_in_distribution": "valid_seen",
    "eval_out_of_distribution": "valid_unseen",
    "valid_seen": "valid_seen",
    "valid_unseen": "valid_unseen",
}
COMPACT_INFO_KEYS = ("admissible_commands", "admissible_actions", "won", "extra.gamefile", "gamefile")


def _assert_safe_fast_environment_process_allowed() -> None:
    if os.environ.get("SLIME_RUNTIME_PROFILE") != "safe-fast":
        return
    mode = os.environ.get("SLIME_EXECUTION_MODE", "")
    requires_live = os.environ.get("SLIME_REQUIRES_LIVE_ENVIRONMENT", "").lower() == "true"
    if not requires_live or mode not in {"live_collection", "live_eval"}:
        raise RuntimeError("safe-fast blocked an ALFWorld environment process outside a live phase")


class AlfWorldDependencyError(RuntimeError):
    pass


@dataclass
class AlfWorldResetResult:
    observation: str
    info: dict[str, Any]
    reset_seconds: float
    worker_id: str
    reused_worker: bool
    worker_created: bool


def resolve_alfworld_cache_dir(cache_dir: str | os.PathLike[str] | None = None) -> Path:
    raw = cache_dir or os.environ.get("ALFWORLD_DATA") or DEFAULT_CACHE_DIR
    return Path(raw).expanduser().resolve()


def ensure_alfworld_cache(cache_dir: str | os.PathLike[str] | None = None, split: str = "train") -> Path:
    cache = resolve_alfworld_cache_dir(cache_dir)
    split_name = TEXTWORLD_SPLITS.get(split, split)
    required = [
        cache / "json_2.1.1" / split_name,
        cache / "logic" / "alfred.pddl",
        cache / "logic" / "alfred.twl2",
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise AlfWorldDependencyError(
            "AlFWorld cache is incomplete. Run "
            f"`ALFWORLD_DATA={cache} alfworld-download -f` first. Missing: {missing}"
        )
    return cache


def check_alfworld_import() -> str:
    env_cls = _alfworld_env_cls()
    return env_cls.__name__


def default_textworld_config(cache_dir: str | os.PathLike[str], max_episode_steps: int = 30) -> dict[str, Any]:
    cache = str(resolve_alfworld_cache_dir(cache_dir))
    return {
        "dataset": {
            "data_path": f"{cache}/json_2.1.1/train",
            "eval_id_data_path": f"{cache}/json_2.1.1/valid_seen",
            "eval_ood_data_path": f"{cache}/json_2.1.1/valid_unseen",
            "num_train_games": -1,
            "num_eval_games": -1,
        },
        "logic": {
            "domain": f"{cache}/logic/alfred.pddl",
            "grammar": f"{cache}/logic/alfred.twl2",
        },
        "env": {
            "type": "AlfredTWEnv",
            "domain_randomization": False,
            "task_types": [1, 2, 3, 4, 5, 6],
            "expert_timeout_steps": 150,
            "expert_type": "handcoded",
            "goal_desc_human_anns_prob": 0.0,
        },
        "general": {
            "random_seed": 42,
            "use_cuda": False,
            "training_method": "dqn",
            "task": "alfred",
        },
        "dagger": {"training": {"max_nb_steps_per_episode": max_episode_steps}},
        "rl": {"training": {"max_nb_steps_per_episode": max_episode_steps}},
    }


def build_alfworld_env(**kwargs: Any) -> "SingleAlfWorldEnv":
    return SingleAlfWorldEnv(**kwargs)


class SingleAlfWorldEnv:
    def __init__(
        self,
        *,
        seed: int = 0,
        split: str = "train",
        cache_dir: str | os.PathLike[str] | None = None,
        config_path: str | os.PathLike[str] | None = None,
        max_episode_steps: int = 30,
        suppress_output: bool = True,
    ) -> None:
        _assert_safe_fast_environment_process_allowed()
        self.seed = seed
        self.split = split
        self.cache_dir = ensure_alfworld_cache(cache_dir, split=split)
        os.environ["ALFWORLD_DATA"] = str(self.cache_dir)
        config = (
            _load_config(config_path) if config_path else default_textworld_config(self.cache_dir, max_episode_steps)
        )
        train_eval = _split_to_train_eval(split)
        with _maybe_suppress_stdout_stderr(suppress_output):
            base_env = _alfworld_env_cls()(config, train_eval=train_eval)
            if getattr(base_env, "num_games", 0) <= 0:
                raise AlfWorldDependencyError(f"No AlfWorld games found for split={split!r} in {self.cache_dir}")
            self._env = base_env.init_env(batch_size=1)
        self._gamefiles = tuple(str(path) for path in self._env.gamefiles)
        self._allowed_gamefiles = frozenset(self._gamefiles)
        if hasattr(self._env, "seed"):
            self._env.seed(seed)
        self.worker_id = f"single-{os.getpid()}"
        self._reset_count = 0

    def reset(self, seed: int | None = None, *, gamefile: str | None = None) -> AlfWorldResetResult:
        if seed is not None and seed != self.seed:
            self.seed = seed
        if gamefile is not None:
            gamefile = str(gamefile)
            if gamefile not in self._allowed_gamefiles:
                raise ValueError(f"Requested ALFWorld gamefile is not in split={self.split!r}: {gamefile}")
            self._env.gamefiles = [gamefile]
        else:
            self._env.gamefiles = list(self._gamefiles)
        if hasattr(self._env, "seed"):
            self._env.seed(self.seed)
        start = time.monotonic()
        obs, info = self._env.reset()
        self._reset_count += 1
        return AlfWorldResetResult(
            observation=_first(obs),
            info=_compact_info(info),
            reset_seconds=time.monotonic() - start,
            worker_id=self.worker_id,
            reused_worker=self._reset_count > 1,
            worker_created=self._reset_count == 1,
        )

    def step(self, action: str) -> tuple[str, float, bool, dict[str, Any]]:
        obs, scores, dones, info = self._env.step([action])
        return _first(obs), float(_first(scores)), bool(_first(dones)), _compact_info(info)

    def close(self) -> None:
        close = getattr(self._env, "close", None)
        if close is not None:
            close()


class ProcessAlfWorldEnvWorker:
    def __init__(
        self,
        *,
        worker_index: int,
        seed: int,
        split: str,
        cache_dir: str | os.PathLike[str] | None,
        max_episode_steps: int,
        suppress_output: bool = True,
    ) -> None:
        _assert_safe_fast_environment_process_allowed()
        ctx = mp.get_context("spawn")
        self.worker_id = f"process-{os.getpid()}-{worker_index}"
        self._request_queue = ctx.Queue()
        self._response_queue = ctx.Queue()
        self._process = ctx.Process(
            target=_process_worker_main,
            kwargs={
                "request_queue": self._request_queue,
                "response_queue": self._response_queue,
                "worker_id": self.worker_id,
                "seed": seed,
                "split": split,
                "cache_dir": str(cache_dir) if cache_dir is not None else None,
                "max_episode_steps": max_episode_steps,
                "suppress_output": suppress_output,
            },
            daemon=True,
        )
        self._process.start()
        self._dirty = False
        self._closed = False
        self._reset_count = 0
        self.pooled = True

    @property
    def dirty(self) -> bool:
        return self._dirty or self._closed or not self._process.is_alive()

    def reset(self, seed: int | None = None, *, gamefile: str | None = None) -> AlfWorldResetResult:
        data = self._request("reset", {"seed": seed, "gamefile": gamefile})
        self._reset_count += 1
        return AlfWorldResetResult(
            observation=data["observation"],
            info=data["info"],
            reset_seconds=data["reset_seconds"],
            worker_id=data["worker_id"],
            reused_worker=self._reset_count > 1,
            worker_created=self._reset_count == 1,
        )

    def step(self, action: str) -> tuple[str, float, bool, dict[str, Any]]:
        data = self._request("step", {"action": action})
        return data["observation"], float(data["reward"]), bool(data["done"]), dict(data["info"])

    def ping(self) -> dict[str, Any]:
        return self._request("ping", {})

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        with contextlib.suppress(Exception):
            self._request_queue.put(("close", {}))
        self._process.join(timeout=5)
        if self._process.is_alive():
            self._process.terminate()
            self._process.join(timeout=5)

    def _request(self, command: str, payload: dict[str, Any]) -> dict[str, Any]:
        if self._closed:
            raise RuntimeError(f"AlfWorld worker {self.worker_id} is closed")
        self._request_queue.put((command, payload))
        try:
            status, data = self._response_queue.get(timeout=300)
        except queue.Empty as exc:
            self._dirty = True
            raise TimeoutError(f"AlfWorld worker {self.worker_id} timed out on {command}") from exc
        if status == "error":
            self._dirty = True
            raise RuntimeError(f"AlfWorld worker {self.worker_id} failed on {command}: {data}")
        return data


class AlfWorldEnvPool:
    def __init__(
        self,
        *,
        pool_size: int,
        split: str,
        cache_dir: str | os.PathLike[str] | None,
        max_episode_steps: int,
        prewarm_batch_size: int = 32,
        suppress_output: bool = True,
    ) -> None:
        _assert_safe_fast_environment_process_allowed()
        self.pool_size = max(1, int(pool_size))
        self.prewarm_batch_size = max(1, int(prewarm_batch_size))
        self.split = split
        self.cache_dir = cache_dir
        self.max_episode_steps = max_episode_steps
        self.suppress_output = suppress_output
        self._workers: list[ProcessAlfWorldEnvWorker] = []
        self._idle_workers: asyncio.Queue[ProcessAlfWorldEnvWorker | None] = asyncio.Queue()
        self._lock = asyncio.Lock()
        self._prewarmed = False

    async def acquire_async(self, *, seed: int) -> ProcessAlfWorldEnvWorker:
        async with self._lock:
            if len(self._workers) < self.pool_size:
                worker = await asyncio.to_thread(
                    ProcessAlfWorldEnvWorker,
                    worker_index=len(self._workers),
                    seed=seed,
                    split=self.split,
                    cache_dir=self.cache_dir,
                    max_episode_steps=self.max_episode_steps,
                    suppress_output=self.suppress_output,
                )
                self._workers.append(worker)
                return worker
        while True:
            worker = await self._idle_workers.get()
            if worker is None or worker.dirty:
                if worker is not None:
                    self._discard(worker)
                async with self._lock:
                    if len(self._workers) < self.pool_size:
                        replacement = await asyncio.to_thread(
                            ProcessAlfWorldEnvWorker,
                            worker_index=len(self._workers),
                            seed=seed,
                            split=self.split,
                            cache_dir=self.cache_dir,
                            max_episode_steps=self.max_episode_steps,
                            suppress_output=self.suppress_output,
                        )
                        self._workers.append(replacement)
                        return replacement
                continue
            return worker

    async def prewarm_async(self, *, seed: int) -> None:
        async with self._lock:
            if self._prewarmed:
                return
            new_workers = []
            try:
                while len(self._workers) < self.pool_size:
                    batch_start = len(self._workers)
                    batch_stop = min(self.pool_size, batch_start + self.prewarm_batch_size)
                    created = await asyncio.gather(
                        *(
                            asyncio.to_thread(
                                ProcessAlfWorldEnvWorker,
                                worker_index=worker_index,
                                seed=seed,
                                split=self.split,
                                cache_dir=self.cache_dir,
                                max_episode_steps=self.max_episode_steps,
                                suppress_output=self.suppress_output,
                            )
                            for worker_index in range(batch_start, batch_stop)
                        )
                    )
                    self._workers.extend(created)
                    new_workers.extend(created)
                if new_workers:
                    ping_results = await asyncio.gather(
                        *(asyncio.to_thread(worker.ping) for worker in new_workers),
                        return_exceptions=True,
                    )
                    for result in ping_results:
                        if isinstance(result, Exception):
                            raise result
                    for worker in new_workers:
                        self._idle_workers.put_nowait(worker)
                self._prewarmed = True
            except Exception:
                for worker in new_workers:
                    self._discard(worker)
                raise

    def release(self, worker: ProcessAlfWorldEnvWorker) -> None:
        if worker.dirty:
            self._discard(worker)
            self._idle_workers.put_nowait(None)
            return
        self._idle_workers.put_nowait(worker)

    def close(self) -> None:
        for worker in list(self._workers):
            worker.close()
        self._workers.clear()

    def _discard(self, worker: ProcessAlfWorldEnvWorker) -> None:
        worker.close()
        if worker in self._workers:
            self._workers.remove(worker)


def get_or_create_alfworld_env_pool(
    owner: Any,
    *,
    split: str,
    cache_dir: str | os.PathLike[str] | None,
    max_episode_steps: int,
    pool_size: int,
    prewarm_batch_size: int = 32,
    suppress_output: bool = True,
) -> AlfWorldEnvPool:
    pools = getattr(owner, "_alfworld_env_pools", None)
    if pools is None:
        pools = {}
        setattr(owner, "_alfworld_env_pools", pools)
        atexit.register(close_alfworld_env_pools, owner)
    key = (
        split,
        str(cache_dir),
        int(max_episode_steps),
        int(pool_size),
        int(prewarm_batch_size),
        bool(suppress_output),
    )
    if key not in pools:
        pools[key] = AlfWorldEnvPool(
            pool_size=pool_size,
            split=split,
            cache_dir=cache_dir,
            max_episode_steps=max_episode_steps,
            prewarm_batch_size=prewarm_batch_size,
            suppress_output=suppress_output,
        )
    return pools[key]


def close_alfworld_env_pools(owner: Any) -> None:
    pools = getattr(owner, "_alfworld_env_pools", None)
    if not pools:
        return
    for pool in pools.values():
        pool.close()
    pools.clear()


def _process_worker_main(
    *,
    request_queue,
    response_queue,
    worker_id: str,
    seed: int,
    split: str,
    cache_dir: str | None,
    max_episode_steps: int,
    suppress_output: bool,
) -> None:
    # Apply only inside the environment subprocess and before ALFWorld imports
    # its numerical stack.  Model-serving and training actors keep their own
    # thread budgets.
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[name] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    with contextlib.suppress(OSError):
        Path("/proc/self/comm").write_text("alfworld-env\n", encoding="utf-8")
    env = None
    try:
        env = SingleAlfWorldEnv(
            seed=seed,
            split=split,
            cache_dir=cache_dir,
            max_episode_steps=max_episode_steps,
            suppress_output=suppress_output,
        )
        env.worker_id = worker_id
        while True:
            command, payload = request_queue.get()
            if command == "close":
                break
            try:
                if command == "reset":
                    result = env.reset(seed=payload.get("seed"), gamefile=payload.get("gamefile"))
                    response_queue.put(("ok", result.__dict__))
                elif command == "ping":
                    response_queue.put(("ok", {"worker_id": worker_id}))
                elif command == "step":
                    observation, reward, done, info = env.step(payload["action"])
                    response_queue.put(
                        ("ok", {"observation": observation, "reward": reward, "done": done, "info": info})
                    )
                else:
                    response_queue.put(("error", f"unknown command {command!r}"))
            except Exception as exc:
                response_queue.put(("error", repr(exc)))
    except Exception as exc:
        response_queue.put(("error", repr(exc)))
    finally:
        if env is not None:
            env.close()


def _alfworld_env_cls():
    try:
        from alfworld.agents.environment import get_environment
    except Exception as exc:
        raise AlfWorldDependencyError(
            "Cannot import `alfworld`. Install docker/tasks/alfworld.requirements.txt in the training container."
        ) from exc
    env_cls = get_environment("AlfredTWEnv")
    if env_cls.__name__ != "AlfredTWEnv":
        raise AlfWorldDependencyError(f"Unexpected AlfWorld env class: {env_cls!r}")
    return env_cls


def _load_config(config_path: str | os.PathLike[str]) -> dict[str, Any]:
    with Path(config_path).expanduser().open() as reader:
        value = yaml.safe_load(reader)
    if not isinstance(value, dict):
        raise ValueError(f"Expected ALFWorld config mapping in {config_path}")
    return value


def _split_to_train_eval(split: str) -> str:
    if split == "train":
        return "train"
    if split in {"eval_in_distribution", "valid_seen"}:
        return "eval_in_distribution"
    if split in {"eval_out_of_distribution", "valid_unseen"}:
        return "eval_out_of_distribution"
    raise ValueError(f"Unsupported ALFWorld split: {split}")


@contextlib.contextmanager
def _maybe_suppress_stdout_stderr(enabled: bool):
    if not enabled:
        yield
        return
    with Path(os.devnull).open("w") as devnull:
        original_stdout = sys.stdout
        original_stderr = sys.stderr
        sys.stdout = devnull
        sys.stderr = devnull
        try:
            yield
        finally:
            sys.stdout = original_stdout
            sys.stderr = original_stderr


def _first(value: Any) -> Any:
    if isinstance(value, (list, tuple)):
        return value[0]
    return value


def _first_info(info: dict[str, Any]) -> dict[str, Any]:
    return {key: _first(value) for key, value in info.items()}


def _compact_info(info: dict[str, Any]) -> dict[str, Any]:
    flat_info = _first_info(info)
    return {key: flat_info[key] for key in COMPACT_INFO_KEYS if key in flat_info}
