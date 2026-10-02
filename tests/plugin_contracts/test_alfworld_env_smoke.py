from __future__ import annotations

import asyncio
import os

import pytest

try:
    from ._shared import install_paths
except ImportError:
    try:
        from plugin_contracts._shared import install_paths
    except ImportError:
        from _shared import install_paths

install_paths()

NUM_GPUS = 0

from slime_plugins.agent_tasks.alfworld.envs import (
    AlfWorldDependencyError,
    AlfWorldEnvPool,
    SingleAlfWorldEnv,
    check_alfworld_import,
    ensure_alfworld_cache,
)


@pytest.mark.system
def test_alfworld_real_env_cache_smoke():
    strict = os.environ.get("ALFWORLD_STRICT_SMOKE", "0") == "1"
    cache_dir = os.environ.get("ALFWORLD_DATA", ".cache/alfworld")
    try:
        env_class_name = check_alfworld_import()
        cache = ensure_alfworld_cache(cache_dir, split="train")
    except AlfWorldDependencyError as exc:
        if strict:
            raise
        pytest.skip(str(exc))

    env = SingleAlfWorldEnv(seed=7, split="train", cache_dir=cache, max_episode_steps=32)
    try:
        reset = env.reset(seed=7)
        assert env_class_name == "AlfredTWEnv"
        assert reset.observation
        assert reset.info.get("admissible_commands")
    finally:
        env.close()


@pytest.mark.system
def test_alfworld_repeated_seed_restarts_same_game():
    strict = os.environ.get("ALFWORLD_STRICT_SMOKE", "0") == "1"
    cache_dir = os.environ.get("ALFWORLD_DATA", ".cache/alfworld")
    try:
        cache = ensure_alfworld_cache(cache_dir, split="eval_in_distribution")
    except AlfWorldDependencyError as exc:
        if strict:
            raise
        pytest.skip(str(exc))

    env = SingleAlfWorldEnv(seed=314160, split="eval_in_distribution", cache_dir=cache, max_episode_steps=30)
    try:
        first = env.reset(seed=314160)
        env.step("look")
        second = env.reset(seed=314160)
        assert first.info.get("extra.gamefile")
        assert second.info.get("extra.gamefile") == first.info["extra.gamefile"]
        assert second.observation == first.observation
    finally:
        env.close()


@pytest.mark.system
def test_alfworld_process_env_pool_prewarm_smoke():
    strict = os.environ.get("ALFWORLD_STRICT_SMOKE", "0") == "1"
    cache_dir = os.environ.get("ALFWORLD_DATA", ".cache/alfworld")
    try:
        ensure_alfworld_cache(cache_dir, split="train")
    except AlfWorldDependencyError as exc:
        if strict:
            raise
        pytest.skip(str(exc))

    async def run_smoke():
        pool = AlfWorldEnvPool(
            pool_size=2,
            split="train",
            cache_dir=cache_dir,
            max_episode_steps=32,
            suppress_output=True,
        )
        try:
            await pool.prewarm_async(seed=7)
            env = await pool.acquire_async(seed=7)
            reset = await asyncio.to_thread(env.reset, 7)
            assert reset.observation
            assert reset.info.get("admissible_commands")
            assert "extra.expert_plan" not in reset.info

            obs, reward, done, info = await asyncio.to_thread(env.step, "look")
            assert obs
            assert isinstance(reward, float)
            assert isinstance(done, bool)
            assert "extra.expert_plan" not in info
            pool.release(env)
        finally:
            pool.close()

    asyncio.run(run_smoke())


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
