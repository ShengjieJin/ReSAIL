from __future__ import annotations

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

from slime_plugins.agent_tasks.textcraft.envs import (
    SingleTextCraftEnv,
    TextCraftDependencyError,
    check_textcraft_cache,
    ensure_textcraft_cache,
    item_id_to_str,
)


def _find_real_one_step_solve(env: SingleTextCraftEnv):
    item_depth_list = sorted(env.crafting_tree.item_recipes_min_depth(1), key=lambda item: item[1])
    for data_idx, (goal, depth) in enumerate(item_depth_list):
        if depth != 1:
            continue
        for recipe in env.crafting_tree.itemid_recipes.get(goal, []):
            get_actions = []
            for item_count in recipe.input_items:
                item_id = item_count.item.item_id
                if item_id is None:
                    break
                if env.crafting_tree.is_craftable(item_id):
                    break
                if not env.crafting_tree.is_valid_item(item_id):
                    break
                get_actions.append(f"get {item_count.count} {item_id_to_str(item_id)}")
            else:
                return data_idx, goal, get_actions, recipe.recipe_str
    raise AssertionError("No one-step solvable TextCraft recipe found in real cache")


@pytest.mark.system
def test_textcraft_real_env_cache_smoke():
    strict = os.environ.get("TEXTCRAFT_STRICT_SMOKE", "0") == "1"
    cache_dir = os.environ.get("TEXTCRAFT_DATA", ".cache/textcraft")
    try:
        cache = ensure_textcraft_cache(cache_dir)
        counts = check_textcraft_cache(cache)
    except TextCraftDependencyError as exc:
        if strict:
            raise
        pytest.skip(str(exc))

    env = SingleTextCraftEnv(seed=7, split="train", cache_dir=cache, data_idx=0, max_episode_steps=30)
    try:
        reset = env.reset(seed=7, data_idx=0)
        assert counts["item_recipe_count"] > 0
        assert counts["valid_item_count"] > 0
        assert reset.observation
        assert "Crafting commands:" in reset.observation
        assert "Goal: craft" in reset.observation
        assert reset.info["goal"]

        obs, reward, done, info = env.step("inventory")
        assert obs.startswith("Inventory:")
        assert isinstance(reward, float)
        assert isinstance(done, bool)
        assert info["action_failed"] is False

        data_idx, goal, get_actions, craft_action = _find_real_one_step_solve(env)
        reset = env.reset(seed=7, data_idx=data_idx)
        assert reset.info["goal"] == goal
        for action in get_actions:
            obs, reward, done, info = env.step(action)
            assert obs.startswith("Got ")
            assert reward == 0.0
            assert done is False
            assert info["action_failed"] is False

        obs, reward, done, info = env.step(craft_action)
        assert obs.startswith("Crafted ")
        assert reward == 1.0
        assert done is True
        assert info["goal"] == goal
        assert info["action_failed"] is False
    finally:
        env.close()


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
