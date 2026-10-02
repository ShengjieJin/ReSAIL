from __future__ import annotations

import json
import os
import random
import re
import time
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

DEFAULT_CACHE_DIR = Path(".cache/textcraft")


class TextCraftDependencyError(RuntimeError):
    pass


class ActionFailed(Exception):
    pass


@dataclass(frozen=True)
class ItemRef:
    tag: str | None = None
    item_id: str | None = None

    @property
    def name(self) -> str:
        if self.item_id is not None:
            return self.item_id
        if self.tag is not None:
            return self.tag
        raise ValueError("ItemRef must contain item_id or tag")


@dataclass(frozen=True)
class ItemCount:
    item: ItemRef
    count: int


@dataclass(frozen=True)
class Recipe:
    input_items: tuple[ItemCount, ...]
    output_item: ItemCount

    @property
    def recipe_str(self) -> str:
        inputs = ", ".join(f"{item.count} {item_id_to_str(item.item.name)}" for item in self.input_items)
        return f"craft {self.output_item.count} {item_id_to_str(self.output_item.item.name)} using {inputs}"


@dataclass
class TextCraftResetResult:
    observation: str
    info: dict[str, Any]
    reset_seconds: float
    worker_id: str
    reused_worker: bool
    worker_created: bool


def item_id_to_str(item_id: str) -> str:
    return item_id.replace("minecraft:", "").replace("_", " ")


def resolve_textcraft_cache_dir(cache_dir: str | os.PathLike[str] | None = None) -> Path:
    raw = cache_dir or os.environ.get("TEXTCRAFT_DATA") or DEFAULT_CACHE_DIR
    return Path(raw).expanduser().resolve()


def ensure_textcraft_cache(cache_dir: str | os.PathLike[str] | None = None) -> Path:
    cache = resolve_textcraft_cache_dir(cache_dir)
    recipe_dir = cache / "recipes"
    if not recipe_dir.exists() or not any(recipe_dir.glob("*.json")):
        raise TextCraftDependencyError(
            "TextCraft cache is incomplete. Expected recipe JSON files under "
            f"{recipe_dir}. Populate repo-local .cache/textcraft/recipes first."
        )
    return cache


def check_textcraft_cache(cache_dir: str | os.PathLike[str] | None = None) -> dict[str, int]:
    tree = load_crafting_tree(str(ensure_textcraft_cache(cache_dir)))
    return {
        "item_recipe_count": len(tree.itemid_recipes),
        "tag_recipe_count": len(tree.tag_recipes),
        "valid_item_count": len(tree.itemid_set),
    }


@lru_cache(maxsize=8)
def load_crafting_tree(cache_dir: str) -> "CraftingTree":
    return CraftingTree(Path(cache_dir) / "recipes")


class CraftingTree:
    def __init__(self, recipe_dir: str | os.PathLike[str]) -> None:
        self.recipe_dir = Path(recipe_dir)
        self.tag_recipes: dict[str, list[Recipe]] = {}
        self.itemid_recipes: dict[str, list[Recipe]] = {}
        self.tag_set: set[str] = set()
        self.itemid_set: set[str] = set()
        self.item_id_to_tag: dict[str, str] = {}
        self.transitive_dependencies: dict[str, set[str]] = {}
        self.min_depth: dict[str, int] = {}
        self._load_recipes()
        self._clean_up_recipes()

    def _load_recipes(self) -> None:
        for filename in os.listdir(self.recipe_dir):
            path = self.recipe_dir / filename
            if not path.is_file() or path.suffix != ".json":
                continue
            data = json.loads(path.read_text(encoding="utf-8"))
            recipe_type = data.get("type")
            if recipe_type == "minecraft:crafting_shaped":
                input_items = self._parse_shaped_inputs(data)
            elif recipe_type == "minecraft:crafting_shapeless":
                input_items = self._parse_shapeless_inputs(data)
            else:
                continue
            if not input_items:
                continue
            output_item = self._parse_result(data)
            self.itemid_set.add(output_item.item.item_id)

            if len(input_items) == 1 and input_items[0].item.name.endswith("_block"):
                continue

            output_tag = None
            if data.get("group"):
                output_tag = "minecraft:" + str(data["group"])
                if output_tag != output_item.item.item_id:
                    self.tag_set.add(output_tag)
                    self.item_id_to_tag[output_item.item.item_id] = output_tag
            if output_tag is not None:
                output_item = ItemCount(ItemRef(tag=output_tag, item_id=output_item.item.item_id), output_item.count)

            self._add_recipe(Recipe(tuple(input_items), output_item))

    def _parse_shaped_inputs(self, data: dict[str, Any]) -> list[ItemCount]:
        pattern = data.get("pattern") or []
        keys = data.get("key") or {}
        input_items = []
        for marker, item_data in keys.items():
            count = sum(str(line).count(marker) for line in pattern)
            if count <= 0:
                continue
            input_items.append(ItemCount(self._parse_item_or_tag(_first_recipe_option(item_data)), count))
        return input_items

    def _parse_shapeless_inputs(self, data: dict[str, Any]) -> list[ItemCount]:
        item_counts: dict[str, tuple[ItemRef, int]] = {}
        for item_data in data.get("ingredients") or []:
            item = self._parse_item_or_tag(_first_recipe_option(item_data))
            existing = item_counts.get(item.name)
            item_counts[item.name] = (item, (existing[1] if existing else 0) + 1)
        return [ItemCount(item, count) for item, count in item_counts.values()]

    def _parse_item_or_tag(self, item_data: dict[str, Any]) -> ItemRef:
        if "item" in item_data:
            item_id = str(item_data["item"])
            self.itemid_set.add(item_id)
            return ItemRef(item_id=item_id)
        if "tag" in item_data:
            tag = str(item_data["tag"])
            self.tag_set.add(tag)
            return ItemRef(tag=tag)
        raise ValueError(f"Unknown TextCraft recipe ingredient: {item_data}")

    def _parse_result(self, data: dict[str, Any]) -> ItemCount:
        result = data.get("result")
        if isinstance(result, str):
            return ItemCount(ItemRef(item_id=result), 1)
        if isinstance(result, dict) and result.get("item"):
            return ItemCount(ItemRef(item_id=str(result["item"])), int(result.get("count") or 1))
        raise ValueError(f"Unknown TextCraft recipe result: {result}")

    def _add_recipe(self, recipe: Recipe) -> None:
        output_id = recipe.output_item.item.item_id
        if output_id is None:
            return
        if self._would_create_cycle(recipe):
            return
        self.itemid_recipes.setdefault(output_id, []).append(recipe)
        self.transitive_dependencies.setdefault(output_id, set())
        for item_count in recipe.input_items:
            name = item_count.item.name
            self.transitive_dependencies[output_id].add(name)
            self.transitive_dependencies[output_id].update(self.transitive_dependencies.get(name, set()))
        if recipe.output_item.item.tag is not None:
            self.tag_recipes.setdefault(recipe.output_item.item.tag, []).append(recipe)

    def _would_create_cycle(self, recipe: Recipe) -> bool:
        output_id = recipe.output_item.item.item_id
        if output_id is None:
            return False
        for item_count in recipe.input_items:
            dependencies = self.transitive_dependencies.get(item_count.item.name)
            if dependencies and output_id in dependencies:
                return True
        return False

    def _clean_up_recipes(self) -> None:
        converted_tags = set()
        for recipes in self.itemid_recipes.values():
            for recipe in recipes:
                for item_count in recipe.input_items:
                    tag = item_count.item.tag
                    if item_count.item.item_id is None and tag is not None:
                        if not any(item in self.itemid_recipes for item in self.get_items_with_tags(tag)):
                            converted_tags.add(tag)
        for tag in converted_tags:
            self.itemid_set.add(tag)
            self.tag_set.discard(tag)

    def craft(self, recipe: Recipe) -> ItemCount | None:
        output_id = recipe.output_item.item.item_id
        if output_id not in self.itemid_recipes:
            return None
        for target_recipe in self.itemid_recipes[output_id]:
            remaining = list(recipe.input_items)
            success = True
            for target_item in target_recipe.input_items:
                match = self._find_matching_item(target_item.item, remaining)
                if match is None or match.count != target_item.count:
                    success = False
                    break
                remaining.remove(match)
            if success and not remaining:
                return target_recipe.output_item
        return None

    def _find_matching_item(self, target: ItemRef, candidates: list[ItemCount]) -> ItemCount | None:
        for candidate in candidates:
            item = candidate.item
            if target.item_id is not None and item.item_id == target.item_id:
                return candidate
            if target.tag is not None and (
                item.tag == target.tag or item.item_id == target.tag or self.item_id_to_tag.get(item.item_id) == target.tag
            ):
                return candidate
        return None

    def is_craftable(self, item: str) -> bool:
        return item in self.itemid_recipes or item in self.tag_recipes

    def is_valid_item(self, item: str) -> bool:
        return item in self.itemid_set

    def is_tag(self, item: str) -> bool:
        return item in self.tag_set

    def get_items_with_tags(self, tag: str):
        for item_id, mapped_tag in self.item_id_to_tag.items():
            if tag == mapped_tag:
                yield item_id

    def get_min_depth(self, item_name: str) -> int:
        if item_name in self.min_depth:
            return self.min_depth[item_name]
        if item_name in self.itemid_recipes:
            self.min_depth[item_name] = self._min_depth_for_recipes(self.itemid_recipes[item_name])
        elif item_name in self.tag_recipes:
            self.min_depth[item_name] = self._min_depth_for_recipes(self.tag_recipes[item_name])
        else:
            self.min_depth[item_name] = 0
        return self.min_depth[item_name]

    def _min_depth_for_recipes(self, recipes: list[Recipe]) -> int:
        depths = []
        for recipe in recipes:
            depths.append(max(self.get_min_depth(item.item.name) + 1 for item in recipe.input_items))
        return min(depths)

    def item_recipes_min_depth(self, min_depth: int):
        for item, _ in self.itemid_recipes.items():
            depth = self.get_min_depth(item)
            if depth >= min_depth:
                yield item, depth

    def create_recipe_set(
        self, item_name: str, rng: random.Random | None = None
    ) -> tuple[list[Recipe], list[Recipe]]:
        item_uses = self._collect_item_uses()
        recipes = self._traverse_recipe_tree(item_name, set())
        distractors = []
        sampler = rng.sample if rng is not None else random.sample
        for recipe in recipes:
            for item in recipe.input_items:
                uses = item_uses.get(item.item.name)
                if uses:
                    distractors.extend(sampler(uses, min(len(uses), 10)))
        return recipes, distractors

    def _traverse_recipe_tree(self, item_name: str, visited: set[str]) -> list[Recipe]:
        if item_name in visited:
            return []
        current_recipes = self.itemid_recipes.get(item_name) or self.tag_recipes.get(item_name) or []
        next_visited = visited | {item_name}
        collected = list(current_recipes)
        for recipe in current_recipes:
            for item in recipe.input_items:
                collected.extend(self._traverse_recipe_tree(item.item.name, set(next_visited)))
        return collected

    def _collect_item_uses(self) -> dict[str, list[Recipe]]:
        item_uses: dict[str, list[Recipe]] = {}
        for recipes in list(self.itemid_recipes.values()) + list(self.tag_recipes.values()):
            for recipe in recipes:
                for item in recipe.input_items:
                    item_uses.setdefault(item.item.name, []).append(recipe)
        return item_uses


class SingleTextCraftEnv:
    action_regexes = {
        "craft": re.compile(r"craft (.*) using (.*)"),
        "get": re.compile(r"get ([0-9]+) (.*)"),
        "inventory": re.compile(r"inventory"),
    }
    count_regex = re.compile(r"([0-9]+) (.*)")

    def __init__(
        self,
        *,
        seed: int = 0,
        split: str = "train",
        cache_dir: str | os.PathLike[str] | None = None,
        data_idx: int = 0,
        max_episode_steps: int = 30,
    ) -> None:
        self.seed = int(seed)
        self.split = split
        self.data_idx = int(data_idx)
        self.max_episode_steps = int(max_episode_steps)
        self.cache_dir = ensure_textcraft_cache(cache_dir)
        self.crafting_tree = load_crafting_tree(str(self.cache_dir))
        self.inventory: dict[str, int] = {}
        self.commands = ""
        self.goal = ""
        self.worker_id = f"single-{os.getpid()}"
        self._reset_count = 0

    def reset(self, seed: int | None = None, data_idx: int | None = None) -> TextCraftResetResult:
        if seed is not None:
            self.seed = int(seed)
        if data_idx is not None:
            self.data_idx = int(data_idx)
        start = time.monotonic()
        self.inventory = {}
        rng = random.Random(self.seed)
        item_depth_list = list(self.crafting_tree.item_recipes_min_depth(1))
        if not item_depth_list:
            raise TextCraftDependencyError(f"No craftable TextCraft items found in {self.cache_dir}")
        sorted_items = sorted(item_depth_list, key=lambda item: item[1])
        goal, goal_depth = sorted_items[self.data_idx % len(sorted_items)]
        self.goal = goal
        recipes, distractors = self.crafting_tree.create_recipe_set(self.goal, rng=rng)
        recipe_set = set()
        distractor_set = set()
        for recipe in recipes:
            recipe_set.add(recipe.recipe_str)
        for distractor in distractors:
            if distractor.recipe_str not in recipe_set:
                distractor_set.add(distractor.recipe_str)
        selected_distractors = (
            rng.sample(list(distractor_set), min(len(distractor_set), 10)) if distractor_set else []
        )
        command_list = list(recipe_set) + selected_distractors
        rng.shuffle(command_list)
        self.commands = "\n".join(command_list)
        observation = f"Crafting commands:\n{self.commands}\n\nGoal: craft {item_id_to_str(self.goal)}."
        self._reset_count += 1
        return TextCraftResetResult(
            observation=observation,
            info=self._info(reward=0.0, done=False, action_failed=False, goal_depth=goal_depth),
            reset_seconds=time.monotonic() - start,
            worker_id=self.worker_id,
            reused_worker=self._reset_count > 1,
            worker_created=self._reset_count == 1,
        )

    def step(self, action: str) -> tuple[str, float, bool, dict[str, Any]]:
        action = str(action).strip()
        observation = None
        reward = 0.0
        terminated = False
        action_failed = False
        try:
            for action_type, regex in self.action_regexes.items():
                match = regex.match(action)
                if not match:
                    continue
                if action_type == "craft":
                    recipe = self.extract_recipe(match.group(1), match.group(2))
                    if not self.has_items(recipe.input_items):
                        raise ActionFailed(f"Could not find enough items to craft {recipe.output_item.item.name}")
                    output_item = self.crafting_tree.craft(recipe)
                    if output_item is None:
                        raise ActionFailed(f"Could not find a valid recipe for {recipe.output_item.item.name}")
                    self.remove_items(recipe.input_items)
                    self.add_item(output_item.item, output_item.count)
                    observation = f"Crafted {output_item.count} {output_item.item.name}"
                    if output_item.item.item_id == self.goal:
                        reward = 1.0
                        terminated = True
                elif action_type == "get":
                    amount = int(match.group(1))
                    item = self.item_str_to_ref(match.group(2))
                    if self.crafting_tree.is_craftable(item.name):
                        raise ActionFailed(f"Could not find {match.group(2)}")
                    if self.crafting_tree.is_tag(item.name) or item.item_id is None:
                        raise ActionFailed(f"Could not find {match.group(2)}")
                    if not self.crafting_tree.is_valid_item(item.item_id):
                        raise ActionFailed(f"Could not find {match.group(2)}")
                    self.add_item(item, amount)
                    observation = f"Got {amount} {match.group(2)}"
                    if item.item_id == self.goal:
                        reward = 1.0
                        terminated = True
                elif action_type == "inventory":
                    observation = self._inventory_observation()
                break
            if observation is None:
                raise ActionFailed(f"Could not execute {action}")
        except ActionFailed as exc:
            observation = str(exc)
            reward = 0.0
            terminated = False
            action_failed = True
        return observation, reward, terminated, self._info(
            reward=reward,
            done=terminated,
            action_failed=action_failed,
        )

    def extract_recipe(self, output_item_str: str, input_items_str: str) -> Recipe:
        output_match = self.count_regex.fullmatch(output_item_str.strip())
        if output_match:
            output_item = self.item_str_to_ref(output_match.group(2))
            output_count = int(output_match.group(1))
        else:
            output_item = self.item_str_to_ref(output_item_str.strip())
            output_count = 1
        input_items = []
        for input_item_count in input_items_str.split(","):
            match = self.count_regex.fullmatch(input_item_count.strip())
            if match is None:
                raise ActionFailed(f"Wrong item format: {input_item_count.strip()}")
            input_items.append(ItemCount(self.item_str_to_ref(match.group(2)), int(match.group(1))))
        return Recipe(tuple(input_items), ItemCount(output_item, output_count))

    def item_str_to_ref(self, item: str) -> ItemRef:
        item_id = "minecraft:" + item.strip().replace(" ", "_")
        if self.crafting_tree.is_tag(item_id):
            return ItemRef(tag=item_id)
        return ItemRef(item_id=item_id)

    def has_items(self, items: tuple[ItemCount, ...]) -> bool:
        return all(self.inventory.get(item.item.item_id, 0) >= item.count for item in items)

    def add_item(self, item: ItemRef, amount: int) -> None:
        if item.item_id is None:
            raise ActionFailed(f"Could not add non-item tag {item.name}")
        self.inventory[item.item_id] = self.inventory.get(item.item_id, 0) + int(amount)

    def remove_items(self, items: tuple[ItemCount, ...]) -> None:
        for item in items:
            if item.item.item_id is None:
                raise ActionFailed(f"Could not remove non-item tag {item.item.name}")
            self.inventory[item.item.item_id] -= item.count
            if self.inventory[item.item.item_id] <= 0:
                del self.inventory[item.item.item_id]

    def close(self) -> None:
        self.inventory.clear()

    def _inventory_observation(self) -> str:
        if not self.inventory:
            return "Inventory: You are not carrying anything."
        return "Inventory: " + "".join(
            f"[{item_id_to_str(item)}] ({amount}) " for item, amount in self.inventory.items()
        )

    def _info(
        self,
        *,
        reward: float,
        done: bool,
        action_failed: bool,
        goal_depth: int | None = None,
    ) -> dict[str, Any]:
        return {
            "reward": float(reward),
            "done": bool(done),
            "goal": self.goal,
            "goal_text": item_id_to_str(self.goal) if self.goal else "",
            "goal_depth": goal_depth,
            "data_idx": self.data_idx,
            "split": self.split,
            "commands_count": len(self.commands.splitlines()) if self.commands else 0,
            "inventory_size": sum(self.inventory.values()),
            "action_failed": bool(action_failed),
        }


def build_textcraft_env(**kwargs: Any) -> SingleTextCraftEnv:
    return SingleTextCraftEnv(**kwargs)


def _first_recipe_option(value: Any) -> dict[str, Any]:
    if isinstance(value, list):
        if not value:
            raise ValueError("empty TextCraft recipe option list")
        value = value[0]
    if not isinstance(value, dict):
        raise ValueError(f"expected TextCraft recipe option dict, got {type(value).__name__}")
    return value
