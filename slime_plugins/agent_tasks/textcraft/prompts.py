from __future__ import annotations

from slime_plugins.agent_tasks.common.history import TextHistory

INITIAL_TEMPLATE = (
    "\n"
    "You are an expert agent operating in TextCraft, a text-only Minecraft crafting environment.\n"
    "Your current observation is:\n{current_observation}\n\n"
    "Valid actions are:\n"
    "- get N item\n"
    "- inventory\n"
    "- craft N target item using N ingredient, N ingredient\n\n"
    "Use only the crafting commands listed in the observation. Always include quantities for get and craft actions.\n"
    "You should first reason step-by-step. This reasoning process MUST be enclosed within <thinking> </thinking> tags.\n"
    "Then choose exactly one action and present it within <action> </action> tags.\n"
)

CHAT_INSTRUCTION = (
    "\n"
    "You are an expert agent operating in TextCraft, a text-only Minecraft crafting environment.\n"
    "Your goal is: {goal_text}\n"
    "{command_context}"
    "Every turn, you will receive the current observation from the environment.\n\n"
    "Valid actions are:\n"
    "- get N item\n"
    "- inventory\n"
    "- craft N target item using N ingredient, N ingredient\n\n"
    "Use only the crafting commands listed in the observation. Always include quantities for get and craft actions.\n"
    "You should first reason step-by-step. This reasoning process MUST be enclosed within <thinking> </thinking> tags.\n"
    "Then choose exactly one action and present it within <action> </action> tags.\n"
)

HISTORY_TEMPLATE = (
    "\n"
    "You are an expert agent operating in TextCraft. Your goal is: {goal_text}\n"
    "Prior to this step, you have already taken {step_count} step(s). Recent history: {action_history}\n"
    "{command_context}"
    "Your current observation is:\n{current_observation}\n\n"
    "Valid actions are: get N item, inventory, or craft N target item using N ingredient, N ingredient.\n"
    "Use only the crafting commands listed above, in the observation, or already visible in the recent history.\n"
    "You should first reason step-by-step within <thinking> </thinking> tags.\n"
    "Then choose exactly one action within <action> </action> tags.\n"
)


def build_prompt(
    *,
    current_observation: str,
    turn_idx: int,
    history: TextHistory,
    goal_text: str,
    crafting_commands: str = "",
) -> str:
    if turn_idx == 0 or len(history) == 0:
        return INITIAL_TEMPLATE.format(current_observation=current_observation)
    return HISTORY_TEMPLATE.format(
        goal_text=goal_text,
        step_count=turn_idx,
        action_history=history.format(),
        command_context=_format_command_context(crafting_commands),
        current_observation=current_observation,
    )


def build_chat_messages(prompt: str) -> list[dict[str, str]]:
    return [{"role": "user", "content": prompt}]


def build_chat_instruction(*, goal_text: str, crafting_commands: str = "") -> str:
    return CHAT_INSTRUCTION.format(
        goal_text=goal_text or "complete the requested craft",
        command_context=_format_command_context(crafting_commands),
    )


def build_messages(
    *,
    current_observation: str,
    turn_idx: int,
    history: TextHistory,
    goal_text: str,
    history_format: str,
    crafting_commands: str = "",
) -> list[dict[str, str]]:
    if history_format == "chat":
        persistent_commands = crafting_commands if history.max_length is not None else ""
        return history.chat_messages(
            current_observation=current_observation,
            instruction=build_chat_instruction(goal_text=goal_text, crafting_commands=persistent_commands),
        )
    prompt = build_prompt(
        current_observation=current_observation,
        turn_idx=turn_idx,
        history=history,
        goal_text=goal_text,
        crafting_commands=crafting_commands,
    )
    return build_chat_messages(prompt)


def _format_command_context(crafting_commands: str) -> str:
    commands = crafting_commands.strip()
    if not commands:
        return ""
    return f"Available crafting commands for this episode:\n{commands}\n\n"
