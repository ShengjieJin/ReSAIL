from __future__ import annotations

from slime_plugins.agent_tasks.common.history import TextHistory

INITIAL_TEMPLATE = (
    "\n"
    "You are an expert agent operating in the ALFRED Embodied Environment.\n"
    "Your current observation is: {current_observation}\n"
    "Your admissible actions of the current situation are: [{admissible_actions}].\n"
    "\n"
    "Now it's your turn to take an action.\n"
    "You should first reason step-by-step about the current situation. This reasoning process MUST be enclosed "
    "within <thinking> </thinking> tags.\n"
    "Once you've finished your reasoning, you should choose an admissible action for current step and present it "
    "within <action> </action> tags.\n"
)

CHAT_INSTRUCTION = (
    "\n"
    "You are an expert agent operating in the ALFRED Embodied Environment.\n"
    "Your task is to: {task_description}\n"
    "Every turn, you will receive the current observation and the admissible actions for the current situation.\n"
    "You must choose one admissible action for the current step.\n\n"
    "You should first reason step-by-step about the current situation. This reasoning process MUST be enclosed "
    "within <thinking> </thinking> tags.\n"
    "Once you've finished your reasoning, you should choose an admissible action for current step and present it "
    "within <action> </action> tags.\n"
)

HISTORY_TEMPLATE = (
    "\n"
    "You are an expert agent operating in the ALFRED Embodied Environment. Your task is to: {task_description}\n"
    "Prior to this step, you have already taken {step_count} step(s). Below are the most recent {history_length} "
    "observations and the corresponding actions you took: {action_history}\n"
    "You are now at step {current_step} and your current observation is: {current_observation}\n"
    "Your admissible actions of the current situation are: [{admissible_actions}].\n"
    "\n"
    "Now it's your turn to take an action.\n"
    "You should first reason step-by-step about the current situation. This reasoning process MUST be enclosed "
    "within <thinking> </thinking> tags.\n"
    "Once you've finished your reasoning, you should choose an admissible action for current step and present it "
    "within <action> </action> tags.\n"
)

def extract_task_description(observation: str) -> str:
    marker = "Your task is to: "
    start = observation.find(marker)
    if start == -1:
        return observation.strip()
    return observation[start + len(marker) :].strip()


def format_admissible_actions(admissible_actions: list[str]) -> str:
    return "\n ".join(f"'{action}'" for action in admissible_actions if action != "help")


def build_observation_message(*, current_observation: str, admissible_actions: list[str]) -> str:
    return (
        f"Your current observation is: {current_observation}\n"
        f"Your admissible actions of the current situation are: [{format_admissible_actions(admissible_actions)}].\n\n"
        "Now it's your turn to take an action."
    )


def build_prompt(
    *,
    current_observation: str,
    admissible_actions: list[str],
    turn_idx: int,
    task_description: str,
    history: TextHistory,
) -> str:
    actions = format_admissible_actions(admissible_actions)
    if turn_idx == 0 or len(history) == 0:
        prompt = INITIAL_TEMPLATE.format(current_observation=current_observation, admissible_actions=actions)
        return prompt
    prompt = HISTORY_TEMPLATE.format(
        task_description=task_description,
        step_count=turn_idx,
        history_length=len(history),
        action_history=history.format(),
        current_step=turn_idx + 1,
        current_observation=current_observation,
        admissible_actions=actions,
    )
    return prompt


def build_chat_messages(prompt: str) -> list[dict[str, str]]:
    return [{"role": "user", "content": prompt}]


def build_chat_instruction(*, task_description: str) -> str:
    return CHAT_INSTRUCTION.format(task_description=task_description)


def build_messages(
    *,
    current_observation: str,
    admissible_actions: list[str],
    turn_idx: int,
    task_description: str,
    history: TextHistory,
    history_format: str,
) -> list[dict[str, str]]:
    if history_format == "chat":
        return history.chat_messages(
            current_observation=build_observation_message(
                current_observation=current_observation,
                admissible_actions=admissible_actions,
            ),
            instruction=build_chat_instruction(task_description=task_description),
        )
    prompt = build_prompt(
        current_observation=current_observation,
        admissible_actions=admissible_actions,
        turn_idx=turn_idx,
        task_description=task_description,
        history=history,
    )
    return build_chat_messages(prompt)
