from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class HistoryTurn:
    observation: str
    action: str
    response: str
    user_message: Any


@dataclass
class TextHistory:
    max_length: int | None = None
    _items: list[HistoryTurn] = field(default_factory=list)

    def append(
        self,
        observation: str,
        action: str,
        response: str | None = None,
        user_message: Any | None = None,
    ) -> None:
        self._items.append(
            HistoryTurn(
                observation=observation,
                action=action,
                response=response if response is not None else action,
                user_message=user_message or observation,
            )
        )
        if self.max_length is not None:
            self._items = self._items[-self.max_length :] if self.max_length > 0 else []

    def format(self) -> str:
        if not self._items:
            return "[]"
        rows = []
        for i, item in enumerate(self._items, start=1):
            rows.append(f"[Observation {i}: '{item.observation}', Action {i}: '{item.action}']")
        return " ".join(rows)

    def format_summaries(self) -> str:
        if not self._items:
            return "[]"
        rows = []
        for i, item in enumerate(self._items, start=1):
            content = item.response.strip()
            if content:
                rows.append(f"[Step {i}: {content}]")
        return " ".join(rows) if rows else "[]"

    def chat_messages(self, *, current_observation: str, instruction: str) -> list[dict[str, Any]]:
        messages = [{"role": "user", "content": instruction}]
        for item in self._items:
            messages.append({"role": "user", "content": item.user_message})
            messages.append({"role": "assistant", "content": item.response})
        messages.append({"role": "user", "content": current_observation})
        return messages

    def tail(self, max_length: int) -> "TextHistory":
        copied = TextHistory(max_length=max(0, int(max_length)))
        copied._items = list(self._items[-copied.max_length :]) if copied.max_length > 0 else []
        return copied

    def __len__(self) -> int:
        return len(self._items)


def assistant_history_content(*, mode: str, action: str, response: str) -> str:
    if mode == "action_only":
        return f"<action>{action}</action>"
    if mode == "full_response":
        return response
    if mode == "summary":
        return extract_summary_content(response)
    raise ValueError(f"Unsupported history assistant content mode: {mode!r}")


def extract_summary_content(response: str) -> str:
    match = re.search(r"<summary>(.*?)</summary>", response, flags=re.IGNORECASE | re.DOTALL)
    return match.group(1).strip() if match else ""


def format_chat_messages(messages: list[dict[str, Any]]) -> str:
    return "\n\n".join(f"{message['role'].upper()}:\n{message['content']}" for message in messages)
