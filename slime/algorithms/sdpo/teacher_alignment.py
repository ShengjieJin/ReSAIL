from __future__ import annotations

import inspect
import json
from collections.abc import Sequence
from typing import Any

import torch

from slime.utils.processing_utils import build_processor_kwargs, process_vision_info

DEFAULT_SDPO_MAX_REPROMPT_TOKENS = 10240


def align_sdpo_teacher_log_probs(
    *,
    sdpo_teacher_log_probs: Any | None = None,
    teacher_log_probs: Any | None = None,
    sdpo_teacher_total_lengths: Sequence[int],
    sdpo_teacher_response_lengths: Sequence[int],
    student_response_lengths: Sequence[int],
    student_loss_masks: Any,
    student_total_lengths: Sequence[int] | None = None,
) -> list[Any]:
    """Slice teacher-prompt logprobs onto student response-token positions."""
    del student_total_lengths
    teacher_rows = sdpo_teacher_log_probs if sdpo_teacher_log_probs is not None else teacher_log_probs
    if teacher_rows is None:
        raise ValueError("sdpo_teacher_log_probs is required for SDPO teacher response alignment.")

    rows = _as_rows(teacher_rows, name="sdpo_teacher_log_probs")
    loss_mask_rows = _as_rows(student_loss_masks, name="student_loss_masks")
    _validate_row_counts(
        rows,
        loss_mask_rows,
        sdpo_teacher_total_lengths,
        sdpo_teacher_response_lengths,
        student_response_lengths,
    )

    aligned: list[Any] = []
    for idx, row in enumerate(rows):
        teacher_total_length = int(sdpo_teacher_total_lengths[idx])
        teacher_response_length = int(sdpo_teacher_response_lengths[idx])
        student_response_length = int(student_response_lengths[idx])
        loss_mask_length = _row_length(loss_mask_rows[idx])

        if student_response_length != loss_mask_length:
            raise ValueError(
                f"response length mismatch at row {idx}: student_response_lengths={student_response_length} "
                f"but student_loss_masks length={loss_mask_length}."
            )
        if teacher_response_length != student_response_length:
            raise ValueError(
                f"sdpo_teacher_log_probs response length mismatch at row {idx}: "
                f"teacher response={teacher_response_length}, student response={student_response_length}."
            )
        if teacher_response_length < 0 or teacher_total_length < teacher_response_length:
            raise ValueError(
                f"sdpo_teacher_log_probs invalid response slice at row {idx}: "
                f"total={teacher_total_length}, response={teacher_response_length}."
            )
        if _row_length(row) < teacher_total_length:
            raise ValueError(
                f"sdpo_teacher_log_probs row {idx} shorter than teacher total response input length: "
                f"row={_row_length(row)}, total={teacher_total_length}."
            )

        start = teacher_total_length - teacher_response_length
        end = teacher_total_length
        sliced = _slice_row(row, start, end)
        if _row_length(sliced) != student_response_length:
            raise ValueError(
                f"sdpo_teacher_log_probs response slice mismatch at row {idx}: "
                f"slice={_row_length(sliced)}, response={student_response_length}."
            )
        aligned.append(sliced)

    return aligned


slice_sdpo_teacher_response_log_probs = align_sdpo_teacher_log_probs


def tokenize_sdpo_teacher_prompt(
    tokenizer: Any,
    text: str,
    *,
    messages: Sequence[dict[str, Any]] | None = None,
    processor: Any | None = None,
    apply_chat_template_kwargs: dict[str, Any] | None = None,
    max_prompt_tokens: int | None = None,
    truncation_side: str = "right",
) -> list[int]:
    """Tokenize SDPO teacher prompt, preferring raw chat-template messages when available."""
    max_prompt_tokens = _normalize_max_prompt_tokens(max_prompt_tokens)
    if messages is not None:
        if not hasattr(tokenizer, "apply_chat_template"):
            raise TypeError("tokenizer must support apply_chat_template(...) when sdpo_teacher_messages is provided.")
        kwargs = _normalize_chat_template_kwargs(apply_chat_template_kwargs)
        has_multimodal_content = _messages_have_multimodal_content(messages)
        if has_multimodal_content and processor is None:
            raise ValueError(
                "processor is required for multimodal SDPO teacher messages; "
                "otherwise image tokens cannot be expanded to match multimodal_train_inputs."
            )
        if has_multimodal_content:
            with _temporary_truncation_side(tokenizer, truncation_side):
                rendered = tokenizer.apply_chat_template(
                    list(messages),
                    tokenize=False,
                    add_generation_prompt=True,
                    **kwargs,
                )
            multimodal_inputs = process_vision_info(list(messages), processor)
            processor_output = processor(text=rendered, **build_processor_kwargs(multimodal_inputs))
            token_ids = _normalize_token_ids(processor_output, name="sdpo_teacher_messages")
            if max_prompt_tokens is not None and len(token_ids) > max_prompt_tokens:
                raise ValueError(
                    "multimodal SDPO teacher prompt cannot be token-truncated because truncation can desynchronize "
                    f"image tokens from multimodal_train_inputs: tokens={len(token_ids)}, "
                    f"max_prompt_tokens={max_prompt_tokens}. Increase --sdpo-max-reprompt-tokens or reduce the "
                    "teacher prompt image/text context."
                )
            return token_ids
        if max_prompt_tokens is not None:
            kwargs.update({"max_length": max_prompt_tokens, "truncation": True})
        with _temporary_truncation_side(tokenizer, truncation_side):
            rendered = tokenizer.apply_chat_template(
                list(messages),
                tokenize=True,
                add_generation_prompt=True,
                **kwargs,
            )
        return _normalize_token_ids(rendered, name="sdpo_teacher_messages")

    input_ids = None
    if callable(tokenizer):
        if _accepts_kwarg(tokenizer, "add_special_tokens"):
            encoded = tokenizer(text, add_special_tokens=False)
        elif not hasattr(tokenizer, "encode"):
            encoded = tokenizer(text)
        else:
            encoded = None
        input_ids = _extract_input_ids(encoded) if encoded is not None else None

    if input_ids is None and hasattr(tokenizer, "encode"):
        if _accepts_kwarg(tokenizer.encode, "add_special_tokens"):
            input_ids = tokenizer.encode(text, add_special_tokens=False)
        else:
            input_ids = tokenizer.encode(text)

    if input_ids is None:
        raise TypeError("tokenizer must support __call__(..., add_special_tokens=False) or encode(...).")

    token_ids = _normalize_token_ids(input_ids, name="sdpo_teacher_prompt_text")
    return _truncate_token_ids(token_ids, max_prompt_tokens=max_prompt_tokens, truncation_side=truncation_side)


def build_sdpo_teacher_rollout_data(
    rollout_data: dict[str, Any],
    tokenizer: Any,
    *,
    processor: Any | None = None,
    apply_chat_template_kwargs: dict[str, Any] | None = None,
    max_prompt_tokens: int | None = DEFAULT_SDPO_MAX_REPROMPT_TOKENS,
    truncation_side: str = "right",
) -> dict[str, Any]:
    """Build teacher-prompt rollout rows as `teacher_prompt_tokens + original_response_tokens`."""
    if "sdpo_teacher_prompt_text" not in rollout_data:
        raise ValueError("sdpo_teacher_prompt_text is required for sdpo_loss teacher log-prob precompute.")
    max_prompt_tokens = _normalize_max_prompt_tokens(max_prompt_tokens)

    tokens = rollout_data["tokens"]
    response_lengths = rollout_data["response_lengths"]
    loss_masks = rollout_data["loss_masks"]
    teacher_prompt_texts = rollout_data["sdpo_teacher_prompt_text"]
    teacher_messages = rollout_data.get("sdpo_teacher_messages")
    teacher_multimodal_train_inputs = rollout_data.get("sdpo_teacher_multimodal_train_inputs")
    if teacher_multimodal_train_inputs is None:
        teacher_multimodal_train_inputs = rollout_data.get("multimodal_train_inputs")
    row_count = len(tokens)
    row_count_fields = {
        "response_lengths": response_lengths,
        "loss_masks": loss_masks,
        "sdpo_teacher_prompt_text": teacher_prompt_texts,
    }
    if teacher_messages is not None:
        row_count_fields["sdpo_teacher_messages"] = teacher_messages
    if teacher_multimodal_train_inputs is not None:
        row_count_fields["sdpo_teacher_multimodal_train_inputs"] = teacher_multimodal_train_inputs
    _validate_teacher_rollout_row_counts(row_count, row_count_fields)

    teacher_tokens = []
    teacher_total_lengths = []
    teacher_response_lengths = []
    teacher_loss_masks = []
    teacher_prompt_token_lengths = []
    teacher_prompt_token_lengths_raw = []
    teacher_prompt_truncated = []
    teacher_prompt_char_lengths = []
    for idx, (token_row, response_length, loss_mask, teacher_prompt_text) in enumerate(
        zip(tokens, response_lengths, loss_masks, teacher_prompt_texts, strict=True)
    ):
        response_length = int(response_length)
        if response_length < 0:
            raise ValueError(f"response_lengths[{idx}] must be non-negative for SDPO teacher precompute.")
        if _row_length(loss_mask) != response_length:
            raise ValueError(
                f"loss_masks[{idx}] length must equal response_lengths[{idx}] for SDPO teacher precompute: "
                f"{_row_length(loss_mask)} != {response_length}."
            )
        if _row_length(token_row) < response_length:
            raise ValueError(
                f"tokens[{idx}] shorter than response_lengths[{idx}] for SDPO teacher precompute: "
                f"{_row_length(token_row)} < {response_length}."
            )

        messages = teacher_messages[idx] if teacher_messages is not None else None
        teacher_prompt_tokens = tokenize_sdpo_teacher_prompt(
            tokenizer,
            teacher_prompt_text,
            messages=messages,
            processor=processor,
            apply_chat_template_kwargs=apply_chat_template_kwargs,
            max_prompt_tokens=max_prompt_tokens,
            truncation_side=truncation_side,
        )
        if max_prompt_tokens is None or len(teacher_prompt_tokens) < max_prompt_tokens:
            raw_teacher_prompt_tokens = teacher_prompt_tokens
        else:
            raw_teacher_prompt_tokens = tokenize_sdpo_teacher_prompt(
                tokenizer,
                teacher_prompt_text,
                messages=messages,
                processor=processor,
                apply_chat_template_kwargs=apply_chat_template_kwargs,
                max_prompt_tokens=None,
                truncation_side=truncation_side,
            )
        response_tokens = _slice_response_tokens(token_row, response_length)
        teacher_token_row = _concat_tokens_like(token_row, teacher_prompt_tokens, response_tokens)

        teacher_tokens.append(teacher_token_row)
        teacher_total_lengths.append(len(teacher_prompt_tokens) + response_length)
        teacher_response_lengths.append(response_length)
        teacher_loss_masks.append(_clone_row(loss_mask))
        teacher_prompt_token_lengths.append(len(teacher_prompt_tokens))
        teacher_prompt_token_lengths_raw.append(len(raw_teacher_prompt_tokens))
        teacher_prompt_truncated.append(float(len(raw_teacher_prompt_tokens) > len(teacher_prompt_tokens)))
        teacher_prompt_char_lengths.append(len(str(teacher_prompt_text)))

    teacher_rollout_data = {
        "tokens": teacher_tokens,
        "total_lengths": teacher_total_lengths,
        "response_lengths": teacher_response_lengths,
        "loss_masks": teacher_loss_masks,
        "sdpo_teacher_prompt_token_lengths": teacher_prompt_token_lengths,
        "sdpo_teacher_prompt_token_lengths_raw": teacher_prompt_token_lengths_raw,
        "sdpo_teacher_prompt_truncated": teacher_prompt_truncated,
        "sdpo_teacher_prompt_char_lengths": teacher_prompt_char_lengths,
    }
    if "dynamic_global_batch_size" in rollout_data:
        teacher_rollout_data["dynamic_global_batch_size"] = rollout_data["dynamic_global_batch_size"]
    if teacher_multimodal_train_inputs is not None:
        teacher_rollout_data["multimodal_train_inputs"] = list(teacher_multimodal_train_inputs)
    return teacher_rollout_data


def _as_rows(value: Any, *, name: str) -> list[Any]:
    if isinstance(value, torch.Tensor):
        if value.ndim == 1:
            return [value.reshape(-1)]
        if value.ndim == 2:
            return [row.reshape(-1) for row in value]
        raise ValueError(f"{name} must be 1D or 2D.")
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        items = list(value)
        if not items:
            return []
        if _is_flat_sequence(items):
            return [items]
        return [item.reshape(-1) if isinstance(item, torch.Tensor) else list(item) for item in items]
    raise TypeError(f"{name} must be a torch.Tensor or a sequence.")


def _validate_row_counts(
    rows: list[Any],
    loss_mask_rows: list[Any],
    sdpo_teacher_total_lengths: Sequence[int],
    sdpo_teacher_response_lengths: Sequence[int],
    student_response_lengths: Sequence[int],
) -> None:
    expected = len(rows)
    lengths = {
        "student_loss_masks": len(loss_mask_rows),
        "sdpo_teacher_total_lengths": len(sdpo_teacher_total_lengths),
        "sdpo_teacher_response_lengths": len(sdpo_teacher_response_lengths),
        "student_response_lengths": len(student_response_lengths),
    }
    mismatches = {key: value for key, value in lengths.items() if value != expected}
    if mismatches:
        raise ValueError(f"sdpo_teacher_log_probs response row count mismatch: expected {expected}, got {mismatches}.")


def _slice_row(row: Any, start: int, end: int) -> Any:
    return row[start:end]


def _row_length(row: Any) -> int:
    if isinstance(row, torch.Tensor):
        return int(row.numel())
    return len(row)


def _is_flat_sequence(items: list[Any]) -> bool:
    return all(not _is_row_like(item) for item in items)


def _is_row_like(item: Any) -> bool:
    if isinstance(item, torch.Tensor):
        return item.ndim > 0
    return isinstance(item, Sequence) and not isinstance(item, (str, bytes))


def _extract_input_ids(encoded: Any) -> Any | None:
    if isinstance(encoded, dict):
        return encoded.get("input_ids")
    return getattr(encoded, "input_ids", None)


def _accepts_kwarg(fn: Any, kwarg: str) -> bool:
    try:
        signature = inspect.signature(fn)
    except (TypeError, ValueError):
        return True
    return any(
        parameter.kind == inspect.Parameter.VAR_KEYWORD or parameter.name == kwarg
        for parameter in signature.parameters.values()
    )


def _normalize_token_ids(input_ids: Any, *, name: str) -> list[int]:
    extracted = _extract_input_ids(input_ids)
    if extracted is not None:
        input_ids = extracted
    if isinstance(input_ids, torch.Tensor):
        if input_ids.ndim == 1:
            input_ids = input_ids.tolist()
        elif input_ids.ndim == 2 and int(input_ids.shape[0]) == 1:
            input_ids = input_ids.reshape(-1).tolist()
        else:
            raise ValueError(f"{name} tokenizer output must contain one input_ids row.")
    elif hasattr(input_ids, "tolist") and not isinstance(input_ids, (str, bytes)):
        input_ids = input_ids.tolist()
    if isinstance(input_ids, Sequence) and not isinstance(input_ids, (str, bytes)):
        ids = list(input_ids)
        if ids and _is_row_like(ids[0]):
            if len(ids) != 1:
                raise ValueError(f"{name} tokenizer output must contain one input_ids row.")
            ids = _normalize_single_token_row(ids[0], name=name)
        return [int(token_id) for token_id in ids]
    raise TypeError(f"{name} tokenizer output input_ids must be a sequence.")


def _normalize_single_token_row(row: Any, *, name: str) -> list[Any]:
    if isinstance(row, torch.Tensor):
        if row.ndim != 1:
            raise ValueError(f"{name} tokenizer output must contain one input_ids row.")
        return list(row.tolist())
    if hasattr(row, "tolist") and not isinstance(row, (str, bytes)):
        row = row.tolist()
    if isinstance(row, Sequence) and not isinstance(row, (str, bytes)):
        ids = list(row)
        if ids and _is_row_like(ids[0]):
            raise ValueError(f"{name} tokenizer output must contain one input_ids row.")
        return ids
    raise TypeError(f"{name} tokenizer output input_ids row must be a sequence.")


def _messages_have_multimodal_content(messages: Sequence[dict[str, Any]]) -> bool:
    for message in messages:
        content = message.get("content") if isinstance(message, dict) else None
        if not isinstance(content, Sequence) or isinstance(content, (str, bytes)):
            continue
        for item in content:
            if isinstance(item, dict) and item.get("type") in {"image", "video"}:
                return True
    return False


def _normalize_chat_template_kwargs(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, str):
        return json.loads(value)
    return dict(value)


def _normalize_max_prompt_tokens(value: int | None) -> int | None:
    if value is None:
        return None
    value = int(value)
    return value if value > 0 else None


def _truncate_token_ids(
    token_ids: list[int],
    *,
    max_prompt_tokens: int | None,
    truncation_side: str = "right",
) -> list[int]:
    if truncation_side not in {"right", "left"}:
        raise ValueError(f"Unsupported SDPO teacher prompt truncation_side={truncation_side!r}.")
    if max_prompt_tokens is None or len(token_ids) <= max_prompt_tokens:
        return token_ids
    if truncation_side == "right":
        return token_ids[:max_prompt_tokens]
    return token_ids[-max_prompt_tokens:]


class _temporary_truncation_side:
    def __init__(self, tokenizer: Any, truncation_side: str) -> None:
        if truncation_side not in {"right", "left"}:
            raise ValueError(f"Unsupported SDPO teacher prompt truncation_side={truncation_side!r}.")
        self.tokenizer = tokenizer
        self.truncation_side = truncation_side
        self.had_attr = hasattr(tokenizer, "truncation_side")
        self.old_value = getattr(tokenizer, "truncation_side", None)

    def __enter__(self):
        if self.had_attr:
            setattr(self.tokenizer, "truncation_side", self.truncation_side)
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if self.had_attr:
            setattr(self.tokenizer, "truncation_side", self.old_value)


def _validate_teacher_rollout_row_counts(row_count: int, values: dict[str, Sequence[Any]]) -> None:
    mismatches = {key: len(value) for key, value in values.items() if len(value) != row_count}
    if mismatches:
        raise ValueError(f"SDPO teacher rollout row count mismatch: expected {row_count}, got {mismatches}.")


def _slice_response_tokens(token_row: Any, response_length: int) -> Any:
    if response_length == 0:
        if isinstance(token_row, torch.Tensor):
            return token_row.new_empty((0,))
        return []
    if isinstance(token_row, torch.Tensor):
        return token_row[-response_length:].clone()
    return list(token_row[-response_length:])


def _concat_tokens_like(source_tokens: Any, prompt_tokens: list[int], response_tokens: Any) -> Any:
    if isinstance(source_tokens, torch.Tensor):
        prompt_tensor = torch.tensor(prompt_tokens, dtype=source_tokens.dtype, device=source_tokens.device)
        if not isinstance(response_tokens, torch.Tensor):
            response_tokens = torch.tensor(response_tokens, dtype=source_tokens.dtype, device=source_tokens.device)
        return torch.cat([prompt_tensor, response_tokens])
    return list(prompt_tokens) + list(response_tokens)


def _clone_row(row: Any) -> Any:
    if isinstance(row, torch.Tensor):
        return row.clone()
    return list(row)
