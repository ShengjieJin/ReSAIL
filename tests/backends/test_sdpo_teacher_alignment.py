from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

NUM_GPUS = 0


class CallableTokenizer:
    def __init__(self, mapping: dict[str, list[int]]):
        self.mapping = mapping
        self.calls: list[tuple[str, bool]] = []

    def __call__(self, text, add_special_tokens=True):
        self.calls.append((text, add_special_tokens))
        return {"input_ids": self.mapping[text]}


class ChatTemplateTokenizer:
    def __init__(self):
        self.calls: list[dict] = []
        self.truncation_side = "left"

    def apply_chat_template(self, messages, tokenize=True, add_generation_prompt=True, **kwargs):
        side_at_call = self.truncation_side
        self.calls.append(
            {
                "messages": messages,
                "tokenize": tokenize,
                "add_generation_prompt": add_generation_prompt,
                "kwargs": kwargs,
                "truncation_side": side_at_call,
            }
        )
        rendered = "|".join(f"{message['role']}:{message['content']}" for message in messages)
        if kwargs.get("enable_thinking") is False:
            rendered += "|<think></think>"
        if add_generation_prompt:
            rendered += "|assistant:"
        if not tokenize:
            return rendered
        token_ids = [ord(char) for char in rendered]
        max_length = kwargs.get("max_length")
        if kwargs.get("truncation") and max_length is not None and len(token_ids) > max_length:
            if side_at_call == "right":
                token_ids = token_ids[:max_length]
            else:
                token_ids = token_ids[-max_length:]
        return token_ids


class BatchEncodingLike:
    def __init__(self, input_ids):
        self.input_ids = input_ids


class BatchChatTemplateTokenizer(ChatTemplateTokenizer):
    def apply_chat_template(self, messages, tokenize=True, add_generation_prompt=True, **kwargs):
        token_ids = super().apply_chat_template(
            messages,
            tokenize=tokenize,
            add_generation_prompt=add_generation_prompt,
            **kwargs,
        )
        return BatchEncodingLike(torch.tensor([token_ids], dtype=torch.long))


@pytest.mark.unit
def test_build_sdpo_teacher_rollout_data_preserves_response_suffix_and_alignment():
    from slime.algorithms.sdpo.teacher_alignment import build_sdpo_teacher_rollout_data

    tokenizer = CallableTokenizer({"short teacher": [1, 2, 3], "longer teacher prompt": [4, 5, 6, 7]})
    rollout_data = {
        "tokens": [torch.tensor([10, 11, 101, 102]), torch.tensor([20, 201])],
        "total_lengths": [4, 2],
        "response_lengths": [2, 1],
        "loss_masks": [torch.tensor([1, 0]), torch.tensor([1])],
        "sdpo_teacher_prompt_text": ["short teacher", "longer teacher prompt"],
    }

    teacher_data = build_sdpo_teacher_rollout_data(rollout_data, tokenizer)

    assert [row.tolist() for row in teacher_data["tokens"]] == [[1, 2, 3, 101, 102], [4, 5, 6, 7, 201]]
    assert teacher_data["total_lengths"] == [5, 5]
    assert teacher_data["response_lengths"] == [2, 1]
    assert [row.tolist() for row in teacher_data["loss_masks"]] == [[1, 0], [1]]
    assert [row.tolist() for row in rollout_data["tokens"]] == [[10, 11, 101, 102], [20, 201]]
    assert all(add_special_tokens is False for _, add_special_tokens in tokenizer.calls)


@pytest.mark.unit
def test_build_sdpo_teacher_rollout_data_applies_chat_template_kwargs_and_right_truncation():
    from slime.algorithms.sdpo.teacher_alignment import build_sdpo_teacher_rollout_data

    tokenizer = ChatTemplateTokenizer()
    rollout_data = {
        "tokens": [torch.tensor([10, 11, 101, 102, 103])],
        "total_lengths": [5],
        "response_lengths": [3],
        "loss_masks": [torch.tensor([1, 0, 1])],
        "sdpo_teacher_prompt_text": ["fallback text"],
        "sdpo_teacher_messages": [
            [
                {"role": "system", "content": "SYSTEM RULES"},
                {"role": "user", "content": "0123456789"},
            ]
        ],
    }

    teacher_data = build_sdpo_teacher_rollout_data(
        rollout_data,
        tokenizer,
        apply_chat_template_kwargs={"enable_thinking": False},
        max_prompt_tokens=8,
        truncation_side="right",
    )

    assert tokenizer.truncation_side == "left"
    assert tokenizer.calls[0]["kwargs"] == {"enable_thinking": False, "max_length": 8, "truncation": True}
    assert tokenizer.calls[0]["truncation_side"] == "right"
    assert teacher_data["tokens"][0].tolist() == [
        ord("s"),
        ord("y"),
        ord("s"),
        ord("t"),
        ord("e"),
        ord("m"),
        ord(":"),
        ord("S"),
        101,
        102,
        103,
    ]
    assert teacher_data["response_lengths"] == [3]
    assert teacher_data["loss_masks"][0].tolist() == [1, 0, 1]
    assert teacher_data["sdpo_teacher_prompt_token_lengths"] == [8]
    assert teacher_data["sdpo_teacher_prompt_truncated"] == [1.0]


@pytest.mark.unit
def test_build_sdpo_teacher_rollout_data_accepts_batch_encoding_chat_template_output():
    from slime.algorithms.sdpo.teacher_alignment import build_sdpo_teacher_rollout_data

    tokenizer = BatchChatTemplateTokenizer()
    rollout_data = {
        "tokens": [torch.tensor([10, 11, 101, 102])],
        "total_lengths": [4],
        "response_lengths": [2],
        "loss_masks": [torch.tensor([1, 1])],
        "sdpo_teacher_prompt_text": ["fallback text"],
        "sdpo_teacher_messages": [[{"role": "user", "content": "abc"}]],
    }

    teacher_data = build_sdpo_teacher_rollout_data(rollout_data, tokenizer, max_prompt_tokens=None)

    expected_prompt = [ord(char) for char in "user:abc|assistant:"]
    assert teacher_data["tokens"][0].tolist() == expected_prompt + [101, 102]
    assert teacher_data["sdpo_teacher_prompt_token_lengths"] == [len(expected_prompt)]


@pytest.mark.unit
def test_build_sdpo_teacher_rollout_data_carries_multimodal_train_inputs(monkeypatch):
    from slime.algorithms.sdpo import teacher_alignment

    tokenizer = ChatTemplateTokenizer()

    class FakeProcessor:
        image_processor = SimpleNamespace(patch_size=14)

        def __call__(self, text, **kwargs):
            return {
                "input_ids": [[4, 5, 6]],
                "attention_mask": [[1, 1, 1]],
                "pixel_values": torch.ones((3, 3)),
                "image_grid_thw": torch.tensor([[1, 1, 3]]),
            }

    monkeypatch.setattr(
        teacher_alignment,
        "process_vision_info",
        lambda messages, processor: {"images": ["decoded-image"], "videos": None},
    )
    multimodal_train_inputs = [
        {
            "pixel_values": torch.ones((1, 3, 2, 2)),
            "image_grid_thw": torch.tensor([[1, 2, 2]]),
        }
    ]
    rollout_data = {
        "tokens": [torch.tensor([10, 11, 101, 102])],
        "total_lengths": [4],
        "response_lengths": [2],
        "loss_masks": [torch.tensor([1, 1])],
        "sdpo_teacher_prompt_text": ["fallback text"],
        "sdpo_teacher_messages": [
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Current observation:\n"},
                        {"type": "image", "image": "placeholder"},
                        {"type": "text", "text": "\nGuidance summary:\n- Minimal plan: move."},
                    ],
                }
            ]
        ],
        "multimodal_train_inputs": multimodal_train_inputs,
    }

    teacher_data = teacher_alignment.build_sdpo_teacher_rollout_data(
        rollout_data,
        tokenizer,
        processor=FakeProcessor(),
        max_prompt_tokens=None,
    )

    assert teacher_data["multimodal_train_inputs"] == multimodal_train_inputs
    assert teacher_data["tokens"][0][-2:].tolist() == [101, 102]


@pytest.mark.unit
def test_build_sdpo_teacher_rollout_data_uses_processor_for_multimodal_messages(monkeypatch):
    from slime.algorithms.sdpo import teacher_alignment

    class FakeProcessor:
        def __init__(self):
            self.calls = []
            self.image_processor = SimpleNamespace(patch_size=14)

        def __call__(self, text, **kwargs):
            self.calls.append({"text": text, "kwargs": kwargs})
            assert kwargs["images"] == ["decoded-image"]
            return {
                "input_ids": [[7, 8, 9, 8]],
                "attention_mask": [[1, 1, 1, 1]],
                "pixel_values": torch.ones((4, 3)),
                "image_grid_thw": torch.tensor([[1, 2, 2]]),
            }

    monkeypatch.setattr(
        teacher_alignment,
        "process_vision_info",
        lambda messages, processor: {"images": ["decoded-image"], "videos": None},
    )
    tokenizer = ChatTemplateTokenizer()
    processor = FakeProcessor()
    rollout_data = {
        "tokens": [torch.tensor([10, 11, 101])],
        "total_lengths": [3],
        "response_lengths": [1],
        "loss_masks": [torch.tensor([1])],
        "sdpo_teacher_prompt_text": ["fallback text"],
        "sdpo_teacher_messages": [
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Current observation:\n"},
                        {"type": "image", "image": "data:image/png;base64,abc"},
                    ],
                }
            ]
        ],
    }

    teacher_data = teacher_alignment.build_sdpo_teacher_rollout_data(
        rollout_data,
        tokenizer,
        processor=processor,
        max_prompt_tokens=None,
    )

    assert tokenizer.calls[0]["tokenize"] is False
    assert processor.calls
    assert teacher_data["tokens"][0].tolist() == [7, 8, 9, 8, 101]
    assert teacher_data["sdpo_teacher_prompt_token_lengths"] == [4]


@pytest.mark.unit
def test_multimodal_teacher_prompt_requires_processor():
    from slime.algorithms.sdpo.teacher_alignment import build_sdpo_teacher_rollout_data

    tokenizer = ChatTemplateTokenizer()
    rollout_data = {
        "tokens": [torch.tensor([10, 11, 101])],
        "total_lengths": [3],
        "response_lengths": [1],
        "loss_masks": [torch.tensor([1])],
        "sdpo_teacher_prompt_text": ["fallback text"],
        "sdpo_teacher_messages": [
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Current observation:\n"},
                        {"type": "image", "image": "data:image/png;base64,abc"},
                    ],
                }
            ]
        ],
    }

    with pytest.raises(ValueError, match="processor is required for multimodal SDPO teacher messages"):
        build_sdpo_teacher_rollout_data(rollout_data, tokenizer, max_prompt_tokens=None)


@pytest.mark.unit
def test_multimodal_teacher_prompt_rejects_token_truncation(monkeypatch):
    from slime.algorithms.sdpo import teacher_alignment

    class FakeProcessor:
        image_processor = SimpleNamespace(patch_size=14)

        def __call__(self, text, **kwargs):
            return {
                "input_ids": [[7, 8, 9, 8]],
                "attention_mask": [[1, 1, 1, 1]],
                "pixel_values": torch.ones((4, 3)),
                "image_grid_thw": torch.tensor([[1, 2, 2]]),
            }

    monkeypatch.setattr(
        teacher_alignment,
        "process_vision_info",
        lambda messages, processor: {"images": ["decoded-image"], "videos": None},
    )
    tokenizer = ChatTemplateTokenizer()
    rollout_data = {
        "tokens": [torch.tensor([10, 11, 101])],
        "total_lengths": [3],
        "response_lengths": [1],
        "loss_masks": [torch.tensor([1])],
        "sdpo_teacher_prompt_text": ["fallback text"],
        "sdpo_teacher_messages": [
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Current observation:\n"},
                        {"type": "image", "image": "data:image/png;base64,abc"},
                    ],
                }
            ]
        ],
    }

    with pytest.raises(ValueError, match="multimodal SDPO teacher prompt cannot be token-truncated"):
        teacher_alignment.build_sdpo_teacher_rollout_data(
            rollout_data,
            tokenizer,
            processor=FakeProcessor(),
            max_prompt_tokens=3,
        )


@pytest.mark.unit
def test_tokenize_sdpo_teacher_prompt_rejects_rank3_chat_template_tokens():
    from slime.algorithms.sdpo.teacher_alignment import tokenize_sdpo_teacher_prompt

    class Rank3ChatTemplateTokenizer:
        def apply_chat_template(self, messages, tokenize=True, add_generation_prompt=True, **kwargs):
            return BatchEncodingLike(torch.tensor([[[1, 2], [3, 4]]], dtype=torch.long))

    with pytest.raises(ValueError, match="one input_ids row"):
        tokenize_sdpo_teacher_prompt(
            Rank3ChatTemplateTokenizer(),
            "fallback",
            messages=[{"role": "user", "content": "abc"}],
        )


@pytest.mark.unit
def test_tokenize_sdpo_teacher_prompt_rejects_deeply_nested_chat_template_tokens():
    from slime.algorithms.sdpo.teacher_alignment import tokenize_sdpo_teacher_prompt

    class NestedListChatTemplateTokenizer:
        def apply_chat_template(self, messages, tokenize=True, add_generation_prompt=True, **kwargs):
            return BatchEncodingLike([[[1, 2, 3]]])

    with pytest.raises(ValueError, match="one input_ids row"):
        tokenize_sdpo_teacher_prompt(
            NestedListChatTemplateTokenizer(),
            "fallback",
            messages=[{"role": "user", "content": "abc"}],
        )


@pytest.mark.unit
def test_align_sdpo_teacher_log_probs_slices_response_suffix_and_rejects_mismatch():
    from slime.algorithms.sdpo.teacher_alignment import align_sdpo_teacher_log_probs

    aligned = align_sdpo_teacher_log_probs(
        sdpo_teacher_log_probs=[torch.tensor([-9.0, -8.0, -0.1, -0.2]), torch.tensor([-7.0, -0.3])],
        sdpo_teacher_total_lengths=[4, 2],
        sdpo_teacher_response_lengths=[2, 1],
        student_response_lengths=[2, 1],
        student_loss_masks=[torch.tensor([1, 0]), torch.tensor([1])],
    )

    assert aligned[0].tolist() == pytest.approx([-0.1, -0.2])
    assert aligned[1].tolist() == pytest.approx([-0.3])

    with pytest.raises(ValueError, match="response length mismatch"):
        align_sdpo_teacher_log_probs(
            sdpo_teacher_log_probs=[torch.tensor([-9.0, -8.0, -0.1])],
            sdpo_teacher_total_lengths=[3],
            sdpo_teacher_response_lengths=[1],
            student_response_lengths=[2],
            student_loss_masks=[torch.tensor([1, 1])],
        )


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
