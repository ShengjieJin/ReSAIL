from slime_plugins.agent_tasks.alfworld.generate import _build_action_content_token_metadata
from slime_plugins.agent_tasks.common.algorithms.sdpo import _add_sgs_fields


class CharacterTokenizer:
    def __call__(self, text, *, add_special_tokens, return_offsets_mapping):
        assert not add_special_tokens and return_offsets_mapping
        return {
            "input_ids": [ord(char) for char in text],
            "offset_mapping": [(index, index + 1) for index in range(len(text))],
        }


def test_action_mask_selects_trimmed_content_only():
    text = "<think>x</think><action>  open door  </action>"
    token_ids = [ord(char) for char in text]
    result = _build_action_content_token_metadata(
        CharacterTokenizer(), response_text=text, response_ids=token_ids, format_valid=True
    )

    selected = "".join(char for char, keep in zip(text, result["sgs_action_token_mask"]) if keep)
    assert result["sgs_action_alignment_valid"] is True
    assert selected == "open door"


def test_invalid_format_is_retained_as_unavailable():
    text = "<action>open door</action>"
    result = _build_action_content_token_metadata(
        CharacterTokenizer(), response_text=text, response_ids=[ord(char) for char in text], format_valid=False
    )

    assert result["sgs_action_alignment_valid"] is False
    assert result["sgs_action_alignment_reason"] == "response_format_invalid"
    assert not any(result["sgs_action_token_mask"])


def test_boundary_straddling_token_is_not_selected():
    text = "<think>x</think><action>open</action>"

    class StraddlingTokenizer:
        def __call__(self, text, *, add_special_tokens, return_offsets_mapping):
            return {"input_ids": [1], "offset_mapping": [(0, len(text))]}

    result = _build_action_content_token_metadata(
        StraddlingTokenizer(), response_text=text, response_ids=[1], format_valid=True
    )
    assert result["sgs_action_alignment_valid"] is False
    assert result["sgs_action_alignment_reason"] == "no_contained_action_content_tokens"


def test_teacher_view_fields_promote_plain_context_and_audit_metadata():
    metadata = {
        "sdpo_current_prompt_text": "plain",
        "sdpo_current_raw_prompt": [{"role": "user", "content": "plain"}],
        "sgs_action_token_mask": [0, 1],
        "sgs_action_alignment_valid": True,
        "sgs_action_alignment_reason": None,
        "sgs_action_match": False,
        "sgs_online_action": "open door",
        "sgs_online_format_valid": True,
        "sgs_frozen_action": "close door",
        "sgs_source_draw_id": 7,
        "sgs_task_id": "pick_and_place/task",
        "sgs_split": "train",
    }
    train_data = {}
    _add_sgs_fields(train_data, [{"sdpo_metadata": metadata}])

    assert train_data["sgs_plain_prompt_text"] == ["plain"]
    assert train_data["sgs_plain_messages"] == [[{"role": "user", "content": "plain"}]]
    assert train_data["sgs_action_token_mask"] == [[0, 1]]
    assert train_data["sgs_action_match"] == [False]
