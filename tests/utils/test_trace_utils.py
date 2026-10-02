# Modified for ReSAIL. See NOTICE and LICENSE for attribution and terms.
import pytest

from slime.utils.trace_utils import build_sglang_meta_trace_attrs


@pytest.mark.unit
def test_build_sglang_meta_trace_attrs_keeps_standard_and_pd_fields():
    meta = {
        "prompt_tokens": 12,
        "completion_tokens": 7,
        "cached_tokens": 3,
        "pd_prefill_forward_duration": 0.125,
        "pd_decode_transfer_duration": None,
        "finish_reason": {"type": "stop"},
        "unused_field": "ignored",
    }

    assert build_sglang_meta_trace_attrs(meta) == {
        "prompt_tokens": 12,
        "completion_tokens": 7,
        "cached_tokens": 3,
        "pd_prefill_forward_duration": 0.125,
        "finish_reason": "stop",
    }
