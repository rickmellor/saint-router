# ABOUTME: SseUsageTracker reads usage from message_start AND message_delta.
# ABOUTME: Regression: litellm's vLLM bridge only reports input_tokens in message_delta.
import json

from saint.anthropic_api import SseUsageTracker


def _ev(t, payload):
    return f"event: {t}\ndata: {json.dumps({'type': t, **payload})}\n\n".encode()


def test_input_tokens_from_message_delta_when_message_start_reports_zero():
    tr = SseUsageTracker()
    tr.feed(_ev("message_start", {"message": {"usage": {
        "input_tokens": 0, "output_tokens": 0,
        "cache_creation_input_tokens": 0, "cache_read_input_tokens": 0}}}))
    tr.feed(_ev("message_delta", {"delta": {}, "usage": {"input_tokens": 58, "output_tokens": 20}}))
    assert tr.usage["input_tokens"] == 58
    assert tr.usage["output_tokens"] == 20


def test_anthropic_shape_keeps_message_start_counts():
    tr = SseUsageTracker()
    tr.feed(_ev("message_start", {"message": {"usage": {
        "input_tokens": 12, "output_tokens": 1,
        "cache_creation_input_tokens": 300, "cache_read_input_tokens": 9000}}}))
    tr.feed(_ev("message_delta", {"delta": {}, "usage": {"output_tokens": 77}}))
    assert tr.usage == {"input_tokens": 12, "output_tokens": 77,
                        "cache_creation_input_tokens": 300, "cache_read_input_tokens": 9000}
