from saint.thinking import should_strip, strip_signed_thinking


def test_should_strip_logic():
    # target validates signatures + continuity unconfirmed → strip
    assert should_strip("anthropic", None, "bedrock-sonnet")
    assert should_strip("bedrock", "bedrock-opus", "bedrock-sonnet")  # switched
    # same backend → keep
    assert not should_strip("bedrock", "bedrock-sonnet", "bedrock-sonnet")
    # openai/local target never validates signatures → never strip
    assert not should_strip("openai", None, "local-coder")


def test_strip_typed_content_blocks():
    msgs = [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": [
            {"type": "thinking", "thinking": "hmm", "signature": "sig"},
            {"type": "text", "text": "answer"},
        ]},
    ]
    out = strip_signed_thinking(msgs)
    assert out[1]["content"] == [{"type": "text", "text": "answer"}]
    assert msgs[1]["content"][0]["type"] == "thinking"  # original untouched


def test_strip_top_level_keys():
    msgs = [
        {"role": "assistant", "content": "answer", "thinking_blocks": [{"x": 1}],
         "reasoning_content": "chain"},
    ]
    out = strip_signed_thinking(msgs)
    assert "thinking_blocks" not in out[0] and "reasoning_content" not in out[0]
    assert out[0]["content"] == "answer"


def test_strip_both_shapes_and_redacted():
    msgs = [{"role": "assistant", "thinking_blocks": [{"x": 1}], "content": [
        {"type": "redacted_thinking", "data": "enc"},
        {"type": "text", "text": "hi"},
    ]}]
    out = strip_signed_thinking(msgs)
    assert "thinking_blocks" not in out[0]
    assert out[0]["content"] == [{"type": "text", "text": "hi"}]


def test_nothing_to_strip_returns_same_object():
    msgs = [{"role": "user", "content": "hi"},
            {"role": "assistant", "content": "no thinking here"}]
    assert strip_signed_thinking(msgs) is msgs  # identity when unchanged


def test_user_messages_untouched():
    msgs = [{"role": "user", "content": [{"type": "thinking", "thinking": "x"}]}]
    # only assistant thinking is signed; user content is left alone
    assert strip_signed_thinking(msgs) is msgs
