import copy

from saint.backends import inject_cache_control
from saint.route_cache import TTLCache, conversation_key, turn_key


# --- TTLCache ---
def test_ttlcache_hit_miss_and_expiry(monkeypatch):
    clock = [1000.0]
    monkeypatch.setattr("saint.route_cache.time.monotonic", lambda: clock[0])
    c = TTLCache(ttl_s=10.0, max_entries=4)
    assert c.get("a") is None
    c.set("a", 1)
    assert c.get("a") == 1
    clock[0] += 9.9
    assert c.get("a") == 1
    clock[0] += 0.2
    assert c.get("a") is None  # expired + evicted
    assert len(c) == 0


def test_ttlcache_lru_eviction(monkeypatch):
    clock = [0.0]
    monkeypatch.setattr("saint.route_cache.time.monotonic", lambda: clock[0])
    c = TTLCache(ttl_s=100.0, max_entries=2)
    c.set("a", 1)
    c.set("b", 2)
    assert c.get("a") == 1  # refreshes recency of 'a'
    c.set("c", 3)           # evicts 'b' (least recently used)
    assert c.get("b") is None
    assert c.get("a") == 1 and c.get("c") == 3


def test_ttlcache_set_refreshes_ttl(monkeypatch):
    clock = [0.0]
    monkeypatch.setattr("saint.route_cache.time.monotonic", lambda: clock[0])
    c = TTLCache(ttl_s=10.0, max_entries=4)
    c.set("a", 1)
    clock[0] += 8
    c.set("a", 2)  # sliding: re-set extends expiry
    clock[0] += 8
    assert c.get("a") == 2


# --- keys ---
def test_turn_key_varies_by_text_and_urgency():
    assert turn_key("fix it", "normal") == turn_key("fix it", "normal")
    assert turn_key("fix it", "normal") != turn_key("fix it", "urgent")
    assert turn_key("fix it", "normal") != turn_key("fix that", "normal")


def test_conversation_key_stable_as_turns_append():
    base = [{"role": "system", "content": "agent"}, {"role": "user", "content": "explore the repo"}]
    grown = base + [{"role": "assistant", "content": "which one?"},
                    {"role": "user", "content": "yes, sorry. saint-router"}]
    assert conversation_key(base, None) == conversation_key(grown, None)


def test_conversation_key_session_id_wins():
    m1 = [{"role": "user", "content": "one thing"}]
    m2 = [{"role": "user", "content": "another thing"}]
    assert conversation_key(m1, "sess-9") == conversation_key(m2, "sess-9")
    assert conversation_key(m1, None) != conversation_key(m2, None)


def test_conversation_key_none_without_non_system_message():
    assert conversation_key([{"role": "system", "content": "agent"}], None) is None
    assert conversation_key([], None) is None


def test_conversation_key_multimodal_first_message_uses_text_parts():
    m = [{"role": "user", "content": [{"type": "text", "text": "hi"},
                                       {"type": "image_url", "image_url": {"url": "x"}}]}]
    assert conversation_key(m, None) == conversation_key(
        [{"role": "user", "content": [{"type": "text", "text": "hi"}]}], None)


# --- inject_cache_control ---
def _big(text: str, n: int = 5000) -> str:
    return (text + " ") * (n // (len(text) + 1) + 1)


def test_inject_marks_system_and_last_message_without_mutation():
    messages = [
        {"role": "system", "content": _big("you are an agent")},
        {"role": "user", "content": "review this"},
        {"role": "assistant", "content": "ok, looking"},
        {"role": "user", "content": "and the tests too"},
    ]
    snapshot = copy.deepcopy(messages)
    out, out_tools = inject_cache_control(messages, None, min_chars=4000)
    assert messages == snapshot  # originals untouched
    assert out[0]["content"][0]["cache_control"] == {"type": "ephemeral"}
    assert out[3]["content"][0]["cache_control"] == {"type": "ephemeral"}
    assert out[1] is messages[1] and out[2] is messages[2]  # non-targets pass by reference
    assert out_tools is None


def test_inject_below_min_chars_is_identity():
    messages = [{"role": "user", "content": "short"}]
    out, out_tools = inject_cache_control(messages, [{"name": "t"}], min_chars=4000)
    assert out is messages and out_tools == [{"name": "t"}]


def test_inject_list_content_marks_last_text_block_only():
    messages = [{"role": "user", "content": [
        {"type": "text", "text": _big("context")},
        {"type": "image_url", "image_url": {"url": "x"}},
        {"type": "text", "text": "the question"},
    ]}]
    out, _ = inject_cache_control(messages, None, min_chars=100)
    blocks = out[0]["content"]
    assert "cache_control" not in blocks[0]
    assert "cache_control" not in blocks[1]
    assert blocks[2]["cache_control"] == {"type": "ephemeral"}


def test_inject_skips_textless_tail_messages():
    messages = [
        {"role": "system", "content": _big("agent")},
        {"role": "user", "content": "do a thing"},
        {"role": "assistant", "content": None, "tool_calls": [{"id": "x"}]},
    ]
    out, _ = inject_cache_control(messages, None, min_chars=100)
    assert out[2] is messages[2]  # content=None tail untouched
    assert out[1]["content"][0]["cache_control"] == {"type": "ephemeral"}


def test_inject_tools_breakpoint_only_without_system():
    tools = [{"name": "a"}, {"name": "b"}]
    messages = [{"role": "user", "content": _big("no system here")}]
    out, out_tools = inject_cache_control(messages, tools, min_chars=100)
    assert out_tools[-1]["cache_control"] == {"type": "ephemeral"}
    assert "cache_control" not in tools[-1]  # copy, not mutation
    # with a system message, tools are left alone (system breakpoint covers them)
    messages2 = [{"role": "system", "content": _big("sys")}, {"role": "user", "content": "q"}]
    _, out_tools2 = inject_cache_control(messages2, tools, min_chars=100)
    assert out_tools2 is tools


def test_inject_one_hour_ttl_propagates():
    messages = [{"role": "user", "content": _big("x")}]
    out, _ = inject_cache_control(messages, None, min_chars=100, ttl="1h")
    assert out[0]["content"][0]["cache_control"] == {"type": "ephemeral", "ttl": "1h"}
