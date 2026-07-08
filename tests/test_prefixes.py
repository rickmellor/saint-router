import pytest

from saint.prefixes import ParsedPrefixes, UnknownPrefixError, parse_prefixes

URGENCIES = {"urgent", "patient", "normal"}
BACKENDS = {"cloud-large": {"opus", "claude"}, "local-coder": {"coder"}, "local-small": set()}


def test_no_prefix():
    out = parse_prefixes("hello world", URGENCIES, BACKENDS)
    assert out == ParsedPrefixes(urgency=None, pinned_backend=None, stripped="hello world", raw="")


def test_urgency_only():
    out = parse_prefixes("!urgent fix bug", URGENCIES, BACKENDS)
    assert out.urgency == "urgent"
    assert out.pinned_backend is None
    assert out.stripped == "fix bug"
    assert out.raw == "!urgent"


def test_backend_alias():
    out = parse_prefixes("!opus refactor", URGENCIES, BACKENDS)
    assert out.pinned_backend == "cloud-large"
    assert out.stripped == "refactor"


def test_combined():
    out = parse_prefixes("!urgent !opus do this", URGENCIES, BACKENDS)
    assert out.urgency == "urgent"
    assert out.pinned_backend == "cloud-large"
    assert out.stripped == "do this"


def test_last_urgency_wins():
    out = parse_prefixes("!urgent !patient go", URGENCIES, BACKENDS)
    assert out.urgency == "patient"
    assert out.stripped == "go"


def test_leading_whitespace_disables():
    out = parse_prefixes(" !opus please", URGENCIES, BACKENDS)
    assert out.urgency is None
    assert out.pinned_backend is None
    assert out.stripped == " !opus please"


def test_unknown_token_raises():
    with pytest.raises(UnknownPrefixError) as e:
        parse_prefixes("!doesnotexist hi", URGENCIES, BACKENDS)
    assert "doesnotexist" in str(e.value)


def test_case_sensitive():
    with pytest.raises(UnknownPrefixError):
        parse_prefixes("!URGENT hi", URGENCIES, BACKENDS)


def test_empty_string():
    out = parse_prefixes("", URGENCIES, BACKENDS)
    assert out.urgency is None
    assert out.stripped == ""


def test_lone_bang_with_space_is_not_a_prefix():
    out = parse_prefixes("! hello", URGENCIES, BACKENDS)
    assert out.urgency is None
    assert out.pinned_backend is None
    assert out.stripped == "! hello"
    assert out.raw == ""


def test_lone_bang_alone():
    out = parse_prefixes("!", URGENCIES, BACKENDS)
    assert out.urgency is None
    assert out.pinned_backend is None
    assert out.stripped == "!"
    assert out.raw == ""


def test_bang_inside_message_after_prefix_not_reparsed():
    out = parse_prefixes("!urgent run !opus thing", URGENCIES, BACKENDS)
    assert out.urgency == "urgent"
    assert out.pinned_backend is None
    assert out.stripped == "run !opus thing"


def test_at_sigil_urgency():
    out = parse_prefixes("@patient review this code", URGENCIES, BACKENDS)
    assert out.urgency == "patient"
    assert out.stripped == "review this code"
    assert out.raw == "@patient"


def test_at_sigil_backend_alias():
    out = parse_prefixes("@coder fix the loop", URGENCIES, BACKENDS)
    assert out.pinned_backend == "local-coder"
    assert out.stripped == "fix the loop"


def test_at_sigil_unknown_token_is_plain_text():
    # '@rick check this' is a mention, not a typo'd prefix — must not raise
    out = parse_prefixes("@rick check this", URGENCIES, BACKENDS)
    assert out.urgency is None and out.pinned_backend is None
    assert out.stripped == "@rick check this"
    assert out.raw == ""


def test_mixed_sigils():
    out = parse_prefixes("!urgent @coder do this", URGENCIES, BACKENDS)
    assert out.urgency == "urgent"
    assert out.pinned_backend == "local-coder"
    assert out.stripped == "do this"
    assert out.raw == "!urgent @coder"


def test_prefix_then_unknown_at_token_stays_in_message():
    out = parse_prefixes("@patient @rick take a look", URGENCIES, BACKENDS)
    assert out.urgency == "patient"
    assert out.stripped == "@rick take a look"


def test_bang_unknown_token_still_raises():
    with pytest.raises(UnknownPrefixError):
        parse_prefixes("!patinet oops", URGENCIES, BACKENDS)
