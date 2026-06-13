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
