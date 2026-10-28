# ABOUTME: toolwatch extracts the hosts tool calls name and records rows, both APIs.
# ABOUTME: Passive: responses must be byte-identical with toolwatch enabled vs disabled.
import json
import threading
import time
from dataclasses import replace
from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient

from saint.config import ToolwatchConfig
from saint.server import build_app
from saint.toolwatch import (
    AnthropicToolTracker,
    OpenAIToolTracker,
    ToolWatch,
    emit,
    extract_hosts,
    scan_tool_input,
)
from tests.test_router import _cfg

_CURL = "curl https://oss-cn-beijing.aliyuncs.com/bucket/x"
# Per-request metadata SAINT itself generates — excluded from the enabled/disabled comparison.
_VOLATILE_HEADERS = {"x-saint-request-id", "content-length", "content-type"}


# ---------------------------------------------------------------- extract_hosts

def test_url_hosts_with_port_userinfo_and_path():
    assert extract_hosts("curl http://admin:pw@backup.example.net:8443/a/b?c=1") == ["backup.example.net"]
    assert extract_hosts("get https://a.example.com/x") == ["a.example.com"]
    assert extract_hosts("u = 'https://q.example.com/x'") == ["q.example.com"]


def test_ipv4_literal():
    assert extract_hosts("nc -lv 192.168.1.7 4444") == ["192.168.1.7"]


def test_ipv6_url_host():
    assert extract_hosts("wget ftp://user@[fd00::1]:21/pub") == ["fd00::1"]


def test_bare_host_order_dedupe_and_trailing_dot():
    assert extract_hosts("ping Host.Example.COM. then api.example.org then host.example.com") == \
        ["host.example.com", "api.example.org"]


def test_ext_and_attribute_stoplist():
    assert extract_hosts("python run.py") == []
    assert extract_hosts("import os.path; print(os.path.join(x))") == []
    assert extract_hosts("cat /etc/foo/bar.sh") == []
    assert extract_hosts("systemctl restart nginx.service") == []
    # Last label not on the stoplist: a code attribute slips through. Accepted noise — the
    # database side decides what is external or watched.
    assert extract_hosts("torch.linalg.norm(x)") == ["torch.linalg.norm"]


def test_email_is_not_a_host_but_a_url_with_the_same_domain_is():
    assert extract_hosts("mail user@example.com about it") == []
    assert extract_hosts("curl https://example.com/login") == ["example.com"]


def test_max_64_hosts():
    hosts = extract_hosts(" ".join(f"h{i}.example.com" for i in range(70)))
    assert len(hosts) == 64
    assert hosts[0] == "h0.example.com" and hosts[-1] == "h63.example.com"


# ------------------------------------------------------------- scan_tool_input

def test_scan_dict_command_and_url():
    assert scan_tool_input("Bash", {"command": _CURL}) == (_CURL, ["oss-cn-beijing.aliyuncs.com"])
    assert scan_tool_input("WebFetch", {"url": "https://a.example.com/x"})[1] == ["a.example.com"]


def test_scan_recurses_into_nested_dicts_and_lists():
    inp = {"todos": [{"note": "see http://deep.example.org"}], "meta": {"x": {"y": "d.example.net"}}}
    assert scan_tool_input("Task", inp)[1] == ["deep.example.org", "d.example.net"]


def test_scan_openai_json_string_and_broken_json():
    assert scan_tool_input("Bash", json.dumps({"command": _CURL}))[1] == ["oss-cn-beijing.aliyuncs.com"]
    excerpt, hosts = scan_tool_input("Bash", '{"command": "curl http://a.example')
    assert hosts == ["a.example"] and "curl" in excerpt


def test_scan_excerpt_is_command_then_url_then_json_and_truncates():
    assert scan_tool_input("B", {"url": "https://a.example.com", "command": "ls"})[0] == "ls"
    assert json.loads(scan_tool_input("B", {"a": 1, "b": "z"})[0]) == {"a": 1, "b": "z"}
    assert scan_tool_input("B", {"command": "x" * 500}, 40)[0] == "x" * 40


def test_keep_rule_needs_a_host_or_a_shellish_tool():
    tw, rows = _capture(ToolwatchConfig(enabled=True))
    emit(tw, api="anthropic", attr={}, client="pi", peer="127.0.0.1", backend="local-coder",
         model="m", calls=[("Read", {"file_path": "/etc/hosts"}),
                           ("Bash", {"command": "ls -la"}),
                           ("WebSearch", {"query": "hello"})])
    assert _until(lambda: len(rows) == 1)
    assert [r["tool_name"] for r in rows] == ["Bash"]   # Read: no host, not a shell; WebSearch: no host


# -------------------------------------------------------- AnthropicToolTracker

def _ev(t, payload):
    return f"event: {t}\ndata: {json.dumps({'type': t, **payload})}\n\n".encode()


def _tool_stream():
    return [
        _ev("message_start", {"message": {"usage": {"input_tokens": 5, "output_tokens": 1}}}),
        _ev("content_block_start", {"index": 0, "content_block": {"type": "text", "text": ""}}),
        _ev("content_block_delta", {"index": 0, "delta": {"type": "text_delta", "text": "hi"}}),
        _ev("content_block_stop", {"index": 0}),
        _ev("content_block_start", {"index": 1, "content_block": {
            "type": "tool_use", "id": "tu1", "name": "Bash", "input": {}}}),
        _ev("content_block_delta", {"index": 1, "delta": {
            "type": "input_json_delta", "partial_json": '{"command": '}}),
        _ev("content_block_delta", {"index": 1, "delta": {
            "type": "input_json_delta", "partial_json": json.dumps(_CURL)[:-1]}}),
        _ev("content_block_delta", {"index": 1, "delta": {
            "type": "input_json_delta", "partial_json": '"}'}}),
        _ev("content_block_stop", {"index": 1}),
        _ev("content_block_start", {"index": 2, "content_block": {
            "type": "tool_use", "id": "tu2", "name": "Read", "input": {}}}),
        _ev("content_block_delta", {"index": 2, "delta": {
            "type": "input_json_delta", "partial_json": '{"file_path": "/x/y'}}),
        _ev("content_block_stop", {"index": 2}),
        _ev("message_delta", {"delta": {"stop_reason": "tool_use"}, "usage": {"output_tokens": 9}}),
        _ev("message_stop", {}),
    ]


def test_anthropic_tracker_accumulates_fragments_and_survives_broken_json():
    tr = AnthropicToolTracker()
    for chunk in _tool_stream():
        tr.feed(chunk)
    assert len(tr.calls) == 2
    assert tr.calls[0][0] == "Bash"
    assert tr.calls[0][1] == {"command": _CURL}      # three fragments, one parse
    assert tr.calls[1][0] == "Read"
    assert isinstance(tr.calls[1][1], str)           # broken JSON -> raw text, scanned later
    assert scan_tool_input("Read", tr.calls[1][1])[0].startswith('{"file_path')


def test_anthropic_tracker_chunk_boundaries_mid_line_and_garbage():
    stream = b"".join(_tool_stream())
    tr = AnthropicToolTracker()
    for i in range(0, len(stream), 7):
        tr.feed(stream[i:i + 7])                     # splits inside event names and JSON
    assert [n for n, _ in tr.calls] == ["Bash", "Read"]
    tr2 = AnthropicToolTracker()
    tr2.feed(b"event: content_block_start\ndata: {not json}\n\n")
    tr2.feed(b"\n\n")
    tr2.feed(None)
    assert tr2.calls == []


# ----------------------------------------------------------- OpenAIToolTracker

def _tc_chunk(idx, name=None, args=None, finish=None):
    tc = {"index": idx}
    if name is not None:
        tc["function"] = {"name": name}
    if args is not None:
        tc.setdefault("function", {})["arguments"] = args
    return {"id": "x", "object": "chat.completion.chunk", "created": 0, "model": "m",
            "choices": [{"index": 0, "delta": {"tool_calls": [tc]}, "finish_reason": finish}]}


def test_openai_tracker_fragments_and_parallel_calls_by_index():
    tr = OpenAIToolTracker()
    tr.feed_chunk(_tc_chunk(0, name="Bash"))
    tr.feed_chunk(_tc_chunk(0, args='{"command": "cur'))
    tr.feed_chunk(_tc_chunk(1, name="Read"))
    tr.feed_chunk(_tc_chunk(0, args='l https://a.example.com/x"}'))
    tr.feed_chunk(_tc_chunk(1, args='{"file_path": "/x"}'))
    tr.feed_chunk({"choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls"}]})
    calls = dict(tr.finish())
    assert json.loads(calls["Bash"]) == {"command": "curl https://a.example.com/x"}
    assert json.loads(calls["Read"]) == {"file_path": "/x"}


def test_openai_tracker_tolerates_garbage():
    tr = OpenAIToolTracker()
    tr.feed_chunk(None)
    tr.feed_chunk({"choices": "nope"})
    tr.feed_chunk({"choices": [{"delta": {"tool_calls": [None, "x"]}}]})
    assert tr.finish() == []


# --------------------------------------------------------------------- the sink

def _capture(cfg):
    """A ToolWatch whose rows land in a test-owned list."""
    rows: list[dict] = []
    return ToolWatch(cfg, "dsn-unused", writer=rows.extend), rows


def _until(pred, timeout=5.0):
    """Poll `pred` until true (the sink's drain thread is asynchronous by design)."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if pred():
            return True
        time.sleep(0.01)
    return pred()


def test_sink_disabled_records_nothing():
    tw, rows = _capture(ToolwatchConfig(enabled=False))
    emit(tw, api="anthropic", attr={}, client="pi", peer="127.0.0.1", backend="b", model="m",
         calls=[("Bash", {"command": _CURL})])
    time.sleep(0.2)
    assert rows == [] and tw.stats() == {"queued": 0, "written": 0, "dropped": 0, "enabled": False}


def test_sink_without_dsn_counts_drops():
    tw = ToolWatch(ToolwatchConfig(enabled=True), None)
    emit(tw, api="anthropic", attr={}, client="pi", peer="127.0.0.1", backend="b", model="m",
         calls=[("Bash", {"command": _CURL})])
    assert tw.stats() == {"queued": 0, "written": 0, "dropped": 1, "enabled": True}


def test_sink_full_queue_drops_without_raising():
    """A sink slower than the request path sheds rows instead of blocking the router."""
    block = threading.Event()

    def writer(batch):
        block.wait(5)

    tw = ToolWatch(ToolwatchConfig(enabled=True, queue_size=1), "dsn-unused", writer=writer)
    try:
        for i in range(60):
            emit(tw, api="anthropic", attr={}, client=None, peer=None, backend=None, model=None,
                 calls=[("Bash", {"command": f"curl http://h{i}.example.com"})])
        assert tw.stats()["dropped"] > 0        # queue_size 1 + a blocked writer
        block.set()
        assert _until(lambda: tw.stats()["queued"] == 0, timeout=8.0)
        stats = tw.stats()
        assert stats["written"] + stats["dropped"] == 60   # every row accounted for once
    finally:
        block.set()
        tw.close()


def test_sink_writer_raises_is_swallowed_and_the_thread_survives():
    state = {"n": 0}

    def writer(batch):
        state["n"] += 1
        if state["n"] == 1:
            raise RuntimeError("netwatch down")

    tw = ToolWatch(ToolwatchConfig(enabled=True), "dsn-unused", writer=writer)
    emit(tw, api="anthropic", attr={}, client=None, peer=None, backend=None, model=None,
         calls=[("Bash", {"command": "curl http://one.example.com"})])
    assert _until(lambda: tw.stats()["dropped"] == 1)
    emit(tw, api="anthropic", attr={}, client=None, peer=None, backend=None, model=None,
         calls=[("Bash", {"command": "curl http://two.example.com"})])
    assert _until(lambda: tw.stats()["written"] == 1)
    tw.close()


# ------------------------------------------------------------------- endpoints

def _app(tmp_path, cfg=None, capture=True):
    """The app under test, plus the list the sink writes into (capture=False -> disabled sink)."""
    cfg = cfg or _cfg()
    app = build_app(cfg, db_path=tmp_path / "log.sqlite")
    rows: list[dict] = []
    if capture:
        app.state.toolwatch = ToolWatch(cfg.toolwatch, "dsn-unused", writer=rows.extend)
    return app, TestClient(app), rows


def _tw_cfg(enabled=True):
    return replace(_cfg(), toolwatch=ToolwatchConfig(enabled=enabled))


_ANTHROPIC_TOOL = {
    "id": "msg_01", "type": "message", "role": "assistant", "model": "m",
    "content": [{"type": "tool_use", "id": "tu1", "name": "Bash",
                 "input": {"command": _CURL}}],
    "stop_reason": "tool_use",
    "usage": {"input_tokens": 10, "output_tokens": 5},
}

_OPENAI_TOOL = {
    "id": "x", "object": "chat.completion", "created": 0, "model": "m",
    "choices": [{"index": 0, "finish_reason": "tool_calls", "message": {
        "role": "assistant", "content": None,
        "tool_calls": [{"id": "c1", "type": "function",
                        "function": {"name": "Bash",
                                     "arguments": json.dumps({"command": _CURL})}}]}}],
    "usage": {"prompt_tokens": 5, "completion_tokens": 3},
}


class _Bytes:
    """Async iterator of pre-canned chunks (litellm's stream shape for both APIs)."""

    def __init__(self, chunks):
        self._c = list(chunks)

    def __aiter__(self):
        return self

    async def __anext__(self):
        if not self._c:
            raise StopAsyncIteration
        return self._c.pop(0)


def _snapshot(resp, raw=None):
    return ({k: v for k, v in resp.headers.items() if k.lower() not in _VOLATILE_HEADERS},
            resp.status_code,
            raw if raw is not None else resp.content)


_ATTR_HEADERS = {"X-Inference-Origin": "specul8", "X-Inference-Session": "sess-7",
                 "X-Inference-Job": "job-42", "User-Agent": "claude-cli/2.0 (specul8)"}


def _messages_requests(client):
    out = []
    for body, headers in (
        ({"model": "saint-cloud-large", "max_tokens": 100,
          "messages": [{"role": "user", "content": "hi"}]}, {}),
        ({"model": "saint-cloud-large", "max_tokens": 100,
          "messages": [{"role": "user", "content": "hi"}]}, _ATTR_HEADERS),
        ({"model": "saint-cloud-large", "max_tokens": 100, "stream": True,
          "messages": [{"role": "user", "content": "hi"}]}, {}),
        ({"model": "saint-cloud-large", "max_tokens": 100, "stream": True,
          "messages": [{"role": "user", "content": "hi"}]}, _ATTR_HEADERS),
    ):
        if body.get("stream"):
            with patch("saint.backends.litellm.anthropic.messages.acreate",
                       AsyncMock(return_value=_Bytes(_tool_stream()))):
                with client.stream("POST", "/v1/messages", json=body, headers=headers) as r:
                    out.append((_snapshot(r, b"".join(r.iter_bytes())), r))
        else:
            with patch("saint.backends.litellm.anthropic.messages.acreate",
                       AsyncMock(return_value=_ANTHROPIC_TOOL)):
                r = client.post("/v1/messages", json=body, headers=headers)
                out.append((_snapshot(r), r))
    return out


def _chat_requests(client):
    """(snapshot, response) for the four shapes: non-stream/stream x with/without attribution."""
    out = []
    for stream in (False, True):
        for headers in ({}, _ATTR_HEADERS):
            body = {"model": "saint-cloud-large",
                    "messages": [{"role": "user", "content": "hi"}]}
            if stream:
                body["stream"] = True
                chunks = [_tc_chunk(0, name="Bash"), _tc_chunk(0, args=json.dumps({"command": _CURL})),
                          {"id": "x", "object": "chat.completion.chunk", "created": 0, "model": "m",
                           "choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls"}]}]
                with patch("saint.backends.litellm.acompletion",
                           AsyncMock(return_value=_Bytes(chunks))):
                    with client.stream("POST", "/v1/chat/completions", json=body,
                                       headers=headers) as r:
                        raw = b"".join(r.iter_bytes())
                        out.append((_snapshot(r, raw), r))
            else:
                with patch("saint.backends.litellm.acompletion",
                           AsyncMock(return_value=_OPENAI_TOOL)):
                    r = client.post("/v1/chat/completions", json=body, headers=headers)
                    out.append((_snapshot(r), r))
    return out


def _assert_rows(rows, api, n=4):
    assert len(rows) == n
    assert all(r["api"] == api and r["tool_name"] == "Bash"
               and r["hosts"] == ["oss-cn-beijing.aliyuncs.com"]
               and r["backend"] == "cloud-large" and r["peer"] == "testclient"
               and r["ts"].endswith("+00:00") for r in rows)
    assert {r["client"] for r in rows} <= {"claude-cli", "python-httpx", "testclient"}


def test_messages_endpoint_records_tool_calls(tmp_path):
    _, client, rows = _app(tmp_path, _tw_cfg())
    for _, r in _messages_requests(client):
        assert r.status_code == 200
    assert _until(lambda: len(rows) == 4)
    _assert_rows(rows, "anthropic")
    assert [r["origin"] for r in rows] == [None, "specul8", None, "specul8"]
    assert [r["session"] for r in rows] == [None, "sess-7", None, "sess-7"]
    assert [r["job_id"] for r in rows] == [None, "job-42", None, "job-42"]
    assert rows[0]["model"] == "saint-cloud-large"


def test_chat_completions_endpoint_records_tool_calls(tmp_path):
    _, client, rows = _app(tmp_path, _tw_cfg())
    for _, r in _chat_requests(client):
        assert r.status_code == 200
    assert _until(lambda: len(rows) == 4)
    _assert_rows(rows, "openai")
    assert [r["origin"] for r in rows] == [None, "specul8", None, "specul8"]
    assert [r["job_id"] for r in rows] == [None, "job-42", None, "job-42"]


def test_responses_are_identical_with_toolwatch_enabled(tmp_path):
    """The whole point: an audit that changes nothing the client can see."""
    for run in (_messages_requests, _chat_requests):
        _, off_client, off_rows = _app(tmp_path, _tw_cfg(enabled=False), capture=False)
        _, on_client, on_rows = _app(tmp_path, _tw_cfg())
        a = [s for s, _ in run(off_client)]
        b = [s for s, _ in run(on_client)]
        assert a == b
        assert _until(lambda: sum(len(r["hosts"]) for r in on_rows) == 4)
        assert off_rows == []


def test_status_reports_toolwatch_stats(tmp_path):
    _, client, _ = _app(tmp_path, _tw_cfg())
    st = client.get("/status").json()
    assert st["toolwatch"] == {"queued": 0, "written": 0, "dropped": 0, "enabled": True}
    assert "choices" in st


def test_default_pg_writer_maps_columns_and_reconnects(tmp_path, monkeypatch):
    """The INSERT's column order/param mapping, and reconnect-after-failure. No real database:
    psycopg.connect is stubbed (importorskip keeps the suite runnable without the extra)."""
    psycopg = pytest.importorskip("psycopg")
    from saint.toolwatch import ToolCallRecord

    writes: list[tuple[str, list]] = []
    state = {"conns": 0}

    class _Cur:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def executemany(self, sql, params):
            if state["conns"] == 1:            # the first connection is the broken one
                raise RuntimeError("connection bad")
            writes.append((sql, list(params)))

    class _Conn:
        def __init__(self):
            self.closed = False

        def cursor(self):
            return _Cur()

        def close(self):
            self.closed = True

    def connect(dsn, **kw):
        assert dsn == "postgresql://netwatch" and kw.get("autocommit") is True
        state["conns"] += 1
        return _Conn()

    monkeypatch.setattr(psycopg, "connect", connect)
    rec = ToolCallRecord(ts="2026-10-28T00:00:00+00:00", api="anthropic", tool_name="Bash",
                         command="curl http://a.example.com", hosts=["a.example.com"], peer="::1")
    tw = ToolWatch(ToolwatchConfig(enabled=True), "postgresql://netwatch")
    tw._write([rec.as_row()])                  # failure: batch dropped, connection closed
    assert tw.stats()["dropped"] == 1
    tw._write([rec.as_row()])                  # next batch reconnects and writes
    assert tw.stats()["written"] == 1 and state["conns"] == 2
    sql, params = writes[-1]
    assert "INSERT INTO tool_calls" in sql and sql.count("%s") == 12
    assert params[0] == ["2026-10-28T00:00:00+00:00", "anthropic", None, "::1", None, None,
                         None, None, None, "Bash", "curl http://a.example.com",
                         ["a.example.com"]]    # hosts stays a list -> psycopg text[]
    tw.close()


def test_toolwatch_section_parses_with_defaults(tmp_path, monkeypatch):
    """[toolwatch] is optional; absent => disabled, and the keys parse when present."""
    from saint.config import load_config
    from tests.test_config import EXAMPLE_TOML
    monkeypatch.setenv("TEST_HOME", str(tmp_path))
    base = tmp_path / "a.toml"
    base.write_text(EXAMPLE_TOML)
    assert load_config(base).toolwatch == ToolwatchConfig()
    with_host = tmp_path / "b.toml"
    with_host.write_text(EXAMPLE_TOML + '\n[toolwatch]\nenabled = true\ndsn_env = "NW_DSN"\n'
                                        'max_command_chars = 500\nqueue_size = 10\n')
    assert load_config(with_host).toolwatch == ToolwatchConfig(
        enabled=True, dsn_env="NW_DSN", max_command_chars=500, queue_size=10)


def test_enabled_without_dsn_starts_and_counts_drops(tmp_path, capsys, monkeypatch):
    """The DSN env var is the only thing that can be missing: warn, keep serving, count drops."""
    monkeypatch.delenv("NW_DSN", raising=False)
    _, client, _ = _app(tmp_path, _tw_cfg(), capture=False)
    assert "NW_DSN is unset" in capsys.readouterr().err
    with patch("saint.backends.litellm.anthropic.messages.acreate",
               AsyncMock(return_value=_ANTHROPIC_TOOL)):
        r = client.post("/v1/messages", json={"model": "saint-cloud-large", "max_tokens": 100,
                                             "messages": [{"role": "user", "content": "hi"}]})
    assert r.status_code == 200
    assert _until(lambda: client.get("/status").json()["toolwatch"]["dropped"] == 1)


def test_a_broken_sink_cannot_change_the_response(tmp_path):
    """The sink raises on every batch: the client still gets the identical 200 response."""
    app, c, rows = _app(tmp_path, _tw_cfg())

    def angry(batch):
        raise RuntimeError("netwatch unreachable")

    app.state.toolwatch = ToolWatch(ToolwatchConfig(enabled=True), "dsn-unused", writer=angry)
    with patch("saint.backends.litellm.anthropic.messages.acreate",
               AsyncMock(return_value=_ANTHROPIC_TOOL)):
        r = c.post("/v1/messages", json={"model": "saint-cloud-large", "max_tokens": 100,
                                         "messages": [{"role": "user", "content": "hi"}]})
    assert r.status_code == 200
    assert r.json()["content"][0]["name"] == "Bash"
    assert _until(lambda: app.state.toolwatch.stats()["dropped"] == 1)
    assert rows == []
