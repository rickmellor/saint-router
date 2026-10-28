"""Passive audit of the tool calls the routed models emit, and the hosts they name.

Why: SAINT is the only place every local-model client's response passes through. A model's
`curl https://oss-cn-beijing.aliyuncs.com/` tool call may be executed by a shell on a machine
the network monitoring cannot see, so the only reliable record of the destination is the
response text itself — read here, at the router.

Nothing here blocks, rewrites or alerts. Rows go to the `netwatch` PostgreSQL database
(`tool_calls`) and a Grafana panel reads them. Every entry point is safe to call on the request
path: no DNS, no sockets except the sink's own Postgres connection, and no exception ever
escapes into a response — the same rule `SseUsageTracker` follows for the SSE relay.
"""
from __future__ import annotations

import json
import queue
import re
import threading
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

# Last-label stoplist for bare dotted tokens: file extensions, code attributes, systemd unit
# types and dict attribute names. A token whose LAST label is here is almost never a hostname
# (`run.py`, `os.path`, `foo.service`, `torch.linalg`). Deliberately generous — noise is
# acceptable, the database side decides what is external or watched.
EXT_STOPLIST: frozenset[str] = frozenset("""
py pyc pyi js mjs cjs ts tsx jsx md txt json jsonl yml yaml toml ini cfg conf sh bash zsh html
htm css scss sql csv tsv xml pdf png jpg jpeg gif svg ico webp mp4 mp3 wav zip gz tgz bz2 xz tar
whl egg lock env log service slice scope target timer socket mount so a o h c cc cpp hpp rs go
java rb php pl lua swift kt dart rst tex bib ipynb parquet db sqlite bak tmp orig patch diff pub
pem key crt example test local localhost arpa internal lan path random linalg array items keys
values get set len min max sum exists join split strip format append extend insert update remove
pop sort copy clear
""".split())

# A tool call is recorded when it names a host OR its tool name looks like a shell (so every
# shell command is kept even when it contains no host at all).
SHELL_TOOL_RE = re.compile(r"bash|shell|command|exec|terminal|run", re.I)

# (a) URL hosts: scheme:// [userinfo@] host. Bracketed form = IPv6 literal.
_URL_RE = re.compile(
    r"[a-z][a-z0-9+.-]*://"                                  # scheme
    r"(?:[^\s/?#:@'\"<>]*(?::[^\s/?#@'\"<>]*)?@)?"           # optional user[:pass]@
    r"(\[[^\]\s]+\]|[^/?#\s:'\"<>]+)",                       # host (or [IPv6])
    re.I)
# (b) IPv4 literals.
_IPV4_RE = re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")
# (c) bare hostnames. Rejected when the last label is an extension/attribute (EXT_STOPLIST) or
# when the match is preceded by `/` (path component), `.` (attribute chain) or `@` (email).
_BARE_RE = re.compile(r"\b(?:[a-z0-9](?:[a-z0-9\-]{0,61}[a-z0-9])?\.)+[a-z]{2,24}\b", re.I)

MAX_HOSTS = 64
MAX_SCAN_CHARS = 65536

_COLUMNS = ("ts", "api", "origin", "peer", "session", "job_id", "client", "backend", "model",
            "tool_name", "command", "hosts")
_INSERT = ("INSERT INTO tool_calls (ts, api, origin, peer, session, job_id, client, backend, "
           "model, tool_name, command, hosts) VALUES "
           "(%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)")


def _norm_host(raw: str) -> str:
    """Lower-case, trailing dot stripped, brackets removed from an IPv6 literal."""
    h = raw.strip().lower().rstrip(".")
    if h.startswith("[") and h.endswith("]"):
        h = h[1:-1]
    return h


def extract_hosts(text: str) -> list[str]:
    """Hosts named in free text — ordered by first appearance, de-duplicated, lower-cased.

    Three patterns: URL hosts, IPv4 literals, bare hostnames (see the module regexes). Names
    are never resolved and no socket is ever opened: this is pure text scanning."""
    found: list[tuple[int, str]] = []
    for m in _URL_RE.finditer(text):
        host = _norm_host(m.group(1))
        if host:
            found.append((m.start(1), host))
    for m in _IPV4_RE.finditer(text):
        found.append((m.start(), m.group()))
    url_hosts = {h for _, h in found}
    for m in _BARE_RE.finditer(text):
        if m.start() > 0 and text[m.start() - 1] in "/.@":
            continue
        host = m.group().lower()
        if host in url_hosts:
            continue                       # already recorded by the URL pass
        if host.rsplit(".", 1)[-1] in EXT_STOPLIST:
            continue
        found.append((m.start(), host))

    out: list[str] = []
    seen: set[str] = set()
    for _, host in sorted(found, key=lambda p: p[0]):
        if host and host not in seen:
            seen.add(host)
            out.append(host)
            if len(out) >= MAX_HOSTS:
                break
    return out


def _iter_strings(node: Any):
    """Every string value in a nested dict/list, recursively."""
    if isinstance(node, str):
        yield node
    elif isinstance(node, dict):
        for v in node.values():
            yield from _iter_strings(v)
    elif isinstance(node, (list, tuple)):
        for v in node:
            yield from _iter_strings(v)


def _excerpt(inp: Any, max_command_chars: int) -> str:
    if isinstance(inp, dict):
        for key in ("command", "url"):
            v = inp.get(key)
            if isinstance(v, str):
                return v[:max_command_chars]
        try:
            text = json.dumps(inp, ensure_ascii=False, separators=(",", ":"), default=str)
        except Exception:
            text = repr(inp)
        return text[:max_command_chars]
    if isinstance(inp, str):
        return inp[:max_command_chars]
    return str(inp)[:max_command_chars]


def scan_tool_input(name: str, inp: Any, max_command_chars: int = 2000) -> tuple[str, list[str]]:
    """(command excerpt, hosts) for one tool call's arguments.

    `inp` is the Anthropic `input` dict or the OpenAI `function.arguments` JSON string; an
    unparseable string is scanned raw (a truncated/partial stream still yields its hosts)."""
    if isinstance(inp, str):
        try:
            parsed = json.loads(inp)
        except Exception:
            return inp[:max_command_chars], extract_hosts(inp[:MAX_SCAN_CHARS])
        inp = parsed
    text = "\n".join(_iter_strings(inp))[:MAX_SCAN_CHARS]
    return _excerpt(inp, max_command_chars), extract_hosts(text)


@dataclass
class ToolCallRecord:
    """One recorded tool call. `hosts` is what the call names; `command` is the excerpt."""
    ts: str
    api: str                     # "anthropic" | "openai"
    tool_name: str
    command: str
    hosts: list[str]
    origin: str | None = None
    session: str | None = None
    job_id: str | None = None
    client: str | None = None
    peer: str | None = None      # the connecting IP — the only attribution interactive has
    backend: str | None = None
    model: str | None = None

    def as_row(self) -> dict[str, Any]:
        return {c: getattr(self, c) for c in _COLUMNS}


class AnthropicToolTracker:
    """tool_use blocks from an Anthropic SSE byte stream, relayed untouched.

    Same `\\n\\n` block buffering as SseUsageTracker: name arrives on content_block_start,
    the JSON arrives in pieces on content_block_delta(input_json_delta) and is parsed on
    content_block_stop (lenient: unparseable input is scanned as raw text). Parsing failures
    must never kill the relay — this is accounting, not correctness."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, Any]] = []
        self._open: dict[int, dict[str, Any]] = {}
        self._buf = b""

    def feed(self, chunk: Any) -> None:
        try:
            self._buf += chunk if isinstance(chunk, bytes) else str(chunk).encode()
            while b"\n\n" in self._buf:
                block, self._buf = self._buf.split(b"\n\n", 1)
                self._parse(block)
        except Exception:
            pass

    def _parse(self, block: bytes) -> None:
        for line in block.split(b"\n"):
            if not line.startswith(b"data:"):
                continue
            try:
                d = json.loads(line[5:].strip())
            except Exception:
                continue
            if not isinstance(d, dict):
                continue
            t = d.get("type")
            if t == "content_block_start":
                cb = d.get("content_block") or {}
                if isinstance(cb, dict) and cb.get("type") == "tool_use":
                    idx = d.get("index")
                    inp = cb.get("input")
                    self._open[idx] = {"name": cb.get("name") or "",
                                       "json": "",
                                       "input": inp if isinstance(inp, dict) and inp else None}
            elif t == "content_block_delta":
                slot = self._open.get(d.get("index"))
                delta = d.get("delta") or {}
                if slot is not None and isinstance(delta, dict):
                    frag = delta.get("partial_json")
                    if isinstance(frag, str):
                        slot["json"] += frag
            elif t == "content_block_stop":
                slot = self._open.pop(d.get("index"), None)
                if slot is None:
                    continue
                if slot["input"] is not None and not slot["json"]:
                    self.calls.append((slot["name"], slot["input"]))
                    continue
                raw = slot["json"]
                if not raw:
                    self.calls.append((slot["name"], slot["input"] or {}))
                    continue
                try:
                    self.calls.append((slot["name"], json.loads(raw)))
                except Exception:
                    self.calls.append((slot["name"], raw))   # scanned raw downstream


class OpenAIToolTracker:
    """delta.tool_calls fragments from an OpenAI chunk stream, keyed by call index."""

    def __init__(self) -> None:
        self._calls: dict[int, dict[str, str]] = {}

    def feed_chunk(self, chunk_dict: Any) -> None:
        try:
            if not isinstance(chunk_dict, dict):
                return
            for choice in chunk_dict.get("choices") or ():
                if not isinstance(choice, dict):
                    continue
                delta = choice.get("delta") or {}
                if not isinstance(delta, dict):
                    continue
                for tc in delta.get("tool_calls") or ():
                    if not isinstance(tc, dict):
                        continue
                    idx = tc.get("index")
                    if not isinstance(idx, int):
                        idx = len(self._calls)
                    slot = self._calls.setdefault(idx, {"name": "", "args": ""})
                    fn = tc.get("function") or {}
                    if isinstance(fn, dict):
                        if isinstance(fn.get("name"), str):
                            slot["name"] = fn["name"]
                        if isinstance(fn.get("arguments"), str):
                            slot["args"] += fn["arguments"]
        except Exception:
            pass

    def finish(self) -> list[tuple[str, str]]:
        return [(c["name"], c["args"]) for c in self._calls.values()
                if c["name"] or c["args"]]


def calls_from_anthropic_payload(payload: Any) -> list[tuple[str, Any]]:
    """(name, input) pairs from a non-streaming /v1/messages response."""
    out: list[tuple[str, Any]] = []
    try:
        content = payload.get("content") if isinstance(payload, dict) else None
        for block in content or ():
            if isinstance(block, dict) and block.get("type") == "tool_use":
                out.append((block.get("name") or "", block.get("input")))
    except Exception:
        pass
    return out


def calls_from_openai_payload(payload: Any) -> list[tuple[str, Any]]:
    """(name, arguments_json_string) pairs from a non-streaming chat completion."""
    out: list[tuple[str, Any]] = []
    try:
        for choice in (payload.get("choices") or ()) if isinstance(payload, dict) else ():
            if not isinstance(choice, dict):
                continue
            msg = choice.get("message") or {}
            if not isinstance(msg, dict):
                continue
            for tc in msg.get("tool_calls") or ():
                if not isinstance(tc, dict):
                    continue
                fn = tc.get("function") or {}
                if isinstance(fn, dict):
                    out.append((fn.get("name") or "", fn.get("arguments")))
    except Exception:
        pass
    return out


class ToolWatch:
    """The sink: request path queues rows, one daemon thread writes batches to netwatch.

    `writer` is injectable (tests never need psycopg). The default writer holds a single
    autocommit psycopg v3 connection and reconnects after any failure. On failure the batch is
    dropped and counted — never retried, never raised into the router."""

    def __init__(self, cfg: Any, dsn: str | None, writer: Any = None) -> None:
        self.enabled = bool(getattr(cfg, "enabled", False))
        self.max_command_chars = int(getattr(cfg, "max_command_chars", 2000))
        self.dsn = dsn
        self.dropped = 0
        self.written = 0
        self._writer = writer
        self._conn: Any = None
        self._last_warn = 0.0
        self._stop = threading.Event()
        # A sink with nowhere to go still counts its drops, so /status shows the bleed.
        self._has_sink = writer is not None or dsn is not None
        if self.enabled and self._has_sink:
            self.queue: queue.Queue = queue.Queue(maxsize=max(1, int(getattr(cfg, "queue_size", 1000))))
            self._thread: threading.Thread | None = threading.Thread(
                target=self._drain, name="toolwatch", daemon=True)
            self._thread.start()
        else:
            self.queue = queue.Queue(maxsize=1)
            self._thread = None

    # ------------------------------------------------------------------ request path

    def record(self, rec: ToolCallRecord) -> None:
        try:
            if not self.enabled:
                return
            if not self._has_sink:
                self.dropped += 1
                return
            self.queue.put_nowait(rec.as_row())
        except Exception:                    # queue.Full (sink slower than the router) or worse
            self.dropped += 1

    def stats(self) -> dict[str, Any]:
        try:
            queued = self.queue.qsize()
        except Exception:
            queued = 0
        return {"queued": queued, "written": self.written, "dropped": self.dropped,
                "enabled": self.enabled}

    def close(self) -> None:
        """Flush with a 2 s bound. The app has no shutdown hook (build_app registers none, and
        the drain thread is a daemon), so nothing calls this at `saint serve` exit — it is here
        for a future lifespan hook and for tests that want a deterministic stop."""
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        self._close_conn()

    # ------------------------------------------------------------------ drain thread

    def _drain(self) -> None:
        while not (self._stop.is_set() and self.queue.empty()):
            try:
                batch = [self.queue.get(timeout=0.1)]
            except queue.Empty:
                continue
            while len(batch) < 50:
                try:
                    batch.append(self.queue.get_nowait())
                except queue.Empty:
                    break
            self._write(batch)
        self._close_conn()

    def _write(self, batch: list[dict[str, Any]]) -> None:
        try:
            (self._writer or self._pg_write)(batch)
            self.written += len(batch)
        except Exception as e:
            self.dropped += len(batch)
            self._close_conn()
            now = time.monotonic()
            if now - self._last_warn >= 60.0:
                self._last_warn = now
                import logging
                logging.getLogger("saint").warning("toolwatch: dropped %d tool-call rows: %s",
                                                   len(batch), e)

    def _pg_write(self, rows: list[dict[str, Any]]) -> None:
        import psycopg   # lazy: the router runs fine without the extra (tests never import it)
        if self._conn is None or getattr(self._conn, "closed", False):
            self._conn = psycopg.connect(self.dsn, autocommit=True, connect_timeout=5)
        with self._conn.cursor() as cur:
            cur.executemany(_INSERT, [[r.get(c) for c in _COLUMNS] for r in rows])

    def _close_conn(self) -> None:
        conn, self._conn = self._conn, None
        if conn is not None:
            try:
                conn.close()
            except Exception:
                pass


def emit(toolwatch: ToolWatch | None, *, api: str, attr: dict[str, Any], client: str | None,
         peer: str | None, backend: str | None, model: str | None,
         calls: list[tuple[str, Any]], max_command_chars: int = 2000) -> None:
    """The one entry point the four hook sites call. Never raises, never blocks."""
    if toolwatch is None or not calls:
        return
    try:
        for name, inp in calls:
            try:
                command, hosts = scan_tool_input(name, inp, max_command_chars)
                if not hosts and not SHELL_TOOL_RE.search(name or ""):
                    continue               # a non-shell tool that names no host: not interesting
                toolwatch.record(ToolCallRecord(
                    ts=datetime.now(UTC).isoformat(), api=api, tool_name=str(name or ""),
                    command=command, hosts=hosts,
                    origin=attr.get("origin"), session=attr.get("session"),
                    job_id=attr.get("job_id"), client=client, peer=peer,
                    backend=backend, model=model))
            except Exception:
                continue
    except Exception:
        pass
