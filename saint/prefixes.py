from __future__ import annotations

from dataclasses import dataclass


class UnknownPrefixError(ValueError):
    """Raised when a !-prefix token doesn't match any urgency or backend."""

    def __init__(self, token: str, known_urgencies: set[str], known_backends: set[str]):
        self.token = token
        super().__init__(
            f"unknown prefix token '{token}'; "
            f"known urgency tokens: {sorted(known_urgencies)}; "
            f"known backend names/aliases: {sorted(known_backends)}"
        )


@dataclass(frozen=True)
class ParsedPrefixes:
    urgency: str | None
    pinned_backend: str | None
    stripped: str
    raw: str


def parse_prefixes(
    content: str,
    urgencies: set[str],
    backends_with_aliases: dict[str, set[str]],
) -> ParsedPrefixes:
    """Parse leading !<token> prefixes. The first '!' must be at index 0.

    Multiple prefixes are space-separated and parsed left-to-right until the
    first non-prefix token. Last urgency wins. Returns the stripped content
    (prefix tokens and any single space after them removed).
    """
    if not content or content[0] != "!":
        return ParsedPrefixes(urgency=None, pinned_backend=None, stripped=content, raw="")

    # Build alias → backend-name lookup
    alias_to_backend: dict[str, str] = {}
    all_known_tokens: set[str] = set()
    for backend_name, aliases in backends_with_aliases.items():
        alias_to_backend[backend_name] = backend_name
        all_known_tokens.add(backend_name)
        for alias in aliases:
            alias_to_backend[alias] = backend_name
            all_known_tokens.add(alias)

    urgency: str | None = None
    pinned: str | None = None
    raw_tokens: list[str] = []
    rest = content
    while rest.startswith("!"):
        # Find end of token: whitespace or end of string
        i = 1
        while i < len(rest) and not rest[i].isspace():
            i += 1
        token = rest[1:i]
        if not token:
            break  # bare '!' is not a prefix; leave the message intact
        raw_tokens.append("!" + token)
        if token in urgencies:
            urgency = token
        elif token in alias_to_backend:
            pinned = alias_to_backend[token]
        else:
            raise UnknownPrefixError(token, urgencies, all_known_tokens)
        # Skip exactly one space if present
        rest = rest[i:].lstrip(" ") if i < len(rest) and rest[i] == " " else rest[i:]

    return ParsedPrefixes(
        urgency=urgency,
        pinned_backend=pinned,
        stripped=rest,
        raw=" ".join(raw_tokens),
    )
