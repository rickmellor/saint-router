import asyncio

from litellm.llms.anthropic.experimental_pass_through.adapters import streaming_iterator as si
from litellm.types.utils import Delta, ModelResponseStream, StreamingChoices

import saint.backends  # noqa: F401  — applies the patches
from saint import litellm_patches

DELTA_FOR = {"thinking": {"thinking_delta", "signature_delta"}, "text": {"text_delta"},
             "tool_use": {"input_json_delta"}}


def _chunk(content=None, reasoning=None, finish=None):
    d = Delta(content=content, role="assistant")
    if reasoning is not None:
        d.reasoning_content = reasoning
    return ModelResponseStream(id="x", model="m",
                               choices=[StreamingChoices(index=0, delta=d, finish_reason=finish)])


def _events(chunks):
    async def gen():
        for c in chunks:
            yield c

    async def collect():
        return [ev async for ev in si.AnthropicStreamWrapper(completion_stream=gen(), model="m")]

    return asyncio.run(collect())


def _check(events):
    """Every delta matches its block's type; return (thinking, text) as streamed."""
    blocks, thinking, text = {}, "", ""
    for ev in events:
        if ev.get("type") == "content_block_start":
            blocks[ev["index"]] = ev["content_block"]["type"]
        elif ev.get("type") == "content_block_delta":
            dt = ev["delta"]["type"]
            assert dt in DELTA_FOR[blocks[ev["index"]]], f"{dt} inside a {blocks[ev['index']]} block"
            thinking += ev["delta"].get("thinking", "")
            text += ev["delta"].get("text", "")
    return thinking, text


def test_reasoning_and_content_in_one_chunk():
    # the vLLM reasoning→answer boundary: one chunk carries the reasoning tail AND the first token
    events = _events([_chunk(reasoning="We think"), _chunk(reasoning=" hard"),
                      _chunk(content="Hello", reasoning="\n"), _chunk(content=" world"), _chunk(finish="stop")])
    assert _check(events) == ("We think hard\n", "Hello world")


def test_boundary_chunk_with_finish_reason():
    events = _events([_chunk(reasoning="ok"), _chunk(content="Done.", reasoning="\n", finish="stop")])
    assert _check(events) == ("ok\n", "Done.")


def test_plain_streams_unchanged():
    events = _events([_chunk(reasoning="a"), _chunk(content="b"), _chunk(content="c"), _chunk(finish="stop")])
    assert _check(events) == ("a", "bc")


def test_apply_is_idempotent():
    split = si._CombinedChunkSplitter._split
    litellm_patches.apply()
    assert si._CombinedChunkSplitter._split is split
