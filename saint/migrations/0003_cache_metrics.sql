-- Provider-side prompt-cache accounting per request.
-- cache_read_tokens:  input tokens served from the provider's prompt cache
--                     (anthropic cache_read_input_tokens at 0.1x price;
--                      openai prompt_tokens_details.cached_tokens at 0.25-0.5x)
-- cache_write_tokens: tokens written to the provider cache this request
--                     (anthropic cache_creation_input_tokens, billed 1.25x — 2x for 1h TTL)
ALTER TABLE requests ADD COLUMN cache_read_tokens  INTEGER;
ALTER TABLE requests ADD COLUMN cache_write_tokens INTEGER;
