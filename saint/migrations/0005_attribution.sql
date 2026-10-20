-- Attribution (hybrid-usage/1, 2026-10-20): WHO caused each request, so `GET /usage/v1` can split
-- one spend report between cloud and local work. Values come from the request headers, are free-form
-- metadata (never prompt content), and are never forwarded upstream.
-- client:         derived from User-Agent — the first matching [clients] rule name, else the UA's
--                 first product token lowercased (curl/8.5 -> curl), else "unknown". Recorded on
--                 EVERY request, whether or not it influenced routing.
-- origin:         X-Inference-Origin — the machine the request came from (e.g. GEN-PW0DXJ6E).
-- session:        X-Inference-Session — the orchestrating session that caused it.
-- job_id:         X-Inference-Job — one unit of delegated work (one worker run).
ALTER TABLE requests ADD COLUMN client   TEXT;
ALTER TABLE requests ADD COLUMN origin   TEXT;
ALTER TABLE requests ADD COLUMN session  TEXT;
ALTER TABLE requests ADD COLUMN job_id   TEXT;
