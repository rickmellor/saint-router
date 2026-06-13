CREATE TABLE IF NOT EXISTS requests (
    id                              INTEGER PRIMARY KEY AUTOINCREMENT,
    ts                              TEXT NOT NULL,
    request_id                      TEXT NOT NULL,
    model_field                     TEXT NOT NULL,
    prefixes_raw                    TEXT,
    pinned_backend                  TEXT,
    urgency_used                    TEXT NOT NULL,
    classifier_used                 TEXT,
    classifier_fallback_reason      TEXT,
    classifier_input_chars          INTEGER,
    classifier_input_truncated_from INTEGER,
    classifier_latency_ms           INTEGER,
    classifier_domain               TEXT,
    classifier_complexity           TEXT,
    classifier_reason               TEXT,
    backend_chosen                  TEXT NOT NULL,
    backend_latency_ms              INTEGER,
    tokens_in                       INTEGER,
    tokens_out                      INTEGER,
    success                         INTEGER NOT NULL,
    error_kind                      TEXT,
    prompt_content                  TEXT,
    prompt_storage_mode             TEXT NOT NULL,
    relabel_backend                 TEXT,
    relabel_ts                      TEXT,
    relabel_note                    TEXT
);
CREATE INDEX IF NOT EXISTS idx_requests_ts      ON requests(ts);
CREATE INDEX IF NOT EXISTS idx_requests_backend ON requests(backend_chosen);
CREATE INDEX IF NOT EXISTS idx_requests_relabel ON requests(relabel_backend) WHERE relabel_backend IS NOT NULL;
