-- johnny integration: attribute each request to a johnny seat + how it was served.
-- johnny_seat:        resolved seat id when the chosen backend was johnny-bound (else NULL)
-- state_at_dispatch:  johnny_ready | static_baseline | while_loading | fallback | NULL(unbound)
ALTER TABLE requests ADD COLUMN johnny_seat TEXT;
ALTER TABLE requests ADD COLUMN state_at_dispatch TEXT;
