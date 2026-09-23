-- Spill accounting (2026-09-23): when a johnny_role backend's own seat was saturated or not ready and
-- another role's seat served the request instead.
-- spilled_from: the role that was asked for (NULL = served by its own seat / not johnny-bound)
-- spilled_to:   the seat that actually served (== johnny_seat; kept explicit for cheap GROUP BY)
-- seat_load:    requests running+waiting on the serving seat at dispatch (NULL = unknown/not vLLM)
ALTER TABLE requests ADD COLUMN spilled_from TEXT;
ALTER TABLE requests ADD COLUMN spilled_to   TEXT;
ALTER TABLE requests ADD COLUMN seat_load    INTEGER;
