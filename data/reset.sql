-- User-scoped data purge for current local user (id = 1).
-- Run:
--   sqlite3 data/central.db < data/reset.sql
--
-- Keeps intact:
-- - tables/schema
-- - items/symptoms/ingredients catalogs
-- - papers/claims evidence corpus
-- - trained model artifacts on disk
--
-- Removes only this user's timeline + computed artifacts/jobs:
-- - events, expansions, derived features, insights, links/verifications,
--   retrieval runs, recurring rules, background jobs, ingest logs
-- - optional: user row itself (enabled below)

PRAGMA foreign_keys = OFF;
PRAGMA busy_timeout = 30000;
BEGIN IMMEDIATE TRANSACTION;

DELETE FROM background_jobs
WHERE user_id = 1;

DELETE FROM recurring_exposure_rules
WHERE user_id = 1;

DELETE FROM retrieval_runs
WHERE user_id = 1;

DELETE FROM insight_event_links
WHERE user_id = 1;

DELETE FROM insight_verifications
WHERE user_id = 1;

DELETE FROM insights
WHERE user_id = 1;

DELETE FROM derived_features
WHERE user_id = 1;

DELETE FROM derived_features_ingredients
WHERE user_id = 1;

DELETE FROM raw_event_ingest
WHERE user_id = 1;

DELETE FROM exposure_expansions
WHERE exposure_event_id IN (
  SELECT id FROM exposure_events WHERE user_id = 1
);

DELETE FROM symptom_events
WHERE user_id = 1;

DELETE FROM exposure_events
WHERE user_id = 1;

-- Keep local dev user id=1 so app inserts continue to work after reset.
INSERT OR IGNORE INTO users (id, created_at, name)
VALUES (1, datetime('now'), 'Local User');

COMMIT;
PRAGMA foreign_keys = ON;
