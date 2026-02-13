-- Wipe all application data while keeping schema intact
-- Usage: sqlite3 data/central.db < data/reset_data.sql
--
/*
Non-destructive re-score of existing insights (does NOT wipe DB).

1) Recompute one user:
curl -X POST "http://127.0.0.1:8000/insights/recompute" \
  -H "Content-Type: application/json" \
  -d '{"user_id":1,"online_enabled":false,"max_papers_per_query":5}'

2) Recompute all users currently in DB:
for uid in $(sqlite3 data/central.db "SELECT id FROM users;"); do
  curl -s -X POST "http://127.0.0.1:8000/insights/recompute" \
    -H "Content-Type: application/json" \
    -d "{\"user_id\":$uid,\"online_enabled\":false,\"max_papers_per_query\":5}" >/dev/null
  echo "recomputed user $uid"
done
*/

PRAGMA foreign_keys = OFF;
BEGIN TRANSACTION;

DELETE FROM background_jobs;
DELETE FROM retrieval_runs;
DELETE FROM insights;
DELETE FROM derived_features_ingredients;
DELETE FROM derived_features;
DELETE FROM claims;
DELETE FROM papers;
DELETE FROM exposure_expansions;
DELETE FROM symptom_events;
DELETE FROM exposure_events;
DELETE FROM raw_event_ingest;
DELETE FROM items_ingredients;
DELETE FROM ingredients_aliases;
DELETE FROM symptoms_aliases;
DELETE FROM items_aliases;
DELETE FROM ingredients;
DELETE FROM symptoms;
DELETE FROM items;
DELETE FROM users;

-- Reset AUTOINCREMENT counters for tables that use INTEGER PRIMARY KEY.
DELETE FROM sqlite_sequence;

COMMIT;
PRAGMA foreign_keys = ON;
