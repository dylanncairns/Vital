BEGIN;

INSERT OR IGNORE INTO users (id, created_at, name) VALUES
  (1, '2026-02-01T09:00:00Z', 'Test User');

INSERT OR IGNORE INTO items (id, name, category) VALUES
  (1, 'Sugar', 'Food'),
  (2, 'General Face Wash', 'Cosmetics');

INSERT OR IGNORE INTO symptoms (id, name, description) VALUES
  (1, 'Acne', 'Skin disorder - inflammation in hair folicles and obstruction of sebaceous glands');

INSERT OR IGNORE INTO exposure_events (id, user_id, item_id, timestamp, route) VALUES
  (1, 1, 1, '2026-02-04T18:00:00Z', 'ingestion'),
  (2, 1, 2, '2026-02-04T21:00:00Z', 'dermal');

INSERT OR IGNORE INTO symptom_events (id, user_id, symptom_id, timestamp, severity) VALUES
  (1, 1, 1, '2026-02-05T08:00:00Z', 4);

COMMIT;
