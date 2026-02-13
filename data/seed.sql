BEGIN;

INSERT OR IGNORE INTO users (id, created_at, name) VALUES
  (1, '2026-02-01T09:00:00Z', 'Test User');

INSERT OR IGNORE INTO items (id, name, category) VALUES
  (1, 'Sugar', 'Food'),
  (2, 'Comedogenic Facial Moisturizer', 'Cosmetics');

INSERT OR IGNORE INTO symptoms (id, name, description) VALUES
  (1, 'Acne', 'Skin disorder - inflammation in hair folicles and obstruction of sebaceous glands');

INSERT OR IGNORE INTO ingredients (id, name, description) VALUES
  (1, 'Refined Sugar', 'Simple carbohydrates and sweeteners'),
  (2, 'Sodium Laureth Sulfate', 'Surfactant used in cleansers');

INSERT OR IGNORE INTO items_ingredients (item_id, ingredient_id) VALUES
  (1, 1),
  (2, 2);

INSERT OR IGNORE INTO papers (id, title, url, abstract, publication_date, source, ingested_at) VALUES
  (
    1,
    'Dietary glycemic load and acne occurrence',
    'https://example.org/papers/dietary-glycemic-load-acne',
    'Observational findings suggest high glycemic load diets may be associated with higher acne prevalence in some cohorts.',
    '2024-03-12',
    'seed',
    '2026-02-01T00:00:00Z'
  ),
  (
    2,
    'Facial cleanser ingredient irritation profile review',
    'https://example.org/papers/cleanser-irritation-review',
    'Review of surfactant irritation potential and skin barrier interaction patterns.',
    '2023-09-02',
    'seed',
    '2026-02-01T00:00:00Z'
  );

INSERT OR IGNORE INTO claims (
  id, ingredient_id, symptom_id, paper_id, claim_type, summary, chunk_index, chunk_text, chunk_hash,
  embedding_model, embedding_vector, citation_title, citation_url, citation_snippet, evidence_polarity_and_strength
) VALUES
  (
    1, 1, 1, 1, 'rag_chunk',
    'Higher glycemic load exposure appears associated with acne flare likelihood in some cohorts.',
    0,
    'In multiple cohorts, higher dietary glycemic load was associated with increased acne lesion counts versus lower glycemic exposure groups.',
    'seed_claim_1_chunk_0',
    'local-token-v1',
    '{}',
    'Dietary glycemic load and acne occurrence',
    'https://example.org/papers/dietary-glycemic-load-acne',
    'Higher dietary glycemic load was associated with increased acne lesion counts.',
    1
  ),
  (
    2, 2, 1, 2, 'rag_chunk',
    'Some surfactants may increase irritation in sensitive users and can overlap with acne-like symptom reporting.',
    0,
    'Irritation-prone surfactants may worsen skin barrier sensitivity in subsets of users, with occasional overlap in acne-like complaints.',
    'seed_claim_2_chunk_0',
    'local-token-v1',
    '{}',
    'Facial cleanser ingredient irritation profile review',
    'https://example.org/papers/cleanser-irritation-review',
    'Irritation-prone surfactants may worsen skin barrier sensitivity in subsets of users.',
    1
  );

INSERT OR IGNORE INTO exposure_events (id, user_id, item_id, timestamp, route) VALUES
  (1, 1, 1, '2026-02-04T18:00:00Z', 'ingestion'),
  (2, 1, 2, '2026-02-04T21:00:00Z', 'dermal');

INSERT OR IGNORE INTO symptom_events (id, user_id, symptom_id, timestamp, severity) VALUES
  (1, 1, 1, '2026-02-05T08:00:00Z', 4);

COMMIT;
