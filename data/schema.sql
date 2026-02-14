CREATE TABLE items (
    id INTEGER NOT NULL PRIMARY KEY,
    name TEXT,
    category TEXT
);
CREATE TABLE items_aliases (
    id INTEGER NOT NULL PRIMARY KEY,
    item_id INTEGER NOT NULL,
    alias TEXT,
    FOREIGN KEY (item_id) REFERENCES items(id)
);
CREATE TABLE ingredients (
    id INTEGER NOT NULL PRIMARY KEY,
    name TEXT,
    description TEXT
);
CREATE TABLE ingredients_aliases (
    id INTEGER NOT NULL PRIMARY KEY,
    ingredient_id INTEGER NOT NULL,
    alias TEXT,
    FOREIGN KEY (ingredient_id) REFERENCES ingredients(id)
);
CREATE TABLE items_ingredients (
    item_id INTEGER NOT NULL,
    ingredient_id INTEGER NOT NULL,
    PRIMARY KEY (item_id, ingredient_id),
    FOREIGN KEY (item_id) REFERENCES items(id),
    FOREIGN KEY (ingredient_id) REFERENCES ingredients(id)
);
CREATE TABLE papers (
    id INTEGER NOT NULL PRIMARY KEY,
    title TEXT NOT NULL,
    url TEXT,
    abstract TEXT,
    publication_date TEXT,
    source TEXT,
    ingested_at TEXT
);
CREATE TABLE users (
    id INTEGER NOT NULL PRIMARY KEY,
    created_at TEXT,
    name TEXT,
    username TEXT,
    password_hash TEXT
);
CREATE UNIQUE INDEX idx_users_username_unique ON users(username);
CREATE TABLE auth_sessions (
    id INTEGER NOT NULL PRIMARY KEY,
    user_id INTEGER NOT NULL,
    token TEXT NOT NULL UNIQUE,
    created_at TEXT NOT NULL,
    expires_at TEXT NOT NULL,
    revoked_at TEXT,
    FOREIGN KEY (user_id) REFERENCES users(id)
);
CREATE INDEX idx_auth_sessions_user ON auth_sessions(user_id);
CREATE INDEX idx_auth_sessions_token ON auth_sessions(token);
CREATE TABLE exposure_events (
    id INTEGER NOT NULL PRIMARY KEY,
    user_id INTEGER NOT NULL,
    item_id INTEGER NOT NULL,
    timestamp TEXT,
    time_range_start TEXT,
    time_range_end TEXT,
    time_confidence TEXT,
    ingested_at TEXT,
    raw_text TEXT,
    route TEXT NOT NULL,
    FOREIGN KEY (user_id) REFERENCES users(id),
    FOREIGN KEY (item_id) REFERENCES items(id)
);
CREATE TABLE symptoms (
    id INTEGER NOT NULL PRIMARY KEY,
    name TEXT,
    description TEXT
);
CREATE TABLE symptoms_aliases (
    id INTEGER NOT NULL PRIMARY KEY,
    symptom_id INTEGER NOT NULL,
    alias TEXT,
    FOREIGN KEY (symptom_id) REFERENCES symptoms(id)
);
CREATE TABLE symptom_events (
    id INTEGER NOT NULL PRIMARY KEY,
    user_id INTEGER NOT NULL,
    symptom_id INTEGER NOT NULL,
    timestamp TEXT,
    time_range_start TEXT,
    time_range_end TEXT,
    time_confidence TEXT,
    ingested_at TEXT,
    raw_text TEXT,
    severity INTEGER,
    FOREIGN KEY (user_id) REFERENCES users(id),
    FOREIGN KEY (symptom_id) REFERENCES symptoms(id)
);
CREATE TABLE exposure_expansions (
    id INTEGER NOT NULL PRIMARY KEY,
    exposure_event_id INTEGER NOT NULL,
    ingredient_id INTEGER,
    FOREIGN KEY (exposure_event_id) REFERENCES exposure_events(id),
    FOREIGN KEY (ingredient_id) REFERENCES ingredients(id)
);
CREATE TABLE raw_event_ingest (
    id INTEGER NOT NULL PRIMARY KEY,
    user_id INTEGER NOT NULL,
    raw_text TEXT NOT NULL,
    ingested_at TEXT,
    parse_status TEXT,
    error TEXT,
    FOREIGN KEY (user_id) REFERENCES users(id)
);
CREATE TABLE claims (
    id INTEGER NOT NULL PRIMARY KEY,
    item_id INTEGER,
    ingredient_id INTEGER,
    symptom_id INTEGER NOT NULL,
    paper_id INTEGER NOT NULL,
    claim_type TEXT,
    summary TEXT,
    chunk_index INTEGER,
    chunk_text TEXT,
    chunk_hash TEXT,
    embedding_model TEXT,
    embedding_vector TEXT,
    citation_title TEXT,
    citation_url TEXT,
    citation_snippet TEXT,
    study_design TEXT,
    study_quality_score REAL,
    population_match REAL,
    temporality_match REAL,
    risk_of_bias REAL,
    llm_confidence REAL,
    evidence_polarity_and_strength INTEGER,
    FOREIGN KEY (item_id) REFERENCES items(id),
    FOREIGN KEY (ingredient_id) REFERENCES ingredients(id),
    FOREIGN KEY (symptom_id) REFERENCES symptoms(id),
    FOREIGN KEY (paper_id) REFERENCES papers(id)
);
CREATE TABLE derived_features (
    id INTEGER NOT NULL PRIMARY KEY,
    user_id INTEGER NOT NULL,
    item_id INTEGER NOT NULL,
    symptom_id INTEGER NOT NULL,
    time_gap_min_minutes REAL,
    time_gap_avg_minutes REAL,
    cooccurrence_count INTEGER,
    cooccurrence_unique_symptom_count INTEGER,
    pair_density REAL,
    exposure_count_7d INTEGER,
    symptom_count_7d INTEGER,
    severity_avg_after REAL,
    computed_at TEXT,
    UNIQUE (user_id, item_id, symptom_id),
    FOREIGN KEY (user_id) REFERENCES users(id),
    FOREIGN KEY (item_id) REFERENCES items(id),
    FOREIGN KEY (symptom_id) REFERENCES symptoms(id)
);
CREATE TABLE derived_features_ingredients (
    id INTEGER NOT NULL PRIMARY KEY,
    user_id INTEGER NOT NULL,
    ingredient_id INTEGER NOT NULL,
    symptom_id INTEGER NOT NULL,
    time_gap_min_minutes REAL,
    time_gap_avg_minutes REAL,
    cooccurrence_count INTEGER,
    cooccurrence_unique_symptom_count INTEGER,
    pair_density REAL,
    exposure_count_7d INTEGER,
    symptom_count_7d INTEGER,
    severity_avg_after REAL,
    computed_at TEXT,
    UNIQUE (user_id, ingredient_id, symptom_id),
    FOREIGN KEY (user_id) REFERENCES users(id),
    FOREIGN KEY (ingredient_id) REFERENCES ingredients(id),
    FOREIGN KEY (symptom_id) REFERENCES symptoms(id)
);
CREATE TABLE derived_features_combos (
    id INTEGER NOT NULL PRIMARY KEY,
    user_id INTEGER NOT NULL,
    combo_key TEXT NOT NULL,
    item_ids_json TEXT NOT NULL,
    symptom_id INTEGER NOT NULL,
    time_gap_min_minutes REAL,
    time_gap_avg_minutes REAL,
    cooccurrence_count INTEGER,
    cooccurrence_unique_symptom_count INTEGER,
    pair_density REAL,
    exposure_count_7d INTEGER,
    symptom_count_7d INTEGER,
    severity_avg_after REAL,
    computed_at TEXT,
    UNIQUE (user_id, combo_key, symptom_id),
    FOREIGN KEY (user_id) REFERENCES users(id),
    FOREIGN KEY (symptom_id) REFERENCES symptoms(id)
);
CREATE TABLE insights (
    id INTEGER NOT NULL PRIMARY KEY,
    user_id INTEGER NOT NULL,
    item_id INTEGER NOT NULL,
    secondary_item_id INTEGER,
    is_combo INTEGER NOT NULL DEFAULT 0,
    combo_key TEXT,
    combo_item_ids_json TEXT,
    source_ingredient_id INTEGER,
    symptom_id INTEGER NOT NULL,
    model_score REAL,
    evidence_score REAL,
    final_score REAL,
    evidence_summary TEXT,
    evidence_strength_score REAL,
    evidence_quality_score REAL,
    model_probability REAL,
    penalty_score REAL,
    display_decision_reason TEXT,
    citations_json TEXT,
    created_at TEXT,
    FOREIGN KEY (user_id) REFERENCES users(id),
    FOREIGN KEY (item_id) REFERENCES items(id),
    FOREIGN KEY (secondary_item_id) REFERENCES items(id),
    FOREIGN KEY (source_ingredient_id) REFERENCES ingredients(id),
    FOREIGN KEY (symptom_id) REFERENCES symptoms(id)
);
CREATE TABLE insight_event_links (
    id INTEGER NOT NULL PRIMARY KEY,
    user_id INTEGER NOT NULL,
    insight_id INTEGER NOT NULL,
    event_type TEXT NOT NULL CHECK (event_type IN ('exposure', 'symptom')),
    event_id INTEGER NOT NULL,
    created_at TEXT,
    UNIQUE (insight_id, event_type, event_id),
    FOREIGN KEY (user_id) REFERENCES users(id),
    FOREIGN KEY (insight_id) REFERENCES insights(id) ON DELETE CASCADE
);
CREATE TABLE insight_verifications (
    id INTEGER NOT NULL PRIMARY KEY,
    user_id INTEGER NOT NULL,
    item_id INTEGER NOT NULL,
    symptom_id INTEGER NOT NULL,
    verified INTEGER NOT NULL DEFAULT 1,
    rejected INTEGER NOT NULL DEFAULT 0,
    created_at TEXT,
    updated_at TEXT,
    UNIQUE (user_id, item_id, symptom_id),
    FOREIGN KEY (user_id) REFERENCES users(id),
    FOREIGN KEY (item_id) REFERENCES items(id),
    FOREIGN KEY (symptom_id) REFERENCES symptoms(id)
);
CREATE TABLE retrieval_runs (
    id INTEGER NOT NULL PRIMARY KEY,
    user_id INTEGER NOT NULL,
    item_id INTEGER NOT NULL,
    symptom_id INTEGER NOT NULL,
    query_key TEXT NOT NULL,
    top_k INTEGER NOT NULL,
    retrieved_count INTEGER NOT NULL,
    created_at TEXT,
    FOREIGN KEY (user_id) REFERENCES users(id),
    FOREIGN KEY (item_id) REFERENCES items(id),
    FOREIGN KEY (symptom_id) REFERENCES symptoms(id)
);
CREATE TABLE model_retrain_state (
    id INTEGER NOT NULL PRIMARY KEY CHECK (id = 1),
    last_trained_total_events INTEGER NOT NULL DEFAULT 0,
    last_enqueued_total_events INTEGER NOT NULL DEFAULT 0,
    updated_at TEXT
);
CREATE TABLE recurring_exposure_rules (
    id INTEGER NOT NULL PRIMARY KEY,
    user_id INTEGER NOT NULL,
    item_id INTEGER NOT NULL,
    route TEXT NOT NULL,
    start_at TEXT NOT NULL,
    interval_hours INTEGER NOT NULL,
    time_confidence TEXT,
    is_active INTEGER NOT NULL DEFAULT 1,
    last_generated_at TEXT,
    notes TEXT,
    created_at TEXT,
    updated_at TEXT,
    FOREIGN KEY (user_id) REFERENCES users(id),
    FOREIGN KEY (item_id) REFERENCES items(id)
);
CREATE INDEX idx_claims_ingredient_symptom_paper ON claims(ingredient_id, symptom_id, paper_id);
CREATE INDEX idx_claims_item_symptom_paper ON claims(item_id, symptom_id, paper_id);
CREATE INDEX idx_insight_event_links_user_event ON insight_event_links(user_id, event_type, event_id);
CREATE INDEX idx_insight_event_links_insight ON insight_event_links(insight_id);
CREATE INDEX idx_insight_verifications_user ON insight_verifications(user_id);
CREATE INDEX idx_retrieval_runs_user_item_symptom ON retrieval_runs(user_id, item_id, symptom_id);
CREATE INDEX idx_derived_features_combos_user_symptom ON derived_features_combos(user_id, symptom_id);
CREATE INDEX idx_insights_combo_lookup ON insights(user_id, is_combo, combo_key, symptom_id);
CREATE INDEX idx_recurring_rules_user_active ON recurring_exposure_rules(user_id, is_active);
INSERT OR IGNORE INTO model_retrain_state (id, last_trained_total_events, last_enqueued_total_events, updated_at)
VALUES (1, 0, 0, datetime('now'));
