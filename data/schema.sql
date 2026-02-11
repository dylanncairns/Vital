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
    name TEXT
);
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
    ingredient_id INTEGER NOT NULL,
    symptom_id INTEGER NOT NULL,
    paper_id INTEGER NOT NULL,
    summary TEXT,
    evidence_polarity_and_strength INTEGER,
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
    exposure_count_7d INTEGER,
    symptom_count_7d INTEGER,
    severity_avg_after REAL,
    computed_at TEXT,
    UNIQUE (user_id, item_id, symptom_id),
    FOREIGN KEY (user_id) REFERENCES users(id),
    FOREIGN KEY (item_id) REFERENCES items(id),
    FOREIGN KEY (symptom_id) REFERENCES symptoms(id)
);
CREATE TABLE insights (
    id INTEGER NOT NULL PRIMARY KEY,
    user_id INTEGER NOT NULL,
    item_id INTEGER NOT NULL,
    symptom_id INTEGER NOT NULL,
    model_score REAL,
    evidence_score REAL,
    final_score REAL,
    created_at TEXT,
    FOREIGN KEY (user_id) REFERENCES users(id),
    FOREIGN KEY (item_id) REFERENCES items(id),
    FOREIGN KEY (symptom_id) REFERENCES symptoms(id)
);
