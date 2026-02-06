CREATE TABLE items (
    id INTEGER NOT NULL PRIMARY KEY,
    name TEXT,
    category TEXT
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
CREATE TABLE evidence (
    id INTEGER NOT NULL PRIMARY KEY,
    title TEXT,
    url TEXT,
    publication_date TEXT
);
CREATE TABLE claims (
    id INTEGER NOT NULL PRIMARY KEY,
    ingredient_id INTEGER NOT NULL,
    evidence_id INTEGER NOT NULL,
    summary TEXT,
    FOREIGN KEY (ingredient_id) REFERENCES ingredients(id),
    FOREIGN KEY (evidence_id) REFERENCES evidence(id)
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
    timestamp TEXT NOT NULL,
    route TEXT NOT NULL,
    FOREIGN KEY (user_id) REFERENCES users(id),
    FOREIGN KEY (item_id) REFERENCES items(id)
);
CREATE TABLE symptoms (
    id INTEGER NOT NULL PRIMARY KEY,
    name TEXT,
    description TEXT
);
CREATE TABLE symptom_events (
    id INTEGER NOT NULL PRIMARY KEY,
    user_id INTEGER NOT NULL,
    symptom_id INTEGER NOT NULL,
    timestamp TEXT,
    severity INTEGER,
    FOREIGN KEY (user_id) REFERENCES users(id),
    FOREIGN KEY (symptom_id) REFERENCES symptoms(id)
);