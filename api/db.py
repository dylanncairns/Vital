import sqlite3
from pathlib import Path
from typing import Callable

DB_PATH = Path(__file__).resolve().parent.parent / "data" / "central.db"
SCHEMA_PATH = Path(__file__).resolve().parent.parent / "data" / "schema.sql"

# connect to sqlite DB - will migrate to postgres in future git
def get_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA foreign_keys = ON")
    return conn

# test table presence before altering
def _table_exists(conn: sqlite3.Connection, table_name: str) -> bool:
    row = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ?",
        (table_name,),
    ).fetchone()
    return row is not None

# prevents second startup after migration from causing duplicate column errors
def _column_exists(conn: sqlite3.Connection, table_name: str, column_name: str) -> bool:
    if not _table_exists(conn, table_name):
        return False
    rows = conn.execute(f"PRAGMA table_info({table_name})").fetchall()
    return any(row["name"] == column_name for row in rows)

# prepare for rag evidence/citation retrival
def _migration_001_insight_and_rag_scaffolding(conn: sqlite3.Connection) -> None:
    # Expand insights for decision transparency and API payloads
    insight_columns = [
        ("evidence_summary", "TEXT"),
        ("evidence_strength_score", "REAL"),
        ("evidence_quality_score", "REAL"),
        ("model_probability", "REAL"),
        ("penalty_score", "REAL"),
        ("display_decision_reason", "TEXT"),
        ("citations_json", "TEXT"),
    ]
    for column_name, column_type in insight_columns:
        if not _column_exists(conn, "insights", column_name):
            conn.execute(f"ALTER TABLE insights ADD COLUMN {column_name} {column_type}")

    # Expand claims so a claim can hold chunk and citation metadata directly
    claim_columns = [
        ("claim_type", "TEXT"),
        ("chunk_index", "INTEGER"),
        ("chunk_text", "TEXT"),
        ("chunk_hash", "TEXT"),
        ("embedding_model", "TEXT"),
        ("embedding_vector", "TEXT"),
        ("citation_title", "TEXT"),
        ("citation_url", "TEXT"),
        ("citation_snippet", "TEXT"),
        ("study_design", "TEXT"),
        ("study_quality_score", "REAL"),
        ("population_match", "REAL"),
        ("temporality_match", "REAL"),
        ("risk_of_bias", "REAL"),
        ("llm_confidence", "REAL"),
    ]
    for column_name, column_type in claim_columns:
        if not _column_exists(conn, "claims", column_name):
            conn.execute(f"ALTER TABLE claims ADD COLUMN {column_name} {column_type}")

    # Retrieval run bookkeeping
    # Specific and unique per user and per run
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS retrieval_runs (
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
        )
        """
    )
    conn.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_retrieval_runs_user_item_symptom
        ON retrieval_runs(user_id, item_id, symptom_id)
        """
    )
    conn.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_claims_ingredient_symptom_paper
        ON claims(ingredient_id, symptom_id, paper_id)
        """
    )

# to add missing columns for existing DBs
def _migration_002_cooccurrence_semantics(conn: sqlite3.Connection) -> None:
    derived_feature_columns = [
        ("cooccurrence_unique_symptom_count", "INTEGER"),
        ("pair_density", "REAL"),
    ]
    for column_name, column_type in derived_feature_columns:
        if not _column_exists(conn, "derived_features", column_name):
            conn.execute(f"ALTER TABLE derived_features ADD COLUMN {column_name} {column_type}")

    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS derived_features_ingredients (
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
        )
        """
    )


def _claims_requires_item_support_rebuild(conn: sqlite3.Connection) -> bool:
    if not _table_exists(conn, "claims"):
        return False
    rows = conn.execute("PRAGMA table_info(claims)").fetchall()
    has_item_id = any(row["name"] == "item_id" for row in rows)
    ingredient_notnull = False
    for row in rows:
        if row["name"] == "ingredient_id":
            ingredient_notnull = bool(row["notnull"])
            break
    return (not has_item_id) or ingredient_notnull


def _migration_003_claims_item_support(conn: sqlite3.Connection) -> None:
    if not _claims_requires_item_support_rebuild(conn):
        conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_claims_item_symptom_paper
            ON claims(item_id, symptom_id, paper_id)
            """
        )
        return

    conn.execute(
        """
        CREATE TABLE claims_new (
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
            evidence_polarity_and_strength INTEGER,
            FOREIGN KEY (item_id) REFERENCES items(id),
            FOREIGN KEY (ingredient_id) REFERENCES ingredients(id),
            FOREIGN KEY (symptom_id) REFERENCES symptoms(id),
            FOREIGN KEY (paper_id) REFERENCES papers(id)
        )
        """
    )
    conn.execute(
        """
        INSERT INTO claims_new (
            id, ingredient_id, symptom_id, paper_id, claim_type, summary,
            chunk_index, chunk_text, chunk_hash, embedding_model, embedding_vector,
            citation_title, citation_url, citation_snippet, evidence_polarity_and_strength
        )
        SELECT
            id, ingredient_id, symptom_id, paper_id, claim_type, summary,
            chunk_index, chunk_text, chunk_hash, embedding_model, embedding_vector,
            citation_title, citation_url, citation_snippet, evidence_polarity_and_strength
        FROM claims
        """
    )
    conn.execute("DROP TABLE claims")
    conn.execute("ALTER TABLE claims_new RENAME TO claims")
    conn.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_claims_ingredient_symptom_paper
        ON claims(ingredient_id, symptom_id, paper_id)
        """
    )
    conn.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_claims_item_symptom_paper
        ON claims(item_id, symptom_id, paper_id)
        """
    )


def _apply_migrations(conn: sqlite3.Connection) -> None:
    migrations: list[Callable[[sqlite3.Connection], None]] = [
        _migration_001_insight_and_rag_scaffolding,
        _migration_002_cooccurrence_semantics,
        _migration_003_claims_item_support,
        _migration_004_background_jobs,
        _migration_005_model_retrain_state,
        _migration_006_recurring_exposure_rules,
        _migration_007_insight_event_links,
        _migration_008_insight_verifications,
        _migration_009_auth,
        _migration_010_insight_rejections,
        _migration_011_insight_source_ingredient,
        _migration_012_combo_candidates,
    ]
    for migration in migrations:
        migration(conn)


def _migration_004_background_jobs(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS background_jobs (
            id INTEGER NOT NULL PRIMARY KEY,
            user_id INTEGER NOT NULL,
            job_type TEXT NOT NULL,
            item_id INTEGER,
            symptom_id INTEGER,
            payload_json TEXT,
            status TEXT NOT NULL DEFAULT 'pending',
            attempts INTEGER NOT NULL DEFAULT 0,
            last_error TEXT,
            created_at TEXT,
            updated_at TEXT,
            FOREIGN KEY (user_id) REFERENCES users(id),
            FOREIGN KEY (item_id) REFERENCES items(id),
            FOREIGN KEY (symptom_id) REFERENCES symptoms(id)
        )
        """
    )
    conn.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_background_jobs_status_created
        ON background_jobs(status, created_at)
        """
    )
    conn.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_background_jobs_user_type
        ON background_jobs(user_id, job_type)
        """
    )


def _migration_005_model_retrain_state(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS model_retrain_state (
            id INTEGER NOT NULL PRIMARY KEY CHECK (id = 1),
            last_trained_total_events INTEGER NOT NULL DEFAULT 0,
            last_enqueued_total_events INTEGER NOT NULL DEFAULT 0,
            updated_at TEXT
        )
        """
    )
    if conn.execute("SELECT 1 FROM model_retrain_state WHERE id = 1").fetchone() is None:
        conn.execute(
            """
            INSERT INTO model_retrain_state (
                id, last_trained_total_events, last_enqueued_total_events, updated_at
            ) VALUES (1, 0, 0, datetime('now'))
            """
        )


def _migration_006_recurring_exposure_rules(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS recurring_exposure_rules (
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
        )
        """
    )
    conn.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_recurring_rules_user_active
        ON recurring_exposure_rules(user_id, is_active)
        """
    )


def _migration_007_insight_event_links(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS insight_event_links (
            id INTEGER NOT NULL PRIMARY KEY,
            user_id INTEGER NOT NULL,
            insight_id INTEGER NOT NULL,
            event_type TEXT NOT NULL CHECK (event_type IN ('exposure', 'symptom')),
            event_id INTEGER NOT NULL,
            created_at TEXT,
            UNIQUE (insight_id, event_type, event_id),
            FOREIGN KEY (user_id) REFERENCES users(id),
            FOREIGN KEY (insight_id) REFERENCES insights(id) ON DELETE CASCADE
        )
        """
    )
    conn.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_insight_event_links_user_event
        ON insight_event_links(user_id, event_type, event_id)
        """
    )
    conn.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_insight_event_links_insight
        ON insight_event_links(insight_id)
        """
    )


def _migration_008_insight_verifications(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS insight_verifications (
            id INTEGER NOT NULL PRIMARY KEY,
            user_id INTEGER NOT NULL,
            item_id INTEGER NOT NULL,
            symptom_id INTEGER NOT NULL,
            verified INTEGER NOT NULL DEFAULT 1,
            created_at TEXT,
            updated_at TEXT,
            UNIQUE (user_id, item_id, symptom_id),
            FOREIGN KEY (user_id) REFERENCES users(id),
            FOREIGN KEY (item_id) REFERENCES items(id),
            FOREIGN KEY (symptom_id) REFERENCES symptoms(id)
        )
        """
    )
    conn.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_insight_verifications_user
        ON insight_verifications(user_id)
        """
    )


def _migration_009_auth(conn: sqlite3.Connection) -> None:
    if not _column_exists(conn, "users", "username"):
        conn.execute("ALTER TABLE users ADD COLUMN username TEXT")
    if not _column_exists(conn, "users", "password_hash"):
        conn.execute("ALTER TABLE users ADD COLUMN password_hash TEXT")
    conn.execute(
        """
        CREATE UNIQUE INDEX IF NOT EXISTS idx_users_username_unique
        ON users(username)
        WHERE username IS NOT NULL
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS auth_sessions (
            id INTEGER NOT NULL PRIMARY KEY,
            user_id INTEGER NOT NULL,
            token TEXT NOT NULL UNIQUE,
            created_at TEXT NOT NULL,
            expires_at TEXT NOT NULL,
            revoked_at TEXT,
            FOREIGN KEY (user_id) REFERENCES users(id)
        )
        """
    )
    conn.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_auth_sessions_user
        ON auth_sessions(user_id)
        """
    )
    conn.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_auth_sessions_token
        ON auth_sessions(token)
        """
    )


def _migration_010_insight_rejections(conn: sqlite3.Connection) -> None:
    if not _column_exists(conn, "insight_verifications", "rejected"):
        conn.execute("ALTER TABLE insight_verifications ADD COLUMN rejected INTEGER NOT NULL DEFAULT 0")


def _migration_011_insight_source_ingredient(conn: sqlite3.Connection) -> None:
    if not _column_exists(conn, "insights", "source_ingredient_id"):
        conn.execute("ALTER TABLE insights ADD COLUMN source_ingredient_id INTEGER")


def _migration_012_combo_candidates(conn: sqlite3.Connection) -> None:
    combo_columns = [
        ("secondary_item_id", "INTEGER"),
        ("is_combo", "INTEGER NOT NULL DEFAULT 0"),
        ("combo_key", "TEXT"),
        ("combo_item_ids_json", "TEXT"),
    ]
    for column_name, column_type in combo_columns:
        if not _column_exists(conn, "insights", column_name):
            conn.execute(f"ALTER TABLE insights ADD COLUMN {column_name} {column_type}")

    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS derived_features_combos (
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
        )
        """
    )
    conn.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_derived_features_combos_user_symptom
        ON derived_features_combos(user_id, symptom_id)
        """
    )
    conn.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_insights_combo_lookup
        ON insights(user_id, is_combo, combo_key, symptom_id)
        """
    )

def initialize_database():
    db_exists = DB_PATH.exists()
    conn = get_connection()
    try:
        if not db_exists:
            with open(SCHEMA_PATH, "r") as f:
                schema_sql = f.read()
            conn.executescript(schema_sql)
        _apply_migrations(conn)
        conn.commit()
    finally:
        conn.close()
