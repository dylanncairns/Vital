import os
from pathlib import Path
from typing import Callable

from psycopg import Connection, connect
from psycopg.rows import dict_row

SCHEMA_PATH = Path(__file__).resolve().parent.parent / "data" / "schema.sql"

# connect to postgres DB
def get_connection():
    database_url = os.getenv("DATABASE_URL", "").strip()
    if not database_url:
        raise RuntimeError("DATABASE_URL is required")
    conn = connect(database_url, row_factory=dict_row)
    return conn

# test table presence before altering
def _table_exists(conn: Connection, table_name: str) -> bool:
    row = conn.execute(
        """
        SELECT 1
        FROM information_schema.tables
        WHERE table_schema = 'public'
          AND table_name = %s
        LIMIT 1
        """,
        (table_name,),
    ).fetchone()
    return row is not None

# prevents second startup after migration from causing duplicate column errors
def _column_exists(conn: Connection, table_name: str, column_name: str) -> bool:
    if not _table_exists(conn, table_name):
        return False
    row = conn.execute(
        """
        SELECT 1
        FROM information_schema.columns
        WHERE table_schema = 'public'
          AND table_name = %s
          AND column_name = %s
        LIMIT 1
        """,
        (table_name, column_name),
    ).fetchone()
    return row is not None


def _column_data_type(conn: Connection, table_name: str, column_name: str) -> str | None:
    if not _table_exists(conn, table_name):
        return None
    row = conn.execute(
        """
        SELECT data_type
        FROM information_schema.columns
        WHERE table_schema = 'public'
          AND table_name = %s
          AND column_name = %s
        LIMIT 1
        """,
        (table_name, column_name),
    ).fetchone()
    if row is None:
        return None
    return str(row["data_type"]).lower()

# prepare for rag evidence/citation retrival
def _migration_001_insight_and_rag_scaffolding(conn: Connection) -> None:
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
            id BIGSERIAL PRIMARY KEY,
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
def _migration_002_cooccurrence_semantics(conn: Connection) -> None:
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
            id BIGSERIAL PRIMARY KEY,
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


def _claims_requires_item_support_rebuild(conn: Connection) -> bool:
    if not _table_exists(conn, "claims"):
        return False
    rows = conn.execute(
        """
        SELECT column_name, is_nullable
        FROM information_schema.columns
        WHERE table_schema = 'public'
          AND table_name = 'claims'
        """
    ).fetchall()
    has_item_id = any(row["column_name"] == "item_id" for row in rows)
    ingredient_notnull = False
    for row in rows:
        if row["column_name"] == "ingredient_id":
            ingredient_notnull = row["is_nullable"] == "NO"
            break
    return (not has_item_id) or ingredient_notnull


def _migration_003_claims_item_support(conn: Connection) -> None:
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
            id BIGSERIAL PRIMARY KEY,
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
            evidence_polarity_and_strength DOUBLE PRECISION,
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
            citation_title, citation_url, citation_snippet,
            study_design, study_quality_score, population_match, temporality_match, risk_of_bias, llm_confidence,
            evidence_polarity_and_strength
        )
        SELECT
            id, ingredient_id, symptom_id, paper_id, claim_type, summary,
            chunk_index, chunk_text, chunk_hash, embedding_model, embedding_vector,
            citation_title, citation_url, citation_snippet,
            study_design, study_quality_score, population_match, temporality_match, risk_of_bias, llm_confidence,
            evidence_polarity_and_strength
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


def _apply_migrations(conn: Connection) -> None:
    migrations: list[Callable[[Connection], None]] = [
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
        _migration_013_rag_source_documents,
        _migration_014_claims_polarity_float,
        _migration_015_claims_evidence_metadata_columns,
    ]
    for migration in migrations:
        migration(conn)


def _migration_004_background_jobs(conn: Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS background_jobs (
            id BIGSERIAL PRIMARY KEY,
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


def _migration_005_model_retrain_state(conn: Connection) -> None:
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
            ) VALUES (1, 0, 0, NOW())
            """
        )


def _migration_006_recurring_exposure_rules(conn: Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS recurring_exposure_rules (
            id BIGSERIAL PRIMARY KEY,
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


def _migration_007_insight_event_links(conn: Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS insight_event_links (
            id BIGSERIAL PRIMARY KEY,
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


def _migration_008_insight_verifications(conn: Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS insight_verifications (
            id BIGSERIAL PRIMARY KEY,
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


def _migration_009_auth(conn: Connection) -> None:
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
            id BIGSERIAL PRIMARY KEY,
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


def _migration_010_insight_rejections(conn: Connection) -> None:
    if not _column_exists(conn, "insight_verifications", "rejected"):
        conn.execute("ALTER TABLE insight_verifications ADD COLUMN rejected INTEGER NOT NULL DEFAULT 0")


def _migration_011_insight_source_ingredient(conn: Connection) -> None:
    if not _column_exists(conn, "insights", "source_ingredient_id"):
        conn.execute("ALTER TABLE insights ADD COLUMN source_ingredient_id INTEGER")


def _migration_012_combo_candidates(conn: Connection) -> None:
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
            id BIGSERIAL PRIMARY KEY,
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


def _migration_013_rag_source_documents(conn: Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS rag_source_documents (
            id BIGSERIAL PRIMARY KEY,
            user_id INTEGER,
            query TEXT,
            original_query TEXT,
            filename TEXT NOT NULL,
            title TEXT NOT NULL,
            url TEXT,
            publication_date TEXT,
            source TEXT,
            abstract TEXT,
            payload_json TEXT NOT NULL,
            source_hash TEXT NOT NULL UNIQUE,
            created_at TEXT,
            updated_at TEXT,
            FOREIGN KEY (user_id) REFERENCES users(id)
        )
        """
    )
    conn.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_rag_source_documents_user_created
        ON rag_source_documents(user_id, created_at)
        """
    )
    conn.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_rag_source_documents_hash
        ON rag_source_documents(source_hash)
        """
    )


def _migration_014_claims_polarity_float(conn: Connection) -> None:
    if not _column_exists(conn, "claims", "evidence_polarity_and_strength"):
        return
    data_type = _column_data_type(conn, "claims", "evidence_polarity_and_strength")
    if data_type in {"smallint", "integer", "bigint"}:
        conn.execute(
            """
            ALTER TABLE claims
            ALTER COLUMN evidence_polarity_and_strength TYPE DOUBLE PRECISION
            USING evidence_polarity_and_strength::double precision
            """
        )


def _migration_015_claims_evidence_metadata_columns(conn: Connection) -> None:
    claim_columns = [
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


def _execute_script(conn: Connection, script: str) -> None:
    with conn.cursor() as cursor:
        cursor.execute(script)


def initialize_database():
    conn = get_connection()
    try:
        if not _table_exists(conn, "users"):
            with open(SCHEMA_PATH, "r") as f:
                schema_sql = f.read()
            _execute_script(conn, schema_sql)
        _apply_migrations(conn)
        conn.commit()
    finally:
        conn.close()
