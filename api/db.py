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
        ("model_probability", "REAL"),
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
def _apply_migrations(conn: sqlite3.Connection) -> None:
    migrations: list[Callable[[sqlite3.Connection], None]] = [
        _migration_001_insight_and_rag_scaffolding,
    ]
    for migration in migrations:
        migration(conn)

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
