import sqlite3
from pathlib import Path

DB_PATH = Path(__file__).resolve().parent.parent / "data" / "central.db"
SCHEMA_PATH = Path(__file__).resolve().parent.parent / "data" / "schema.sql"

# connect to sqlite DB - will migrate to postgres in future git
def get_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA foreign_keys = ON")
    return conn

def initialize_database():
    if not DB_PATH.exists():
        conn = get_connection()
        with open(SCHEMA_PATH, "r") as f:
            schema_sql = f.read()
        conn.executescript(schema_sql)
        conn.close()

