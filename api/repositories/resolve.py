# convert user text names into normalized IDs
# eventually need to make normalized names and aliases UNIQUE in schema

import re
from api.db import get_connection

_NON_ALNUM = re.compile(r"[^a-z0-9]+")

# standardize formatting for input
def _normalize_name(value: str) -> str:
    value = value.strip().lower()
    value = _NON_ALNUM.sub(" ", value)
    return " ".join(value.split())

# check for canonical name lookup and alias lookup
# if missing then create a new name and alias row in respective tables
def resolve_item_id(item_name: str) -> int:
    # normalize input
    normalized = _normalize_name(item_name)
    conn = get_connection()
    cursor = conn.cursor()
    # try a canonical match
    cursor.execute(
        """
        SELECT id FROM items
        WHERE lower(name) = lower(?)
        LIMIT 1
        """,
        (normalized,),
    )
    row = cursor.fetchone()
    if row:
        conn.close()
        return row["id"]
    # try alias match
    cursor.execute(
        """
        SELECT item_id FROM items_aliases
        WHERE lower(alias) = lower(?)
        LIMIT 1
        """,
        (normalized,),
    )
    row = cursor.fetchone()
    if row:
        conn.close()
        return row["item_id"]
    # if not found, create new canonical name and alias
    cursor.execute(
        """
        INSERT INTO items (name, category)
        VALUES (?, ?)
        """,
        (normalized, "Uncategorized"),
    )
    new_id = cursor.lastrowid
    cursor.execute(
        """
        INSERT INTO items_aliases (item_id, alias)
        VALUES (?, ?)
        """,
        (new_id, normalized),
    )
    conn.commit()
    conn.close()
    return new_id

# same thing for symptoms as was done for items
def resolve_symptom_id(symptom_name: str) -> int:
    normalized = _normalize_name(symptom_name)
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        """
        SELECT id FROM symptoms
        WHERE lower(name) = lower(?)
        LIMIT 1
        """,
        (normalized,),
    )
    row = cursor.fetchone()
    if row:
        conn.close()
        return row["id"]

    cursor.execute(
        """
        SELECT symptom_id FROM symptoms_aliases
        WHERE lower(alias) = lower(?)
        LIMIT 1
        """,
        (normalized,),
    )
    row = cursor.fetchone()
    if row:
        conn.close()
        return row["symptom_id"]

    cursor.execute(
        """
        INSERT INTO symptoms (name, description)
        VALUES (?, ?)
        """,
        (normalized, None),
    )
    new_id = cursor.lastrowid
    cursor.execute(
        """
        INSERT INTO symptoms_aliases (symptom_id, alias)
        VALUES (?, ?)
        """,
        (new_id, normalized),
    )
    conn.commit()
    conn.close()
    return new_id
