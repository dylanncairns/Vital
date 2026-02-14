from __future__ import annotations

import argparse
import json
from pathlib import Path

from api.db import get_connection


def _norm(value: str) -> str:
    return " ".join((value or "").strip().lower().split())


def load_catalog(path: Path) -> dict[str, int]:
    payload = json.loads(path.read_text())
    if not isinstance(payload, list):
        raise ValueError("Catalog must be a JSON array")

    conn = get_connection()
    try:
        item_lookup = {
            _norm(row["name"]): int(row["id"])
            for row in conn.execute("SELECT id, name FROM items WHERE name IS NOT NULL").fetchall()
        }
        ing_lookup = {
            _norm(row["name"]): int(row["id"])
            for row in conn.execute("SELECT id, name FROM ingredients WHERE name IS NOT NULL").fetchall()
        }
        created_items = 0
        created_ingredients = 0
        created_links = 0
        for row in payload:
            if not isinstance(row, dict):
                continue
            item_name = str(row.get("item") or "").strip()
            if not item_name:
                continue
            category = str(row.get("category") or "general").strip() or "general"
            ingredient_names = row.get("ingredients") or []
            if not isinstance(ingredient_names, list):
                continue

            item_key = _norm(item_name)
            item_id = item_lookup.get(item_key)
            if item_id is None:
                cur = conn.execute(
                    "INSERT INTO items (name, category) VALUES (?, ?)",
                    (item_name, category),
                )
                item_id = int(cur.lastrowid)
                item_lookup[item_key] = item_id
                created_items += 1

            for ingredient_name_raw in ingredient_names:
                ingredient_name = str(ingredient_name_raw or "").strip()
                if not ingredient_name:
                    continue
                ing_key = _norm(ingredient_name)
                ingredient_id = ing_lookup.get(ing_key)
                if ingredient_id is None:
                    cur = conn.execute(
                        "INSERT INTO ingredients (name, description) VALUES (?, ?)",
                        (ingredient_name, "catalog"),
                    )
                    ingredient_id = int(cur.lastrowid)
                    ing_lookup[ing_key] = ingredient_id
                    created_ingredients += 1

                before = conn.total_changes
                conn.execute(
                    "INSERT OR IGNORE INTO items_ingredients (item_id, ingredient_id) VALUES (?, ?)",
                    (int(item_id), int(ingredient_id)),
                )
                if conn.total_changes > before:
                    created_links += 1

        conn.commit()
        return {
            "created_items": created_items,
            "created_ingredients": created_ingredients,
            "created_links": created_links,
            "catalog_rows": len(payload),
        }
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Load curated item->ingredient catalog into DB.")
    parser.add_argument(
        "--path",
        type=str,
        default="data/item_ingredient_catalog.json",
        help="Path to catalog JSON file",
    )
    args = parser.parse_args()
    print(load_catalog(Path(args.path)))


if __name__ == "__main__":
    main()

