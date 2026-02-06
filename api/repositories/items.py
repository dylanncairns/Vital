from api.db import get_connection

def list_items():
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM items")
    rows = cursor.fetchall()
    item_list = []
    for row in rows:
        item_list.append({"id": row["id"], "name": row["name"], "category": row["category"]})
    conn.close()
    return item_list