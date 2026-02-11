from api.db import get_connection

# after exposure event insert, look up the item's ingredients via items_ingredients table
# expand one exposure event into many exposure_expansions rows (references ingredients in that item)
# participates in DB connection passed by caller
# if none passed then this function owns open/commit/close of new connection
# enables atomic transactions and handles partial writing failures
def expand_exposure_event(exposure_event_id: int, conn=None) -> None:
    own_connection = conn is None
    if conn is None:
        conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        """
        SELECT e.id AS exposure_event_id, ii.ingredient_id
        FROM exposure_events e
        JOIN items_ingredients ii ON ii.item_id = e.item_id
        WHERE e.id = ?
        """,
        (exposure_event_id,),
    )
    rows = cursor.fetchall()
    if not rows:
        # nothing to expand for this exposure event
        if own_connection:
            conn.close()
        return
    # else expand all ingredients of an item into entries of individual exposures
    conn.executemany(
        """
        INSERT INTO exposure_expansions (exposure_event_id, ingredient_id)
        VALUES (?, ?)
        """,
        [(row["exposure_event_id"], row["ingredient_id"]) for row in rows],
    )
    if own_connection:
        # commit only when this function owns the transaction
        conn.commit()
        conn.close()
