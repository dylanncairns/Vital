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
        WHERE e.id = %s
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
    cursor.executemany(
        """
        INSERT INTO exposure_expansions (exposure_event_id, ingredient_id)
        VALUES (%s, %s)
        ON CONFLICT (exposure_event_id, ingredient_id) DO NOTHING
        """,
        [(row["exposure_event_id"], row["ingredient_id"]) for row in rows],
    )
    if own_connection:
        # commit only when this function owns the transaction
        conn.commit()
        conn.close()


def backfill_missing_exposure_expansions(user_id: int | None = None, conn=None) -> int:
    own_connection = conn is None
    if conn is None:
        conn = get_connection()
    cursor = conn.cursor()
    if user_id is None:
        cursor.execute(
            """
            SELECT e.id AS exposure_event_id
            FROM exposure_events e
            LEFT JOIN exposure_expansions x ON x.exposure_event_id = e.id
            WHERE x.id IS NULL
            ORDER BY e.id ASC
            """
        )
    else:
        cursor.execute(
            """
            SELECT e.id AS exposure_event_id
            FROM exposure_events e
            LEFT JOIN exposure_expansions x ON x.exposure_event_id = e.id
            WHERE e.user_id = %s
              AND x.id IS NULL
            ORDER BY e.id ASC
            """,
            (user_id,),
        )
    exposure_ids = [int(row["exposure_event_id"]) for row in cursor.fetchall()]
    for exposure_id in exposure_ids:
        expand_exposure_event(exposure_id, conn=conn)
    if own_connection:
        conn.commit()
        conn.close()
    return len(exposure_ids)
