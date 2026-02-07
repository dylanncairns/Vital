from api.db import get_connection

# Pulls list of events when /events reached via get for a given user id to create timeline
def list_events(user_id: int):
    conn = get_connection()
    cursor = conn.cursor
    # First select reads exposure events, joins items and exposures at item_name, sets event_type to exposure
    # Second select reads symptom events, joins symptoms and events at symptom_name, sets event_type to symptom
    # Union all merges the two lists while preserving all events (doesnt remove duplicate events)
    cursor.execute("""
        SELECT
            e.id AS id,
            'exposure' AS event_type,
            e.user_id AS user_id,
            e.timestamp AS timestamp,
            e.item_id AS item_id,
            i.name AS item_name,
            e.route AS route,
            NULL AS symptom_id,
            NULL AS symptom_name,
            NULL AS severity
        FROM exposure_events e JOIN items i ON i.id = e.item_id
        WHERE e.user_id = ?
                   
        UNION ALL
                   
        SELECT
            s.id AS id,
            'symptom' AS event_type,
            s.user_id AS user_id,
            s.timestamp AS timestamp,
            NULL AS item_id,
            NULL AS item_name,
            NULL AS route,
            s.symptom_id AS symptom_id,
            sy.name AS symptom_name,
            s.severity AS severity
        FROM symptom_events s
        JOIN symptoms sy ON sy.id = s.symptom_id
        WHERE s.user_id = ?
        
        ORDER BY timestamp
        """,
        (user_id, user_id,)
    )
    rows = cursor.fetchall()
    conn.close()
    # JSON friendly return
    return [dict(row) for row in rows]