from __future__ import annotations

import api.db


_TEST_TABLES = (
    "auth_sessions",
    "background_jobs",
    "claims",
    "derived_features",
    "derived_features_combos",
    "derived_features_ingredients",
    "exposure_events",
    "exposure_expansions",
    "ingredients",
    "ingredients_aliases",
    "insight_event_links",
    "insight_verifications",
    "insights",
    "items",
    "items_aliases",
    "items_ingredients",
    "model_retrain_state",
    "papers",
    "raw_event_ingest",
    "rag_source_documents",
    "recurring_exposure_rules",
    "retrieval_runs",
    "symptom_events",
    "symptoms",
    "symptoms_aliases",
    "users",
)


def reset_test_database() -> None:
    api.db.initialize_database()
    conn = api.db.get_connection()
    try:
        conn.execute(f"TRUNCATE TABLE {', '.join(_TEST_TABLES)} RESTART IDENTITY CASCADE")
        conn.execute(
            """
            INSERT INTO model_retrain_state (
                id, last_trained_total_events, last_enqueued_total_events, updated_at
            )
            VALUES (1, 0, 0, NOW())
            ON CONFLICT (id) DO NOTHING
            """
        )
        conn.commit()
    finally:
        conn.close()
