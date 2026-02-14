from __future__ import annotations

import unittest

import ingestion.ingest_text as ingest_text_mod


class IngestTextRulesTest(unittest.TestCase):
    def setUp(self) -> None:
        self._orig_resolve_symptom_id = ingest_text_mod.resolve_symptom_id
        self._orig_resolve_item_id = ingest_text_mod.resolve_item_id

    def tearDown(self) -> None:
        ingest_text_mod.resolve_symptom_id = self._orig_resolve_symptom_id
        ingest_text_mod.resolve_item_id = self._orig_resolve_item_id

    def test_parse_high_blood_pressure_text_as_symptom(self) -> None:
        ingest_text_mod.resolve_symptom_id = lambda _name: 999
        parsed = ingest_text_mod.parse_with_rules("high blood pressure this morning")
        self.assertIsNotNone(parsed)
        assert parsed is not None
        self.assertEqual(parsed.event_type, "symptom")
        self.assertEqual(parsed.symptom_id, 999)
        self.assertIsNotNone(parsed.timestamp)

    def test_parse_three_sentence_mixed_burb(self) -> None:
        item_ids = {"watermelon": 101}
        symptom_ids = {"high blood pressure": 201, "headache": 202}
        ingest_text_mod.resolve_item_id = lambda name: item_ids.get(name, 999)
        ingest_text_mod.resolve_symptom_id = lambda name: symptom_ids.get(name, 998)
        text = "Had high blood pressure this morning. Then I ate watermelon for breakfast. After lunch I had headache."
        rows = ingest_text_mod.parse_text_events(text)
        self.assertEqual(len(rows), 3)
        self.assertEqual(rows[0].event_type, "symptom")
        self.assertEqual(rows[0].symptom_id, 201)
        self.assertEqual(rows[1].event_type, "exposure")
        self.assertEqual(rows[1].item_id, 101)
        self.assertEqual(rows[2].event_type, "symptom")
        self.assertEqual(rows[2].symptom_id, 202)

    def test_two_days_ago_evening_is_not_today(self) -> None:
        ingest_text_mod.resolve_item_id = lambda _name: 101
        parsed = ingest_text_mod.parse_with_rules("Two days ago, I drank alcohol in the evening.")
        self.assertIsNotNone(parsed)
        assert parsed is not None
        self.assertEqual(parsed.event_type, "exposure")
        self.assertIsNotNone(parsed.timestamp)
        event_ts = ingest_text_mod.datetime.fromisoformat(str(parsed.timestamp).replace("Z", "+00:00"))
        now_utc = ingest_text_mod.datetime.now(tz=ingest_text_mod.timezone.utc)
        delta_days = (now_utc.date() - event_ts.date()).days
        self.assertIn(delta_days, {2, 3})

    def test_normalize_symptom_candidate_from_phrase(self) -> None:
        normalized = ingest_text_mod._normalize_symptom_candidate("also some new acne spots")
        self.assertEqual(normalized, "acne")

    def test_split_exposure_items_handles_repeated_had_clause(self) -> None:
        items = ingest_text_mod._split_exposure_items(
            "For dinner tonight, I had a steak and I had sweet potatoes."
        )
        self.assertIn("steak", items)
        self.assertIn("sweet potatoes", items)

    def test_split_exposure_items_handles_plain_conjunction_list(self) -> None:
        items = ingest_text_mod._split_exposure_items("chicken and rice")
        self.assertIn("chicken", items)
        self.assertIn("rice", items)

    def test_parse_time_handles_absolute_month_day(self) -> None:
        ts, start, end = ingest_text_mod._parse_time("On February 10, I had mango in the evening.")
        self.assertIsNotNone(ts)
        self.assertIsNone(start)
        self.assertIsNone(end)
        assert ts is not None
        parsed = ingest_text_mod.datetime.fromisoformat(ts.replace("Z", "+00:00"))
        local = parsed.astimezone()
        self.assertEqual(local.month, 2)
        self.assertEqual(local.day, 10)

    def test_clean_candidate_strips_also_prefix(self) -> None:
        self.assertEqual(ingest_text_mod._clean_candidate_text("also weed"), "weed")


if __name__ == "__main__":
    unittest.main()
