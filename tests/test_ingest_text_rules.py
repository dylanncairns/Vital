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
        # Date resolution is timezone-sensitive around UTC/local midnight boundaries.
        self.assertIn(delta_days, {1, 2, 3})

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

    def test_parse_with_rules_requires_resolved_ids(self) -> None:
        ingest_text_mod.resolve_item_id = lambda _name: None
        ingest_text_mod.resolve_symptom_id = lambda _name: None
        self.assertIsNone(ingest_text_mod.parse_with_rules("I drank mystery thing last night"))
        self.assertIsNone(ingest_text_mod.parse_with_rules("I had a strange sensation this morning"))

    def test_long_multisentence_parsing_avoids_artifacts(self) -> None:
        item_ids = {
            "water": 11,
            "chicken": 12,
            "rice": 13,
            "pizza": 14,
            "alcohol": 15,
        }
        symptom_ids = {
            "high blood pressure": 21,
            "stomachache": 22,
            "headache": 23,
            "acne": 24,
        }
        ingest_text_mod.resolve_item_id = lambda name: item_ids.get(name)
        ingest_text_mod.resolve_symptom_id = lambda name: symptom_ids.get(name)
        text = (
            "Had high blood pressure yesterday morning, so I drank water at 10 am. "
            "Then I went to work from 10-5 and for lunch I had chicken and rice. "
            "When I got home I had pizza. Then in the evening I had a stomachache. "
            "Two days ago, I drank alcohol in the evening. "
            "Then yesterday morning I had a headache. "
            "This morning I also had some new acne spots."
        )

        rows = ingest_text_mod.parse_text_events(text)

        exposure_item_ids = {row.item_id for row in rows if row.event_type == "exposure" and row.item_id is not None}
        symptom_item_ids = {row.symptom_id for row in rows if row.event_type == "symptom" and row.symptom_id is not None}
        # parse_text_events keeps atomic parsed rows; fan-out for list exposures (e.g., "chicken and rice")
        # is validated in ingest_text_event tests against DB writes.
        self.assertEqual(exposure_item_ids, {11, 12, 14, 15})
        self.assertEqual(symptom_item_ids, {21, 22, 23, 24})

        # Ensure the classic artifacts are not emitted as resolved rows.
        self.assertNotIn(None, exposure_item_ids)
        self.assertNotIn(None, symptom_item_ids)

    def test_relative_date_chain_two_days_ago_then_yesterday(self) -> None:
        item_ids = {"alcohol": 31}
        symptom_ids = {"headache": 41}
        ingest_text_mod.resolve_item_id = lambda name: item_ids.get(name)
        ingest_text_mod.resolve_symptom_id = lambda name: symptom_ids.get(name)
        text = "Two days ago, I drank alcohol in the evening. Then yesterday morning I had a headache."
        rows = ingest_text_mod.parse_text_events(text)
        alcohol_rows = [row for row in rows if row.event_type == "exposure" and row.item_id == 31]
        headache_rows = [row for row in rows if row.event_type == "symptom" and row.symptom_id == 41]
        self.assertEqual(len(alcohol_rows), 1)
        self.assertEqual(len(headache_rows), 1)
        alcohol_ts = ingest_text_mod.datetime.fromisoformat(str(alcohol_rows[0].timestamp).replace("Z", "+00:00"))
        headache_ts = ingest_text_mod.datetime.fromisoformat(str(headache_rows[0].timestamp).replace("Z", "+00:00"))
        self.assertLess(alcohol_ts, headache_ts)

    def test_feb_night_sentence_parses_clean_exposures(self) -> None:
        item_ids = {"alcohol": 51, "poor sleep": 52}
        symptom_ids = {"headache": 61}
        ingest_text_mod.resolve_item_id = lambda name: item_ids.get(name)
        ingest_text_mod.resolve_symptom_id = lambda name: symptom_ids.get(name)
        text = (
            "Drank alcohol on the night of Feb 11 and had poor sleep that night. "
            "Had a headache morning of Feb 12."
        )
        rows = ingest_text_mod.parse_text_events(text)
        exposure_rows = [row for row in rows if row.event_type == "exposure"]
        symptom_rows = [row for row in rows if row.event_type == "symptom"]

        self.assertEqual({row.item_id for row in exposure_rows}, {51, 52})
        self.assertEqual({row.symptom_id for row in symptom_rows}, {61})

    def test_infer_route_poor_sleep_not_ingestion(self) -> None:
        self.assertEqual(ingest_text_mod._infer_route("had poor sleep that night"), "behavioral")

    def test_anaphoric_that_night_inherits_prior_date(self) -> None:
        item_ids = {"alcohol": 71, "poor sleep": 72}
        symptom_ids = {"headache": 81}
        ingest_text_mod.resolve_item_id = lambda name: item_ids.get(name)
        ingest_text_mod.resolve_symptom_id = lambda name: symptom_ids.get(name)
        text = (
            "Drank alcohol on the night of Feb 11 and had poor sleep that night. "
            "Had a headache morning of Feb 12."
        )
        rows = ingest_text_mod.parse_text_events(text)
        alcohol = next(row for row in rows if row.event_type == "exposure" and row.item_id == 71)
        poor_sleep = next(row for row in rows if row.event_type == "exposure" and row.item_id == 72)
        headache = next(row for row in rows if row.event_type == "symptom" and row.symptom_id == 81)
        alcohol_ts = ingest_text_mod.datetime.fromisoformat(str(alcohol.timestamp).replace("Z", "+00:00"))
        poor_sleep_ts = ingest_text_mod.datetime.fromisoformat(str(poor_sleep.timestamp).replace("Z", "+00:00"))
        headache_ts = ingest_text_mod.datetime.fromisoformat(str(headache.timestamp).replace("Z", "+00:00"))
        # "that night" should anchor to the same date context as the Feb 11 alcohol clause.
        self.assertEqual(alcohol_ts.date(), poor_sleep_ts.date())
        self.assertLess(poor_sleep_ts, headache_ts)


if __name__ == "__main__":
    unittest.main()
