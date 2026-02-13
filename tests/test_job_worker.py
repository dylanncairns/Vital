from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import api.db
from api.job_worker import run_once


class JobWorkerTests(unittest.TestCase):
    def setUp(self) -> None:
        self._orig_db_path = api.db.DB_PATH
        self._tmpdir = tempfile.TemporaryDirectory()
        api.db.DB_PATH = Path(self._tmpdir.name) / "test.db"
        api.db.initialize_database()

    def tearDown(self) -> None:
        api.db.DB_PATH = self._orig_db_path
        self._tmpdir.cleanup()

    def test_run_once_with_no_jobs(self) -> None:
        result = run_once(limit=10, max_papers_per_query=1)
        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["jobs_claimed"], 0)
        self.assertEqual(result["jobs_done"], 0)
        self.assertEqual(result["jobs_failed"], 0)


if __name__ == "__main__":
    unittest.main()
