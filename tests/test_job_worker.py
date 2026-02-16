from __future__ import annotations

import unittest

import api.db
from api.job_worker import run_once
from tests.db_test_utils import reset_test_database


class JobWorkerTests(unittest.TestCase):
    def setUp(self) -> None:
        reset_test_database()

    def test_run_once_with_no_jobs(self) -> None:
        result = run_once(limit=10, max_papers_per_query=1)
        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["jobs_claimed"], 0)
        self.assertEqual(result["jobs_done"], 0)
        self.assertEqual(result["jobs_failed"], 0)


if __name__ == "__main__":
    unittest.main()
