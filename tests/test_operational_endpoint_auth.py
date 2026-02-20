from __future__ import annotations

import unittest
from unittest.mock import patch

from fastapi import HTTPException

from api.main import CitationAuditIn, ProcessJobsIn, audit_citations, enqueue_citation_audit, process_background_jobs


class OperationalEndpointAuthTests(unittest.TestCase):
    def test_audit_citations_requires_auth(self) -> None:
        with self.assertRaises(HTTPException) as ctx:
            audit_citations(CitationAuditIn(), authorization=None)
        self.assertEqual(ctx.exception.status_code, 401)

    def test_enqueue_citations_requires_auth(self) -> None:
        with self.assertRaises(HTTPException) as ctx:
            enqueue_citation_audit(CitationAuditIn(), authorization=None)
        self.assertEqual(ctx.exception.status_code, 401)

    def test_process_jobs_requires_auth(self) -> None:
        with self.assertRaises(HTTPException) as ctx:
            process_background_jobs(ProcessJobsIn(), authorization=None)
        self.assertEqual(ctx.exception.status_code, 401)

    def test_process_jobs_delegates_after_auth(self) -> None:
        with (
            patch("api.main._resolve_request_user_id", return_value=1) as resolve_mock,
            patch("api.main.process_background_jobs_batch", return_value={"status": "ok"}) as batch_mock,
        ):
            result = process_background_jobs(ProcessJobsIn(limit=5, max_papers_per_query=2), authorization="Bearer t")
        resolve_mock.assert_called_once()
        batch_mock.assert_called_once()
        self.assertEqual(result, {"status": "ok"})


if __name__ == "__main__":
    unittest.main()
