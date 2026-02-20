from __future__ import annotations

import unittest
from unittest.mock import patch

from fastapi import HTTPException

from api.main import _resolve_request_user_id


class AuthRequestResolutionTests(unittest.TestCase):
    def test_requires_auth_when_no_token(self) -> None:
        with self.assertRaises(HTTPException) as ctx:
            _resolve_request_user_id(explicit_user_id=None, authorization=None)
        self.assertEqual(ctx.exception.status_code, 401)

    def test_rejects_mismatched_explicit_user_id(self) -> None:
        with patch("api.main.resolve_user_from_token", return_value={"id": 7}):
            with self.assertRaises(HTTPException) as ctx:
                _resolve_request_user_id(
                    explicit_user_id=8,
                    authorization="Bearer token-value",
                )
        self.assertEqual(ctx.exception.status_code, 403)

    def test_accepts_matching_explicit_user_id(self) -> None:
        with patch("api.main.resolve_user_from_token", return_value={"id": 7}):
            resolved = _resolve_request_user_id(
                explicit_user_id=7,
                authorization="Bearer token-value",
            )
        self.assertEqual(resolved, 7)

    def test_uses_token_user_when_explicit_missing(self) -> None:
        with patch("api.main.resolve_user_from_token", return_value={"id": 12}):
            resolved = _resolve_request_user_id(
                explicit_user_id=None,
                authorization="Bearer token-value",
            )
        self.assertEqual(resolved, 12)

    def test_does_not_allow_explicit_without_valid_token(self) -> None:
        with patch("api.main.resolve_user_from_token", return_value=None):
            with self.assertRaises(HTTPException) as ctx:
                _resolve_request_user_id(
                    explicit_user_id=42,
                    authorization="Bearer invalid-token",
                )
        self.assertEqual(ctx.exception.status_code, 401)


if __name__ == "__main__":
    unittest.main()
