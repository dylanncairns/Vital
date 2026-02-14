from __future__ import annotations

import hashlib
import hmac
import os
import secrets
from datetime import datetime, timedelta, timezone

from api.db import get_connection

SESSION_TTL_DAYS = int(os.getenv("AUTH_SESSION_TTL_DAYS", "30"))


def _now() -> datetime:
    return datetime.now(tz=timezone.utc)


def _now_iso() -> str:
    return _now().isoformat()


def _hash_password(password: str, *, salt: bytes | None = None) -> str:
    if salt is None:
        salt = os.urandom(16)
    rounds = 210_000
    digest = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, rounds)
    return f"pbkdf2_sha256${rounds}${salt.hex()}${digest.hex()}"


def verify_password(password: str, password_hash: str | None) -> bool:
    if not password_hash:
        return False
    try:
        algo, rounds_str, salt_hex, expected_hex = password_hash.split("$", 3)
        if algo != "pbkdf2_sha256":
            return False
        rounds = int(rounds_str)
        salt = bytes.fromhex(salt_hex)
        expected = bytes.fromhex(expected_hex)
    except Exception:
        return False
    computed = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, rounds)
    return hmac.compare_digest(computed, expected)


def create_user(*, username: str, password: str, name: str | None = None) -> dict:
    uname = username.strip().lower()
    if not uname or len(uname) < 3:
        raise ValueError("username must be at least 3 chars")
    if len(password) < 8:
        raise ValueError("password must be at least 8 chars")
    conn = get_connection()
    try:
        existing = conn.execute(
            "SELECT id FROM users WHERE username = ? LIMIT 1",
            (uname,),
        ).fetchone()
        if existing is not None:
            raise ValueError("username already exists")
        cursor = conn.execute(
            """
            INSERT INTO users (created_at, name, username, password_hash)
            VALUES (?, ?, ?, ?)
            """,
            (_now_iso(), (name or uname).strip(), uname, _hash_password(password)),
        )
        conn.commit()
        user_id = int(cursor.lastrowid)
        return {"id": user_id, "username": uname, "name": (name or uname).strip()}
    finally:
        conn.close()


def _create_session(*, user_id: int) -> str:
    token = secrets.token_urlsafe(48)
    now = _now()
    expires_at = (now + timedelta(days=SESSION_TTL_DAYS)).isoformat()
    conn = get_connection()
    try:
        conn.execute(
            """
            INSERT INTO auth_sessions (user_id, token, created_at, expires_at, revoked_at)
            VALUES (?, ?, ?, ?, NULL)
            """,
            (int(user_id), token, now.isoformat(), expires_at),
        )
        conn.commit()
        return token
    finally:
        conn.close()


def login_user(*, username: str, password: str) -> dict | None:
    uname = username.strip().lower()
    conn = get_connection()
    try:
        row = conn.execute(
            "SELECT id, username, name, password_hash FROM users WHERE username = ? LIMIT 1",
            (uname,),
        ).fetchone()
    finally:
        conn.close()
    if row is None:
        return None
    if not verify_password(password, row["password_hash"]):
        return None
    token = _create_session(user_id=int(row["id"]))
    return {
        "token": token,
        "user": {"id": int(row["id"]), "username": row["username"], "name": row["name"]},
    }


def resolve_user_from_token(token: str | None) -> dict | None:
    if not token:
        return None
    conn = get_connection()
    try:
        row = conn.execute(
            """
            SELECT u.id AS id, u.username AS username, u.name AS name
            FROM auth_sessions s
            JOIN users u ON u.id = s.user_id
            WHERE s.token = ?
              AND s.revoked_at IS NULL
              AND s.expires_at > ?
            LIMIT 1
            """,
            (token, _now_iso()),
        ).fetchone()
        return dict(row) if row else None
    finally:
        conn.close()


def revoke_session(token: str | None) -> None:
    if not token:
        return
    conn = get_connection()
    try:
        conn.execute(
            "UPDATE auth_sessions SET revoked_at = ? WHERE token = ? AND revoked_at IS NULL",
            (_now_iso(), token),
        )
        conn.commit()
    finally:
        conn.close()

