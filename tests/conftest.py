from __future__ import annotations

import os
from urllib.parse import urlparse, urlunparse
from uuid import uuid4

import pytest
from psycopg import connect, sql

import api.db
from tests.db_test_utils import reset_test_database


def _module_uses_reset_test_database(module: object) -> bool:
    return "reset_test_database" in getattr(module, "__dict__", {})


def _build_test_database_url(base_url: str, test_db_name: str) -> str:
    parsed = urlparse(base_url)
    if parsed.scheme not in {"postgres", "postgresql"}:
        raise RuntimeError("Test database URL must use postgres/postgresql scheme")
    return urlunparse(parsed._replace(path=f"/{test_db_name}"))


def pytest_collection_modifyitems(session, config, items) -> None:
    _ = session, config
    for item in items:
        if hasattr(item, "module") and _module_uses_reset_test_database(item.module):
            item.add_marker("needs_db")


def pytest_configure(config) -> None:
    config.addinivalue_line("markers", "needs_db: test requires isolated PostgreSQL test database")


@pytest.fixture(scope="session")
def _isolated_test_database_session():
    os.environ["APP_ENV"] = "test"
    admin_url = os.getenv("TEST_DATABASE_ADMIN_URL", "").strip() or "postgresql://postgres:postgres@127.0.0.1:5432/postgres"
    test_db_name = f"vital_test_{uuid4().hex[:12]}"
    test_database_url = _build_test_database_url(admin_url, test_db_name)

    admin_conn = connect(admin_url, autocommit=True)
    try:
        with admin_conn.cursor() as cur:
            cur.execute(sql.SQL("CREATE DATABASE {}").format(sql.Identifier(test_db_name)))
    finally:
        admin_conn.close()

    previous_database_url = os.getenv("DATABASE_URL")
    os.environ["DATABASE_URL"] = test_database_url
    api.db.assert_test_database_safety()
    api.db.initialize_database()

    try:
        yield
    finally:
        admin_conn = connect(admin_url, autocommit=True)
        try:
            with admin_conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT pg_terminate_backend(pid)
                    FROM pg_stat_activity
                    WHERE datname = %s
                      AND pid <> pg_backend_pid()
                    """,
                    (test_db_name,),
                )
                cur.execute(sql.SQL("DROP DATABASE IF EXISTS {}").format(sql.Identifier(test_db_name)))
        finally:
            admin_conn.close()

        if previous_database_url is None:
            os.environ.pop("DATABASE_URL", None)
        else:
            os.environ["DATABASE_URL"] = previous_database_url


@pytest.fixture(autouse=True)
def reset_database_between_tests(request) -> None:
    if not _module_uses_reset_test_database(request.node.module):
        return
    request.getfixturevalue("_isolated_test_database_session")
    reset_test_database()
