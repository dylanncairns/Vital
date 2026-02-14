from __future__ import annotations

from typing import Any, Callable
from urllib import error as urlerror
from urllib import request as urlrequest


def _url_status(url: str, *, timeout_seconds: float = 6.0) -> str:
    candidate = (url or "").strip()
    if not candidate:
        return "error"

    def _probe(method: str) -> tuple[bool, int | None]:
        req = urlrequest.Request(candidate, method=method, headers={"User-Agent": "vital-citation-audit/1.0"})
        try:
            with urlrequest.urlopen(req, timeout=timeout_seconds) as resp:
                code = int(getattr(resp, "status", 200) or 200)
        except urlerror.HTTPError as exc:
            code = int(getattr(exc, "code", 0) or 0)
        except Exception:
            return False, None
        return True, code

    ok, code = _probe("HEAD")
    if not ok:
        return "error"
    if code is None:
        return "error"
    if code in {401, 403}:
        return "exists"
    if 200 <= code < 400:
        return "exists"
    if code in {404, 410, 451}:
        return "missing"
    if code in {405, 429, 500, 501, 502, 503, 504}:
        ok_get, code_get = _probe("GET")
        if not ok_get or code_get is None:
            return "error"
        if code_get in {401, 403}:
            return "exists"
        if 200 <= code_get < 400:
            return "exists"
        if code_get in {404, 410, 451}:
            return "missing"
        return "error"
    return "error"


def audit_claim_citations(
    conn,
    *,
    limit: int = 300,
    delete_missing: bool = True,
    checker: Callable[[str], str] | None = None,
) -> dict[str, Any]:
    probe = checker or (lambda value: _url_status(value))
    rows = conn.execute(
        """
        SELECT citation_url, COUNT(*) AS claim_count
        FROM claims
        WHERE citation_url IS NOT NULL
          AND TRIM(citation_url) <> ''
        GROUP BY citation_url
        ORDER BY claim_count DESC, citation_url ASC
        LIMIT ?
        """,
        (max(1, int(limit)),),
    ).fetchall()

    scanned = 0
    missing_urls: list[str] = []
    errors = 0
    for row in rows:
        url = str(row["citation_url"] or "").strip()
        if not url:
            continue
        scanned += 1
        status = str(probe(url) or "error").strip().lower()
        if status == "missing":
            missing_urls.append(url)
        elif status != "exists":
            errors += 1

    deleted_claims = 0
    deleted_papers = 0
    if delete_missing and missing_urls:
        placeholders = ",".join("?" for _ in missing_urls)
        deleted_claims = int(
            conn.execute(
                f"DELETE FROM claims WHERE citation_url IN ({placeholders})",
                tuple(missing_urls),
            ).rowcount
            or 0
        )
        # Keep paper rows only when they still back at least one claim.
        deleted_papers = int(
            conn.execute(
                """
                DELETE FROM papers
                WHERE id IN (
                    SELECT p.id
                    FROM papers p
                    LEFT JOIN claims c ON c.paper_id = p.id
                    WHERE c.id IS NULL
                )
                """
            ).rowcount
            or 0
        )

    return {
        "scanned_urls": scanned,
        "missing_urls": len(missing_urls),
        "deleted_claims": deleted_claims,
        "deleted_papers": deleted_papers,
        "errors": errors,
    }

