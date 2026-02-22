from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from api.db import get_connection
from ml.insights import list_rag_sync_candidates
from ml.rag import (
    _extract_json_text_from_responses_payload,
    _get_openai_client,
    _to_plain_dict,
    add_files_to_vector_store,
    create_vector_store,
    fetch_ingredient_name_map,
    fetch_item_name_map,
    fetch_symptom_name_map,
    upload_file,
)

PAPER_DISCOVERY_SCHEMA = {
    "name": "rag_source_discovery",
    "schema": {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "papers": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "title": {"type": "string"},
                        "url": {"type": ["string", "null"]},
                        "publication_date": {"type": ["string", "null"]},
                        "source": {"type": ["string", "null"]},
                        "abstract": {"type": ["string", "null"]},
                        "snippet": {"type": ["string", "null"]},
                    },
                    "required": ["title", "url", "publication_date", "source", "abstract", "snippet"],
                },
            }
        },
        "required": ["papers"],
    },
}

_LAG_BUCKET_PHRASE = {
    "0_6h": "within 6 hours",
    "6_24h": "within 24 hours",
    "24_72h": "within 72 hours",
    "72h_7d": "within 7 days",
}


def ensure_vector_store_id(*, provided_id: str | None = None, name: str = "vital-rag-store") -> str:
    if provided_id:
        return provided_id
    env_id = os.getenv("RAG_VECTOR_STORE_ID")
    if env_id:
        return env_id
    return create_vector_store(name)


def _poll_vector_store_file(vector_store_id: str, file_id: str, *, timeout_seconds: int = 180) -> str:
    client = _get_openai_client()
    if client is None:
        raise RuntimeError("OpenAI client unavailable or OPENAI_API_KEY missing.")

    deadline = time.time() + timeout_seconds
    status = "in_progress"
    while time.time() < deadline:
        vs_file = client.vector_stores.files.retrieve(vector_store_id=vector_store_id, file_id=file_id)
        status = str(getattr(vs_file, "status", "unknown"))
        if status in {"completed", "failed", "cancelled"}:
            return status
        time.sleep(1.2)
    return status


def upload_paths_to_vector_store(
    *,
    vector_store_id: str,
    paths: list[Path],
    poll: bool = True,
    timeout_seconds: int = 180,
) -> dict[str, Any]:
    uploaded_file_ids: list[str] = []
    skipped: list[str] = []

    for path in paths:
        if not path.exists() or not path.is_file():
            skipped.append(str(path))
            continue
        uploaded_file_ids.append(upload_file(str(path)))

    if uploaded_file_ids:
        add_files_to_vector_store(vector_store_id, uploaded_file_ids)

    statuses: dict[str, str] = {}
    if poll:
        for file_id in uploaded_file_ids:
            statuses[file_id] = _poll_vector_store_file(
                vector_store_id,
                file_id,
                timeout_seconds=timeout_seconds,
            )

    return {
        "vector_store_id": vector_store_id,
        "uploaded_count": len(uploaded_file_ids),
        "uploaded_file_ids": uploaded_file_ids,
        "statuses": statuses,
        "skipped": skipped,
    }


def export_papers_table_to_text_files(output_dir: Path) -> list[Path]:
    conn = get_connection()
    try:
        rows = conn.execute(
            """
            SELECT id, title, url, abstract, publication_date, source
            FROM papers
            ORDER BY id ASC
            """
        ).fetchall()
    finally:
        conn.close()

    output_dir.mkdir(parents=True, exist_ok=True)
    exported: list[Path] = []
    for row in rows:
        paper_id = int(row["id"])
        title = row["title"] or f"paper-{paper_id}"
        filename = f"paper_{paper_id}.txt"
        payload = {
            "paper_id": paper_id,
            "title": title,
            "url": row["url"],
            "publication_date": row["publication_date"],
            "source": row["source"],
            "abstract": row["abstract"],
        }
        path = output_dir / filename
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        exported.append(path)
    return exported


def _safe_filename(value: str) -> str:
    cleaned = "".join(char if char.isalnum() else "_" for char in value.strip().lower())
    cleaned = "_".join(token for token in cleaned.split("_") if token)
    return cleaned[:64] if cleaned else "paper"


def _source_hash(*, user_id: int | None, payload: dict[str, Any]) -> str:
    material = {
        "user_id": user_id,
        "query": payload.get("query"),
        "original_query": payload.get("original_query"),
        "title": payload.get("title"),
        "url": payload.get("url"),
        "publication_date": payload.get("publication_date"),
        "source": payload.get("source"),
        "abstract": payload.get("abstract"),
        "snippet": payload.get("snippet"),
    }
    encoded = json.dumps(material, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _persist_discovered_sources(
    *,
    user_id: int | None,
    records: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    if not records:
        return []
    conn = get_connection()
    try:
        persisted: list[dict[str, Any]] = []
        for record in records:
            payload = record["payload"]
            payload_json = json.dumps(payload, indent=2, ensure_ascii=False)
            source_hash = _source_hash(user_id=user_id, payload=payload)
            row = conn.execute(
                """
                INSERT INTO rag_source_documents (
                    user_id, query, original_query, filename, title, url, publication_date,
                    source, abstract, payload_json, source_hash, created_at, updated_at
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, NOW(), NOW())
                ON CONFLICT (source_hash) DO UPDATE SET
                    user_id = EXCLUDED.user_id,
                    query = EXCLUDED.query,
                    original_query = EXCLUDED.original_query,
                    filename = EXCLUDED.filename,
                    title = EXCLUDED.title,
                    url = EXCLUDED.url,
                    publication_date = EXCLUDED.publication_date,
                    source = EXCLUDED.source,
                    abstract = EXCLUDED.abstract,
                    payload_json = EXCLUDED.payload_json,
                    updated_at = EXCLUDED.updated_at
                RETURNING id, filename, payload_json
                """,
                (
                    user_id,
                    payload.get("query"),
                    payload.get("original_query"),
                    record["filename"],
                    payload.get("title"),
                    payload.get("url"),
                    payload.get("publication_date"),
                    payload.get("source"),
                    payload.get("abstract"),
                    payload_json,
                    source_hash,
                ),
            ).fetchone()
            persisted.append(dict(row))
        conn.commit()
        return persisted
    finally:
        conn.close()


def _materialize_source_rows(rows: list[dict[str, Any]], output_dir: Path) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for row in rows:
        filename = str(row.get("filename") or f"rag_source_{int(row['id'])}.txt")
        path = output_dir / filename
        path.write_text(str(row["payload_json"]), encoding="utf-8")
        paths.append(path)
    return paths


def _normalize_discovery_query(query: str) -> str:
    compact = " ".join(query.split())
    return compact[:240].strip()


_DISCOVERY_STOPWORDS = {
    "and",
    "or",
    "the",
    "a",
    "an",
    "in",
    "on",
    "for",
    "with",
    "to",
    "of",
    "from",
    "humans",
    "human",
    "adverse",
    "effect",
    "effects",
    "cohort",
    "trial",
    "onset",
    "linked",
    "interaction",
    "causes",
    "side",
}


def _tokenize_terms(value: str | None) -> set[str]:
    if not value:
        return set()
    out: set[str] = set()
    for raw in value.lower().split():
        token = "".join(char for char in raw if char.isalnum())
        if len(token) < 3:
            continue
        if token in _DISCOVERY_STOPWORDS:
            continue
        out.add(token)
    return out


def _text_contains_any_token(text: str, tokens: set[str]) -> bool:
    if not text or not tokens:
        return False
    normalized = " ".join("".join(char if char.isalnum() else " " for char in text.lower()).split())
    if not normalized:
        return False
    padded = f" {normalized} "
    return any(f" {token} " in padded for token in tokens)


_HIGH_TRUST_SOURCE_TOKENS = {
    "pubmed",
    "nih",
    "nejm",
    "lancet",
    "jamanetwork",
    "bmj",
    "nature",
    "science",
    "wiley",
    "springer",
    "oxford",
    "elsevier",
}


_YEAR_RE = re.compile(r"(19|20)\d{2}")


def _extract_year(value: Any) -> int | None:
    text = str(value or "").strip()
    if not text:
        return None
    match = _YEAR_RE.search(text)
    if not match:
        return None
    try:
        year = int(match.group(0))
    except (TypeError, ValueError):
        return None
    if 1900 <= year <= 2100:
        return year
    return None


def _source_quality_score(*, url: str | None, source: str | None) -> float:
    score = 0.0
    host = ""
    if isinstance(url, str) and url.strip():
        try:
            host = (urlparse(url.strip()).hostname or "").lower()
        except Exception:
            host = ""
    haystack = f"{host} {str(source or '').lower()}"
    for token in _HIGH_TRUST_SOURCE_TOKENS:
        if token in haystack:
            score = max(score, 1.0)
    if host.endswith(".gov"):
        score = max(score, 0.9)
    if host.endswith(".edu"):
        score = max(score, 0.8)
    return score


def _paper_rank_score(
    *,
    paper: dict[str, Any],
    symptom_tokens: set[str],
    exposure_tokens: set[str],
) -> float:
    text = " ".join(
        str(value or "")
        for value in [paper.get("title"), paper.get("abstract"), paper.get("snippet")]
    )
    text_lower = text.lower()
    symptom_hits = sum(1 for token in symptom_tokens if token and f" {token} " in f" {text_lower} ")
    exposure_hits = sum(1 for token in exposure_tokens if token and f" {token} " in f" {text_lower} ")
    coverage = min(1.0, 0.5 * min(1.0, symptom_hits / max(1, len(symptom_tokens))) + 0.5 * min(1.0, exposure_hits / max(1, len(exposure_tokens))))

    year = _extract_year(paper.get("publication_date"))
    now_year = datetime.now(tz=timezone.utc).year
    if year is None:
        recency = 0.25
    else:
        age = max(0, now_year - year)
        recency = max(0.0, 1.0 - min(age, 20) / 20.0)
    source_score = _source_quality_score(url=paper.get("url"), source=paper.get("source"))
    return 0.55 * coverage + 0.30 * recency + 0.15 * source_score


def _primary_lag_phrase(lag_bucket_counts: dict[str, int]) -> str | None:
    if not lag_bucket_counts:
        return None
    dominant_bucket = max(
        lag_bucket_counts.items(),
        key=lambda pair: (pair[1], pair[0]),
    )[0]
    return _LAG_BUCKET_PHRASE.get(dominant_bucket)


def _route_context(routes: list[str]) -> str | None:
    if not routes:
        return None
    cleaned = [route.strip().lower() for route in routes if route and route.strip()]
    if not cleaned:
        return None
    primary = cleaned[0]
    if primary == "ingestion":
        return "oral exposure"
    if primary == "dermal":
        return "topical exposure"
    if primary == "inhalation":
        return "inhalation exposure"
    if primary == "injection":
        return "injection exposure"
    if primary in {"proximity_environment", "proximity/environment"}:
        return "environmental exposure"
    return f"{primary} exposure"


def _build_structured_queries_for_candidate(
    *,
    item_name: str | None,
    secondary_item_name: str | None,
    symptom_name: str,
    ingredient_names: list[str],
    routes: list[str],
    lag_bucket_counts: dict[str, int],
) -> list[str]:
    terms: list[str] = []
    if ingredient_names:
        terms.extend(ingredient_names)
    if item_name and item_name.strip():
        terms.append(item_name.strip())
    combo_phrase: str | None = None
    if (
        item_name
        and secondary_item_name
        and item_name.strip()
        and secondary_item_name.strip()
    ):
        combo_phrase = f"{item_name.strip()} and {secondary_item_name.strip()}"
        terms.extend([combo_phrase, secondary_item_name.strip()])
    if not terms:
        return []
    route_phrase = _route_context(routes)
    lag_phrase = _primary_lag_phrase(lag_bucket_counts)

    queries: list[str] = []
    for term in terms:
        term_value = term.strip()
        if not term_value:
            continue
        queries.extend(
            [
                f"{term_value} {symptom_name} adverse effect humans",
                f"{term_value} {symptom_name} cohort trial",
                f"{term_value} causes {symptom_name} side effect",
            ]
        )
        if route_phrase:
            queries.append(f"{term_value} {symptom_name} {route_phrase} humans")
        if lag_phrase:
            queries.append(f"{term_value} {symptom_name} onset {lag_phrase}")
    if combo_phrase:
        queries.extend(
            [
                f"{combo_phrase} {symptom_name} interaction adverse effect humans",
                f"{combo_phrase} linked to {symptom_name} cohort",
            ]
        )

    deduped: list[str] = []
    seen: set[str] = set()
    for query in queries:
        normalized = _normalize_discovery_query(query).lower()
        if normalized in seen:
            continue
        seen.add(normalized)
        deduped.append(_normalize_discovery_query(query))
    return deduped


def _discover_papers_for_query(
    *,
    query: str,
    symptom_name: str | None = None,
    exposure_terms: list[str] | None = None,
    max_papers: int = 5,
) -> tuple[list[dict[str, Any]], str | None]:
    client = _get_openai_client()
    if client is None:
        return [], "OpenAI client unavailable or OPENAI_API_KEY missing."

    model = os.getenv("RAG_OPENAI_MODEL", "gpt-4.1")
    system_prompt = (
        "You are a biomedical literature retrieval assistant. "
        "Find real publications relevant to the query using web search. "
        "Return strict JSON only. "
        "Do not invent URLs, titles, journals, or publication dates. "
        "Prioritize human medical evidence and avoid veterinary/agricultural papers "
        "unless the query is explicitly about animals/agriculture. "
        "If uncertain, return fewer papers."
    )
    normalized_query = _normalize_discovery_query(query)
    user_payload = {
        "query": normalized_query,
        "original_query": query,
        "max_papers": max_papers,
        "requirements": [
            "Prefer peer-reviewed publications and high-quality scientific sources.",
            "Each paper must include title; include URL when available.",
            "abstract should be concise and faithful to source text.",
            "snippet should capture a short, directly relevant passage from the source.",
            "Exclude papers that do not clearly discuss the candidate exposure and symptom in humans.",
        ],
    }
    response_obj: Any | None = None
    last_error: Exception | None = None
    for mode in ("response_format", "text_format"):
        try:
            if mode == "response_format":
                response_obj = client.responses.create(
                    model=model,
                    input=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": json.dumps(user_payload, sort_keys=True)},
                    ],
                    tools=[{"type": "web_search"}],
                    response_format={"type": "json_schema", "json_schema": PAPER_DISCOVERY_SCHEMA},
                )
            else:
                response_obj = client.responses.create(
                    model=model,
                    input=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": json.dumps(user_payload, sort_keys=True)},
                    ],
                    tools=[{"type": "web_search"}],
                    text={
                        "format": {
                            "type": "json_schema",
                            "name": PAPER_DISCOVERY_SCHEMA["name"],
                            "schema": PAPER_DISCOVERY_SCHEMA["schema"],
                            "strict": True,
                        }
                    },
                )
            break
        except Exception as exc:
            response_obj = None
            last_error = exc
            continue
    if response_obj is None:
        message = str(last_error) if last_error is not None else "unknown discovery error"
        if os.getenv("RAG_DEBUG", "0") == "1":
            print(f"[vector_ingest] discovery failed for query '{query}': {message}")
        return [], message

    payload = _to_plain_dict(response_obj)
    raw_json = _extract_json_text_from_responses_payload(payload)
    if not raw_json:
        return [], "No structured JSON payload in model response."
    try:
        parsed = json.loads(raw_json)
    except json.JSONDecodeError:
        return [], "Model output was not valid JSON."
    papers = parsed.get("papers")
    if not isinstance(papers, list):
        return [], "Structured payload missing papers array."

    deduped: list[dict[str, Any]] = []
    seen_keys: set[tuple[str, str]] = set()
    for paper in papers:
        if not isinstance(paper, dict):
            continue
        title = paper.get("title")
        if not isinstance(title, str) or not title.strip():
            continue
        url = paper.get("url")
        url_value = url.strip() if isinstance(url, str) and url.strip() else ""
        key = (title.strip().lower(), url_value.lower())
        if key in seen_keys:
            continue
        seen_keys.add(key)
        deduped.append(
            {
                "title": title.strip(),
                "url": url_value or None,
                "publication_date": paper.get("publication_date"),
                "source": paper.get("source"),
                "abstract": paper.get("abstract"),
                "snippet": paper.get("snippet"),
            }
        )
    symptom_tokens = _tokenize_terms(symptom_name)
    exposure_tokens: set[str] = set()
    for term in (exposure_terms or []):
        exposure_tokens.update(_tokenize_terms(term))

    if symptom_tokens or exposure_tokens:
        filtered: list[dict[str, Any]] = []
        for paper in deduped:
            text = " ".join(
                str(value or "")
                for value in [paper.get("title"), paper.get("abstract"), paper.get("snippet")]
            )
            symptom_ok = True if not symptom_tokens else _text_contains_any_token(text, symptom_tokens)
            exposure_ok = True if not exposure_tokens else _text_contains_any_token(text, exposure_tokens)
            if symptom_ok and exposure_ok:
                filtered.append(paper)
        deduped = filtered

    deduped.sort(
        key=lambda paper: _paper_rank_score(
            paper=paper,
            symptom_tokens=symptom_tokens,
            exposure_tokens=exposure_tokens,
        ),
        reverse=True,
    )

    return deduped[:max_papers], None


def auto_generate_sources_for_user(
    *,
    user_id: int,
    output_dir: Path,
    max_queries: int,
    max_papers_per_query: int,
) -> tuple[list[Path], list[dict[str, str]]]:
    conn = get_connection()
    try:
        candidates = list_rag_sync_candidates(user_id)
        if not candidates:
            return [], [{"query": "", "error": "No candidates found for user."}]
        ingredient_name_map = fetch_ingredient_name_map(conn)
        symptom_name_map = fetch_symptom_name_map(conn)
        item_name_map = fetch_item_name_map(conn)
    finally:
        conn.close()

    queries: list[dict[str, Any]] = []
    seen_queries: set[str] = set()
    for candidate in candidates:
        symptom_name = symptom_name_map.get(int(candidate["symptom_id"]))
        if not symptom_name:
            continue
        item_name = item_name_map.get(int(candidate["item_id"]))
        secondary_item_name = None
        if candidate.get("secondary_item_id") is not None:
            secondary_item_name = item_name_map.get(int(candidate["secondary_item_id"]))
        ingredient_names = [
            ingredient_name_map[int(ingredient_id)]
            for ingredient_id in sorted(candidate.get("ingredient_ids", set()))
            if int(ingredient_id) in ingredient_name_map
        ]
        candidate_queries = _build_structured_queries_for_candidate(
            item_name=item_name,
            secondary_item_name=secondary_item_name,
            symptom_name=symptom_name,
            ingredient_names=ingredient_names,
            routes=list(candidate.get("routes", [])),
            lag_bucket_counts=dict(candidate.get("lag_bucket_counts", {})),
        )
        exposure_terms = list(ingredient_names)
        if item_name:
            exposure_terms.append(item_name)
        if secondary_item_name:
            exposure_terms.append(secondary_item_name)
        for query in candidate_queries:
            key = query.lower()
            if key in seen_queries:
                continue
            seen_queries.add(key)
            queries.append(
                {
                    "query": query,
                    "symptom_name": symptom_name,
                    "exposure_terms": exposure_terms,
                }
            )

    if not queries:
        return [], [{"query": "", "error": "No literature queries built from candidates."}]

    source_records: list[dict[str, Any]] = []
    written_keys: set[tuple[str, str]] = set()
    errors: list[dict[str, str]] = []
    for query_info in queries[:max_queries]:
        query = str(query_info["query"])
        query_symptom = str(query_info.get("symptom_name") or "")
        query_exposure_terms = list(query_info.get("exposure_terms") or [])
        normalized_query = _normalize_discovery_query(query)
        papers, error = _discover_papers_for_query(
            query=normalized_query,
            symptom_name=query_symptom,
            exposure_terms=query_exposure_terms,
            max_papers=max_papers_per_query,
        )
        if error:
            errors.append({"query": query, "error": error})
            continue
        for index, paper in enumerate(papers):
            title = paper.get("title")
            if not isinstance(title, str) or not title.strip():
                continue
            url = paper.get("url")
            url_value = url if isinstance(url, str) and url.strip() else ""
            key = (title.strip().lower(), url_value.lower())
            if key in written_keys:
                continue
            written_keys.add(key)
            filename = f"{_safe_filename(query)}_{index+1}_{_safe_filename(title)}.txt"
            payload = {
                "query": normalized_query,
                "original_query": query,
                "title": title.strip(),
                "url": url if isinstance(url, str) and url.strip() else None,
                "publication_date": paper.get("publication_date"),
                "source": paper.get("source"),
                "abstract": paper.get("abstract"),
                "snippet": paper.get("snippet"),
            }
            source_records.append(
                {
                    "filename": filename,
                    "payload": payload,
                }
            )

    persisted_rows = _persist_discovered_sources(
        user_id=user_id,
        records=source_records,
    )
    written_paths = _materialize_source_rows(persisted_rows, output_dir)
    return written_paths, errors


def ingest_sources_for_candidates(
    *,
    candidates: list[dict[str, Any]],
    vector_store_id: str | None = None,
    vector_store_name: str = "vital-rag-store",
    output_dir: str = "data/rag_sources",
    max_queries: int = 8,
    max_papers_per_query: int = 5,
) -> dict[str, Any]:
    if not candidates:
        return {
            "status": "error",
            "reason": "no_candidates",
            "uploaded_count": 0,
            "source_files": [],
            "auto_generated_files": [],
            "auto_discovery_errors": [{"query": "", "error": "No candidates supplied."}],
        }

    conn = get_connection()
    try:
        ingredient_name_map = fetch_ingredient_name_map(conn)
        symptom_name_map = fetch_symptom_name_map(conn)
        item_name_map = fetch_item_name_map(conn)
    finally:
        conn.close()

    queries: list[dict[str, Any]] = []
    seen_queries: set[str] = set()
    for candidate in candidates:
        symptom_name = symptom_name_map.get(int(candidate["symptom_id"]))
        if not symptom_name:
            continue
        item_name = item_name_map.get(int(candidate["item_id"]))
        secondary_item_name = None
        if candidate.get("secondary_item_id") is not None:
            secondary_item_name = item_name_map.get(int(candidate["secondary_item_id"]))
        ingredient_names = [
            ingredient_name_map[int(ingredient_id)]
            for ingredient_id in sorted(candidate.get("ingredient_ids", set()))
            if int(ingredient_id) in ingredient_name_map
        ]
        candidate_queries = _build_structured_queries_for_candidate(
            item_name=item_name,
            secondary_item_name=secondary_item_name,
            symptom_name=symptom_name,
            ingredient_names=ingredient_names,
            routes=list(candidate.get("routes", [])),
            lag_bucket_counts=dict(candidate.get("lag_bucket_counts", {})),
        )
        exposure_terms = list(ingredient_names)
        if item_name:
            exposure_terms.append(item_name)
        if secondary_item_name:
            exposure_terms.append(secondary_item_name)
        for query in candidate_queries:
            key = query.lower()
            if key in seen_queries:
                continue
            seen_queries.add(key)
            queries.append(
                {
                    "query": query,
                    "symptom_name": symptom_name,
                    "exposure_terms": exposure_terms,
                }
            )
            if len(queries) >= max_queries:
                break
        if len(queries) >= max_queries:
            break

    if not queries:
        return {
            "status": "error",
            "reason": "no_queries",
            "uploaded_count": 0,
            "source_files": [],
            "auto_generated_files": [],
            "auto_discovery_errors": [{"query": "", "error": "No discovery queries built."}],
        }

    candidate_user_ids = {int(candidate["user_id"]) for candidate in candidates if candidate.get("user_id") is not None}
    generated_user_id = next(iter(candidate_user_ids)) if len(candidate_user_ids) == 1 else None
    generated_records: list[dict[str, Any]] = []
    generated_keys: set[tuple[str, str]] = set()
    errors: list[dict[str, str]] = []

    for query_info in queries:
        query = str(query_info["query"])
        query_symptom = str(query_info.get("symptom_name") or "")
        query_exposure_terms = list(query_info.get("exposure_terms") or [])
        normalized_query = _normalize_discovery_query(query)
        papers, error = _discover_papers_for_query(
            query=normalized_query,
            symptom_name=query_symptom,
            exposure_terms=query_exposure_terms,
            max_papers=max_papers_per_query,
        )
        if error:
            errors.append({"query": query, "error": error})
            continue
        for index, paper in enumerate(papers):
            title = paper.get("title")
            if not isinstance(title, str) or not title.strip():
                continue
            url = paper.get("url")
            url_value = url if isinstance(url, str) and url.strip() else ""
            key = (title.strip().lower(), url_value.lower())
            if key in generated_keys:
                continue
            generated_keys.add(key)
            filename = f"{_safe_filename(query)}_{index+1}_{_safe_filename(title)}.txt"
            payload = {
                "query": normalized_query,
                "original_query": query,
                "title": title.strip(),
                "url": url if isinstance(url, str) and url.strip() else None,
                "publication_date": paper.get("publication_date"),
                "source": paper.get("source"),
                "abstract": paper.get("abstract"),
                "snippet": paper.get("snippet"),
            }
            generated_records.append(
                {
                    "filename": filename,
                    "payload": payload,
                }
            )

    persisted_rows = _persist_discovered_sources(
        user_id=generated_user_id,
        records=generated_records,
    )
    generated_paths: list[Path] = []
    if persisted_rows:
        with tempfile.TemporaryDirectory(prefix="vital_rag_sources_") as tmp:
            generated_paths = _materialize_source_rows(persisted_rows, Path(tmp))
            store_id = ensure_vector_store_id(provided_id=vector_store_id, name=vector_store_name)
            upload_result = upload_paths_to_vector_store(
                vector_store_id=store_id,
                paths=generated_paths,
            )
    else:
        store_id = ensure_vector_store_id(provided_id=vector_store_id, name=vector_store_name)
        upload_result = {
            "vector_store_id": store_id,
            "uploaded_count": 0,
            "uploaded_file_ids": [],
            "statuses": {},
            "skipped": [],
        }

    upload_result["source_files"] = [
        f"rag_source_documents/{int(row['id'])}:{row['filename']}"
        for row in persisted_rows
    ]
    upload_result["auto_generated_files"] = list(upload_result["source_files"])
    upload_result["output_dir"] = output_dir
    upload_result["auto_discovery_errors"] = errors
    if not persisted_rows:
        upload_result["status"] = "error"
        upload_result["reason"] = "auto_discovery_produced_no_sources"
    return upload_result


def ingest_sources(
    *,
    vector_store_id: str | None,
    vector_store_name: str,
    include_papers_table: bool,
    user_id: int | None = None,
    auto_discover_when_empty: bool = False,
    max_queries: int = 8,
    max_papers_per_query: int = 5,
) -> dict[str, Any]:
    final_vector_store_id = ensure_vector_store_id(provided_id=vector_store_id, name=vector_store_name)
    all_paths: list[Path] = []

    auto_generated: list[str] = []
    auto_discovery_errors: list[dict[str, str]] = []
    if auto_discover_when_empty and user_id is not None:
        auto_dir = Path("data/rag_sources_auto")
        generated, auto_discovery_errors = auto_generate_sources_for_user(
            user_id=user_id,
            output_dir=auto_dir,
            max_queries=max_queries,
            max_papers_per_query=max_papers_per_query,
        )
        all_paths.extend(generated)
        auto_generated = [str(path) for path in generated]

    if include_papers_table:
        with tempfile.TemporaryDirectory(prefix="vital_rag_papers_") as tmp:
            exported = export_papers_table_to_text_files(Path(tmp))
            all_paths.extend(exported)
            result = upload_paths_to_vector_store(
                vector_store_id=final_vector_store_id,
                paths=all_paths,
            )
            result["source_files"] = [str(path) for path in all_paths]
            result["auto_generated_files"] = auto_generated
            result["auto_discovery_errors"] = auto_discovery_errors
            return result

    result = upload_paths_to_vector_store(
        vector_store_id=final_vector_store_id,
        paths=all_paths,
    )
    result["source_files"] = [str(path) for path in all_paths]
    result["auto_generated_files"] = auto_generated
    result["auto_discovery_errors"] = auto_discovery_errors
    if auto_discover_when_empty and user_id is not None and not result["source_files"]:
        result["status"] = "error"
        result["reason"] = "auto_discovery_produced_no_sources"
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Upload source files into an OpenAI vector store for RAG.")
    parser.add_argument("--vector-store-id", default=None, help="Existing vector store ID (uses env if omitted)")
    parser.add_argument("--vector-store-name", default="vital-rag-store", help="Name when creating a new vector store")
    parser.add_argument(
        "--include-papers-table",
        action="store_true",
        help="Also export current papers rows to temporary .txt docs before upload",
    )
    parser.add_argument("--user-id", type=int, default=None, help="User id for candidate-driven auto discovery")
    parser.add_argument(
        "--auto-discover-when-empty",
        action="store_true",
        help="If no local files found, auto-discover papers from user candidates using OpenAI",
    )
    parser.add_argument("--max-queries", type=int, default=8, help="Max candidate queries for auto discovery")
    parser.add_argument("--max-papers-per-query", type=int, default=5, help="Max papers per discovery query")
    args = parser.parse_args()

    result = ingest_sources(
        vector_store_id=args.vector_store_id,
        vector_store_name=args.vector_store_name,
        include_papers_table=args.include_papers_table,
        user_id=args.user_id,
        auto_discover_when_empty=args.auto_discover_when_empty,
        max_queries=args.max_queries,
        max_papers_per_query=args.max_papers_per_query,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
