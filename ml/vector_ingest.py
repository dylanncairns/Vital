from __future__ import annotations

import argparse
import json
import os
import tempfile
import time
from pathlib import Path
from typing import Any

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
                    },
                    "required": ["title", "url", "publication_date", "source", "abstract"],
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


def collect_local_sources(source_dir: Path) -> list[Path]:
    patterns = ("*.pdf", "*.txt", "*.md")
    paths: list[Path] = []
    for pattern in patterns:
        paths.extend(sorted(source_dir.rglob(pattern)))
    deduped: list[Path] = []
    seen: set[str] = set()
    for path in paths:
        key = str(path.resolve())
        if key in seen:
            continue
        seen.add(key)
        deduped.append(path)
    return deduped


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


def _normalize_discovery_query(query: str) -> str:
    compact = " ".join(query.split())
    return compact[:240].strip()


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
    return f"{primary} exposure"


def _build_structured_queries_for_candidate(
    *,
    item_name: str | None,
    symptom_name: str,
    ingredient_names: list[str],
    routes: list[str],
    lag_bucket_counts: dict[str, int],
) -> list[str]:
    terms = ingredient_names if ingredient_names else ([item_name] if item_name else [])
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
            }
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

    queries: list[str] = []
    seen_queries: set[str] = set()
    for candidate in candidates:
        symptom_name = symptom_name_map.get(int(candidate["symptom_id"]))
        if not symptom_name:
            continue
        item_name = item_name_map.get(int(candidate["item_id"]))
        ingredient_names = [
            ingredient_name_map[int(ingredient_id)]
            for ingredient_id in sorted(candidate.get("ingredient_ids", set()))
            if int(ingredient_id) in ingredient_name_map
        ]
        candidate_queries = _build_structured_queries_for_candidate(
            item_name=item_name,
            symptom_name=symptom_name,
            ingredient_names=ingredient_names,
            routes=list(candidate.get("routes", [])),
            lag_bucket_counts=dict(candidate.get("lag_bucket_counts", {})),
        )
        for query in candidate_queries:
            key = query.lower()
            if key in seen_queries:
                continue
            seen_queries.add(key)
            queries.append(query)

    if not queries:
        return [], [{"query": "", "error": "No literature queries built from candidates."}]

    output_dir.mkdir(parents=True, exist_ok=True)
    written_paths: list[Path] = []
    written_keys: set[tuple[str, str]] = set()
    errors: list[dict[str, str]] = []
    for query in queries[:max_queries]:
        normalized_query = _normalize_discovery_query(query)
        papers, error = _discover_papers_for_query(query=normalized_query, max_papers=max_papers_per_query)
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
            }
            path = output_dir / filename
            path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
            written_paths.append(path)
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

    queries: list[str] = []
    seen_queries: set[str] = set()
    for candidate in candidates:
        symptom_name = symptom_name_map.get(int(candidate["symptom_id"]))
        if not symptom_name:
            continue
        item_name = item_name_map.get(int(candidate["item_id"]))
        ingredient_names = [
            ingredient_name_map[int(ingredient_id)]
            for ingredient_id in sorted(candidate.get("ingredient_ids", set()))
            if int(ingredient_id) in ingredient_name_map
        ]
        candidate_queries = _build_structured_queries_for_candidate(
            item_name=item_name,
            symptom_name=symptom_name,
            ingredient_names=ingredient_names,
            routes=list(candidate.get("routes", [])),
            lag_bucket_counts=dict(candidate.get("lag_bucket_counts", {})),
        )
        for query in candidate_queries:
            key = query.lower()
            if key in seen_queries:
                continue
            seen_queries.add(key)
            queries.append(query)
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

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    generated_paths: list[Path] = []
    generated_keys: set[tuple[str, str]] = set()
    errors: list[dict[str, str]] = []

    for query in queries:
        normalized_query = _normalize_discovery_query(query)
        papers, error = _discover_papers_for_query(query=normalized_query, max_papers=max_papers_per_query)
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
            }
            path = output / filename
            path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
            generated_paths.append(path)

    store_id = ensure_vector_store_id(provided_id=vector_store_id, name=vector_store_name)
    upload_result = upload_paths_to_vector_store(
        vector_store_id=store_id,
        paths=generated_paths,
    )
    upload_result["source_files"] = [str(path) for path in generated_paths]
    upload_result["auto_generated_files"] = [str(path) for path in generated_paths]
    upload_result["auto_discovery_errors"] = errors
    if not generated_paths:
        upload_result["status"] = "error"
        upload_result["reason"] = "auto_discovery_produced_no_sources"
    return upload_result


def ingest_sources(
    *,
    source_dir: str | None,
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

    if source_dir:
        all_paths.extend(collect_local_sources(Path(source_dir)))

    auto_generated: list[str] = []
    auto_discovery_errors: list[dict[str, str]] = []
    if auto_discover_when_empty and not all_paths and user_id is not None:
        auto_dir = Path(source_dir) if source_dir else Path("data/rag_sources_auto")
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
    parser.add_argument("--source-dir", default="data/rag_sources", help="Directory to scan for .pdf/.txt/.md")
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
        source_dir=args.source_dir,
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
