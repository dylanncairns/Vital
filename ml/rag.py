from __future__ import annotations

import json
import math
import os
import re
from collections import Counter
from datetime import datetime, timezone
from hashlib import sha256
from typing import Any

_TOKEN_RE = re.compile(r"[a-z0-9]+")
_NON_ALNUM_RE = re.compile(r"[^a-z0-9]+")

RAG_SCHEMA = {
    "name": "rag_cited_answer",
    "schema": {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "answer": {"type": "string"},
            "confidence": {"type": "number", "minimum": 0, "maximum": 1},
            "citations": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "citation_id": {"type": "string"},
                        "title": {"type": "string"},
                        "authors": {"type": "array", "items": {"type": "string"}},
                        "year": {"type": "integer"},
                        "doi": {"type": ["string", "null"]},
                        "url": {"type": ["string", "null"]},
                        "file_id": {"type": "string"},
                    },
                    "required": ["citation_id", "title", "authors", "year", "doi", "url", "file_id"],
                },
            },
            "evidence": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "claim": {"type": "string"},
                        "supports": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "additionalProperties": False,
                                "properties": {
                                    "citation_id": {"type": "string"},
                                    "snippet": {"type": "string"},
                                    "chunk_id": {"type": "string"},
                                },
                                "required": ["citation_id", "snippet", "chunk_id"],
                            },
                        },
                    },
                    "required": ["claim", "supports"],
                },
            },
        },
        "required": ["answer", "confidence", "citations", "evidence"],
    },
}


def fetch_item_name_map(conn) -> dict[int, str]:
    rows = conn.execute("SELECT id, name FROM items").fetchall()
    return {int(row["id"]): row["name"] for row in rows if row["name"] is not None}


def fetch_symptom_name_map(conn) -> dict[int, str]:
    rows = conn.execute("SELECT id, name FROM symptoms").fetchall()
    return {int(row["id"]): row["name"] for row in rows if row["name"] is not None}


def fetch_ingredient_name_map(conn) -> dict[int, str]:
    rows = conn.execute("SELECT id, name FROM ingredients").fetchall()
    return {int(row["id"]): row["name"] for row in rows if row["name"] is not None}


def tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall((text or "").lower())


def _normalize_phrase(text: str) -> str:
    value = _NON_ALNUM_RE.sub(" ", (text or "").lower())
    return " ".join(value.split())


def chunk_text(text: str, *, chunk_chars: int = 420, overlap_chars: int = 80) -> list[str]:
    content = (text or "").strip()
    if not content:
        return []
    if chunk_chars <= 0:
        return [content]
    if overlap_chars < 0:
        overlap_chars = 0

    chunks: list[str] = []
    start = 0
    length = len(content)
    while start < length:
        end = min(length, start + chunk_chars)
        chunks.append(content[start:end].strip())
        if end == length:
            break
        start = max(start + 1, end - overlap_chars)
    return [chunk for chunk in chunks if chunk]


def text_hash(text: str) -> str:
    return sha256(text.encode("utf-8")).hexdigest()


def embed_text(text: str) -> dict[str, float]:
    tokens = tokenize(text)
    if not tokens:
        return {}
    counts = Counter(tokens)
    norm = math.sqrt(sum(value * value for value in counts.values()))
    if norm == 0:
        return {}
    return {token: value / norm for token, value in counts.items()}


def cosine_similarity(vec_a: dict[str, float], vec_b: dict[str, float]) -> float:
    if not vec_a or not vec_b:
        return 0.0
    if len(vec_a) > len(vec_b):
        vec_a, vec_b = vec_b, vec_a
    return sum(value * vec_b.get(token, 0.0) for token, value in vec_a.items())


def build_candidate_query(
    *,
    item_name: str | None,
    symptom_name: str | None,
    routes: set[str],
    lag_bucket_counts: dict[str, int],
) -> str:
    route_text = ", ".join(sorted(routes)) if routes else "unknown route"
    lag_text = ", ".join(f"{bucket}:{count}" for bucket, count in sorted(lag_bucket_counts.items()))
    return (
        f"item {item_name or 'unknown'} symptom {symptom_name or 'unknown'} "
        f"routes {route_text} lag_buckets {lag_text}"
    ).strip()


def _get_openai_client():
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return None
    try:
        from openai import OpenAI  # type: ignore
    except Exception:
        return None
    return OpenAI(api_key=api_key)


def create_vector_store(name: str) -> str:
    client = _get_openai_client()
    if client is None:
        raise RuntimeError("OpenAI client unavailable or OPENAI_API_KEY missing.")
    store = client.vector_stores.create(name=name)
    return str(store.id)


def upload_file(filepath: str) -> str:
    client = _get_openai_client()
    if client is None:
        raise RuntimeError("OpenAI client unavailable or OPENAI_API_KEY missing.")
    with open(filepath, "rb") as handle:
        uploaded = client.files.create(file=handle, purpose="assistants")
    return str(uploaded.id)


def add_files_to_vector_store(vector_store_id: str, file_ids: list[str]) -> None:
    client = _get_openai_client()
    if client is None:
        raise RuntimeError("OpenAI client unavailable or OPENAI_API_KEY missing.")
    for file_id in file_ids:
        client.vector_stores.files.create(vector_store_id=vector_store_id, file_id=file_id)


def _upsert_paper(conn, paper: dict[str, Any], *, ingested_at: str) -> int:
    paper_url = paper.get("url")
    if paper_url is not None and isinstance(paper_url, str) and paper_url.strip():
        existing = conn.execute(
            "SELECT id FROM papers WHERE url = ? LIMIT 1",
            (paper_url.strip(),),
        ).fetchone()
    else:
        # URL can be null from structured RAG output; use title + date as fallback identity.
        existing = conn.execute(
            """
            SELECT id
            FROM papers
            WHERE title = ?
              AND COALESCE(publication_date, '') = COALESCE(?, '')
            LIMIT 1
            """,
            (paper.get("title"), paper.get("publication_date")),
        ).fetchone()
    if existing:
        return int(existing["id"])
    cursor = conn.execute(
        """
        INSERT INTO papers (title, url, abstract, publication_date, source, ingested_at)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (
            paper["title"],
            paper["url"],
            paper.get("abstract"),
            paper.get("publication_date"),
            paper.get("source"),
            ingested_at,
        ),
    )
    return int(cursor.lastrowid)


def _claim_text(row: dict[str, Any]) -> str:
    fields = [
        row.get("title"),
        row.get("abstract"),
        row.get("summary"),
        row.get("chunk_text"),
        row.get("citation_snippet"),
    ]
    return " ".join(field for field in fields if isinstance(field, str) and field.strip())


def retrieve_claim_evidence(
    conn,
    *,
    ingredient_ids: set[int],
    item_id: int | None,
    symptom_id: int,
    query_text: str,
    top_k: int = 5,
) -> list[dict[str, Any]]:
    if top_k <= 0:
        return []
    where_filters: list[str] = []
    params: list[Any] = [symptom_id]
    if ingredient_ids:
        placeholders = ",".join("?" for _ in ingredient_ids)
        where_filters.append(f"c.ingredient_id IN ({placeholders})")
        params.extend(sorted(ingredient_ids))
    if item_id is not None:
        where_filters.append("c.item_id = ?")
        params.append(item_id)
        # Fallback path: if claims are ingredient-linked and exposure expansion rows are missing,
        # recover ingredient IDs from item composition.
        if not ingredient_ids:
            ingredient_rows = conn.execute(
                """
                SELECT ingredient_id
                FROM items_ingredients
                WHERE item_id = ?
                """,
                (item_id,),
            ).fetchall()
            fallback_ingredient_ids = {
                int(row["ingredient_id"])
                for row in ingredient_rows
                if row["ingredient_id"] is not None
            }
            if fallback_ingredient_ids:
                placeholders = ",".join("?" for _ in fallback_ingredient_ids)
                where_filters.append(f"c.ingredient_id IN ({placeholders})")
                params.extend(sorted(fallback_ingredient_ids))
    if not where_filters:
        return []
    where_clause = " OR ".join(where_filters)
    rows = conn.execute(
        f"""
        SELECT
            c.id AS claim_id,
            c.item_id AS item_id,
            c.ingredient_id AS ingredient_id,
            c.symptom_id AS symptom_id,
            c.paper_id AS paper_id,
            c.summary AS summary,
            c.chunk_text AS chunk_text,
            c.citation_snippet AS citation_snippet,
            c.citation_title AS citation_title,
            c.citation_url AS citation_url,
            c.evidence_polarity_and_strength AS evidence_polarity_and_strength,
            p.title AS title,
            p.abstract AS abstract,
            p.publication_date AS publication_date
        FROM claims c
        JOIN papers p ON p.id = c.paper_id
        WHERE c.symptom_id = ?
          AND ({where_clause})
        """,
        tuple(params),
    ).fetchall()

    query_embedding = embed_text(query_text)
    scored: list[dict[str, Any]] = []
    for raw in rows:
        row = dict(raw)
        claim_embedding = embed_text(_claim_text(row))
        relevance = cosine_similarity(query_embedding, claim_embedding)
        row["relevance"] = relevance
        scored.append(row)

    scored.sort(
        key=lambda row: (
            row["relevance"],
            abs(row.get("evidence_polarity_and_strength") or 0),
            row.get("publication_date") or "",
            row["claim_id"],
        ),
        reverse=True,
    )
    return scored[:top_k]


def ingest_paper_claim_chunks(
    conn,
    *,
    paper_id: int,
    item_id: int | None,
    ingredient_id: int | None,
    symptom_id: int,
    summary: str,
    evidence_polarity_and_strength: int,
    citation_title: str | None,
    citation_url: str | None,
    source_text: str,
) -> int:
    chunks = chunk_text(source_text)
    inserted = 0
    if not chunks:
        chunks = [summary]
    for index, chunk in enumerate(chunks):
        embedding = embed_text(chunk)
        conn.execute(
            """
            INSERT INTO claims (
                item_id, ingredient_id, symptom_id, paper_id, claim_type, summary,
                chunk_index, chunk_text, chunk_hash, embedding_model, embedding_vector,
                citation_title, citation_url, citation_snippet, evidence_polarity_and_strength
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                item_id,
                ingredient_id,
                symptom_id,
                paper_id,
                "rag_chunk",
                summary,
                index,
                chunk,
                text_hash(chunk),
                "local-token-v1",
                json_dumps_safe(embedding),
                citation_title,
                citation_url,
                (chunk[:280] + "...") if len(chunk) > 280 else chunk,
                evidence_polarity_and_strength,
            ),
        )
        inserted += 1
    return inserted


def json_dumps_safe(value: Any) -> str:
    return json.dumps(value, sort_keys=True)


def _infer_polarity_strength(text: str) -> int:
    normalized = _normalize_phrase(text)
    negative_cues = (
        "no association",
        "not associated",
        "not linked",
        "no increase",
        "did not increase",
        "decrease",
        "reduced",
    )
    for cue in negative_cues:
        if cue in normalized:
            return -1
    return 1


def _to_plain_dict(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if hasattr(value, "model_dump"):
        dumped = value.model_dump()  # type: ignore[attr-defined]
        if isinstance(dumped, dict):
            return dumped
    if hasattr(value, "to_dict"):
        dumped = value.to_dict()  # type: ignore[attr-defined]
        if isinstance(dumped, dict):
            return dumped
    return {}


def _extract_json_text_from_responses_payload(payload: dict[str, Any]) -> str | None:
    output_json = payload.get("output_json")
    if isinstance(output_json, dict):
        return json.dumps(output_json)
    if isinstance(output_json, list):
        return json.dumps({"evidence": output_json})

    output_text = payload.get("output_text")
    if isinstance(output_text, str) and output_text.strip():
        return output_text
    output = payload.get("output")
    if not isinstance(output, list):
        return None
    for entry in output:
        if not isinstance(entry, dict):
            continue
        content = entry.get("content")
        if not isinstance(content, list):
            continue
        for part in content:
            if not isinstance(part, dict):
                continue
            json_value = part.get("json")
            if isinstance(json_value, dict):
                return json.dumps(json_value)
            parsed_value = part.get("parsed")
            if isinstance(parsed_value, dict):
                return json.dumps(parsed_value)
            text_value = part.get("text")
            if isinstance(text_value, str) and text_value.strip():
                return text_value
    return None


def _collect_ids(node: Any, *, key: str, out: set[str]) -> None:
    if isinstance(node, dict):
        value = node.get(key)
        if isinstance(value, str) and value.strip():
            out.add(value.strip())
        for child in node.values():
            _collect_ids(child, key=key, out=out)
        return
    if isinstance(node, list):
        for child in node:
            _collect_ids(child, key=key, out=out)


def _extract_grounded_ids(payload: dict[str, Any]) -> tuple[set[str], set[str]]:
    file_ids: set[str] = set()
    chunk_ids: set[str] = set()
    _collect_ids(payload.get("output"), key="file_id", out=file_ids)
    _collect_ids(payload.get("output"), key="chunk_id", out=chunk_ids)

    # Some SDK payloads place grounding metadata under annotations with IDs.
    _collect_ids(payload.get("output"), key="id", out=chunk_ids)
    return file_ids, chunk_ids


def _llm_retrieve_evidence_rows(
    *,
    symptom_name: str,
    ingredient_names: list[str],
    item_name: str | None,
    routes: list[str] | None = None,
    lag_bucket_counts: dict[str, int] | None = None,
    max_evidence_rows: int,
    http_post_json=None,
    client_override=None,
    vector_store_id_override: str | None = None,
) -> list[dict[str, Any]]:
    _ = http_post_json
    client = client_override or _get_openai_client()
    if client is None:
        return []
    vector_store_id = vector_store_id_override or os.getenv("RAG_VECTOR_STORE_ID")
    if not vector_store_id:
        if os.getenv("RAG_DEBUG", "0") == "1":
            print("[RAG] RAG_VECTOR_STORE_ID missing; skipping online retrieval.")
        return []

    model = os.getenv("RAG_OPENAI_MODEL", "gpt-4.1")
    system_prompt = (
        "You are a scientific evidence assistant. "
        "Use ONLY retrieved file_search chunks from the configured vector store. "
        "Return JSON that exactly matches the schema. No extra keys, no missing keys. "
        "If evidence is insufficient, return empty citations/evidence arrays and explain limits in answer. "
        "DO NOT invent citation_ids, file_ids, chunk_ids, DOI, URLs, titles, or years. "
        "Each evidence.supports entry must reference a citation_id present in citations."
    )
    question_payload = {
        "symptom_name": symptom_name,
        "item_name": item_name,
        "ingredient_names": ingredient_names,
        "routes": routes or [],
        "lag_bucket_counts": lag_bucket_counts or {},
        "instructions": [
            "Retrieve evidence linking exposure to symptom.",
            "Return JSON matching schema exactly.",
            "Citations/supports must map to retrieved results.",
        ],
    }

    response_obj: Any | None = None
    last_error: Exception | None = None
    request_modes: list[str] = ["response_format", "text_format"]
    for mode in request_modes:
        try:
            if mode == "response_format":
                response_obj = client.responses.create(
                    model=model,
                    input=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": json.dumps(question_payload, sort_keys=True)},
                    ],
                    tools=[{"type": "file_search", "vector_store_ids": [vector_store_id]}],
                    response_format={"type": "json_schema", "json_schema": RAG_SCHEMA},
                )
            else:
                response_obj = client.responses.create(
                    model=model,
                    input=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": json.dumps(question_payload, sort_keys=True)},
                    ],
                    tools=[{"type": "file_search", "vector_store_ids": [vector_store_id]}],
                    text={
                        "format": {
                            "type": "json_schema",
                            "name": RAG_SCHEMA["name"],
                            "schema": RAG_SCHEMA["schema"],
                            "strict": True,
                        }
                    },
                )
            break
        except Exception as exc:
            last_error = exc
            response_obj = None
            continue

    if response_obj is None:
        if os.getenv("RAG_DEBUG", "0") == "1" and last_error is not None:
            print(f"[RAG] Responses API failed: {last_error}")
        return []

    response_payload = _to_plain_dict(response_obj)
    raw_json_text = _extract_json_text_from_responses_payload(response_payload)
    if not raw_json_text:
        if os.getenv("RAG_DEBUG", "0") == "1":
            print("[RAG] Responses output had no parseable JSON payload.")
        return []

    try:
        parsed = json.loads(raw_json_text)
    except json.JSONDecodeError:
        if os.getenv("RAG_DEBUG", "0") == "1":
            print("[RAG] Structured payload was not valid JSON.")
        return []
    if not isinstance(parsed, dict):
        return []

    citations = parsed.get("citations")
    evidence_blocks = parsed.get("evidence")
    if not isinstance(citations, list) or not isinstance(evidence_blocks, list):
        return []

    retrieved_file_ids, retrieved_chunk_ids = _extract_grounded_ids(response_payload)

    citation_map: dict[str, dict[str, Any]] = {}
    first_valid_citation: dict[str, Any] | None = None
    for raw in citations:
        if not isinstance(raw, dict):
            continue
        citation_id = raw.get("citation_id")
        title = raw.get("title")
        file_id = raw.get("file_id")
        if not isinstance(citation_id, str) or not citation_id.strip():
            continue
        if not isinstance(title, str) or not title.strip():
            continue
        if not isinstance(file_id, str) or not file_id.strip():
            continue
        if retrieved_file_ids and file_id not in retrieved_file_ids:
            continue
        normalized_id = citation_id.strip()
        citation_map[normalized_id] = raw
        if first_valid_citation is None:
            first_valid_citation = raw

    output: list[dict[str, Any]] = []
    skipped_by_chunk_filter = 0
    for block in evidence_blocks:
        if not isinstance(block, dict):
            continue
        claim = block.get("claim")
        supports = block.get("supports")
        if not isinstance(claim, str) or not claim.strip() or not isinstance(supports, list):
            continue
        polarity = _infer_polarity_strength(claim)
        for support in supports:
            if not isinstance(support, dict):
                continue
            citation_id = support.get("citation_id")
            snippet = support.get("snippet")
            chunk_id = support.get("chunk_id")
            if not isinstance(citation_id, str) or citation_id.strip() not in citation_map:
                continue
            if not isinstance(snippet, str) or not snippet.strip():
                continue
            if not isinstance(chunk_id, str) or not chunk_id.strip():
                continue
            if retrieved_chunk_ids and chunk_id not in retrieved_chunk_ids:
                skipped_by_chunk_filter += 1
                # Fallback: keep row if snippet exists and citation is otherwise grounded.
                # Some SDK payloads surface chunk IDs in a different field than our parser.
                if os.getenv("RAG_STRICT_CHUNK_GROUNDING", "0") == "1":
                    continue

            citation = citation_map[citation_id.strip()]
            citation_url = citation.get("url")
            citation_url_value = citation_url.strip() if isinstance(citation_url, str) and citation_url.strip() else None
            output.append(
                {
                    "title": citation["title"],
                    "url": citation_url_value,
                    "publication_date": str(citation.get("year")) if citation.get("year") is not None else None,
                    "source": "openai_file_search",
                    "item_name": item_name,
                    "ingredient_name": ingredient_names[0] if ingredient_names else None,
                    "symptom_name": symptom_name,
                    "summary": claim.strip(),
                    "snippet": snippet.strip(),
                    "evidence_polarity_and_strength": polarity,
                }
            )
            if len(output) >= max_evidence_rows:
                return output

    # Secondary fallback: if supports were too strict, still salvage claim->citation linkage.
    if not output and evidence_blocks and citation_map:
        for block in evidence_blocks:
            if not isinstance(block, dict):
                continue
            claim = block.get("claim")
            supports = block.get("supports")
            if not isinstance(claim, str) or not claim.strip() or not isinstance(supports, list):
                continue
            polarity = _infer_polarity_strength(claim)
            for support in supports:
                if not isinstance(support, dict):
                    continue
                snippet = support.get("snippet")
                if not isinstance(snippet, str) or not snippet.strip():
                    continue
                citation_id = support.get("citation_id")
                citation: dict[str, Any] | None = None
                if isinstance(citation_id, str):
                    citation = citation_map.get(citation_id.strip())
                if citation is None:
                    citation = first_valid_citation
                if citation is None:
                    continue
                citation_url = citation.get("url")
                citation_url_value = citation_url.strip() if isinstance(citation_url, str) and citation_url.strip() else None
                output.append(
                    {
                        "title": citation.get("title") or "Untitled citation",
                        "url": citation_url_value,
                        "publication_date": str(citation.get("year")) if citation.get("year") is not None else None,
                        "source": "openai_file_search",
                        "item_name": item_name,
                        "ingredient_name": ingredient_names[0] if ingredient_names else None,
                        "symptom_name": symptom_name,
                        "summary": claim.strip(),
                        "snippet": snippet.strip(),
                        "evidence_polarity_and_strength": polarity,
                    }
                )
                if len(output) >= max_evidence_rows:
                    return output

    if os.getenv("RAG_DEBUG", "0") == "1":
        print(
            "[RAG] retrieval summary:",
            {
                "citations": len(citations),
                "citation_map": len(citation_map),
                "evidence_blocks": len(evidence_blocks),
                "rows_out": len(output),
                "skipped_by_chunk_filter": skipped_by_chunk_filter,
                "grounded_file_ids": len(retrieved_file_ids),
                "grounded_chunk_ids": len(retrieved_chunk_ids),
            },
        )
    return output


def build_literature_queries(
    *,
    candidates: list[dict[str, Any]],
    ingredient_name_map: dict[int, str],
    symptom_name_map: dict[int, str],
    item_name_map: dict[int, str],
) -> list[str]:
    queries: list[str] = []
    seen_queries: set[str] = set()
    for candidate in candidates:
        symptom_name = symptom_name_map.get(int(candidate["symptom_id"]))
        if not symptom_name:
            continue
        ingredient_ids = candidate.get("ingredient_ids", set())
        item_name = item_name_map.get(int(candidate["item_id"]))
        if ingredient_ids:
            for ingredient_id in sorted(ingredient_ids):
                ingredient_name = ingredient_name_map.get(int(ingredient_id))
                if not ingredient_name:
                    continue
                query = f"{ingredient_name} {symptom_name}"
                if query not in seen_queries:
                    seen_queries.add(query)
                    queries.append(query)
        elif item_name:
            query = f"{item_name} {symptom_name}"
            if query not in seen_queries:
                seen_queries.add(query)
                queries.append(query)
    return queries


def _normalize_name_to_id_lookup(conn, table: str, alias_table: str, id_col: str) -> dict[str, int]:
    rows = conn.execute(
        f"""
        SELECT id AS entity_id, name AS term
        FROM {table}
        WHERE name IS NOT NULL
        UNION ALL
        SELECT {id_col} AS entity_id, alias AS term
        FROM {alias_table}
        WHERE alias IS NOT NULL
        """
    ).fetchall()
    mapping: dict[str, int] = {}
    for row in rows:
        key = _normalize_phrase(row["term"])
        if key and key not in mapping:
            mapping[key] = int(row["entity_id"])
    return mapping


def sync_claims_for_candidates(
    conn,
    *,
    candidates: list[dict[str, Any]],
    ingredient_name_map: dict[int, str],
    symptom_name_map: dict[int, str],
    item_name_map: dict[int, str],
    online_enabled: bool = True,
    max_papers_per_query: int = 3,
    http_get=None,
    llm_retriever=_llm_retrieve_evidence_rows,
) -> dict[str, int]:
    _ = http_get
    queries = build_literature_queries(
        candidates=candidates,
        ingredient_name_map=ingredient_name_map,
        symptom_name_map=symptom_name_map,
        item_name_map=item_name_map,
    )

    ingredient_lookup = _normalize_name_to_id_lookup(
        conn,
        table="ingredients",
        alias_table="ingredients_aliases",
        id_col="ingredient_id",
    )

    papers_added = 0
    claims_added = 0

    if online_enabled:
        now_iso = datetime.now(tz=timezone.utc).isoformat()
        for candidate in candidates:
            symptom_id = int(candidate["symptom_id"])
            symptom_name = symptom_name_map.get(symptom_id)
            if not symptom_name:
                continue
            candidate_item_id = int(candidate["item_id"])
            item_name = item_name_map.get(candidate_item_id)
            ingredient_ids = sorted(int(value) for value in candidate.get("ingredient_ids", set()))
            ingredient_names = [
                ingredient_name_map[ingredient_id]
                for ingredient_id in ingredient_ids
                if ingredient_id in ingredient_name_map
            ]

            llm_rows = llm_retriever(
                symptom_name=symptom_name,
                ingredient_names=ingredient_names,
                item_name=item_name,
                routes=sorted(candidate.get("routes", [])),
                lag_bucket_counts=candidate.get("lag_bucket_counts"),
                max_evidence_rows=max_papers_per_query,
            )
            for row in llm_rows:
                row_ingredient_id: int | None = None
                row_item_id: int | None = None
                if ingredient_ids:
                    ingredient_name = row.get("ingredient_name")
                    if isinstance(ingredient_name, str) and ingredient_name.strip():
                        normalized_ingredient = _normalize_phrase(ingredient_name)
                        mapped_ingredient_id = ingredient_lookup.get(normalized_ingredient)
                        if mapped_ingredient_id in ingredient_ids:
                            row_ingredient_id = mapped_ingredient_id
                    if row_ingredient_id is None:
                        row_item_id = candidate_item_id
                else:
                    row_item_id = candidate_item_id
                row_symptom_id = symptom_id

                paper_url = row.get("url")
                if isinstance(paper_url, str) and paper_url.strip():
                    existing_paper = conn.execute(
                        "SELECT id FROM papers WHERE url = ? LIMIT 1",
                        (paper_url.strip(),),
                    ).fetchone()
                else:
                    existing_paper = conn.execute(
                        """
                        SELECT id FROM papers
                        WHERE title = ?
                          AND COALESCE(publication_date, '') = COALESCE(?, '')
                        LIMIT 1
                        """,
                        (row["title"], row.get("publication_date")),
                    ).fetchone()
                paper_id = _upsert_paper(
                    conn,
                    {
                        "title": row["title"],
                        "url": row["url"],
                        "abstract": row["summary"],
                        "publication_date": row.get("publication_date"),
                        "source": row.get("source") or "openai_rag",
                    },
                    ingested_at=now_iso,
                )
                if existing_paper is None:
                    papers_added += 1

                snippet = row["snippet"]
                snippet_hash = text_hash(snippet)
                duplicate = conn.execute(
                    """
                    SELECT 1 FROM claims
                    WHERE paper_id = ?
                      AND symptom_id = ?
                      AND chunk_hash = ?
                      AND ((item_id IS NULL AND ? IS NULL) OR item_id = ?)
                      AND ((ingredient_id IS NULL AND ? IS NULL) OR ingredient_id = ?)
                    LIMIT 1
                    """,
                    (
                        paper_id,
                        row_symptom_id,
                        snippet_hash,
                        row_item_id,
                        row_item_id,
                        row_ingredient_id,
                        row_ingredient_id,
                    ),
                ).fetchone()
                if duplicate:
                    continue
                claims_added += ingest_paper_claim_chunks(
                    conn,
                    paper_id=paper_id,
                    item_id=row_item_id,
                    ingredient_id=row_ingredient_id,
                    symptom_id=row_symptom_id,
                    summary=row["summary"],
                    evidence_polarity_and_strength=row["evidence_polarity_and_strength"],
                    citation_title=row["title"],
                    citation_url=row["url"],
                    source_text=snippet,
                )

    return {"queries_built": len(queries), "papers_added": papers_added, "claims_added": claims_added}


def enrich_claims_for_candidates(
    conn,
    *,
    candidates: list[dict[str, Any]],
    ingredient_name_map: dict[int, str],
    symptom_name_map: dict[int, str],
    item_name_map: dict[int, str],
    online_enabled: bool = True,
    max_papers_per_query: int = 3,
    http_get=None,
    llm_retriever=_llm_retrieve_evidence_rows,
) -> dict[str, int]:
    return sync_claims_for_candidates(
        conn,
        candidates=candidates,
        ingredient_name_map=ingredient_name_map,
        symptom_name_map=symptom_name_map,
        item_name_map=item_name_map,
        online_enabled=online_enabled,
        max_papers_per_query=max_papers_per_query,
        http_get=http_get,
        llm_retriever=llm_retriever,
    )


def aggregate_evidence(retrieved_claims: list[dict[str, Any]]) -> dict[str, Any]:
    if not retrieved_claims:
        return {
            "evidence_score": 0.0,
            "evidence_strength_score": 0.0,
            "evidence_summary": "No matching evidence found for this symptom and exposure pattern.",
            "citations": [],
        }

    weighted_sum = 0.0
    total_weight = 0.0
    citations: list[dict[str, Any]] = []

    for claim in retrieved_claims:
        polarity = float(claim.get("evidence_polarity_and_strength") or 0.0)
        weight = max(0.05, float(claim.get("relevance") or 0.0))
        weighted_sum += polarity * weight
        total_weight += weight

        snippet = (
            claim.get("citation_snippet")
            or claim.get("summary")
            or claim.get("chunk_text")
            or ""
        )
        snippet_text = (snippet[:280] + "...") if len(snippet) > 280 else snippet
        citations.append(
            {
                "title": claim.get("citation_title") or claim.get("title"),
                "url": claim.get("citation_url"),
                "snippet": snippet_text,
                "evidence_polarity_and_strength": int(claim.get("evidence_polarity_and_strength") or 0),
            }
        )

    evidence_score = (weighted_sum / total_weight) if total_weight > 0 else 0.0
    evidence_score = max(-1.0, min(1.0, evidence_score))
    evidence_strength_score = abs(evidence_score)
    direction = "mixed"
    if evidence_score > 0.15:
        direction = "supportive"
    elif evidence_score < -0.15:
        direction = "contradictory"

    return {
        "evidence_score": evidence_score,
        "evidence_strength_score": evidence_strength_score,
        "evidence_summary": (
            f"{len(retrieved_claims)} claim(s) retrieved; overall evidence is {direction}."
        ),
        "citations": citations,
    }
