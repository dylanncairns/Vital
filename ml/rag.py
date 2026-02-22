from __future__ import annotations

import json
import os
import re
from datetime import datetime, timezone
from hashlib import sha256
from typing import Any
from urllib.parse import urlparse

from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.metrics.pairwise import cosine_similarity as sklearn_cosine_similarity

_NON_ALNUM_RE = re.compile(r"[^a-z0-9]+")
_SECRET_REDACTION_PATTERNS = [
    (re.compile(r"sk-[A-Za-z0-9_\-]{12,}"), "[REDACTED_API_KEY]"),
    (re.compile(r"(Bearer\s+)[A-Za-z0-9_\-\.]+", re.I), r"\1[REDACTED_TOKEN]"),
]
_RELEVANCE_VECTORIZER = HashingVectorizer(
    n_features=4096,
    alternate_sign=False,
    norm="l2",
    ngram_range=(1, 2),
)


def _sanitize_error_message(message: str) -> str:
    text = str(message or "")
    for pattern, repl in _SECRET_REDACTION_PATTERNS:
        text = pattern.sub(repl, text)
    if "Incorrect API key provided" in text:
        text = re.sub(
            r"Incorrect API key provided:\s*[^\.]+",
            "Incorrect API key provided: [REDACTED_API_KEY]",
            text,
        )
    return text[:1000]


def _strip_nul_bytes(value: str | None) -> str | None:
    if value is None:
        return None
    return value.replace("\x00", "")


def _sanitize_db_text_fields(value: Any) -> Any:
    if isinstance(value, str):
        return _strip_nul_bytes(value)
    if isinstance(value, list):
        return [_sanitize_db_text_fields(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_sanitize_db_text_fields(item) for item in value)
    if isinstance(value, dict):
        return {key: _sanitize_db_text_fields(item) for key, item in value.items()}
    return value

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
                                    "study_design": {
                                        "type": "string",
                                        "enum": [
                                            "rct",
                                            "cohort",
                                            "case_control",
                                            "cross_sectional",
                                            "systematic_review",
                                            "meta_analysis",
                                            "case_report",
                                            "animal",
                                            "in_vitro",
                                            "other",
                                        ],
                                    },
                                    "study_quality_score": {"type": "number", "minimum": 0, "maximum": 1},
                                    "population_match": {"type": "number", "minimum": 0, "maximum": 1},
                                    "temporality_match": {"type": "number", "minimum": 0, "maximum": 1},
                                    "risk_of_bias": {"type": "number", "minimum": 0, "maximum": 1},
                                    "llm_confidence": {"type": "number", "minimum": 0, "maximum": 1},
                                    "support_direction_score": {"type": "number", "minimum": -1, "maximum": 1},
                                },
                                "required": [
                                    "citation_id",
                                    "snippet",
                                    "chunk_id",
                                    "study_design",
                                    "study_quality_score",
                                    "population_match",
                                    "temporality_match",
                                    "risk_of_bias",
                                    "llm_confidence",
                                    "support_direction_score",
                                ],
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


def _normalize_phrase(text: str) -> str:
    value = _NON_ALNUM_RE.sub(" ", (text or "").lower())
    return " ".join(value.split())


def chunk_text(text: str, *, chunk_chars: int = 420, overlap_chars: int = 80) -> list[str]:
    content = (_strip_nul_bytes(text or "") or "").strip()
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


def _relevance_scores(query_text: str, claim_texts: list[str]) -> list[float]:
    if not claim_texts:
        return []
    matrix = _RELEVANCE_VECTORIZER.transform([query_text, *claim_texts])
    query_vec = matrix[0:1]
    claim_vecs = matrix[1:]
    sims = sklearn_cosine_similarity(query_vec, claim_vecs)
    if sims.size == 0:
        return [0.0 for _ in claim_texts]
    scores = sims[0].tolist()
    return [float(max(0.0, min(1.0, value))) for value in scores]


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
    paper = _sanitize_db_text_fields(paper)
    paper_url = paper.get("url")
    if paper_url is not None and isinstance(paper_url, str) and paper_url.strip():
        existing = conn.execute(
            "SELECT id FROM papers WHERE url = %s LIMIT 1",
            (paper_url.strip(),),
        ).fetchone()
    else:
        # URL can be null from structured RAG output; use title + date as fallback identity.
        existing = conn.execute(
            """
            SELECT id
            FROM papers
            WHERE title = %s
              AND COALESCE(publication_date, '') = COALESCE(%s, '')
            LIMIT 1
            """,
            (paper.get("title"), paper.get("publication_date")),
        ).fetchone()
    if existing:
        return int(existing["id"])
    # Some tests seed explicit IDs; align identity sequence before default insert.
    conn.execute(
        """
        SELECT setval(
            pg_get_serial_sequence('papers', 'id'),
            COALESCE((SELECT MAX(id) FROM papers), 1),
            (SELECT MAX(id) IS NOT NULL FROM papers)
        )
        """
    )
    cursor = conn.execute(
        """
        INSERT INTO papers (title, url, abstract, publication_date, source, ingested_at)
        VALUES (%s, %s, %s, %s, %s, %s)
        RETURNING id
        """,
        (
            _strip_nul_bytes(paper["title"]),
            _strip_nul_bytes(paper["url"]),
            _strip_nul_bytes(paper.get("abstract")),
            _strip_nul_bytes(paper.get("publication_date")),
            _strip_nul_bytes(paper.get("source")),
            ingested_at,
        ),
    )
    return int(cursor.fetchone()["id"])


def _claim_text(row: dict[str, Any]) -> str:
    fields = [
        row.get("title"),
        row.get("abstract"),
        row.get("summary"),
        row.get("chunk_text"),
        row.get("citation_snippet"),
    ]
    return " ".join(field for field in fields if isinstance(field, str) and field.strip())


def _candidate_tokens(value: str | None) -> list[str]:
    normalized = _normalize_phrase(value or "")
    if not normalized:
        return []
    stop = {
        "a",
        "an",
        "the",
        "and",
        "or",
        "to",
        "of",
        "in",
        "on",
        "for",
        "with",
        "symptom",
        "exposure",
        "item",
    }
    return [token for token in normalized.split() if len(token) >= 3 and token not in stop]


def _contains_token_like(text: str, token: str) -> bool:
    if not text or not token:
        return False
    pattern = re.compile(rf"\b{re.escape(token)}(?:s|es|er|ers|ing|ed)?\b")
    return pattern.search(text) is not None


def retrieve_claim_evidence(
    conn,
    *,
    ingredient_ids: set[int],
    item_id: int | None,
    symptom_id: int,
    query_text: str,
    top_k: int = 12,
) -> list[dict[str, Any]]:
    if top_k <= 0:
        return []
    where_filters: list[str] = []
    params: list[Any] = [symptom_id]
    if ingredient_ids:
        placeholders = ",".join("%s" for _ in ingredient_ids)
        where_filters.append(f"c.ingredient_id IN ({placeholders})")
        params.extend(sorted(ingredient_ids))
    if item_id is not None:
        where_filters.append("c.item_id = %s")
        params.append(item_id)
        # Fallback path: if claims are ingredient-linked and exposure expansion rows are missing,
        # recover ingredient IDs from item composition.
        if not ingredient_ids:
            ingredient_rows = conn.execute(
                """
                SELECT ingredient_id
                FROM items_ingredients
                WHERE item_id = %s
                """,
                (item_id,),
            ).fetchall()
            fallback_ingredient_ids = {
                int(row["ingredient_id"])
                for row in ingredient_rows
                if row["ingredient_id"] is not None
            }
            if fallback_ingredient_ids:
                placeholders = ",".join("%s" for _ in fallback_ingredient_ids)
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
            c.study_design AS study_design,
            c.study_quality_score AS study_quality_score,
            c.population_match AS population_match,
            c.temporality_match AS temporality_match,
            c.risk_of_bias AS risk_of_bias,
            c.llm_confidence AS llm_confidence,
            c.evidence_polarity_and_strength AS evidence_polarity_and_strength,
            p.title AS title,
            p.abstract AS abstract,
            p.publication_date AS publication_date,
            p.source AS source
        FROM claims c
        JOIN papers p ON p.id = c.paper_id
        WHERE c.symptom_id = %s
          AND ({where_clause})
        """,
        tuple(params),
    ).fetchall()

    row_dicts = [dict(raw) for raw in rows]
    item_tokens: list[str] = []
    ingredient_tokens: list[str] = []
    symptom_tokens: list[str] = []
    if item_id is not None:
        item_row = conn.execute("SELECT name FROM items WHERE id = %s LIMIT 1", (item_id,)).fetchone()
        if item_row and item_row["name"]:
            item_tokens = _candidate_tokens(str(item_row["name"]))
    if ingredient_ids:
        placeholders = ",".join("%s" for _ in ingredient_ids)
        ing_rows = conn.execute(
            f"SELECT name FROM ingredients WHERE id IN ({placeholders})",
            tuple(sorted(ingredient_ids)),
        ).fetchall()
        for ing_row in ing_rows:
            ingredient_tokens.extend(_candidate_tokens(str(ing_row["name"] or "")))
    symptom_row = conn.execute("SELECT name FROM symptoms WHERE id = %s LIMIT 1", (symptom_id,)).fetchone()
    if symptom_row and symptom_row["name"]:
        symptom_tokens = _candidate_tokens(str(symptom_row["name"]))

    claim_texts = [_claim_text(row) for row in row_dicts]
    scores = _relevance_scores(query_text, claim_texts)
    scored: list[dict[str, Any]] = []
    base_min_relevance = max(0.0, min(1.0, float(os.getenv("RAG_MIN_DB_RELEVANCE", "0.12"))))
    min_relevance = base_min_relevance if len(row_dicts) > max(3, top_k) else 0.0
    for row, relevance, claim_text in zip(row_dicts, scores, claim_texts):
        normalized_text = _normalize_phrase(claim_text)
        if relevance < min_relevance:
            continue
        if symptom_tokens and not any(_contains_token_like(normalized_text, token) for token in symptom_tokens):
            continue
        exposure_tokens = item_tokens or ingredient_tokens
        if exposure_tokens and not any(_contains_token_like(normalized_text, token) for token in exposure_tokens):
            continue
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
    study_design: str | None = None,
    study_quality_score: float | None = None,
    population_match: float | None = None,
    temporality_match: float | None = None,
    risk_of_bias: float | None = None,
    llm_confidence: float | None = None,
    citation_title: str | None = None,
    citation_url: str | None = None,
    source_text: str = "",
) -> int:
    summary = _strip_nul_bytes(summary) or ""
    citation_title = _strip_nul_bytes(citation_title)
    citation_url = _strip_nul_bytes(citation_url)
    source_text = _strip_nul_bytes(source_text) or ""
    chunks = chunk_text(source_text)
    inserted = 0
    if not chunks:
        chunks = [summary]
    for index, chunk in enumerate(chunks):
        conn.execute(
            """
            INSERT INTO claims (
                item_id, ingredient_id, symptom_id, paper_id, claim_type, summary,
                chunk_index, chunk_text, chunk_hash, embedding_model, embedding_vector,
                citation_title, citation_url, citation_snippet,
                study_design, study_quality_score, population_match, temporality_match, risk_of_bias, llm_confidence,
                evidence_polarity_and_strength
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
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
                "hashing_vectorizer_v1",
                None,
                citation_title,
                citation_url,
                _strip_nul_bytes((chunk[:280] + "...") if len(chunk) > 280 else chunk),
                study_design,
                study_quality_score,
                population_match,
                temporality_match,
                risk_of_bias,
                llm_confidence,
                evidence_polarity_and_strength,
            ),
        )
        inserted += 1
    return inserted


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


def _bounded_metric(value: Any, *, default: float = 0.5) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        parsed = default
    if parsed < 0.0:
        return 0.0
    if parsed > 1.0:
        return 1.0
    return parsed


_HAZARD_CONTEXT_TERMS = {
    "outbreak",
    "waterborne",
    "foodborne",
    "contaminated",
    "pathogen",
    "infection",
    "salmonella",
    "campylobacter",
    "e coli",
    "listeria",
    "undercooked",
    "spoiled",
}
_EXPLICIT_HAZARD_ITEM_TERMS = {
    "contaminated",
    "undercooked",
    "raw",
    "spoiled",
    "unsafe",
    "tainted",
}
_GENERIC_ITEM_TOKENS = {
    "food",
    "drink",
    "meal",
    "medication",
    "supplement",
    "exposure",
    "item",
}

_CONTRADICTORY_CUE_TERMS = {
    "no evidence",
    "no association",
    "not associated",
    "did not",
    "does not",
    "not linked",
    "no data",
    "not a cause",
    "not cause",
    "unlikely",
}
_LIMITED_EVIDENCE_CUE_TERMS = {
    "limited evidence",
    "insufficient evidence",
    "more research is needed",
    "more research needed",
    "unclear",
    "inconclusive",
}
_SYMPTOM_TEXT_TERMS = {
    "headache",
    "migraine",
    "nausea",
    "vomiting",
    "stomachache",
    "stomach pain",
    "abdominal pain",
    "diarrhea",
    "constipation",
    "acne",
    "rash",
    "itch",
    "itchy",
    "hives",
    "fatigue",
    "brain fog",
    "dizziness",
    "anxiety",
    "insomnia",
    "fever",
    "cough",
    "sore throat",
}


def _is_hazard_context_mismatch(*, item_name: str | None, snippet: str, title: str) -> bool:
    item = " ".join((item_name or "").strip().lower().split())
    if not item:
        return False
    if any(token in item for token in _EXPLICIT_HAZARD_ITEM_TERMS):
        return False
    text = f"{title} {snippet}".strip().lower()
    return any(term in text for term in _HAZARD_CONTEXT_TERMS)


def _item_tokens_for_match(item_name: str | None) -> list[str]:
    normalized = " ".join((item_name or "").strip().lower().split())
    if not normalized:
        return []
    tokens = [t for t in re.findall(r"[a-z0-9]+", normalized) if len(t) >= 3]
    return [t for t in tokens if t not in _GENERIC_ITEM_TOKENS]


_ITEM_MATCH_ALIASES: dict[str, list[str]] = {
    "poor sleep": [
        "sleep deprivation",
        "sleep deprived",
        "sleep restriction",
        "insufficient sleep",
        "sleep loss",
        "sleep deficit",
    ],
    "fasting": [
        "fasting",
        "meal skipping",
        "skipped meals",
        "skipped meal",
        "food deprivation",
    ],
    "long shift": [
        "shift work",
        "shiftworker",
        "shift worker",
        "night shift",
        "overnight shift",
        "extended work hours",
        "long work hours",
    ],
}


def _item_match_phrases(item_name: str | None) -> list[str]:
    normalized = " ".join((item_name or "").strip().lower().split())
    if not normalized:
        return []
    phrases: list[str] = [normalized]
    for alias in _ITEM_MATCH_ALIASES.get(normalized, []):
        alias_norm = " ".join(alias.strip().lower().split())
        if alias_norm and alias_norm not in phrases:
            phrases.append(alias_norm)
    return phrases


def _item_context_mismatch(*, item_name: str | None, text: str) -> bool:
    phrases = _item_match_phrases(item_name)
    tokens: list[str] = []
    for phrase in phrases:
        for token in _item_tokens_for_match(phrase):
            if token not in tokens:
                tokens.append(token)
    if not tokens and not phrases:
        return False
    normalized = " ".join((text or "").strip().lower().split())
    if not normalized:
        return True
    for phrase in phrases:
        if phrase and phrase in normalized:
            return False
    # For stem-like terms, allow basic inflections ("work" -> "worker", "working").
    for token in tokens:
        if token in normalized:
            return False
        if any(form in normalized for form in (f"{token}er", f"{token}ers", f"{token}ing", f"{token}ed")):
            return False
    return True


def _combo_item_context_mismatch(*, item_name: str | None, secondary_item_name: str | None, text: str) -> bool:
    return _item_context_mismatch(item_name=item_name, text=text) or _item_context_mismatch(
        item_name=secondary_item_name, text=text
    )


def _apply_support_direction_cues(
    *,
    support_direction_score: float,
    claim: str,
    snippet: str,
    title: str,
) -> float:
    text = f"{title} {claim} {snippet}".strip().lower()
    if any(term in text for term in _CONTRADICTORY_CUE_TERMS):
        # Explicit non-supportive wording must not become positive correlation support.
        return min(support_direction_score, -0.6)
    if any(term in text for term in _LIMITED_EVIDENCE_CUE_TERMS):
        return min(support_direction_score, 0.0)
    return support_direction_score


def _polarity_from_text_cues(
    *,
    fallback_polarity: float,
    text: str,
) -> float:
    normalized = " ".join((text or "").strip().lower().split())
    if not normalized:
        return max(-1.0, min(1.0, fallback_polarity))
    if any(term in normalized for term in _CONTRADICTORY_CUE_TERMS):
        return min(max(-1.0, fallback_polarity), -0.7)
    if any(term in normalized for term in _LIMITED_EVIDENCE_CUE_TERMS):
        return min(max(-1.0, fallback_polarity), 0.0)
    return max(-1.0, min(1.0, fallback_polarity))


def _symptom_context_mismatch(
    *,
    symptom_name: str | None,
    text: str,
) -> bool:
    symptom = " ".join((symptom_name or "").strip().lower().split())
    normalized = " ".join((text or "").strip().lower().split())
    if not symptom or not normalized:
        return False
    if symptom in normalized:
        return False
    # If another known symptom term appears but the target symptom doesn't, treat as mismatch.
    for term in _SYMPTOM_TEXT_TERMS:
        if term == symptom:
            continue
        if term in normalized:
            return True
    return False


def _study_design_multiplier(study_design: Any) -> float:
    normalized = str(study_design or "").strip().lower()
    weights = {
        "meta_analysis": 1.0,
        "systematic_review": 0.9,
        "rct": 0.9,
        "cohort": 0.75,
        "case_control": 0.65,
        "cross_sectional": 0.45,
        "case_report": 0.25,
        "animal": 0.20,
        "in_vitro": 0.10,
        "other": 0.35,
    }
    return weights.get(normalized, 0.35)


def _citation_source_label(citation: dict[str, Any]) -> str | None:
    url_value = citation.get("url")
    if isinstance(url_value, str) and url_value.strip():
        try:
            host = urlparse(url_value.strip()).hostname or ""
            host = host.lower()
            if host.startswith("www."):
                host = host[4:]
            if host:
                return host
        except Exception:
            pass
    doi_value = citation.get("doi")
    if isinstance(doi_value, str) and doi_value.strip():
        return "doi"
    return None


def _llm_retrieve_evidence_rows(
    *,
    symptom_name: str,
    ingredient_names: list[str],
    item_name: str | None,
    secondary_item_name: str | None = None,
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
    is_combo = bool((secondary_item_name or "").strip())
    system_prompt = (
        "You are a scientific evidence assistant. "
        "Use ONLY retrieved file_search chunks from the configured vector store. "
        "Return JSON that exactly matches the schema. No extra keys, no missing keys. "
        "If evidence is insufficient, return empty citations/evidence arrays and explain limits in answer. "
        "Treat the candidate exposure literally and specifically; do not substitute broader hazard contexts. "
        "You MAY use close clinical or lay aliases for the same exposure/symptom concept "
        "(e.g., sleep deprivation for poor sleep, shift work for long shift, meal skipping for fasting, "
        "palpitations for racing heart), but only when the cited snippet/title clearly refers to the same concept. "
        "Do not broaden to a different exposure category or a generic risk factor. "
        "Example: ordinary 'water' should NOT be treated as contaminated/waterborne outbreak exposure "
        "unless the candidate text explicitly indicates contamination or unsafe water. "
        "Do NOT treat loosely related occupational/cohort populations as evidence unless the candidate exposure "
        "itself is explicitly present in the cited snippet/title. "
        "If the citation discusses a different exposure than the candidate, exclude it. "
        "DO NOT invent citation_ids, file_ids, chunk_ids, DOI, URLs, titles, or years. "
        "Each evidence.supports entry must reference a citation_id present in citations. "
        "For each support, assign study_design and numeric metrics from the retrieved text only: "
        "study_quality_score, population_match, temporality_match, risk_of_bias, llm_confidence in [0,1]. "
        "Also assign support_direction_score in [-1,1] where -1 is contradictory, 0 is mixed/unclear, "
        "and +1 is strongly supportive for the exact exposure-symptom linkage and temporal pattern. "
        + (
            "This candidate is a two-exposure combination. "
            "Only include supports where BOTH exposures are explicitly present in the same citation context. "
            "Exclude single-exposure papers."
            if is_combo
            else ""
        )
    )
    question_payload = {
        "symptom_name": symptom_name,
        "item_name": item_name,
        "secondary_item_name": secondary_item_name,
        "ingredient_names": ingredient_names,
        "routes": routes or [],
        "lag_bucket_counts": lag_bucket_counts or {},
        "instructions": [
            "Retrieve evidence linking exposure to symptom.",
            "Return JSON matching schema exactly.",
            "Citations/supports must map to retrieved results.",
            "Reject citations where candidate exposure term is not explicitly present in snippet/title.",
            "Clinical/lay aliases are allowed only when clearly equivalent to the provided exposure/symptom in the retrieved snippet/title.",
            (
                "For combo candidates, every support must explicitly mention both exposures."
                if is_combo
                else "For single candidates, supports must explicitly mention the candidate exposure."
            ),
            f"Return up to {max_evidence_rows} strongest evidence rows; fewer is preferred over weak matches.",
            "Populate support-level study quality and match metrics strictly in [0,1].",
            "Populate support_direction_score in [-1,1] for each support.",
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
            print(f"[RAG] Responses API failed: {_sanitize_error_message(str(last_error))}")
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
        claim_polarity = _infer_polarity_strength(claim)
        for support in supports:
            if not isinstance(support, dict):
                continue
            citation_id = support.get("citation_id")
            snippet = support.get("snippet")
            chunk_id = support.get("chunk_id")
            study_design = support.get("study_design")
            study_quality_score = _bounded_metric(support.get("study_quality_score"))
            population_match = _bounded_metric(support.get("population_match"))
            temporality_match = _bounded_metric(support.get("temporality_match"))
            risk_of_bias = _bounded_metric(support.get("risk_of_bias"))
            llm_confidence = _bounded_metric(support.get("llm_confidence"))
            try:
                support_direction_score = float(support.get("support_direction_score"))
            except (TypeError, ValueError):
                support_direction_score = 0.0
            support_direction_score = max(-1.0, min(1.0, support_direction_score))
            if not isinstance(citation_id, str) or citation_id.strip() not in citation_map:
                continue
            if not isinstance(snippet, str) or not snippet.strip():
                continue
            if not isinstance(chunk_id, str) or not chunk_id.strip():
                continue
            if not isinstance(study_design, str) or not study_design.strip():
                study_design = "other"
            if retrieved_chunk_ids and chunk_id not in retrieved_chunk_ids:
                skipped_by_chunk_filter += 1
                # Fallback: keep row if snippet exists and citation is otherwise grounded.
                # Some SDK payloads surface chunk IDs in a different field than our parser.
                if os.getenv("RAG_STRICT_CHUNK_GROUNDING", "0") == "1":
                    continue

            citation = citation_map[citation_id.strip()]
            citation_url = citation.get("url")
            citation_url_value = citation_url.strip() if isinstance(citation_url, str) and citation_url.strip() else None
            citation_title = str(citation.get("title") or "")
            snippet_value = snippet.strip()
            cue_text = f"{citation_title} {claim.strip()} {snippet_value}".strip()
            if is_combo:
                if _combo_item_context_mismatch(
                    item_name=item_name,
                    secondary_item_name=secondary_item_name,
                    text=cue_text,
                ):
                    continue
            elif _item_context_mismatch(item_name=item_name, text=cue_text):
                continue
            if _symptom_context_mismatch(symptom_name=symptom_name, text=cue_text):
                # Do not ingest support rows for a different symptom than the candidate.
                continue
            support_direction_score = _apply_support_direction_cues(
                support_direction_score=support_direction_score,
                claim=claim.strip(),
                snippet=snippet_value,
                title=citation_title,
            )
            if _is_hazard_context_mismatch(
                item_name=item_name,
                snippet=snippet_value,
                title=citation_title,
            ):
                # Drop contaminated/outbreak context for generic exposures (e.g., plain "water").
                continue
            if abs(support_direction_score) >= 0.05:
                polarity = support_direction_score
            else:
                polarity = 0.25 * float(claim_polarity)
            output.append(
                {
                    "title": citation["title"],
                    "url": citation_url_value,
                    "publication_date": str(citation.get("year")) if citation.get("year") is not None else None,
                    "source": _citation_source_label(citation),
                    "item_name": item_name,
                    "ingredient_name": ingredient_names[0] if ingredient_names else None,
                    "symptom_name": symptom_name,
                    "summary": claim.strip(),
                    "snippet": snippet_value,
                    "evidence_polarity_and_strength": polarity,
                    "study_design": study_design.strip().lower(),
                    "study_quality_score": study_quality_score,
                    "population_match": population_match,
                    "temporality_match": temporality_match,
                    "risk_of_bias": risk_of_bias,
                    "llm_confidence": llm_confidence,
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
                        "source": _citation_source_label(citation),
                        "item_name": item_name,
                        "ingredient_name": ingredient_names[0] if ingredient_names else None,
                        "symptom_name": symptom_name,
                        "summary": claim.strip(),
                        "snippet": snippet.strip(),
                        "evidence_polarity_and_strength": polarity,
                        "study_design": "other",
                        "study_quality_score": 0.5,
                        "population_match": 0.5,
                        "temporality_match": 0.5,
                        "risk_of_bias": 0.5,
                        "llm_confidence": 0.5,
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
        secondary_item_id = candidate.get("secondary_item_id")
        secondary_item_name = (
            item_name_map.get(int(secondary_item_id))
            if secondary_item_id is not None
            else None
        )
        if ingredient_ids:
            for ingredient_id in sorted(ingredient_ids):
                ingredient_name = ingredient_name_map.get(int(ingredient_id))
                if not ingredient_name:
                    continue
                query = f"{ingredient_name} {symptom_name}"
                if query not in seen_queries:
                    seen_queries.add(query)
                    queries.append(query)
        elif item_name and secondary_item_name:
            query = f"{item_name} + {secondary_item_name} {symptom_name} interaction"
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


def _claim_row_passes_quality_floor(row: dict[str, Any]) -> bool:
    min_relevance = max(0.0, min(1.0, float(os.getenv("RAG_MIN_ONLINE_ROW_RELEVANCE", "0.10"))))
    min_quality = max(0.0, min(1.0, float(os.getenv("RAG_MIN_ONLINE_ROW_QUALITY", "0.35"))))
    min_population = max(0.0, min(1.0, float(os.getenv("RAG_MIN_ONLINE_ROW_POPULATION_MATCH", "0.30"))))
    min_temporality = max(0.0, min(1.0, float(os.getenv("RAG_MIN_ONLINE_ROW_TEMPORALITY_MATCH", "0.25"))))
    min_confidence = max(0.0, min(1.0, float(os.getenv("RAG_MIN_ONLINE_ROW_LLM_CONFIDENCE", "0.35"))))
    max_bias = max(0.0, min(1.0, float(os.getenv("RAG_MAX_ONLINE_ROW_RISK_OF_BIAS", "0.90"))))
    min_abs_direction = max(0.0, min(1.0, float(os.getenv("RAG_MIN_ONLINE_ROW_ABS_DIRECTION", "0.04"))))
    min_snippet_chars = max(20, int(float(os.getenv("RAG_MIN_ONLINE_ROW_SNIPPET_CHARS", "40"))))

    snippet = str(row.get("snippet") or row.get("summary") or "")
    relevance = max(0.0, min(1.0, float(row.get("relevance") or 0.0)))
    study_quality = _bounded_metric(row.get("study_quality_score"))
    population_match = _bounded_metric(row.get("population_match"))
    temporality_match = _bounded_metric(row.get("temporality_match"))
    risk_of_bias = _bounded_metric(row.get("risk_of_bias"))
    llm_confidence = _bounded_metric(row.get("llm_confidence"))
    direction = max(-1.0, min(1.0, float(row.get("evidence_polarity_and_strength") or 0.0)))

    composite = (
        0.30 * relevance
        + 0.25 * study_quality
        + 0.15 * population_match
        + 0.10 * temporality_match
        + 0.10 * llm_confidence
        + 0.10 * (1.0 - risk_of_bias)
    )
    if len(snippet.strip()) < min_snippet_chars:
        return False
    if abs(direction) < min_abs_direction:
        return False
    if risk_of_bias > max_bias:
        return False
    if composite >= min_quality:
        return True
    # Soft pass path for sparse candidates if several key metrics clear floors.
    if (
        relevance >= min_relevance
        and llm_confidence >= min_confidence
        and population_match >= min_population
        and temporality_match >= min_temporality
    ):
        return True
    # Sparse/novel fallback: allow a strong, grounded row when some support metrics
    # are missing/underestimated but relevance and confidence are high.
    sparse_min_relevance = max(min_relevance, 0.35)
    sparse_min_confidence = max(min_confidence, 0.45)
    sparse_min_study_quality = 0.20
    return (
        relevance >= sparse_min_relevance
        and llm_confidence >= sparse_min_confidence
        and study_quality >= sparse_min_study_quality
        and risk_of_bias <= max_bias
    )


def sync_claims_for_candidates(
    conn,
    *,
    candidates: list[dict[str, Any]],
    ingredient_name_map: dict[int, str],
    symptom_name_map: dict[int, str],
    item_name_map: dict[int, str],
    online_enabled: bool = True,
    max_papers_per_query: int = 8,
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
    rows_rejected_quality = 0
    duplicate_claim_rows_skipped = 0
    retrieval_stage_attempts = 0
    retrieval_stage_rows = 0
    candidates_without_rows = 0

    if online_enabled:
        now_iso = datetime.now(tz=timezone.utc).isoformat()
        for candidate in candidates:
            symptom_id = int(candidate["symptom_id"])
            symptom_name = symptom_name_map.get(symptom_id)
            if not symptom_name:
                continue
            candidate_item_id = int(candidate["item_id"])
            item_name = item_name_map.get(candidate_item_id)
            secondary_item_id = candidate.get("secondary_item_id")
            secondary_item_name = (
                item_name_map.get(int(secondary_item_id))
                if secondary_item_id is not None
                else None
            )
            ingredient_ids = sorted(int(value) for value in candidate.get("ingredient_ids", set()))
            ingredient_names = [
                ingredient_name_map[ingredient_id]
                for ingredient_id in ingredient_ids
                if ingredient_id in ingredient_name_map
            ]
            llm_rows: list[dict[str, Any]] = []
            candidate_rows_raw = 0
            def _call_llm_retriever(*, ingredient_names_arg: list[str]) -> list[dict[str, Any]]:
                kwargs = {
                    "symptom_name": symptom_name,
                    "ingredient_names": ingredient_names_arg,
                    "item_name": item_name,
                    "secondary_item_name": secondary_item_name,
                    "routes": sorted(candidate.get("routes", [])),
                    "lag_bucket_counts": candidate.get("lag_bucket_counts"),
                    "max_evidence_rows": max_papers_per_query,
                }
                try:
                    return llm_retriever(**kwargs)
                except TypeError:
                    kwargs.pop("secondary_item_name", None)
                    return llm_retriever(**kwargs)

            def _call_llm_retriever_relaxed_for_item(*, item_name_arg: str) -> list[dict[str, Any]]:
                kwargs = {
                    "symptom_name": symptom_name,
                    "ingredient_names": [],
                    "item_name": item_name_arg,
                    "secondary_item_name": None,
                    "routes": sorted(candidate.get("routes", [])),
                    "lag_bucket_counts": candidate.get("lag_bucket_counts"),
                    "max_evidence_rows": max_papers_per_query,
                }
                try:
                    return llm_retriever(**kwargs)
                except TypeError:
                    kwargs.pop("secondary_item_name", None)
                    return llm_retriever(**kwargs)
            if ingredient_ids:
                # Per expanded ingredient: use existing DB claims when present; call LLM only for missing coverage.
                missing_ingredient_names: list[str] = []
                for ingredient_id in ingredient_ids:
                    has_existing = conn.execute(
                        """
                        SELECT 1
                        FROM claims
                        WHERE ingredient_id = %s AND symptom_id = %s
                        LIMIT 1
                        """,
                        (int(ingredient_id), int(symptom_id)),
                    ).fetchone()
                    if has_existing is None and int(ingredient_id) in ingredient_name_map:
                        missing_ingredient_names.append(str(ingredient_name_map[int(ingredient_id)]))
                for ingredient_name in missing_ingredient_names:
                    retrieval_stage_attempts += 1
                    ingredient_rows = _call_llm_retriever(ingredient_names_arg=[ingredient_name])
                    candidate_rows_raw += len(ingredient_rows)
                    retrieval_stage_rows += len(ingredient_rows)
                    llm_rows.extend(ingredient_rows)
                # When ingredient coverage exists globally but does not yield useful evidence for this item context,
                # still query with item-level context as a first-line retrieval path.
                if not llm_rows and item_name:
                    retrieval_stage_attempts += 1
                    llm_rows = _call_llm_retriever(ingredient_names_arg=[])
                    candidate_rows_raw += len(llm_rows)
                    retrieval_stage_rows += len(llm_rows)
            else:
                retrieval_stage_attempts += 1
                llm_rows = _call_llm_retriever(ingredient_names_arg=ingredient_names)
                candidate_rows_raw += len(llm_rows)
                retrieval_stage_rows += len(llm_rows)
                # Combo retrieval is intentionally strict and may return no rows.
                # Fallback to single-item retrieval so per-item claims can still be acquired.
                if not llm_rows and secondary_item_id is not None:
                    relaxed_rows: list[dict[str, Any]] = []
                    if item_name:
                        retrieval_stage_attempts += 1
                        for row in _call_llm_retriever_relaxed_for_item(item_name_arg=str(item_name)):
                            row_copy = dict(row)
                            row_copy["_force_item_id"] = int(candidate_item_id)
                            relaxed_rows.append(row_copy)
                    if secondary_item_name:
                        retrieval_stage_attempts += 1
                        for row in _call_llm_retriever_relaxed_for_item(item_name_arg=str(secondary_item_name)):
                            row_copy = dict(row)
                            row_copy["_force_item_id"] = int(secondary_item_id)
                            relaxed_rows.append(row_copy)
                    llm_rows = relaxed_rows
                    candidate_rows_raw += len(llm_rows)
                    retrieval_stage_rows += len(llm_rows)
            if candidate_rows_raw == 0:
                candidates_without_rows += 1
            for row in llm_rows:
                if not _claim_row_passes_quality_floor(row):
                    rows_rejected_quality += 1
                    continue
                row_ingredient_id: int | None = None
                target_item_ids: list[int | None] = []
                forced_item_id_raw = row.get("_force_item_id")
                forced_item_id: int | None = None
                try:
                    if forced_item_id_raw is not None:
                        forced_item_id = int(forced_item_id_raw)
                except (TypeError, ValueError):
                    forced_item_id = None
                if ingredient_ids:
                    ingredient_name = row.get("ingredient_name")
                    if isinstance(ingredient_name, str) and ingredient_name.strip():
                        normalized_ingredient = _normalize_phrase(ingredient_name)
                        mapped_ingredient_id = ingredient_lookup.get(normalized_ingredient)
                        if mapped_ingredient_id in ingredient_ids:
                            row_ingredient_id = mapped_ingredient_id
                    if row_ingredient_id is None:
                        target_item_ids = [candidate_item_id]
                else:
                    if forced_item_id is not None:
                        target_item_ids = [forced_item_id]
                    else:
                        target_item_ids = [candidate_item_id]
                        if secondary_item_id is not None:
                            target_item_ids.append(int(secondary_item_id))
                row_symptom_id = symptom_id

                paper_url = row.get("url")
                if isinstance(paper_url, str) and paper_url.strip():
                    existing_paper = conn.execute(
                        "SELECT id FROM papers WHERE url = %s LIMIT 1",
                        (paper_url.strip(),),
                    ).fetchone()
                else:
                    existing_paper = conn.execute(
                        """
                        SELECT id FROM papers
                        WHERE title = %s
                          AND COALESCE(publication_date, '') = COALESCE(%s, '')
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

                snippet = str(row.get("snippet") or row.get("summary") or "").strip()
                if not snippet:
                    rows_rejected_quality += 1
                    continue
                snippet_hash = text_hash(snippet)
                if row_ingredient_id is not None:
                    target_item_ids = [None]
                for row_item_id in target_item_ids:
                    duplicate = conn.execute(
                        """
                        SELECT 1 FROM claims
                        WHERE paper_id = %s
                          AND symptom_id = %s
                          AND chunk_hash = %s
                          AND item_id IS NOT DISTINCT FROM %s
                          AND ingredient_id IS NOT DISTINCT FROM %s
                        LIMIT 1
                        """,
                        (
                            paper_id,
                            row_symptom_id,
                            snippet_hash,
                            row_item_id,
                            row_ingredient_id,
                        ),
                    ).fetchone()
                    if duplicate:
                        duplicate_claim_rows_skipped += 1
                        continue
                    claims_added += ingest_paper_claim_chunks(
                        conn,
                        paper_id=paper_id,
                        item_id=row_item_id,
                        ingredient_id=row_ingredient_id,
                        symptom_id=row_symptom_id,
                        summary=row["summary"],
                        evidence_polarity_and_strength=row["evidence_polarity_and_strength"],
                        study_design=row.get("study_design"),
                        study_quality_score=row.get("study_quality_score"),
                        population_match=row.get("population_match"),
                        temporality_match=row.get("temporality_match"),
                        risk_of_bias=row.get("risk_of_bias"),
                        llm_confidence=row.get("llm_confidence"),
                        citation_title=row["title"],
                        citation_url=row["url"],
                        source_text=snippet,
                    )

    return {
        "queries_built": len(queries),
        "papers_added": papers_added,
        "claims_added": claims_added,
        "rows_rejected_quality": rows_rejected_quality,
        "duplicate_claim_rows_skipped": duplicate_claim_rows_skipped,
        "retrieval_stage_attempts": retrieval_stage_attempts,
        "retrieval_stage_rows": retrieval_stage_rows,
        "candidates_without_rows": candidates_without_rows,
    }


def enrich_claims_for_candidates(
    conn,
    *,
    candidates: list[dict[str, Any]],
    ingredient_name_map: dict[int, str],
    symptom_name_map: dict[int, str],
    item_name_map: dict[int, str],
    online_enabled: bool = True,
    max_papers_per_query: int = 8,
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


def aggregate_evidence(
    retrieved_claims: list[dict[str, Any]],
    *,
    item_name: str | None = None,
    symptom_name: str | None = None,
) -> dict[str, Any]:
    if not retrieved_claims:
        return {
            "evidence_score": 0.0,
            "evidence_strength_score": 0.0,
            "avg_relevance": 0.0,
            "evidence_summary": "No matching evidence found for this symptom and exposure pattern.",
            "citations": [],
        }

    weighted_sum = 0.0
    total_weight = 0.0
    relevance_sum = 0.0
    citations: list[dict[str, Any]] = []
    seen_citation_keys: set[tuple[str, str, str]] = set()
    seen_scoring_keys: set[tuple[str, str, str]] = set()
    unique_claim_count = 0

    for claim in retrieved_claims:
        scoring_text = (
            claim.get("citation_snippet")
            or claim.get("chunk_text")
            or claim.get("summary")
            or ""
        )
        scoring_key = (
            str(claim.get("citation_title") or claim.get("title") or "").strip().lower(),
            str(claim.get("citation_url") or "").strip().lower(),
            str(scoring_text or "").strip().lower(),
        )
        if scoring_key in seen_scoring_keys:
            continue
        seen_scoring_keys.add(scoring_key)
        unique_claim_count += 1

        base_polarity = max(-1.0, min(1.0, float(claim.get("evidence_polarity_and_strength") or 0.0)))
        cue_text = " ".join(
            part
            for part in [
                str(claim.get("citation_title") or ""),
                str(claim.get("summary") or ""),
                str(claim.get("citation_snippet") or ""),
                str(claim.get("chunk_text") or ""),
            ]
            if part
        )
        polarity = _polarity_from_text_cues(
            fallback_polarity=base_polarity,
            text=cue_text,
        )
        if _is_hazard_context_mismatch(
            item_name=item_name,
            snippet=str(claim.get("citation_snippet") or ""),
            title=str(claim.get("citation_title") or claim.get("title") or ""),
        ):
            polarity = min(polarity, 0.0)
        if _symptom_context_mismatch(symptom_name=symptom_name, text=cue_text):
            polarity = min(polarity, 0.0)
        design_multiplier = _study_design_multiplier(claim.get("study_design"))
        relevance = max(0.05, float(claim.get("relevance") or 0.0))
        study_quality = _bounded_metric(claim.get("study_quality_score"))
        population_match = _bounded_metric(claim.get("population_match"))
        temporality_match = _bounded_metric(claim.get("temporality_match"))
        risk_of_bias = _bounded_metric(claim.get("risk_of_bias"))
        llm_confidence = _bounded_metric(claim.get("llm_confidence"))
        quality_weight = (
            0.35 * relevance
            + 0.20 * study_quality
            + 0.15 * population_match
            + 0.15 * temporality_match
            + 0.10 * (1.0 - risk_of_bias)
            + 0.05 * llm_confidence
        )
        weight = max(0.05, min(1.0, quality_weight))
        relevance_sum += weight
        weighted_sum += polarity * design_multiplier * weight
        total_weight += weight

        snippet = (
            claim.get("citation_snippet")
            or claim.get("summary")
            or claim.get("chunk_text")
            or ""
        )
        snippet_text = (snippet[:280] + "...") if len(snippet) > 280 else snippet
        citation_title = claim.get("citation_title") or claim.get("title")
        citation_url = claim.get("citation_url")
        citation_key = (
            str(citation_title or "").strip().lower(),
            str(citation_url or "").strip().lower(),
            str(snippet_text or "").strip().lower(),
        )
        if citation_key not in seen_citation_keys:
            seen_citation_keys.add(citation_key)
            citations.append(
                {
                    "title": citation_title,
                    "source": claim.get("source"),
                    "url": citation_url,
                    "snippet": snippet_text,
                    # Store the adjusted polarity actually used in aggregation so downstream
                    # quality metrics (support/contradict/neutral ratios) are consistent
                    # with the signed evidence_score gate.
                    "evidence_polarity_and_strength": polarity,
                    "raw_evidence_polarity_and_strength": max(
                        -1.0, min(1.0, float(claim.get("evidence_polarity_and_strength") or 0.0))
                    ),
                    "study_design": claim.get("study_design"),
                    "study_quality_score": study_quality,
                    "population_match": population_match,
                    "temporality_match": temporality_match,
                    "risk_of_bias": risk_of_bias,
                    "llm_confidence": llm_confidence,
                }
            )

    evidence_score = (weighted_sum / total_weight) if total_weight > 0 else 0.0
    evidence_score = max(-1.0, min(1.0, evidence_score))
    avg_relevance = (relevance_sum / unique_claim_count) if unique_claim_count > 0 else 0.0
    avg_relevance = max(0.0, min(1.0, avg_relevance))
    # Strength should not max out from a single claim.
    # One strong claim can still be meaningful, but multi-claim support increases confidence.
    claim_count = unique_claim_count
    coverage_factor = min(1.0, 0.35 + (0.16 * max(0, claim_count - 1)))
    # Blend coverage and grounding quality so single strong claims are possible
    # but weak/low-coverage evidence does not saturate.
    quality_factor = avg_relevance
    strength_factor = (0.6 * quality_factor) + (0.4 * coverage_factor)
    evidence_strength_score = max(0.0, min(1.0, abs(evidence_score) * strength_factor))
    direction = "mixed"
    if evidence_score > 0.25:
        direction = "supportive"
    elif evidence_score < -0.25:
        direction = "contradictory"

    return {
        "evidence_score": evidence_score,
        "evidence_support_score": evidence_score,
        "evidence_strength_score": evidence_strength_score,
        "avg_relevance": avg_relevance,
        "evidence_summary": (
            f"{claim_count} claim(s) retrieved; overall evidence is {direction}."
        ),
        "citations": citations,
    }


def generate_user_evidence_summary(
    *,
    item_name: str | None,
    symptom_name: str | None,
    citations: list[dict[str, Any]],
    evidence_score: float,
) -> str | None:
    # One concise, product-safe sentence generated from citation snippets.
    if not citations:
        return None

    client = _get_openai_client()
    if client is None:
        return None

    short_citations: list[dict[str, Any]] = []
    for citation in citations[:4]:
        short_citations.append(
            {
                "title": citation.get("title"),
                "source": citation.get("source"),
                "snippet": citation.get("snippet"),
                "url": citation.get("url"),
            }
        )
    support_direction = "supportive" if float(evidence_score) >= 0 else "contradictory_or_mixed"
    schema = {
        "name": "insight_user_summary",
        "strict": True,
        "schema": {
            "type": "object",
            "additionalProperties": False,
            "properties": {"summary": {"type": "string"}},
            "required": ["summary"],
        },
    }
    try:
        response = client.chat.completions.create(
            model=os.getenv("OPENAI_SUMMARY_MODEL", os.getenv("RAG_MODEL", "gpt-4o-mini")),
            temperature=0.1,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Write one concise sentence for a health insight card. "
                        "Use only provided snippets; do not invent facts. "
                        "The sentence must describe whether the evidence supports or contradicts "
                        "a linkage between the provided item_name and symptom_name. "
                        "Must start exactly with 'Supportive evidence states that ' "
                        "if direction is supportive, otherwise start with "
                        "'Current evidence suggests that '. "
                        "Do not describe study design, sample demographics, journal metadata, or methods. "
                        "Do not mention claim counts, confidence scores, or model details. "
                        "Prefer plain language effect wording about the linkage itself. "
                        "Keep under 24 words."
                    ),
                },
                {
                    "role": "user",
                    "content": json.dumps(
                        {
                            "item_name": item_name,
                            "symptom_name": symptom_name,
                            "direction": support_direction,
                            "citations": short_citations,
                        }
                    ),
                },
            ],
            response_format={"type": "json_schema", "json_schema": schema},
        )
    except Exception:
        return None

    try:
        content = response.choices[0].message.content or ""
        payload = json.loads(content)
        summary = str(payload.get("summary") or "").strip()
    except Exception:
        return None
    if not summary:
        return None
    summary = " ".join(summary.split())
    summary = re.sub(
        r"(?i)^supportive evidence states that\s+",
        "Supportive evidence states that ",
        summary,
    )
    summary = re.sub(
        r"(?i)^current evidence suggests that\s+",
        "Current evidence suggests that ",
        summary,
    )
    expected_prefix = (
        "Supportive evidence states that "
        if float(evidence_score) >= 0
        else "Current evidence suggests that "
    )
    if not summary.lower().startswith(expected_prefix.lower()):
        summary = f"{expected_prefix}{summary.lstrip()}"
    if summary.startswith(expected_prefix) and len(summary) > len(expected_prefix):
        first_char_idx = len(expected_prefix)
        summary = summary[:first_char_idx] + summary[first_char_idx].lower() + summary[first_char_idx + 1 :]
    words = summary.split()
    if len(words) > 24:
        summary = " ".join(words[:24]).rstrip(" .,:;") + "."
    if not summary.endswith((".", "!", "?")):
        summary = f"{summary}."
    return summary
