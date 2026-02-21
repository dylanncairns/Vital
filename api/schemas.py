# standardize base schemas for repeated payload patterns

import os
from typing import Optional

from pydantic import BaseModel, Field


# Validate / standardize input JSON payload format for /events
# Will contain user_id and either item_id + route or symptom_id + severity
# supports id or name resolution fields, exact or range time fields, can be raw text
class EventIn(BaseModel):
    event_type: str = Field(pattern="^(exposure|symptom)$")
    user_id: Optional[int] = None
    timestamp: Optional[str] = None
    time_range_start: Optional[str] = None
    time_range_end: Optional[str] = None
    time_confidence: Optional[str] = Field(default=None, pattern="^(exact|approx|backfilled)$")
    raw_text: Optional[str] = None
    item_id: int | None = None
    item_name: Optional[str] = None
    route: str | None = None
    symptom_id: int | None = None
    symptom_name: Optional[str] = None
    severity: int | None = None


# Validate / standardize output JSON payload format for /events - adds event id to enable echoing data after an insert
class EventOut(EventIn):
    id: int
    status: Optional[str] = None
    resolution: Optional[str] = None


# Define the JSON format for an entity in timeline (a timeline row that get/events returns)
class TimelineEvent(BaseModel):
    id: int
    event_type: str
    user_id: int
    timestamp: Optional[str] = None
    time_range_start: Optional[str] = None
    time_range_end: Optional[str] = None
    time_confidence: Optional[str] = None
    raw_text: Optional[str] = None
    item_id: Optional[int] = None
    item_name: Optional[str] = None
    route: Optional[str] = None
    symptom_id: Optional[int] = None
    symptom_name: Optional[str] = None
    severity: Optional[int] = None


# Shape of ingested text from user text blurb input
class TextIngestIn(BaseModel):
    user_id: Optional[int] = None
    raw_text: str
    timezone_offset_minutes: Optional[int] = Field(default=None, ge=-840, le=840)


# Response shape for /events/ingest_text
class TextIngestOut(BaseModel):
    status: str
    event_type: Optional[str] = None
    resolution: Optional[str] = None
    reason: Optional[str] = None


class RecomputeInsightsIn(BaseModel):
    user_id: Optional[int] = None
    online_enabled: bool = Field(
        default_factory=lambda: os.getenv("RAG_ENABLE_ONLINE_RETRIEVAL", "1") == "1"
    )
    max_papers_per_query: int = Field(default=8, ge=1, le=30)


class RecomputeInsightsOut(BaseModel):
    status: str
    user_id: int
    candidates_considered: int
    pairs_evaluated: int
    insights_written: int


class RagSyncIn(BaseModel):
    user_id: Optional[int] = None
    online_enabled: bool = True
    max_papers_per_query: int = Field(default=8, ge=1, le=30)


class RagSyncOut(BaseModel):
    status: str
    user_id: int
    candidates_considered: int
    queries_built: int
    papers_added: int
    claims_added: int


class ProcessJobsIn(BaseModel):
    limit: int = Field(default=20, ge=1, le=200)
    max_papers_per_query: int = Field(default=8, ge=1, le=30)


class ProcessJobsOut(BaseModel):
    status: str
    jobs_claimed: int
    jobs_done: int
    jobs_failed: int
    recompute_jobs_done: int
    evidence_jobs_done: int
    model_retrain_jobs_done: int
    citation_audit_jobs_done: int


class CitationAuditIn(BaseModel):
    user_id: Optional[int] = None
    limit: int = Field(default=300, ge=1, le=5000)
    delete_missing: bool = True


class CitationAuditOut(BaseModel):
    status: str
    scanned_urls: int
    missing_urls: int
    deleted_claims: int
    deleted_papers: int
    errors: int


class CitationAuditEnqueueOut(BaseModel):
    status: str
    job_id: Optional[int] = None


class EventPatchIn(BaseModel):
    timestamp: Optional[str] = None
    time_range_start: Optional[str] = None
    time_range_end: Optional[str] = None
    time_confidence: Optional[str] = Field(default=None, pattern="^(exact|approx|backfilled)$")
    raw_text: Optional[str] = None
    item_id: int | None = None
    item_name: Optional[str] = None
    route: str | None = None
    symptom_id: int | None = None
    symptom_name: Optional[str] = None
    severity: int | None = None


class EventMutationOut(BaseModel):
    status: str
    event_id: int
    event_type: str
    jobs_queued: int


class RecurringExposureRuleIn(BaseModel):
    user_id: Optional[int] = None
    item_id: int | None = None
    item_name: Optional[str] = None
    route: str = "ingestion"
    start_at: str
    interval_hours: int = Field(ge=1, le=24 * 30)
    time_confidence: str = Field(default="approx", pattern="^(exact|approx|backfilled)$")
    notes: Optional[str] = None


class RecurringExposureRulePatchIn(BaseModel):
    route: str | None = None
    start_at: str | None = None
    interval_hours: int | None = Field(default=None, ge=1, le=24 * 30)
    time_confidence: str | None = Field(default=None, pattern="^(exact|approx|backfilled)$")
    is_active: bool | None = None
    notes: Optional[str] = None


class RecurringExposureRuleOut(BaseModel):
    id: int
    user_id: int
    item_id: int
    item_name: str
    route: str
    start_at: str
    interval_hours: int
    time_confidence: Optional[str] = None
    is_active: int
    last_generated_at: Optional[str] = None
    notes: Optional[str] = None
    created_at: Optional[str] = None
    updated_at: Optional[str] = None


class InsightCitationOut(BaseModel):
    source: Optional[str] = None
    title: Optional[str] = None
    url: Optional[str] = None
    snippet: Optional[str] = None
    evidence_polarity_and_strength: Optional[float] = None


class InsightOut(BaseModel):
    id: int
    user_id: int
    item_id: int
    item_name: str
    secondary_item_id: Optional[int] = None
    secondary_item_name: Optional[str] = None
    is_combo: bool = False
    combo_key: Optional[str] = None
    combo_item_ids: Optional[list[int]] = None
    source_ingredient_id: Optional[int] = None
    source_ingredient_name: Optional[str] = None
    symptom_id: int
    symptom_name: str
    model_probability: Optional[float] = None
    evidence_support_score: Optional[float] = None
    evidence_strength_score: Optional[float] = None
    evidence_quality_score: Optional[float] = None
    penalty_score: Optional[float] = None
    overall_confidence_score: Optional[float] = None
    evidence_summary: Optional[str] = None
    display_decision_reason: Optional[str] = None
    display_status: Optional[str] = None
    user_verified: bool = False
    user_rejected: bool = False
    created_at: Optional[str] = None
    citations: list[InsightCitationOut]


class EventInsightLinkOut(BaseModel):
    event_type: str
    event_id: int
    insight_id: int


class InsightVerifyIn(BaseModel):
    user_id: Optional[int] = None
    verified: bool


class InsightVerifyOut(BaseModel):
    status: str
    insight_id: int
    user_id: int
    item_id: int
    symptom_id: int
    verified: bool
    rejected: bool = False


class InsightRejectIn(BaseModel):
    user_id: Optional[int] = None
    rejected: bool


class InsightFeedbackStatsOut(BaseModel):
    user_id: int
    verified_count: int
    rejected_count: int
    total_count: int
    surfaced_only: bool = True


class AuthRegisterIn(BaseModel):
    username: str
    password: str
    name: Optional[str] = None


class AuthLoginIn(BaseModel):
    username: str
    password: str


class AuthUserOut(BaseModel):
    id: int
    username: str
    name: Optional[str] = None


class AuthTokenOut(BaseModel):
    token: str
    user: AuthUserOut


class AuthMePatchIn(BaseModel):
    name: str
