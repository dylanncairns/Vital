# Vital

## Purpose and Scope
***Vital is a system for longitudinal health symptom & exposure logging that exists to provide personalized, evidence-backed insights into patterns in a user's log that established research may contextualize as worthy of further investigation, and does not make medical, safety, or causal claims.***


## User Experience
- User logs exposure and symptom events either as structured input or as free text.
    - Exposure events include food/drink, oral medication/supplements, lifestyle/physiology, topical/dermal interaction, inhalation, injection, nasal/sinus ingestion, proximity/environment interaction
- Events are stored and structured on the user's timeline, where they can view and edit their events
- Candidate exposure-symptom linkage insights are generated from temporal patterns
- Published research and analysis pertaining to a candidate insight is attached
- Insights are scored and gated with algorithms that compute user-specific probability, evidence quality and support, and penalties
- Insights that pass the score gate are shown to user, who can verify and reject the insight to improve the model's accuracy globally and user-specifically

## Run it Locally
- Set env vars: `OPENAI_API_KEY`=__, `RAG_VECTOR_STORE_ID`=__, `RAG_OPENAI_MODEL`=__, `EXPO_PUBLIC_API_BASE_URL=http://127.0.0.1:8000`
- Install Python dependencies: `pip install -r requirements.txt`
- Terminal 1 (from repo root): `uvicorn api.main:app --reload`
- Terminal 2 (from apps/mobile or apps/web): `npm install && npm start`
- Terminal 3 (from repo root): `python3 -m api.job_worker --limit 100 --max-papers-per-query 5`
- Test suite (from repo root): `PYTHONPATH=. pytest -q`

## How it Works
- [Backend is built on FastAPI](api/main.py)
    - User authentication and sessions
    - Input validation and routing to normalization endpoints
    - Event CRUD
    - Insight reading and recomputation endpoints
    - Endpoints for evidence retrieval background jobs and citation audits/validation jobs
    - Endpoints for authentication/login/logout
    - Model retraining job endpoint
- [Ingestion pipeline](ingestion/)
    - Free text parsing into structured events
    - Timestamp normalization (local <--> UTC)
    - Exposure and symptom alias resolution
    - Exposure to a multi-ingredient item event expansion into events for ingredients
- [Data is stored in SQLite with migrations](data/schema.sql)
    - Event domain with all events, user timeline, failed parsing capture, recurring exposure rules
    - Knowledge and evidence domain with items, ingredients, aliases, items_ingredients, evidence + abstracts, retrieval history
    - Feature and output domain with derived features for candidate-specific temporal features, insights with score, insight-event links to connect timeline to insights, insight verifications
    - Operations domain with background jobs for worker and state of model retraining trigger
- [Retrieval-Augmented Generation evidence retrieval pipeline](ml/rag.py)
    - rag.py sends OpenAI Responses API a targeted evidence retrieval query
    - OpenAI GPT-4.1 model is allowed to search only a configured document index vector store and is given context for item, symptom, route, lag buckets
    - If this process acquires no evidence for candidate, it sends OpenAI web search API a query to add evidence to the vector store and is given the same context
    - Evidence validation filters citations with weak relevance to candidate linkage
    - Evidence is aggregated into support direction, strength, relevance, citation payload, and user-facing summary
    - Computes overall evidence support strength score (E)
- [Insight Generation](ml/insights.py) and [Score Computation](ml/final_score.py)
    - Linkage candidates are built from each user’s exposure/symptom history using lag windows and route context
    - Time-based and recurrence features are computed, including co-occurrence patterns, lag statistics, density, route-temporal shares, severity-after signal
    - Evidence is retrieved for a candidate linkage from claims
    - Probability of linkage (P) between exposure(s) and symptom based on the user's personal timeline data is computed by a gradient-boosted (XGBoost) classification model
    - Evidence quality score (q) is computed with aggregated features from RAG
    - Penalty score is computed (p) for lack of info, low co-occurence count, high pair density to address confounding variables, low timestamp delta confidence, contradictory citations
    - Final score is computed by fusion logistic regression model that receives metrics that involve E, P, q, p
- [Background Worker](api/job_worker.py)
    - recompute_candidate job queued to recompute insight metrics for a specific user/symptom/item group
    - evidence_acquire_candidate job queued when trying to retrieve evidence for candidate via OpenAI API vector search and if none returned then retrieving new evidence via OpenAI web search, heavily interacting with rag.py
    - model_retrain job queued when number of total events for a user hits trigger 
    - citation_audit job repeatedly queued to check if stored citation URLs are valid, removing links that are missing paper or claims or summaries or sources


## Security
- Password hashing
- Random bearer session tokens with expiry and revocation
- Token to user mismatch protection
- Parameterized SQL queries across repositories
- Currently phasing out explicit user_id fallback when token auth fails


## Tests
- Included unit tests for
    - background job processing and requeue behavior
    - job enqueueing
    - citation auditing
    - ingestion text parsing rules
    - insights recomputation
    - A+B=C candidate insight logic and handling
    - RAG evidence retrieval behavior
    - Vector-ingest query construction
    - Model logic
    - Feature construction
    - Insight scoring with guardrails
    - Training data generation
- Included integration tests for
    - Insight life cycle
    - Job life cycle
    - Candidate linkage life cycle
    - Input -> ingestion -> candidate cycle
    - Candidate -> evidence -> scoring cycle


## Build History
- Personal annotations on project evolution and notable changes over time can be found in the [Build Log](docs/buildlog.md)


## What's Next?
- Voice-to-text event logging
- Migration to PostgreSQL to support more users with disk encryption
- Improved models
- Improved token security and endpoint authentication guard
- Admin and user role separation
- Publication!
- Long term vision includes integration of bloodwork, wearable data, and genetic sequencing
