Build Log:

# Commit 1
- Established framework for what system and repo will look like upon end of phase 1 (first completed and deployable version)

# Commit 2
- Archived test run model (prototype v1) in iOS/VitalPrototype
- Built functional barebones python api and tested health

# Commits 3-5
- Built basic swift <-> python <-> sql prototype v2 and tested successful integration across frontend & backend

# Commits 6-8
- Updated sqlite table schemas from foundational products & ingredients & products_ingredients to items, ingredients, aliases, claims, evidence, users, symptoms, exposure_events, symptom_events
    - User table is referenced by exposure_events (also references items) and symptom_events (also references symptoms)
    - Items_ingredients references items and ingredients
    - Claims references ingredients and evidence 
- Integrated updated schema.db with db.py using pathlib
- Created api/__init__.py and api/repositories/__init__.py to initialize api repos as packages

# Commits 9-10
- Added initial data flow where /events endpoint accepts JSON input with event details, data is then validated and stored in SQLite symptom_event or exposure_event table, and then saved event info and id is returned

# Commit 11
- Created backend for events data display
    - Added get/events endpoint to create timeline that displays exposures and symptoms longitudinally
    - Inserted seed exposure and symptom event data for a single user to test /events flow for get and post
        - schema.db renamed to central.db
        - schema.sql serves as schema template
        - seed.sql used for testing

# Commit 12
- Established end-to-end loop for "Log Event" input form page (post/events endpoint) and a timeline display page (get/events endpoint)
    - archived swiftui clients
    - created apps folder for react native mobile and react web

# Commit 13
- updated schema.sql
    - evidence --> papers (for ingestion of published evidence linking a symptom and ingredient)
    - derived_features table for storing inputs to model that will compute confidence score for there being strong evidence supporting linkage between a symptom and ingredient based on a user's specific exposures and symptoms patterns
    - insights table for user-facing confidence score, cited sources, and linked symptoms and ingredients
    - claims table updated to reference ingredient, symptom, paper, and summary of a potential linkage + evidence_polarity_and_strength (-1 to 1)
    - added alias tables to handle ingestion of user-logged events/symptoms with different names that reference the same item/event
        - “Head ache”, “head-ache”, “Headache” all normalize to head ache
    - added raw_event_ingest staging table
    - central.db recreated and verified to match new schema
- implemented ingestion pipeline
    - Rule‑based parsing with placeholder for external API call
    - writes structured events to db or stores raw text if parsing fails
        - user input is submitted to endpoint POST/events/ingest_text 
        - Parser tries to extract event type, time or time range, severity, candidate name which it resolves to canonical ID
        - if it succeeds it writes to exposure_events or symptoms_events
        - if it fails it writes to raw_event_ingest
    - resolve_item_id / resolve_symptom_id normalize ingested input and create canonical IDs
    - timestamp normalization supports exact timestamp or ingested time range w confidence
    - room for implementation of voice-to-text input that will be ingested through this same pipeline with parse_with_api function in ingest_text.py
    - implemented table exposure_expansions, which takes a logged event and references all of the ingredients contained in an item
        - items_ingredients and ingredients can be joined and rows are inserted into exposure_expansions, which allows candidate symptom/exposure pairs to reference ingredient ids and/or item ids
        - one feature action = exposure insert and expansion rows succeed or fail together to ensure full writes
    - incorporated openai API for text parsing first pass, if it fails then use the regex previously established in the bullets above
- updated client.ts, events.ts, logeventscreen.tsx to implement the text blurb input which will later be transferred to voice-to-text
- debugged and improved end-to-end flow
    - Mobile sends either structured event to POST /events or free text to POST /events/ingest_text
    - /events:
        - validates/normalizes payload with normalize_event.py
        - failures sent to raw_event_ingest table
        - resolves IDs by name if needed with resolve.py
        - inserts into exposure_events or symptom_events in main.py if normalized
        - for exposures, expands item to ingredients via expand_exposure.py
        - success insert response echoes event payload
    - /events/ingest_text:
        - calls ingest_text.py where OpenAI API is called first to attempt to parse input
        - if API call fails, regex parsing fallback is used
        - OpenAI output is schema contrained, and if successful event is written as a structured event entry into event table 
        - if both parsing mechanisms fail to compose output that is valid in schema, text input is written to raw_event_ingest table
    - JSON status field used as truth source for success/failure of event logging 
        - NormalizationError catches incompatibilities in event normalization
        - errors storing events into DB, parsing failure, ingestion failure all handled
    - insertions into tables are atomic operations (executes as a single complete step to prevent data corruption)
- update UI for timeline to properly handle multiple dates
    - ingestion now handles user input as according to their local time, stores in backend with standardized UTC time, and displays on timeline with their local time
- lots of ingestion pipeline debugging
    - fixed route identification
    - allowed for a multi-item exposure split 
        - ie "ate steak and potatoes" splits into exposure event for ingestion of steak and exposure event for ingestion of potatoes
        - fixed logging failure when timestamp given as "morning", "afternoon" etc
    - next version will remove most of the regex in ingestion pipeline and outsource parsing to LLM api calls with strict formatting rules, as regex cannot keep up with the variety in user input

# Commit 14
- upgraded from pure event logging to a foundation for insight generation
    - production skeleton for linkage insight generation built, UI unchanged
    - Updated route alias mapping with token normalization
    - Added migration framework and startup migrations in db.py
        - Implemented scaffolding for future RAG evidence/citation retrival
        - Migration runner allows existing DB to continually get upgraded as schema evolves, since model code will expect columns that old DB versions may not have
            - prevents runtime failures
    - Refined normalization calls in ingestion pipeline and ensure normalized output before writing to DB, ensuring text ingest and /events share canonical route names
        - standardized at ingestion so downstream analytics are consistent, preventing semantic drift from fragmenting generated candidates
    - Added insights API endpoint to main.py
        - post/insights/recompute computes insights for a user and get/insights lists structured computed insights
            - backend endpoints created, frontend will eventually display insights on timeline but yet to be implemented
    - Added ML folder currently containing insights.py
        - generates candidate linkages between symptoms and exposure patterns
        - define unit of analysis as exposure-symptom pairs and preseve temporal shape as opposed to raw cooccurence, letting lag buckets capture timing patterns that can affect linkage plausability
            - lag bucket logic supports a multitude of candidate linkages depending on temporal patterns of exposure even if between the same symptom and exposure by aggregating per item id and symptom id, storing bucket counts in lag_bucket_counts rather than separating into persisted candidate rows for each temporal candidate
            - derived_features generates and stores derived features for model input into DB
            - insights and retrival_runs are placeholder structured output
                - next will build RAG retrival of citation and evidence
                - for now just locked API contracts between insights and main endpoints
    - updated schema for claims and insights
        - expansion to prepare for evidence retrival
        - added retrival_runs for context of where candidate linkages are generated from when retriving evidence
    - most recent commit is fully backend logic and preparation for insight generation models as stated above, no frontend implementation yet
    - next versions will include RAG evidence retrival, then model scoring, then UI update to add insights to timeline 

# Commit 15
- cleaned up some duplicate engineering logic
    - removed old uncalled files (items.py and symptoms.py)
    - added raw_event_ingest and removed duplicate logic found in main and ingest_text
    - added time_utils and removed duplicate logic found in normalize_event and ingest_text

# Commit 16
- validated feature computation
    - added tests for feature math and time window logic
        - verified lag minimum and average, coocurrence handling, number of features generated, severity inference average, insight scores, retrival run endpoint
        - validated recompute handling of improper timestamps and ensured it does not crash upon invalid context for feature generation
- added defensive handling in insight generation to cover invalid rows, empty lag arrays, overeager error throwing
- added new derived feature fields (cooccurrence_unique_symptom_count and pair_density) while keeping existing cooccurrence_count as raw pair count for backward compatibility
- implemented dual and equal feature generation for items and ingredients with rollback from ingredient to item only where items_ingredients event expansion occurred
- implemented migration that add missing columns for existing DBs with _column_exists checks
- order of operations from current state
    - implement RAG evidence/citation retrival
    - implement XGBoost inference path and model correlation score + evidence strength score
    - update UI to properly display insight computation output

# Commit 17
- Implemented retrival of evidence and the claim it supports tied to each candidate pair, with citations returned in /insights
    - Added rag.py pipeline to retrieve evidence with structured querie into OpenAI API using environment-configurable model (currently gpt-4.1) to retrieve relevant citations and return structured output when given a clear candidate linkage and instruction as part of the prompt
        - claims DB currently stores local token embeddings (local-token-v1 JSON vector) and OpenAI vector store is used during retrival path
        - vector_ingest.py pipeline contains vector storing/setup/upload/polling and auto discovery query generation, along with structured output parsing
        - query retrieves top-k chunks per candidate from insights.py including ingredient/item name, symptom, temporal pattern, route, and lag bucket context
        - Ingredient specific retrival preffered when expansions exist and item level retrival default for non expanable exposures
    - Built citations_json from retrieved chunks and algo to derive evidence_score from retrieved claim/chunk signals
        - Weighted average over all chunks for matched evidence polarity/strength and relevance used to compute score, with each chunk containing a title, url, snippet and strength plus a summary of the implications
        - Retrival provenance stored in retrival_runs table and evidence outputs stored into insights
    - Candidate generation is user-specific and temporal, while evidence data is shared and reusable so recompute is fast and deterministic while still producing candidate-specific citations and scores
    - active online acquisition paths (/rag/sync and background evidence jobs) used so acquisition is incremental, not strictly one-time
        - implemented retrieval-only recomputation for finding relevant evidence for an individual candidate linkage via background jobs when evidence is insufficient 
        - background jobs added to allow for new evidence retrival upon entry of new symptoms or events
            - /jobs/process processes recompute_candidate and evidence_acquire_candidate
            - job_worker.py is continous and can serve as temp background worker but would eventually be moved to cloud so i dont have to run it in terminal to enable continuous evidence retrival
    - added claims schema migration for item-level support
        - claims.item_id added and migration/index updates (idx_claims_item_symptom_paper)
    - /insights/recompute now locally recomputes from the persisted DB of evidence which is synced, while new evidence ingestion is handled by async background jobs
- UI updates: insights tab added and is now user facing, which displays insights that pass gating
    - gating now sets candidate linkage + evidence as an entity to supported, suppressed_no_citations, or suppressed_low_evidence_strength
        - more accurate evidence score functionality and threshold will be added next
        - display_status is returned per entity, only allowing supported entities to pass to UI
- unit tests: added for background jobs, insights recomputation, background evidence retrival in jobs, evidence retrival in rag.py, vector ingestion (construction of queries for OpenAI API), and full insights integration
    - Manual integration test to recompute insights as user inputs exposures and symptoms and ensure citations/evidence fields are populated properly
        - ingest_text.py queues scoped recompute_candidate jobs after successful ingest
            - exposure ingest -> queue jobs for existing user symptoms
            - symptom ingest -> queue jobs for existing user exposures
- full current end-to-end flow:
    - Novel pair logged by user
    - Worker tries retrieval from existing database of evidence/citations
    - If evidence is insufficient then worker auto-runs acquisition for that exact pair with OpenAI API LLM query and ingests papers/claims
    - Worker recomputes candidates
    - Insight appears in UI if evidence passes thresholds

# Commit 18
- Developed model that uses gradient boosting to predict whether a candidate linkage is likely true for a user
    - Input features include:
        - user's personal timeline
        - candidate context 
        - rag evidence strength score
        - body of evidence citation count, stance polarity, abstract interpretation
    - Outputs probability from 0 to 1 for above context
    - For each candidate an exact feature vector is built in identical order
- Overall UI-displayed Confidence Score S = q(E) + P - p
    - E is the OpenAI API evidence retrival generated score for how supportive evidence is of candidate linkage
        - extracting evaluation from study design, stance, applicability, effect statements is a strong suit of LLM
    - q measures the quality of the total body of evidence retrieved for a candidate linkage
        - manually written deterministic algo more reliable, auditable, stable for health-regulatory-guideline association 
    - P is the probability that XGBoost model computes for if the given candidate linkage is likely true for the specific user
        - gradient boosting strong with tabular, sparse, noisy, temporal data with strong calibration and control
    - p is any penalties that originate from quality of user input, meaning that the less information they give per timeline entry involved in the candidate linkage then the more penalized the confidence score will be
        - manual algo to calculate based on explicit pitfalls of user interaction behavior
- trimmed a lot of hardcoding with the help of a multitude of imported packages 
- model training and eval + improved standing algorithms
    - used deterministic, LLM generated training data for candidates with obvious determinism to train and test model
        - only very obvious correlates backed by evidence and very obvious non correlates that can be 99% sure no linkage
        - any ambiguous training data would not be appropriate for health oriented setting where precision and confidence in insights is of the upmost importance, as mislabeling/overconfidence cannot be tolerated
    - ran training, verified artifacts exist, validated runtime health and migration safety
    - verified OpenAI API responses include required support fields and worker repeatedly retries failed jobs while stale running jobs are requeued and long LLM calls are timed out + placed back into queue
    - verified that get/insights always returns P, E, q, p and thresholds are loaded from XGBoost artifact
        - ensures insight display gating is functional and accurate
    - model retraining is triggered when enough new timeline data is added
        - handled by existing background job worker
        - trained on pooled global data but uses user's features for personalized inference
        - curated_linkages.json is a handcrafted dataset of obvious positives/negatives
        - training rows are also built from stored insights and derived features
        - training_data.py builds many rows from real event windows per user
            - Case rows are anchored to established symptom episodes, with y=1
            - Control rows are anchored to time-aware windows that lack nearby symptom episodes, with y=0
            - Features are computed identically for both so model can learn pattern differences in symptom onset vs baseline windows
        - train_model.py combines these with --dataset-source hybrid
    - curated dataset made to align with the purpose of the app, which is to surface insight into underlying reactions to logged exposures that might be causing QOL diminishing health symptoms which user may not previously have correlated
- implemented a dedicated fusion score calculator in final_score.py, which trains a fusion logistic regression model over P, q (E alr factored in), p, P*q, q*(1-p), contradiction ratio, citation count
- standing functionality: ingestion, candidate generation, RAG-backed evidence, persisted insights, background jobs, and calibrated scoring all integrated and working together