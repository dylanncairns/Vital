Build Log:


- Established framework for what system and repo will look like upon end of phase 1 (first completed and deployable version)

- Archived test run model (prototype v1) in iOS/VitalPrototype
- Built functional barebones python api and tested health

- Built basic swift <-> python <-> sql prototype v2 and tested successful integration across frontend & backend

- Updated sqlite table schemas from foundational products & ingredients & products_ingredients to items, ingredients, aliases, claims, evidence, users, symptoms, exposure_events, symptom_events
    - User table is referenced by exposure_events (also references items) and symptom_events (also references symptoms)
    - Items_ingredients references items and ingredients
    - Claims references ingredients and evidence 

- Integrated updated schema.db with db.py using pathlib
- Created api/__init__.py and api/repositories/__init__.py to initialize api repos as packages

- Added initial data flow where /events endpoint accepts JSON input with event details, data is then validated and stored in SQLite symptom_event or exposure_event table, and then saved event info and id is returned

- Created backend for events data display
    - Added get/events endpoint to create timeline that displays exposures and symptoms longitudinally
    - Inserted seed exposure and symptom event data for a single user to test /events flow for get and post
        - schema.db renamed to central.db
        - schema.sql serves as schema template
        - seed.sql used for testing

- Established end-to-end loop for "Log Event" input form page (post/events endpoint) and a timeline display page (get/events endpoint)
    - archived swiftui clients
    - created apps folder for react native mobile and react web

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