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