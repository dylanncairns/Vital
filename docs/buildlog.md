Build Log:


- Established framework for what system and repo will look like upon end of phase 1 (first completed and deployable version)

- Archived test run model (prototype v1) in iOS/VitalPrototype
- Built functional barebones python api and tested health

- Built basic swift <-> python <-> sql prototype v2 and tested successful integration across frontend & backend

- Updated sqlite table schemas from foundational products & ingredients & products_ingredients to items, ingredients, aliases, claims, evidence, users, symptoms, exposure_events, symptom_events
- User table exists, which is referenced by exposure_events (also references items) and symptom_events (also references symptoms)
- items_ingredients references items and ingredients, claims references ingredients and evidence 

- integrate updated schema.db with db.py using pathlib
- create api/__init__.py and api/repositories/__init__.py to initialize api repos as packages

- add initial data flow where /events endpoint accepts JSON input with event details, data is then validated and stored in SQLite symptom_event or exposure_event table, and then saved event info and id is returned