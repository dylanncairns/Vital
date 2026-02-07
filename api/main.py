from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Optional

from api.db import get_connection, initialize_database
from api.repositories.events import list_events
import sqlite3

app = FastAPI(
    title = "Vital API",
    version = "0.1.0",
    )

# Ensure SQLite schema exists on startup
@app.on_event("startup")
def on_startup():
    initialize_database()

# Validate / standardize input JSON payload format for /events
# Will contain user_id and either item_id + route or symptom_id + severity
class EventIn(BaseModel):
    event_type: str = Field(pattern="^(exposure|symptom)$")
    user_id: int
    timestamp: str
    item_id: int | None = None
    route: str | None = None
    symptom_id: int | None = None
    severity: int | None = None

# Validate / standardize output JSON payload format for /events - adds event id to enable echoing data after an insert
class EventOut(EventIn):
    id: int

# Define the JSON format for an entity in timeline (a timeline row that get/events returns)
class TimelineEvent(BaseModel):
    id: int
    event_type: str
    user_id: int
    timestamp: str
    item_id: Optional[int] = None
    item_name: Optional[str] = None
    route: Optional[str] = None
    symptom_id: Optional[int] = None
    symptom_name: Optional[str] = None
    severity: Optional[int] = None

@app.get("/health")
def healthcheck():
    return {
        "status": "on",
        }

@app.get("/version")
def version():
    return {
        "version": app.version,
        }

@app.post("/events", response_model = EventOut)
def create_event(payload: EventIn):
        # string validation for timestamp
        if not payload.timestamp.strip():
             raise HTTPException(status_code = 400, detail = "timestamp is required")
        
        # branch by event type and validate input format
        if payload.event_type == "exposure":
             if payload.item_id is None or payload.route_id is None or not payload.route.strip():
                raise HTTPException(status_code = 400, detail = "exposure event requires item and route")
             
             insert_sql = """
                    INSERT INTO exposure_events (user_id, item_id, timestamp, route) 
                    VALUES (?, ?, ?, ?)
                    """
             insert_params = (payload.user_id, payload.item_id, payload.timestamp, payload.route,)
        
        else:
             if payload.symptom_id is None:
                  raise HTTPException(status_code = 400, detail = "symptom event requires health symptom title")
             insert_sql = """
                    INSERT INTO symptom_events (user_id, symptom_id, timestamp, severity) 
                    VALUES (?, ?, ?, ?)
                    """
             insert_params = (payload.user_id, payload.item_id, payload.timestamp, payload.severity,)
            
        # insert into the appropriate table - determined by exposure type
        conn = get_connection()
        try:
            cursor = conn.execute(insert_sql, insert_params)
            conn.commit()
            created_id = cursor.lastrowid
        except sqlite3.IntegrityError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        conn.close()

        return {
            "id": created_id,
            "event_type": payload.event_type,
            "user_id": payload.user_id,
            "timestamp": payload.timestamp,
            "item_id": payload.item_id,
            "route": payload.route,
            "symptom_id": payload.symptom_id,
            "severity": payload.severity,
    }

@app.get("/events", response_model = list[TimelineEvent])
def get_events(user_id: int):
     return list_events(user_id)
