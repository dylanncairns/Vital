from fastapi import FastAPI
# from api.db import get_connection

app = FastAPI(
    title = "Vital API",
    version = "0.1.0",
    )

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
