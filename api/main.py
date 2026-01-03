from fastapi import FastAPI
app = FastAPI()

@app.get("/health")
def healthcheck():
    return {"hows it going boss"}

@app.get("/products")
def productlist():
    productsDB = [
    {"id": 1, "name": "Face Wash"},
    {"id": 2, "name": "Moisturizer"},
    {"id": 3, "name": "Spot Treatment"}
    ]
    return productsDB
