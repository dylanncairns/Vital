from fastapi import FastAPI
import sqlite3

app = FastAPI()

@app.get("/health")
def healthcheck():
    return {"hows it going boss"}

@app.get("/products")
def product_display():
    conn = sqlite3.connect("data/vital.db")
    cursor = conn.cursor()

    cursor.execute("SELECT * FROM products")
    rows = cursor.fetchall()
    product_list = []
    for row in rows:
        product_list.append({"id": row[0], "product_name": row[1]})
        
    conn.close()
    return product_list
