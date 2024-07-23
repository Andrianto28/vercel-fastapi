from fastapi import FastAPI

app = FastAPI()

@app.get("/")
async def heatlt_check():
    return "the health is good"

@app.get("/ping/")
def ping():
    return {"status": "oke"}