from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

import random

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins="*",
    allow_credentials=True,
    allow_methods=["GET"],
    allow_headers=["*"],
)


@app.get("/probability/{ticker}")
def get_probability(ticker: str):
    return {"probability": random.randint(0, 101), "ticker": ticker}


app.mount("/", StaticFiles(directory="static",html = True), name="static")
