import re

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

from api.db import insert_probability_request, get_probability_request
from data_collection.historical_data import get_exchange

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
    ticker = ticker.upper()
    ticker = re.sub(r"[A-Z\.]", "", ticker)

    cache_result = get_probability_request(ticker)
    if cache_result:
        return {"probability": cache_result["probability"], "ticker": ticker}
    
    exchange_name = get_exchange(ticker)
    

    
    res = {"probability": random.randint(0, 101), "ticker": ticker}


app.mount("/", StaticFiles(directory="static",html = True), name="static")
