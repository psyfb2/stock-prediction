import sys
import os
import re
import logging
import logging.handlers
from requests.exceptions import HTTPError
from functools import wraps

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

from api.db import insert_probability_request, get_probability_request
from api.predict import predict
from data_collection.historical_data import get_exchange


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins="*",
    allow_credentials=True,
    allow_methods=["GET"],
    allow_headers=["*"],
)

def set_up_logging(log_filename: str):
    # set-up logging on root-logger
    logger = logging.getLogger()  
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - {%(filename)s:%(lineno)d} - %(levelname)s - %(message)s')

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    console_handler.setLevel(logging.INFO)
    logger.addHandler(console_handler)

    file_handler = logging.handlers.RotatingFileHandler(log_filename, maxBytes=8 * 1024 * 10, backupCount=10)
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.INFO)
    logger.addHandler(file_handler)


set_up_logging("logs" + os.sep + "api.log")
logger = logging.getLogger(__name__)


@app.get("/probability/{ticker}")
def get_probability(ticker: str):
    logging.info(f"GET to /probability/{ticker}")

    ticker = ticker.upper()
    ticker = re.sub(r"[^A-Z\.-]", "", ticker)

    cache_result = get_probability_request(ticker=ticker)

    if cache_result:
        res = {"probability": cache_result["probability"], "ticker": ticker}
    else:
        try:
            exchange_name = get_exchange(ticker)
            probability = predict(ticker=ticker, exchange_name=exchange_name)
        except HTTPError as ex:
            logger.exception(f"Failed predict probability for ticker '{ticker}'")
            raise HTTPException(
                status_code=422, detail=f"Failed to predict probability for '{ticker}'. "
                f"Ensure the ticker is correct, is not delisted and exists on yahoo finance. "
                f"Exact exception: {repr(ex)}"
            )
        except Exception as ex:
            logger.exception(f"Failed predict probability for ticker '{ticker}'")
            raise HTTPException(
                status_code=500, detail=f"Failed to predict probability for '{ticker}'. Exact exception: {repr(ex)}"
            )

        insert_probability_request(ticker=ticker, exchange_name=exchange_name, probability=probability)

        res = {"probability": probability, "ticker": ticker}
    
    logger.info(f"Return result {res} for ticker '{ticker}'")
    return res


app.mount("/", StaticFiles(directory="static",html = True), name="static")
