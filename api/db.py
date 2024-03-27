import logging
import os
from datetime import datetime, date, timedelta, timezone
from typing import Optional

import pandas_market_calendars as mcal
from pymongo import MongoClient
from pymongo.server_api import ServerApi


logger = logging.getLogger(__name__)

client = MongoClient(os.environ["MONGO_DB_URI"], server_api=ServerApi('1'))
client.admin.command('ping')
logger.info("Successfully connected to MongoDB!")

api_db = client.get_database("sm_api_db")


def insert_probability_request(ticker: str, exchange_name: str, probability: float):
    """ Insert request info from "/probability/{ticker}" endpoint into DB.
    Uses TTL index so that the inserted document will be automatically removed
    15 minutes after the next market close date. This is to automatically
    clear up the DB and so that it can be used as a cache.

    Also insert probability into probability_predictions collection, this
    can be used to monitor model performance with real life inferences.

    Args:
        ticker (str): ticker symbol (e.g. 'AAPL')
        exchange_name (str): exchange which ticker is listen on (e.g. 'NASDAQ')
        probability (float): bullish probability for ticker
    """
    logger.info(f"Caching result for ticker '{ticker}' with exchange '{exchange_name}' and probability {probability}")
    
    try:
        exchange = mcal.get_calendar(exchange_name)
    except RuntimeError as ex:
        logger.exception(f"Unknown exchange '{exchange_name}'. Will skip saving to cache.")
        return
    
    tz         = exchange.tz
    close_time = exchange.close_time  # datetime.time object with tz of exchange
    close_time = datetime.combine(date.today(), close_time) + timedelta(minutes=15)
    close_time = close_time.time()
    now        = datetime.now(tz)

    # expires after next close (close price will be available so model prediction can be different)
    expiresAt = None
    if now.time() < close_time:
        expiresAt = datetime.combine(now, close_time)
    else:
        expiresAt = datetime.combine(now.date() + timedelta(days=1), close_time)
    
    # convert expiresAt to utc
    expiresAt = tz.normalize(tz.localize(expiresAt)).astimezone(timezone.utc)

    document = {
        "createdAt": datetime.now(timezone.utc),
        "expiresAt": expiresAt,
        "ticker": ticker,
        "probability": probability
    }
    logger.info(f"Inserting document into 'probability_cache' collection: {document}")
    api_db["probability_cache"].insert_one(document)

    document.pop("expiresAt")
    api_db["probability_predictions"].insert_one(document)


def get_probability_request(ticker: str) -> Optional[dict]:
    """ Get request info for endpoint "/probability/{ticker}" 
    which was inserted into DB. Use this as a retrieve from cache function.

    Args:
        ticker (str): ticker symbol (e.g. 'AAPL')

    Returns:
        Optional[dict]: document with keys "createdAt", "expiresAt", "ticker", "probability".
            Will return None if ticker is not in DB (it can expire due to TTL).
    """
    doc = api_db["probability_cache"].find_one({"ticker": ticker})
    logger.info(f"Got cached result for ticker '{ticker}':\n{doc}")
    return doc
