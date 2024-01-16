import logging
import os
from datetime import datetime, date, timedelta, timezone

import pandas_market_calendars as mcal
from pymongo import MongoClient
from pymongo.server_api import ServerApi

from data_collection.historical_data import get_exchange


logger = logging.getLogger(__name__)

client = MongoClient(os.environ["MONGO_DB_URI"], server_api=ServerApi('1'))
client.admin.command('ping')
logger.info("Successfully connected to MongoDB!")

api_db = client.get_database("sm_api_db")

def insert_probability_request(ticker: str, probability: float):
    """ Insert request info from "/probability/{ticker}" endpoint into DB.
    Uses TTL index so that the inserted document will be automatically removed
    15 minutes after the next market close date. This is to automatically
    clear up the DB and so that it can be used as a cache.

    Args:
        ticker (str): _description_
        probability (float): _description_
    """
    exchange = get_exchange(ticker)
    logger.info(f"Caching result for ticker '{ticker}' and probability {probability}")

    try:
        exchange = mcal.get_calendar(exchange)
        logger.info(f"Using exchange '{exchange}' for ticker '{ticker}'")
    except RuntimeError as ex:
        logger.exception(f"Unknown exchange '{exchange}'.")
        return
    
    close_time = exchange.close_time  # datetime.time object with tz of exchange
    close_time = datetime.combine(date.today(), close_time) + timedelta(minutes=15)
    close_time = close_time.time()
    now        = datetime.now(exchange.tz)

    expiresAt = None
    if now.time() < close_time:
        expiresAt = datetime.combine(now, close_time).replace(tzinfo=exchange.tz)
        
        




    
    

    api_db["probability_cache"].insert_one(
        {
            "createdAt": datetime.utcnow(),
            "expiresAt": None,
            "ticker": ticker,
            "probability": probability
        }
    )