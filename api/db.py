import logging
import os
from datetime import datetime

from pymongo import MongoClient
from pymongo.server_api import ServerApi

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
    # assuming using ma
    api_db["probability_cache"].insert_one(
        {
            "createdAt": datetime.utcnow(),
            "expiresAt": None,
            "ticker": ticker,
            "probability": probability
        }
    )