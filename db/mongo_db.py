import logging

from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi

logger = logging.getLogger(__name__)

uri = "mongodb+srv://fadyben98:Jc1Ho8bbyxGUlNc6@sm-prediction.dad5sjr.mongodb.net/?retryWrites=true&w=majority"
client = MongoClient(uri, server_api=ServerApi('1'))
client.admin.command('ping')
logger.info("Successfully connected to MongoDB!")
