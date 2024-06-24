import os
import json
from typing import List, Callable, Any, Dict
from dotenv import load_dotenv
from multiprocessing import Pool
from datetime import datetime, timezone, timedelta, date

import pandas as pd
import pandas_market_calendars as mcal

from data_preprocessing import labelling
from api.db import get_probability_predictions
from data_collection.historical_data import get_historical_data, get_most_recent_trading_day


load_dotenv()


def inference_classification_report(start_date: str, 
                                    end_date: str, 
                                    label_func: Callable[[pd.DataFrame, Any], pd.Series],
                                    label_kwargs: Dict[str, Any],
                                    thresholds: List[float] = [0.25, 0.4, 0.5, 0.6, 0.75],
                                    before_close: float = 10) -> str:
    """ Get classification report on model predictions for a given time period.

    Args:
        start_date (str): date string in the format "YYYY-MM-DD". Will only consider predictions
            made after this date. Inclusive.
        end_date (str): date stirng in the format "YYYY-MM-DD". Will only consider predictions
            made before this date. Exclusive.
        label_func (Callable[[pd.Dataframe, Any], pd.Series]): function to calculate labels
        label_kwargs (Dict[str, Any]): keyword arguments to pass to label_func
        thresholds (float, optional): model outputs probability.
            this is the threshold to use for classifications. Defaults to 0.5.
        before_close (float, optional): Num minutes before close for prediction to be classed as
            happening on that day.
            For example, if a prediction happens 20 mins before the close, and before_close is 10,
            the prediction will be relative to the previous close. If the prediction happens anywhere
            between 10 mins before the close and 10 mins before the next close, the prediction will be relative to this close.
            Defaults to 10.
    """ 
    docs = get_probability_predictions(start_date=start_date, end_date=end_date)
    unique_tickers = list(set(doc["ticker"] for doc in docs))

    with Pool() as p:
        # TODO: exchange data needs to be passed to get_historical_data (by adding it to probability predictions, change interface to allow user to enter exchange)
        # if no exchange is passed assume nyse or nasdaq
        results = p.starmap(get_historical_data, [(ticker, start_date, datetime.now().strftime("%Y-%m-%d")) for ticker in unique_tickers])
        tickers_to_data = dict(zip(unique_tickers, results))

        all_label_series = p.starmap(label_func_with_kwargs, [(label_func, tickers_to_data[ticker], label_kwargs) for ticker in unique_tickers])

        for ticker, label_series in zip(unique_tickers, all_label_series):
            tickers_to_data[ticker]["label"] = label_series
    
    for ticker, df in tickers_to_data.items():
        tickers_to_data[ticker] = df.set_index("t")

    probs  = []
    labels = []

    for doc in docs:
        exchange = mcal.get_calendar("NYSE")
        adjusted_close_time = (
            datetime.combine(date(1, 1, 1), exchange.close_time) - timedelta(minutes=before_close)
        ).time().replace(tzinfo=exchange.tz)

        created_at = doc["createdAt"].replace(tzinfo=timezone.utc).astimezone(exchange.tz)

        most_recent_trading_day = get_most_recent_trading_day(created_at, "NYSE")
        prediction_day = None

        # if the prediction was made on a non-trading day or it was made after adjusted_close_time, 
        # the prediction will be relative to the most recent trading day
        if most_recent_trading_day.date() != created_at.date() or created_at.time() >= adjusted_close_time:
            prediction_day = most_recent_trading_day
        else:
            prediction_day = get_most_recent_trading_day(created_at - timedelta(days=1), "NYSE")
        
        probs.append(doc["probability"])
        labels.append(tickers_to_data[doc["ticker"]]["label"].loc[prediction_day])


def label_func_with_kwargs(label_func: Callable[[pd.DataFrame, Any], pd.Series], 
                           df: pd.DataFrame, 
                           kwargs: Dict[str, Any]) -> pd.Series:
    return label_func(df, **kwargs)


if __name__ == "__main__":
    MODEL_FILES_DIR = os.environ["MODEL_FILES_DIR"] + "/"

    with open(MODEL_FILES_DIR +  "config.json") as json_file:
        train_config = json.load(json_file)

    label_func  = getattr(labelling, train_config["label_config"]["label_function"])
    label_kwargs = train_config["label_config"]["kwargs"]

    inference_classification_report("2024-01-01", "2025-01-31", label_func, label_kwargs)

