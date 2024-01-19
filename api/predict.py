import os
import json
import logging
from datetime import timedelta

import numpy as np
import torch

from models.classification_transformer import ClassificationTransformer
from data_preprocessing.asset_preprocessor import AssetPreprocessor
from data_preprocessing.vix_preprocessor import VixPreprocessor
from data_collection.historical_data import get_last_full_trading_day, get_vix_daily_data
from data_preprocessing.dataset import StocksDatasetInMem
from models.mlp import MLP
from utils.get_device import get_device


logger = logging.getLogger(__name__)

# model file dir should contain config.json, normalisation_info.json and model.pth files
MODEL_FILES_DIR = os.environ["MODEL_FILES_DIR"]

with open(MODEL_FILES_DIR + os.sep + "config.json") as json_file:
    train_config = json.load(json_file)
model_cfg = train_config["model_config"]

with open(MODEL_FILES_DIR + os.sep + "normalisation_info.json") as json_file:
    normalisation_info = json.load(json_file)


# load model
device = get_device()
if model_cfg["model_type"] == "classification_transformer":
    classifier = ClassificationTransformer(
        seq_len=train_config["num_candles_to_stack"], in_features=normalisation_info["in_features"],
        num_classes=2, d_model=model_cfg["d_model"], nhead=model_cfg["nhead"], num_encoder_layers=model_cfg["num_encoder_layers"],
        dim_feedforward=model_cfg["dim_feedforward"], dropout=model_cfg["dropout"], layer_norm_eps=model_cfg["layer_norm_eps"],
        norm_first=model_cfg["norm_first"]
    ).to(device)

elif model_cfg["model_type"] == "mlp":
    classifier = MLP(
        dropout=model_cfg["dropout"], seq_len=train_config["num_candles_to_stack"], 
        in_features=normalisation_info["in_features"], num_classes=2, 
        hidden_layers=model_cfg["hidden_layers"]
    ).to(device)
    
else:
    raise ValueError(f"model_type '{model_cfg['model_type']}' is not recognised.")
classifier.load_state_dict(torch.load(MODEL_FILES_DIR + os.sep + "model.pth", map_location=device))
classifier.eval()

# initialise preprocessors
asset_preprocessor = AssetPreprocessor(
    features_to_use=train_config["features_to_use"], 
    candle_size=train_config["candle_size"]
)
vix_preprocessor = VixPreprocessor(
    features_to_use=train_config["vix_features_to_use"]
)


def predict(ticker: str, exchange_name: str) -> float:
    """ Predict bullish probability using model.

    Args:
        ticker (str): ticker for which to predict bullish probability (e.g. 'AAPL').
        exchange_name (str): name of exchange for ticker (e.g. 'NASDAQ')

    Returns:
        float: bullish probability.
    """
    try:
        start_date = get_last_full_trading_day(exchange_name)
    except RuntimeError as ex:
        logger.exception(f"Exchange '{exchange_name}' is not recognised. Will assume NASDAQ schedule.")
        start_date = get_last_full_trading_day("NASDAQ")
        
    # load preprocessed data for ticker
    df = StocksDatasetInMem.preprocess_ticker(
        ticker=ticker, 
        exchange=exchange_name, 
        start_date=start_date.strftime("%Y-%m-%d"),
        end_date=(start_date + timedelta(days=1)).strftime("%Y-%m-%d"), 
        preprocessor=asset_preprocessor,
        num_candles_to_stack=train_config["num_candles_to_stack"],
        candle_size=train_config["candle_size"],
        raise_invalid_data_exception=False,
        tp=None,
        tsl=None
    )

    # normalise df
    asset_preprocessor.normalise_df(
        df=df, 
        means=normalisation_info["means"]["asset"], 
        stds=normalisation_info["stds"]["asset"]
    )

    # merge VIX data
    if train_config["vix_features_to_use"]:
        df = StocksDatasetInMem.merge_vix_data(
            vix_preprocessor=vix_preprocessor, 
            data_df=df,
            calc_means=False,
            means=normalisation_info["means"],
            stds=normalisation_info["stds"],
            num_candles_to_stack=train_config["num_candles_to_stack"]
        )

        # sometimes VIX data lags behind by one day (CBOE update VIX one day late)
        # if this is case vix features in df on last candle will be None, fix this by 
        # replace those NaNs with last value
        num_nans = df.isna().sum().sum()
        if num_nans != 0:
            logger.warning(f"Preprocessed df has NaNs! This is most likely because the last day of VIX data hasn't been uploaded yet. "
                           f"NaNs per col:\n{dict(df.isna().sum())}\ndf:\n{df}")
            df = df.fillna(method="ffill")
            logger.warning(f"Filled df:\n{df}")
        
    num_nans = df.isna().sum().sum()
    assert num_nans == 0, f"Expected Number of NaNs in df to be 0, but was {num_nans}. NaNs per col:\n{dict(df.isna().sum())}"

    # X has shape (1, num_candles_to_stack, D)
    X = df.drop(columns=list("toclhv")).to_numpy(dtype=np.float32)
    X = X[np.newaxis, -train_config["num_candles_to_stack"]:, :]
    X = torch.from_numpy(X)

    # use model to predict bullish probability
    with torch.no_grad():
        X = X.to(device) # (1, num_candles_to_stack, D)

        logits = classifier(X)  # (1, 2)

        probs  = torch.nn.functional.softmax(logits, dim=1) # (1, 2)
        probs  = probs[:, 1]  # (1, )
        probs  = probs.item()

    return probs
