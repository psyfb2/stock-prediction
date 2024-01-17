import os
import json

import torch

from models.classification_transformer import ClassificationTransformer
from data_preprocessing.asset_preprocessor import AssetPreprocessor
from utils.get_device import get_device


with open(os.environ["TRAIN_CONFIG_PATH"]) as json_file:
    train_config = json.load(json_file)
model_cfg = train_config["model_config"]

with open(os.environ["NORMALISATION_INFO_PATH"]) as json_file:
    normalisation_info = json.load(json_file)

MODEL_PATH = os.environ["CLASSIFIER_MODEL_PATH"]

# load model
device = get_device()
classifier = ClassificationTransformer(
        seq_len=train_config["num_candles_to_stack"], in_features=normalisation_info["in_features"],
        num_classes=2, d_model=model_cfg["d_model"], nhead=model_cfg["nhead"], num_encoder_layers=model_cfg["num_encoder_layers"],
        dim_feedforward=model_cfg["dim_feedforward"], dropout=model_cfg["dropout"], layer_norm_eps=model_cfg["layer_norm_eps"],
        norm_first=model_cfg["norm_first"]
).to(device)
classifier.load_state_dict(torch.load(MODEL_PATH))
classifier.eval()

# initialise preprocessors
asset_preprocessor = AssetPreprocessor(candle_size=train_config["candle_size"])


def predict(ticker: str) -> float:
    """ Predict bullish probability using model.

    Args:
        ticker (str): ticker for which to predict bullish probability.

    Returns:
        float: bullish probability.
    """
    # load data for ticker
    pass