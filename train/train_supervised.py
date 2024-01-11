import logging
import sys
import os
import json
import time
from typing import List, Tuple
from argparse import ArgumentParser
from datetime import datetime

import numpy as np
import pandas as pd
from torch.utils.data import IterableDataset

from data_collection.historical_data import get_historical_data, get_all_sp500_tickers, get_vix_daily_data
from data_preprocessing.asset_preprocessor import AssetPreprocessor
from data_preprocessing.vix_preprocessor import VixPreprocessor
from data_preprocessing.labelling import binary_label_tp_tsl

"""
TODO: classification:
        classify:
            - labels {0 = sell, 1 = buy}:
                - 1 if TP is hit before trailing SL
                - 0 otherwise
        use stacking of T candles, so X shape is (N, T, D) and labels have shape (N, 1)
        use daily candles going back to 2014, with standardised TI's + candle_stick patterns 
        
        Train using encoder only transformer using data from 100+ stocks using data going.

        ROC Curve to find threshold which maximizes TPR - FPR
        
        
        example strategy:
            - if p > j_threshold:
                - buy with 20% of original cash amount allocated for this stock
                  with TP and (trailing) SL used during labelling
            - otherwise:
                - sell all shares
"""
logger = logging.getLogger(__name__)

TICKERS = get_all_sp500_tickers() + [
    # some extra symbols outside the S&P 500
    "GOLD", "AEM", "WPM", "FNV", "GFI", "RGLD",
    "GLD", "AU", "KGC", "PAAS", "AGI", "OR", "WAF",
    "EQX", "CEY", "OGC", "CG", "AIG", "FSM", "ORA",

    "NHC", "PLS", "YAL", "ALK", "TER", "SJT", "PDN", "NRT",
    "EFR", "PLL", "NXE", "U-UN.TO"
]


def load_dataset(tickers: List[str], start_date: str, end_date: str, 
                 tp: float, tsl: float, candles_to_stack: int, 
                 local_storage_dir: str, candle_size="1d") -> IterableDataset:
    """ load preprocessed dataset. Does not perform windowing to transform
    X shape from (N, D) to (N, T, D). This should be done at train time
    whenever a batch is loaded to save memory.

    Args:
        tickers (List[str]): ticker symbols to be included in the dataset
        start_date (str): start date for data in yyyy-mm-dd format
        end_date (str): end date for data in yyyy-mm-dd format (exclusive)
        tp (float): take profit value used for labelling (e.g. 0.05 for 5% take profit)
        tsl (float): trailing stop loss value used for labelling (e.g. 0.05 for 5% trailing stop loss)
        candles_to_stack (int): number of time-steps for a single data-point
        local_storage_dir (str): path to directory containing training files for this experiment
        candle_size (str): frequency of candles. "1d" or "1h"

    Returns:
        IterableDataset: (X, y). X has shape (N, D), 
            y has shape (N, ) and contains binary labels.
    """
    all_dfs = []
    preprocessor = AssetPreprocessor(candle_size=candle_size)
    
    for ticker in tickers:
        df = get_historical_data(symbol=ticker, start_date=start_date, end_date=end_date, candle_size=candle_size)
        df = preprocessor.preprocess_ochl_df(df)

        df["labels"] = binary_label_tp_tsl(df, tp, tsl)
        logger.info(f"labels for ticker '{ticker}' has {(df['labels'].isna().sum() / len(df)) * 100}% NaNs, these will be removed.")
        df = df.dropna().reset_index(drop=True)
        
        all_dfs.append(df)
    
    data_df = pd.concat(all_dfs, ignore_index=True)

    asset_means = data_df.mean(numeric_only=True)
    asset_stds  = data_df.std(numeric_only=True)
    preprocessor.normalise_df(data_df, asset_means, asset_stds)

    # add VIX and VVIX data as broad market sentiment indicators
    vix_preprocessor = VixPreprocessor()
    vix_df = get_vix_daily_data("VIX")
    vix_df = vix_preprocessor.preprocess_ochl_df(vix_df)
    vix_df = vix_df.rename(columns={col : f"vix_{col}" for col in vix_df.columns})
    
    vix_means = vix_df.mean(numeric_only=True)
    vix_stds  = vix_df.std(numeric_only=True)
    vix_preprocessor.normalise_df(vix_df, means=vix_means, stds=vix_stds)

    vvix_df = get_vix_daily_data("VVIX")
    vvix_df = vix_preprocessor.preprocess_ochl_df(vvix_df)
    vvix_df = vvix_df.rename(columns={col : f"vvix_{col}" for col in vvix_df.columns})
    
    vvix_means = vvix_df.mean(numeric_only=True)
    vvix_stds  = vvix_df.std(numeric_only=True)
    vix_preprocessor.normalise_df(vvix_df, means=vvix_means, stds=vvix_stds)

    data_df = data_df.merge(vix_df, on="t").merge(vvix_df, on="t")
    

    return None, None


def main(train_config: dict):
    start_time = time.time()
    logger.info(f"Starting training with config:\n{train_config}")

    experiment_name = train_config["experiment_name"] + "_" + datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    local_storage_dir = train_config['storage_path'] + os.sep + experiment_name + os.sep
    os.makedirs(local_storage_dir, exist_ok=False)

    with open(local_storage_dir + "config.json", "w") as fp:
        json.dump(train_config, fp)
    logger.info(f"Storing all train files to: {local_storage_dir}")

    load_dataset(tickers=train_config["tickers"], start_date=train_config["train_start_date"], 
                 end_date=train_config["test_end_date"], tp=train_config["tp"], tsl=train_config["tsl"], 
                 num_candles_to_stack=train_config["num_candles_to_stack"], local_storage_dir=local_storage_dir,
                 candle_size=train_config["candle_size"])


def set_up_logging():
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - {%(filename)s:%(lineno)d} - %(levelname)s - %(message)s')

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    console_handler.setLevel(logging.INFO)
    logger.addHandler(console_handler)


if __name__ == "__main__":
    # if running using ssh, can use 'nohup python -m train.train_supervised &'
    # to keep the process running even after exiting the ssh session.
    set_up_logging()

    parser = ArgumentParser()
    parser.add_argument("-c", "--config", default="train/configs/supervised_base_config.json", 
                        help="train config json file")
    
    args = parser.parse_args()

    with open(args.config) as json_file:
        train_config = json.load(json_file)
    
    main(train_config)
