import logging
import sys
import os
import json
import time
from argparse import ArgumentParser
from datetime import datetime
from dateutil.parser import parse

from torch.utils.data import DataLoader

from data_collection.historical_data import get_historical_data
from data_preprocessing.asset_preprocessor import AssetPreprocessor
from data_preprocessing.dataset import StocksDatasetInMem


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


def main(train_config: dict):
    start_time = time.time()

    experiment_name = train_config["experiment_name"] + "_" + datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    local_storage_dir = train_config['storage_path'] + os.sep + experiment_name + os.sep
    os.makedirs(local_storage_dir, exist_ok=False)

    set_up_logging(local_storage_dir + "log.log")
    logger.info(f"Starting training with config:\n{train_config}")

    with open(local_storage_dir + "config.json", "w") as fp:
        json.dump(train_config, fp)
    logger.info(f"Storing all train files to: {local_storage_dir}")

    # load all data for tickers into file cache (this is only neccessary because of yfinance 2K requests rate limit per hour)
    # loading into file cache means 1 req per ticker instead of 3 (train, val, test)
    # preprocessor = AssetPreprocessor(candle_size=train_config["candle_size"])
    # adjusted_start_date = preprocessor.adjust_start_date(
    #     parse(train_config["train_start_date"]), train_config["num_candles_to_stack"]
    # ).strftime("%Y-%m-%d")
    # for ticker in train_config["tickers"]:
    #     get_historical_data(symbol=ticker, start_date=adjusted_start_date, end_date=train_config["test_end_date"],
    #                         candle_size=train_config["candle_size"])
    
    # load preprocessed dataset
    train_config["tickers"] = train_config["tickers"][:3] # REMOVE ME
    train_dataset = StocksDatasetInMem(
        tickers=train_config["tickers"], start_date=train_config["train_start_date"], 
        end_date=train_config["val_start_date"], tp=train_config["tp"], tsl=train_config["tsl"],
        num_candles_to_stack=train_config["num_candles_to_stack"], candle_size=train_config["candle_size"]
    )

    # save means and stds, these will be required at inference time
    means, stds = train_dataset.means, train_dataset.stds
    with open(local_storage_dir + "normalisation_info.json", "w") as fp:
        json.dump({"means": means, "stds": stds}, fp)

    train_dataloader = DataLoader(train_dataset, batch_size=train_config["batch_size"], shuffle=True)

    


def set_up_logging(log_filename: str):
    logger = logging.getLogger()  # set-up logging on root-logger
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - {%(filename)s:%(lineno)d} - %(levelname)s - %(message)s')

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    console_handler.setLevel(logging.INFO)
    logger.addHandler(console_handler)

    file_handler = logging.FileHandler(log_filename)
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.INFO)
    logger.addHandler(file_handler)


if __name__ == "__main__":
    # if running using ssh, can use 'nohup python -m train.train_supervised &'
    # to keep the process running even after exiting the ssh session.    

    parser = ArgumentParser()
    parser.add_argument("-c", "--config", default="train/configs/supervised_base_config.json", 
                        help="train config json file")
    
    args = parser.parse_args()

    with open(args.config) as json_file:
        train_config = json.load(json_file)
    
    main(train_config)
