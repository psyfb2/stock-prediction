import logging
import sys
import os
import json
import time
from argparse import ArgumentParser
from datetime import datetime
from copy import deepcopy
from dateutil.parser import parse

import torch
from torch.utils.data import DataLoader
from torch import nn
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter
from torchinfo import summary

from data_collection.historical_data import get_historical_data
from data_preprocessing.asset_preprocessor import AssetPreprocessor
from data_preprocessing.dataset import StocksDatasetInMem
from models.classification_transformer import ClassificationTransformer
from utils.get_device import get_device
from utils.scheduled_optim import ScheduledOptim
from utils.early_stopping import EarlyStopper


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

    model_cfg = train_config["model_config"]

    # load all data for tickers into file cache (this is only neccessary because of yfinance 2K requests rate limit per hour)
    # loading into file cache means 1 req per ticker instead of 3 (train, val, test)
    preprocessor = AssetPreprocessor(candle_size=train_config["candle_size"])
    adjusted_start_date = preprocessor.adjust_start_date(
        parse(train_config["train_start_date"]), train_config["num_candles_to_stack"]
    ).strftime("%Y-%m-%d")
    for ticker in train_config["tickers"]:
        get_historical_data(symbol=ticker, start_date=adjusted_start_date, end_date=train_config["test_end_date"],
                            candle_size=train_config["candle_size"])
    
    # load preprocessed datasets (train, val, test)
    train_dataset = StocksDatasetInMem(
        tickers=train_config["tickers"], start_date=train_config["train_start_date"], 
        end_date=train_config["val_start_date"], tp=train_config["tp"], tsl=train_config["tsl"],
        num_candles_to_stack=train_config["num_candles_to_stack"], candle_size=train_config["candle_size"]
    )

    # save means and stds, these will be required at inference time
    means, stds = train_dataset.means, train_dataset.stds
    with open(local_storage_dir + "normalisation_info.json", "w") as fp:
        json.dump({"means": means, "stds": stds}, fp)

    val_dataset = StocksDatasetInMem(
        tickers=train_config["tickers"], start_date=train_config["val_start_date"], 
        end_date=train_config["test_start_date"], tp=train_config["tp"], tsl=train_config["tsl"],
        num_candles_to_stack=train_config["num_candles_to_stack"],
        means=means, stds=stds, candle_size=train_config["candle_size"]
    )

    # test_dataset = StocksDatasetInMem(
    #     tickers=train_config["tickers"], start_date=train_config["test_start_date"], 
    #     end_date=train_config["test_end_date"], tp=train_config["tp"], tsl=train_config["tsl"],
    #     num_candles_to_stack=train_config["num_candles_to_stack"],
    #     means=means, stds=stds, candle_size=train_config["candle_size"]
    # )

    train_dataloader = DataLoader(train_dataset, batch_size=model_cfg["batch_size"], shuffle=True)
    val_dataloader   = DataLoader(val_dataset,   batch_size=model_cfg["batch_size"])
    # test_dataloader  = DataLoader(test_dataset,  batch_size=model_cfg["batch_size"])

    # initialise model
    device = get_device()
    in_features = train_dataset.features.shape[-1]
    classifier = ClassificationTransformer(
        seq_len=train_config["num_candles_to_stack"], in_features=in_features,
        num_classes=2, d_model=model_cfg["d_model"], nhead=model_cfg["nhead"], num_encoder_layers=model_cfg["num_encoder_layers"],
        dim_feedforward=model_cfg["dim_feedforward"], dropout=model_cfg["dropout"], layer_norm_eps=model_cfg["layer_norm_eps"],
        norm_first=model_cfg["norm_first"]
    ).to(device)
    summary(classifier, input_size=(model_cfg["batch_size"], train_config["num_candles_to_stack"], in_features), device=device)

    # initialise loss function and optimizer
    loss_fn = nn.CrossEntropyLoss(label_smoothing=model_cfg["label_smoothing"])

    optimizer = ScheduledOptim(
        optimizer=Adam(classifier.parameters(), betas=(0.9, 0.98), eps=1e-09),
        lr_mul=model_cfg["lr_mul"],
        d_model=model_cfg["d_model"],
        n_warmup_steps=model_cfg["n_warmup_steps"]
    )

    # train model using earling stopping
    early_stopper = EarlyStopper(patience=model_cfg["patience"])
    best_model_params = None

    train_batches_per_epoch = len(train_dataset) / model_cfg["batch_size"]
    val_batches_per_epoch = len(val_dataset) / model_cfg["batch_size"]
    writer = SummaryWriter(log_dir=local_storage_dir)
    logger.info(f"Starting training. View TensorBoard logs at dir: {local_storage_dir}")

    for epoch in range(model_cfg["max_epochs"]):
        train_loss = 0
        train_correct = 0
        classifier.train()
        for batch, (X, y) in enumerate(train_dataloader):
            X, y = X.to(device), y.to(device)
            print(f"Shape of X: {X.shape} {X.dtype}")
            print(f"Shape of y: {y.shape} {y.dtype}")

            pred = classifier(X)
            loss = loss_fn(pred, y)

            train_loss += loss.item()
            train_correct += (y == pred.max(dim=1).indices).sum().item()

            loss.backward()
            optimizer.step_and_update_lr()
            optimizer.zero_grad()

            if batch % 10:
                loss, current = loss.item(), (batch + 1) * len(X)
                print(f"loss: {loss:>7f}  [{current:>5d}/{len(train_dataset):>5d}]")

        train_loss = train_loss / train_batches_per_epoch
        train_acc  = train_correct / len(train_dataset)
        writer.add_scalar("Train Loss", train_loss , epoch)
        writer.add_scalar("Train Acc", train_acc, epoch)
        
        # calculate validation loss
        val_loss = 0
        val_correct = 0
        classifier.eval()
        with torch.no_grad():
            for batch, (X, y) in enumerate(val_dataloader):
                X, y = X.to(device), y.to(device)

                pred = classifier(X)
                loss = loss_fn(pred, y)

                val_loss += loss.item()
                val_correct += (y == pred.max(dim=1).indices).sum().item()

        val_loss = val_loss / val_batches_per_epoch
        val_acc  = val_correct / len(val_dataset)
        writer.add_scalar("Val Loss", val_loss, epoch)
        writer.add_scalar("Val Acc", val_acc , epoch)

        logger.info(f"epoch #{epoch}: train_loss={train_loss}, train_acc={train_acc}, val_loss={val_loss}, val_acc={val_acc}")

        if early_stopper.early_stop(val_loss):
            logger.info(f"Stopping training after {epoch} epochs, due to early stopping.")
            break

        if early_stopper.counter == 0:
            best_model_params = deepcopy(classifier.state_dict())

    writer.flush()
    logger.info(f"Saving model with lowest val loss to {local_storage_dir}model.pth")
    torch.save(best_model_params, local_storage_dir + "model.pth")

    # TODO: find best thresholds using ROC on val set

    # TODO: create function which finds acc and F1 score on test set and plot ROC on test set



    



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
