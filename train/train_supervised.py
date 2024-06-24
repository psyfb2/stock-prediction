import logging
import sys
import os
import json
import time
from typing import Optional, Tuple
from argparse import ArgumentParser
from datetime import datetime
from dateutil.parser import parse

import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torch import nn
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter
from torchinfo import summary

from xgboost import XGBClassifier
from sklearn.metrics import roc_curve, classification_report
from sklearn.base import ClassifierMixin
from data_collection.historical_data import get_historical_data
from data_preprocessing.asset_preprocessor import AssetPreprocessor
from data_preprocessing.dataset import StocksDatasetInMem
from models.classification_transformer import ClassificationTransformer
from models.mlp import MLP
from utils.get_device import get_device
from utils.scheduled_optim import ScheduledOptim
from utils.early_stopping import EarlyStopper


logger = logging.getLogger(__name__)

"""
2 new metrics:
    1. avg daily return (profit per candle) = (candle_return_1 + ... + candle_return_n) / n
       This tells you how much profit you should expect to make per candle (close to close).
       Can also calculate std deviation of daily returns and give sharpe ratio (expected daily return - daily risk free rate / std),
       which will tell you how good the returns are relative to the uncertatainty around those returns.
       Obviously want to maximize expected daily return, but have a small std is really good too,
       because it means you are having consistent returns and you can be more sure that your returns
       are because your strategy is good and not just lucky. Btw daily risk free rate for 5.25% 3 month treassury is 0.02

    2. risk adjusted return = profit / exposure
       This tells you how much profit you expect to make if you were long the whole time by utilising multiple assets.
       When trading a single stock, your risk adjusted return should be higher than the B&H profit of that stock.
       The rar should also be higher than S&P-500 return in the same timeframe. So if trading for 252 candles,
       rar should be higher than S&P-500 avg annual return. If trading for 20 candles, rar should be higher
       than S&p500 avg 20-candle return (can estimate this by knowing avg daily S&P500 return and compounding over 20 candles).
    
    btw, a note on diversifying trades. Assume we have X_i ~ {a w.p. p, -b w.p. 1-p}, which represents a TP-SL bet from our model.
    If we just bet 100% of our equity at a time, then profit will be (ignored compounding) S_n = X_1 + ... + X_n.
    E[S_n] = n*E[X], Var[S_n] = n*Var(x)
    The probability of losing money (i.e. S_n < 0) as n increases goes to 0, however the variance increases. This is not great because
    it means there is a lot of uncertainaty, around what the profit will be, say on an annual basis. Instead it would be better
    to bet 1/k of equity on k trades at the same time. So we have Y_i = (X_1 + .... + X_k) / k.
    E[Y_i] = E[X], Var[Y_i] = Var(X) / k
    as k increases, Y_i approaches our expected profit of a trade in probability. This is great as it removes the uncertainty.
    So now if S_n = Y_1 + ... + Y_n , our annual profit will have much less variance and for example, annual profits should be
    relatively consistent. 
    Practically what this means is that you should have as many high confidence trades on as possible at the same time,
    but making sure not to have any unitilised capital 
    (i.e. maximize k with constraint of not having any equity not being used in a trade, meaning there is k high confidence trades always available).
    

TO DO:
    - sep up model drift monitoring for mlp

    - refactor code
    
    - Create XGBoost, MLP models to predict if 
        - c_{t} < c_{t + 1},
        - c_{t} < c_{t + 5},
        - c_{t} < c_{t + 10},
        - price change in 10 candles is more than 5%
        - tp-tsl ([6, 6], [6, 9])
        - tp-tsl only close ([6, 6], [6, 9])
        - tp-sl ([6, 4], [6, 5], [6, 6], [5, 4])
        - tp-sl only close ([6, 4], [6, 5], [6, 6], [5, 4])
        
      use marketcap > 6B
      Make tp-tsl and tp-sl label functions more efficient 
      Use ^VIX to get VIX data, also make data cache more efficient (DONE)
      hyperparemeter tuning on validation set and early stopping
      model selection on validation set (using ppc, ppc calculation can be on a subsample to save time, plot ppc histogram,
        also classification report, plot ROC curve, expectation calculations, just manually validate the best ppc model)
      test set classification report, ROC, expectation calculations, ppc on best model using tuned threshold
      accept if ppc_tuned_thresh > spx_pcc, ppc_tuned_thresh > long_term_ppc, precision_tuned_thresh > random precision, acc_0.5_thresh > most_common_acc + 0.03
      
      A high precision model is preferred (even if it has low recall), since there are hundreds of different stocks to possibly trade.
      The same is true for a high profit factor model. Ultimately, we want to maximize profit, but when considering individual stocks,
      we can always assume there will always be an oppurtinity to be long out of all stocks, therefore, highest profit factor or precision models
      are the best even though they may have a low recall.
      
    - RL model:
        - env:
            - daily candles, observation and action on day close
            - reward is differential sharpe ratio
            - episodes is full length of train data for a single stock
        
        - model:
            - GTrXLNet
            - APPO
            - validation set early stopping (episode reward mean)
    
        - hyperparameter tuning (pick combination which maximizes validation episode reward mean):
            - try small, medium, big GTrXLNet
            - try action spaces:
                - {0: neutral position, 1: long position}
            - change in profit per candle reward (if you have time), 
              change in rar (profit / exposure)
        
        - test single best model for each reward:
            - stats (avg B&H profit, avg profit, avg sharpe ratio, avg profit factor, win rate, profit per candle when long, etc)
            - plot trades for each ticker
            - plot profit per candle when long distribution (histogram)

            - check exposure on validation set to determine correct N for each model.
            
            - accept model if profit per candle when long > S&P-500 profit per candle

    Another thing is you must consider how these RL models will be used. For example:
        1. portfolio of n stocks. Each stock has capital/n weighting (equally weighted). In this case,
           CIPV would be the best reward, since a high precision but low recall model would not be
           able to take advantage of those high confidence bets because of the equal weighting.
        
        2. If you think of a real life trader. They find good trade setups out of hundreds of stocks, then
           bet big on the promising trades. They do not use equal weighting. This is exactly what a high precision
           and low recall model needs, such as CIPV + drawdown penalty. So one could use the following trading strategy:
            - maximum of n stocks which can be traded at the same time
            - just before close at each candle, preprocess data, get output of each model,
              go long on n - num_positions_open with size 1/n on the stocks whose model has long output.

    One thing to keep in mind is that in the backtesting we assume we can know the true close price and volume
    before placing a trade on the close. However in reality this is not the case since we trade e.g. 5 mins before close.
    Calculations for 15 stocks using yfinance querying 5 mins before close and then the day after shows
    an average price deviation of the price 5 mins before close and actual close is 0.07% and average volume increase is 18.35%.
    So in a real life application it's recomended to artificially raise the voume by 18%, the deviation in the close is acceptable.
    Once a broker is decided I would retest this with chosed broker data collection.
"""


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

    # load and preprocess data
    if train_config["load_data_path"] is not None:
        train_dataset = StocksDatasetInMem(
            tickers=None,
            sectors=train_config["sectors"],
            features_to_use=train_config["features_to_use"],
            vix_features_to_use=train_config["vix_features_to_use"],
            start_date=None, 
            end_date=None, 
            label_config=train_config["label_config"],
            num_candles_to_stack=train_config["num_candles_to_stack"],
            candle_size=None, 
            load_path=train_config["load_data_path"] + "train_dataset/",
            recalculate_labels=train_config["recalculate_labels"],
            perform_normalisation=train_config["perform_feature_normalisation"]
        )

        val_dataset = StocksDatasetInMem(
            tickers=None, 
            sectors=train_config["sectors"],
            features_to_use=train_config["features_to_use"],
            vix_features_to_use=train_config["vix_features_to_use"],
            start_date=None, 
            end_date=None, 
            label_config=train_config["label_config"],
            num_candles_to_stack=train_config["num_candles_to_stack"],
            candle_size=None, 
            load_path=train_config["load_data_path"] + "val_dataset/",
            recalculate_labels=train_config["recalculate_labels"],
            perform_normalisation=train_config["perform_feature_normalisation"]
        )

        test_dataset = StocksDatasetInMem(
            tickers=None, 
            sectors=train_config["sectors"],
            features_to_use=train_config["features_to_use"],
            vix_features_to_use=train_config["vix_features_to_use"],
            start_date=None, 
            end_date=None, 
            label_config=train_config["label_config"],
            num_candles_to_stack=train_config["num_candles_to_stack"],
            candle_size=None, 
            load_path=train_config["load_data_path"] + "test_dataset/",
            recalculate_labels=train_config["recalculate_labels"],
            perform_normalisation=train_config["perform_feature_normalisation"]
        )

    else:
        # load all data for tickers into file cache (this is only neccessary 
        # because of yfinance 2K requests rate limit per hour)
        # loading into file cache means 1 req per ticker instead of 3 (train, val, test)
        # preprocessor = AssetPreprocessor(features_to_use=train_config["features_to_use"], 
        #                                  candle_size=train_config["candle_size"])
    
        # adjusted_start_date = preprocessor.adjust_start_date(
        #     parse(train_config["train_start_date"]), train_config["num_candles_to_stack"]
        # ).strftime("%Y-%m-%d")

        # for ticker, exchange in train_config["tickers"]:
        #     get_historical_data(
        #         symbol=ticker, 
        #         start_date=adjusted_start_date, 
        #         end_date=train_config["test_end_date"],
        #         candle_size=train_config["candle_size"], 
        #         exchange=exchange
        #     )

        os.makedirs(local_storage_dir + "train_dataset/")
        os.makedirs(local_storage_dir + "val_dataset/")
        os.makedirs(local_storage_dir + "test_dataset/")

        # load preprocessed datasets (train, val, test)
        logger.info("Loading training data")
        train_dataset = StocksDatasetInMem(
            tickers=train_config["tickers"], 
            sectors=train_config["sectors"],
            features_to_use=train_config["features_to_use"],
            vix_features_to_use=train_config["vix_features_to_use"],
            start_date=train_config["train_start_date"], 
            end_date=train_config["val_start_date"], 
            label_config=train_config["label_config"],
            num_candles_to_stack=train_config["num_candles_to_stack"], 
            candle_size=train_config["candle_size"],
            save_path=local_storage_dir + "train_dataset/",
            perform_normalisation=train_config["perform_feature_normalisation"]
        )

        # save means and stds, these will be required at inference time
        means, stds = train_dataset.means, train_dataset.stds
        with open(local_storage_dir + "normalisation_info.json", "w") as fp:
            json.dump({"means": means, "stds": stds, "in_features": train_dataset.features.shape[-1]}, fp)

        logger.info("Loading validation data")
        val_dataset = StocksDatasetInMem(
            tickers=train_config["tickers"], 
            sectors=train_config["sectors"],
            features_to_use=train_config["features_to_use"],
            vix_features_to_use=train_config["vix_features_to_use"],
            start_date=train_config["val_start_date"], 
            end_date=train_config["test_start_date"],
            label_config=train_config["label_config"],
            num_candles_to_stack=train_config["num_candles_to_stack"],
            means=means, 
            stds=stds, 
            candle_size=train_config["candle_size"],
            save_path=local_storage_dir + "val_dataset/",
            perform_normalisation=train_config["perform_feature_normalisation"]
        )

        logger.info("Loading test data")
        test_dataset = StocksDatasetInMem(
            tickers=train_config["tickers"], 
            sectors=train_config["sectors"],
            features_to_use=train_config["features_to_use"],
            vix_features_to_use=train_config["vix_features_to_use"],
            start_date=train_config["test_start_date"], 
            end_date=train_config["test_end_date"], 
            label_config=train_config["label_config"],
            num_candles_to_stack=train_config["num_candles_to_stack"],
            means=means, 
            stds=stds, 
            candle_size=train_config["candle_size"],
            save_path=local_storage_dir + "test_dataset/",
            perform_normalisation=train_config["perform_feature_normalisation"]
        )
    
    logger.info(f"Train dataset length = {len(train_dataset)}, with label counts: {train_dataset.label_counts}")
    logger.info(f"Val dataset length = {len(val_dataset)}, with label counts: {val_dataset.label_counts}")
    logger.info(f"Test dataset length = {len(test_dataset)}, with label counts: {test_dataset.label_counts}")


    # initialise model
    device = get_device()
    torch_model = True  # if False, will assume it's a model that follows SKlearn interface.
    logger.info(f"Using device '{device}'")

    in_features = train_dataset.features.shape[-1]
    if model_cfg["model_type"] == "classification_transformer":
        classifier = ClassificationTransformer(
            seq_len=train_config["num_candles_to_stack"], in_features=in_features,
            num_classes=2, d_model=model_cfg["d_model"], nhead=model_cfg["nhead"], num_encoder_layers=model_cfg["num_encoder_layers"],
            dim_feedforward=model_cfg["dim_feedforward"], dropout=model_cfg["dropout"], layer_norm_eps=model_cfg["layer_norm_eps"],
            norm_first=model_cfg["norm_first"]
        ).to(device)

    elif model_cfg["model_type"] == "mlp":
        classifier = MLP(dropout=model_cfg["dropout"], seq_len=train_config["num_candles_to_stack"], in_features=in_features, 
                         num_classes=2, hidden_layers=model_cfg["hidden_layers"]).to(device)
    
    elif model_cfg["model_type"] == "xgboost":
        torch_model = False
        classifier = XGBClassifier(**{k: v for k, v in model_cfg.items() if k != "model_type"})
        
    else:
        raise ValueError(f"model_type '{model_cfg['model_type']}' is not recognised.")


    # train model and eval model
    if torch_model:
        summary(classifier, input_size=(model_cfg["batch_size"], train_config["num_candles_to_stack"], in_features), device=device)

        train_dataloader = DataLoader(train_dataset, batch_size=model_cfg["batch_size"], shuffle=True)
        val_dataloader   = DataLoader(val_dataset,   batch_size=model_cfg["batch_size"])
        test_dataloader  = DataLoader(test_dataset,  batch_size=model_cfg["batch_size"])

        train_torch_model(
            classifier=classifier,
            device=device,
            model_cfg=model_cfg,
            local_storage_dir=local_storage_dir,
            train_dataloader=train_dataloader,
            val_dataloader=val_dataloader
        )
    
        # load the best model
        classifier.load_state_dict(torch.load(local_storage_dir + "model.pth"))
        classifier.eval()

        eval_model(
            classifier=classifier, 
            val_dataloader=val_dataloader, 
            test_dataloader=test_dataloader, 
            val_roc_path=local_storage_dir + "val_ROC.pdf",
            test_roc_path=local_storage_dir + "test_ROC.pdf",
            classification_report_path=local_storage_dir + "classification_report.txt"
        )
    else:
        train_X, train_y = train_dataset.get_full_data_matrix()
        val_X, val_y = val_dataset.get_full_data_matrix()
        
        train_X[(train_X == -np.inf) | (train_X == np.inf)] = np.nan
        val_X[(val_X == -np.inf) | (val_X == np.inf)] = np.nan

        classifier.fit(train_X, train_y, eval_set=[(train_X, train_y), (val_X, val_y)])
        classifier.save_model(local_storage_dir + "model.json")

        # plot train and val loss
        results = classifier.evals_result()

        plt.clf()
        plt.figure(figsize=(10,7))
        plt.plot(results["validation_0"]["logloss"], label="Training loss")
        plt.plot(results["validation_1"]["logloss"], label="Validation loss")
        plt.axvline(classifier.best_iteration, color="gray", label="Optimal tree number")
        plt.xlabel("Number of trees")
        plt.ylabel("Loss")
        plt.legend()
        plt.savefig(local_storage_dir + "loss.pdf")

        test_X, test_y = test_dataset.get_full_data_matrix()

        test_X[(test_X == -np.inf) | (test_X == np.inf)] = np.nan

        eval_sklearn_model(
            classifier=classifier,
            val_X=val_X,
            val_y=val_y,
            test_X=test_X,
            test_y=test_y,
            val_roc_path=local_storage_dir + "val_ROC.pdf",
            test_roc_path=local_storage_dir + "test_ROC.pdf",
            classification_report_path=local_storage_dir + "classification_report.txt"
        )

        if model_cfg["model_type"] == "xgboost":
            # save feature importance
            feature_names = []
            for t in range(train_config["num_candles_to_stack"] - 1, -1, -1):
                for name in train_config["features_to_use"]:
                    feature_names.append(f"{name}_t-{t}")
            
            sorted_idx = classifier.feature_importances_.argsort()
            with open(local_storage_dir + "feature_importance.txt", "w") as out_file:
                out_file.writelines(
                    [f"{feature_names[i]}: \t\t\t{classifier.feature_importances_[i]}\n" 
                     for i in sorted_idx]
                )

    logger.info(f"Finished training and evaluation in {time.time() - start_time}s. Find related files in {local_storage_dir}")


def train_torch_model(classifier: torch.nn.Module, device: str,
                      model_cfg: dict, local_storage_dir: str,
                      train_dataloader: DataLoader, 
                      val_dataloader: DataLoader):
    """ Train torch model using early stopping. Model which achieved
    lowest validation loss is saved to local_storage_dir/model.pth

    Args:
        classifier (torch.nn.Module): Torch model
        device (str): device to use for training.
        model_cfg (dict): model config (contains training config params such as early stopping param etc)
        local_storage_dir (str): where to store train and validation metrics with training. Also
            best model will be stored here in with filename model.pth
        train_dataloader (DataLoader): train dataset
        val_dataloader (DataLoader): validation dataset
    """
    # initialise loss function and optimizer
    loss_fn = nn.CrossEntropyLoss(label_smoothing=model_cfg["label_smoothing"])
    val_loss_fn = nn.CrossEntropyLoss()  # no label smoothing

    optimizer = ScheduledOptim(
        optimizer=Adam(classifier.parameters(), betas=(0.9, 0.98), eps=1e-09),
        lr_mul=model_cfg["lr_mul"],
        d_model=model_cfg.get("d_model", 512),
        n_warmup_steps=model_cfg["n_warmup_steps"]
    )

    # train model using earling stopping
    early_stopper = EarlyStopper(patience=model_cfg["patience"], min_delta=model_cfg["min_delta"])
    writer = SummaryWriter(log_dir=local_storage_dir)
    logger.info(f"View TensorBoard logs at dir: {local_storage_dir}")

    for epoch in range(1, model_cfg["max_epochs"] + 1):
        train_loss = train_correct = 0
        classifier.train()
        for batch, (X, y) in enumerate(train_dataloader):
            X, y = X.to(device), y.to(device)

            pred = classifier(X)
            loss = loss_fn(pred, y)

            train_loss    += loss.item()
            train_correct += (pred.argmax(1) == y).type(torch.float).sum().item()

            loss.backward()
            nn.utils.clip_grad_norm_(classifier.parameters(), model_cfg["grad_clip"])
            optimizer.step_and_update_lr()
            optimizer.zero_grad()

        train_loss     /= len(train_dataloader)
        train_correct  /= len(train_dataloader.dataset)
        writer.add_scalar("Train Loss", train_loss , epoch)
        writer.add_scalar("Train Acc",  train_correct, epoch)
        
        # calculate validation loss
        val_loss = val_correct = 0
        classifier.eval()
        with torch.no_grad():
            for batch, (X, y) in enumerate(val_dataloader):
                X, y = X.to(device), y.to(device)

                pred = classifier(X)
                loss = val_loss_fn(pred, y)

                val_loss    += loss.item()
                val_correct += (pred.argmax(1) == y).type(torch.float).sum().item()

        val_loss     /= len(val_dataloader)
        val_correct  /= len(val_dataloader.dataset)
        writer.add_scalar("Val Loss", val_loss, epoch)
        writer.add_scalar("Val Acc",  val_correct , epoch)

        logger.info(f"Epoch [{epoch}/{model_cfg['max_epochs']}], Patience [{early_stopper.counter}/{early_stopper.patience}] : "
                    f"[train_loss={train_loss}], [train_acc={train_correct}], [val_loss={val_loss}], [val_acc={val_correct}]")

        if early_stopper.early_stop(val_loss):
            logger.info(f"Stopping training after {epoch} epochs, due to early stopping.")
            break

        if early_stopper.counter == 0:
            logger.info(f"Saving model with lowest val loss so far to {local_storage_dir}model.pth")
            torch.save(classifier.state_dict(), local_storage_dir + "model.pth")

    writer.flush()
    writer.close()


def eval_model(classifier: nn.Module, val_dataloader: DataLoader, 
               test_dataloader: DataLoader, val_roc_path: str, 
               test_roc_path: str, classification_report_path: str):
    """ Evaluate model by finding best thresholds on validation ROC,
    best_threshold maximizes TPR - FPR, safe_threshold maximizes TPR - 2 * FPR, 
    risky_threshold maximizes 2 * TPR - FPR. Then save a classification report for
    the three thresholds on the test set. Also plot ROC on test set.

    Args:
        classifier (nn.Module): classifier to test.
        val_dataloader (DataLoader): validation set dataloader
        test_dataloader (DataLoader): test set dataloader
        val_roc_path (str): file name path for which to save validation set ROC (.pdf file)
        test_roc_path (str): file name path for which to save test set ROC (.pdf file)
        classification_report_path (str): file name path for which to save classification report (.txt file)
    """
    device = get_device()

    # find optimal thresholds using val set  
    best_thresh, safe_thresh, risky_thresh = calc_optimal_threshold(val_dataloader, classifier, device, val_roc_path)
    logger.info(f"best_thresh = {best_thresh}, safe_thresh = {safe_thresh}, risky_thresh = {risky_thresh}")

    # plot ROC on test set and report results:
    #    - classification report using best_thresh
    #    - classification report using safe_thresh
    #    - classification report using risky_thresh
    calc_optimal_threshold(test_dataloader, classifier, device, test_roc_path)

    ys                  = np.zeros( (len(test_dataloader.dataset), ) )
    preds_best_thresh   = np.zeros( (len(test_dataloader.dataset), ) )
    preds_safe_thresh   = np.zeros( (len(test_dataloader.dataset), ) )
    preds_safe_thresh2  = np.zeros( (len(test_dataloader.dataset), ) )
    preds_risky_thresh  = np.zeros( (len(test_dataloader.dataset), ) )
    preds_risky_thresh2 = np.zeros( (len(test_dataloader.dataset), ) )
    idx = 0
    classifier.eval()
    with torch.no_grad():
        for batch, (X, y) in enumerate(test_dataloader):
            X, y = X.to(device), y.to(device)

            logits = classifier(X)  # (batch_size, 2)
            probs  = nn.functional.softmax(logits, dim=1) # (batch_size, 2)
            probs  = probs[:, 1]  # (batch_size, )

            batch_size = probs.size(dim=0)

            ys[                 idx: idx + batch_size]  = y.numpy(force=True)
            preds_best_thresh[  idx: idx + batch_size]  = (probs > best_thresh ).long().numpy(force=True)
            preds_safe_thresh[  idx: idx + batch_size]  = (probs > safe_thresh ).long().numpy(force=True)
            preds_safe_thresh2[ idx: idx + batch_size]  = (probs > 0.75        ).long().numpy(force=True)
            preds_risky_thresh[ idx: idx + batch_size]  = (probs > risky_thresh).long().numpy(force=True)
            preds_risky_thresh2[idx: idx + batch_size]  = (probs > 0.25        ).long().numpy(force=True)
    
            idx += batch_size
    
    assert idx == len(test_dataloader.dataset), f"Expected final idx to be {len(test_dataloader.dataset)} not {idx}"
    
    best_thresh_report   = classification_report(ys, preds_best_thresh,   target_names=["Don't Buy", "Buy"])
    safe_thresh_report   = classification_report(ys, preds_safe_thresh,   target_names=["Don't Buy", "Buy"])
    safe_thresh_report2  = classification_report(ys, preds_safe_thresh2,  target_names=["Don't Buy", "Buy"])
    risky_thresh_report  = classification_report(ys, preds_risky_thresh,  target_names=["Don't Buy", "Buy"])
    risky_thresh_report2 = classification_report(ys, preds_risky_thresh2, target_names=["Don't Buy", "Buy"])

    with open(classification_report_path, mode="w") as f:
        f.writelines([
            f"best_thresh = {best_thresh}\n",
            "best_thresh_report:\n", best_thresh_report,

            f"\nsafe_thresh = {safe_thresh}\n",
            "safe_thresh_report:\n", safe_thresh_report,

            f"\nsafe_thresh2 = 0.75\n",
            "safe_thresh_report:\n", safe_thresh_report2,

            f"\nrisky_thresh = {risky_thresh}\n",
            "risky_thresh_report:\n", risky_thresh_report,

            f"\nrisky_thresh2 = 0.25\n",
            "risky_thresh_report:\n", risky_thresh_report2
        ])


def eval_sklearn_model(classifier: ClassifierMixin,
                       val_X: np.ndarray, val_y: np.ndarray,
                       test_X: np.ndarray, test_y: np.ndarray,
                       val_roc_path: str, test_roc_path: str, 
                       classification_report_path: str):
    """ Evaluate model by finding best thresholds on validation ROC,
    best_threshold maximizes TPR - FPR, safe_threshold maximizes TPR - 2 * FPR, 
    risky_threshold maximizes 2 * TPR - FPR. Then save a classification report for
    the three thresholds on the test set. Also plot ROC on test set. 

    Args:
        classifier (ClassifierMixin): sklearn classifier
        val_X (np.ndarray): (N, D) np array, validation features
        val_y (np.ndarray): (N, ) np array, validation labels
        test_X (np.ndarray): (N', D) np array, test features
        test_y (np.ndarray): (N', ) np array, test labels
        val_roc_path (str): file name path for which to save validation set ROC (.pdf file)
        test_roc_path (str): file name path for which to save test set ROC (.pdf file)
        classification_report_path (str): file name path for which to save classification report (.txt file)
    """
    # find optimal thresholds using val set  
    best_thresh, safe_thresh, risky_thresh = calculate_optimal_threshold_sklearn_model(
        classifier, val_X, val_y, val_roc_path
    )
    logger.info(f"best_thresh = {best_thresh}, safe_thresh = {safe_thresh}, risky_thresh = {risky_thresh}")

    # plot ROC on test set and report results:
    #    - classification report using best_thresh
    #    - classification report using safe_thresh
    #    - classification report using risky_thresh
    calculate_optimal_threshold_sklearn_model(
        classifier, test_X, test_y, test_roc_path
    )
    
    probs = classifier.predict_proba(test_X)[:, 1]

    best_thresh_report   = classification_report(test_y, probs > best_thresh,  target_names=["Don't Buy", "Buy"])
    def_thresh_report    = classification_report(test_y, probs > 0.5,          target_names=["Don't Buy", "Buy"])
    safe_thresh_report   = classification_report(test_y, probs > safe_thresh,  target_names=["Don't Buy", "Buy"])
    safe_thresh_report2  = classification_report(test_y, probs > 0.75,         target_names=["Don't Buy", "Buy"])
    risky_thresh_report  = classification_report(test_y, probs > risky_thresh, target_names=["Don't Buy", "Buy"])
    risky_thresh_report2 = classification_report(test_y, probs > 0.25,         target_names=["Don't Buy", "Buy"])

    with open(classification_report_path, mode="w") as f:
        f.writelines([
            f"best_thresh = {best_thresh}\n",
            "best_thresh_report:\n", 
            best_thresh_report,

            f"\ndefault thresh = 0.5\n",
            "default_thresh_report:\n", 
            def_thresh_report,

            f"\nsafe_thresh = {safe_thresh}\n",
            "safe_thresh_report:\n", 
            safe_thresh_report,

            f"\nsafe_thresh2 = 0.75\n",
            "safe_thresh_report:\n", 
            safe_thresh_report2,

            f"\nrisky_thresh = {risky_thresh}\n",
            "risky_thresh_report:\n", 
            risky_thresh_report,

            f"\nrisky_thresh2 = 0.25\n",
            "risky_thresh_report:\n", 
            risky_thresh_report2
        ])


def calc_optimal_threshold(data_loader: DataLoader, classifier: nn.Module, 
                           device: str, plot_fn: Optional[str]) -> Tuple[float, float, float]:
    """ Calculate threshold which maximizes TPR - FPR

    Args:
        data_loader (DataLoader):  dataset used for calculating best threshold
        classifier (nn.Module): torch model which has output (batch_size, 2) (binary classification)
        device (str): device ("cpu" or "cuda")
        plot_fn (Optional[str]): path to save ROC plot, if None won't plot.
    Returns:
        Tuple[float, float, float]: (optimal_thresh, safe_thresh, risky_thresh):
            optimal_thresh maximizes TPR - FPR
            safe_thresh maximizes 0.25 * TPR - 0.75 * FPR
            risky_thresh maximizes 0.75 * TPR - 0.25 * FPR
    """
    ys        = np.zeros( (len(data_loader.dataset), ) )
    all_probs = np.zeros( (len(data_loader.dataset), ) )
    idx = 0
    classifier.eval()
    with torch.no_grad():
        for batch, (X, y) in enumerate(data_loader):
            X, y = X.to(device), y.to(device)

            logits = classifier(X)  # (batch_size, 2)
            probs  = nn.functional.softmax(logits, dim=1) # (batch_size, 2)
            probs  = probs[:, 1]  # (batch_size, )

            batch_size = probs.size(dim=0)

            ys[idx: idx + batch_size]        = y.numpy(force=True)
            all_probs[idx: idx + batch_size] = probs.numpy(force=True)

            idx += batch_size

    assert idx == len(data_loader.dataset), f"Expected final idx to be {len(data_loader.dataset)} not {idx}"

    fpr, tpr, thresholds = roc_curve(ys, all_probs)

    best_thresh_idx = np.argmax(tpr - fpr)
    best_thresh = thresholds[best_thresh_idx]

    safe_thresh_idx = np.argmax(tpr - 2 * fpr)
    safe_thresh = thresholds[safe_thresh_idx]

    risky_thresh_idx = np.argmax(2 * tpr - fpr)
    risky_thresh = thresholds[risky_thresh_idx]

    if plot_fn is not None:
        plt.clf()
        plt.plot([0,1], [0,1], linestyle='--', label='No Skill')
        plt.plot(fpr, tpr, marker='.', label=type(classifier).__name__)
        plt.scatter(fpr[best_thresh_idx], tpr[best_thresh_idx], marker='o', color='black', label=f'Best={round(best_thresh, 2)}')
        plt.scatter(fpr[safe_thresh_idx], tpr[safe_thresh_idx], marker='o', color='green', label=f'Safe={round(safe_thresh, 2)}')
        plt.scatter(fpr[risky_thresh_idx], tpr[risky_thresh_idx], marker='o', color='red', label=f'Risky={round(risky_thresh, 2)}')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.legend()
        plt.savefig(plot_fn)

    return best_thresh, safe_thresh, risky_thresh


def calculate_optimal_threshold_sklearn_model(classifier: ClassifierMixin, 
                                              X: np.ndarray, y: np.ndarray, 
                                              plot_fn: Optional[str]):
    """Calculate threshold which maximizes TPR - FPR on sklearn model.

    Args:
        classifier (ClassifierMixin): sklearn model
        X (np.ndarray): numpy array with shape (N, D) used to optimise threshold
        y (np.ndarray): numpy array with shape (N, ) used to optimise threshold
        plot_fn (Optional[str]): path to save ROC plot, if None won't plot.
    """
    probs = classifier.predict_proba(X)[:, 1]
    fpr, tpr, thresholds = roc_curve(y, probs)

    best_thresh_idx = np.argmax(tpr - fpr)
    best_thresh = thresholds[best_thresh_idx]

    safe_thresh_idx = np.argmax(tpr - 1.5 * fpr)
    safe_thresh = thresholds[safe_thresh_idx]

    risky_thresh_idx = np.argmax(1.5 * tpr - fpr)
    risky_thresh = thresholds[risky_thresh_idx]

    if plot_fn is not None:
        plt.clf()
        plt.plot([0,1], [0,1], linestyle='--', label='No Skill')
        plt.plot(fpr, tpr, marker='.', label=type(classifier).__name__)
        plt.scatter(fpr[best_thresh_idx], tpr[best_thresh_idx], marker='o', color='black', label=f'Best={round(best_thresh, 2)}')
        plt.scatter(fpr[safe_thresh_idx], tpr[safe_thresh_idx], marker='o', color='green', label=f'Safe={round(safe_thresh, 2)}')
        plt.scatter(fpr[risky_thresh_idx], tpr[risky_thresh_idx], marker='o', color='red', label=f'Risky={round(risky_thresh, 2)}')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.legend()
        plt.savefig(plot_fn)

    return best_thresh, safe_thresh, risky_thresh


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
