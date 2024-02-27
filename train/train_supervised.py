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

from sklearn.metrics import roc_curve, classification_report
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
TO DO:
    how to test each strategy?

    - Create models to predict if close_{t + 1} > close_{t}, close_{t + 1} > open_{t + 1}, open_{t + 1} > close_{t}
      hyperparemeter tuning on validation set and early stopping
      test set to pick best model (from MLP, 1D-CNN, XGBoost, Ensemble (majority vote)) for each of the three predictions.

      Also use a reduced feature set with T in {2, 3, 4, 5}, with and without normalisation.
        (l to o, o to c, c to high, c_perc_change, v_perc_change,
         percentile_6month, percentile_1y, 
         bbands, mfi, uo, adx (no +- signal), macd, stoch_rsi (no ma), stoch_osc (no ma), natr, 
         50day_sma,  100day_sma, 200day_sma, c_perc_change_ema_10, 
         month_of_year)
      and use a lot of data (5000+ stocks from US, Canada, UK, EU)
    
    - Create Markov chain for last 3 candles where each candle has the following posibilities:
        - close_t > close_{t - 1}
        - close_t > open_t
        - open_t  > close_{t - 1}
        - neck > body
        - tail > body
        - volume_t > volume_{t - 1} * 1.25
        - NATR > 3.0
        - ADX > 25
        - MACD > 0
        - MACD_HIST > 0
        - low_t < bbands_lower_t
        - high_t > bbands_upper_t
        - bbands_width_t > bbands_width_{t - 1}
        - STOCH_RSI > 0.9
        - STOCH_RSI < 0.1
        - probably need to be candle specific about this

        Make sure there is enough data to estimate probability of each transition. If not
        try last 2 candles. Test Markov chain on same test used for ML models, is it competetive
        in terms of accuracy and F1 for the three prediction tasks? 

        Then calculate confidence interval adjusted transition probabilities,
        each transition prob in markov chain = prob - 95% CI. The markov chain probabilities will no
        longer add to one, but this will be used as a kind of safety net, especially when we are uncertain
        about the estimated transition probability.
        

    - Testing a trading strategy
        use a second test set for testing trading strategy
        Strategy 1:
            if transition to a close_t > close_{t - 1} state has probability > 50
                
                if CI adjusted transition probability > 51 and ML predicts close_t > close_{t - 1}
                    Given that close_t > close_{t - 1}, 
                    if P(open_t < close_{t - 1}) > 50 buy on the next open, otherwise buy on the close, sell on next close
            
            elif transition to a close_t > open_t state has probability > 50
                if CI adjusted transition probability > 51 and ML predicts close_t > open_t 
                    buy on next open, sell on next close
            
            elif transition to a open_t > close_{t - 1} state has probability > 50
                if CI adjusted transition probability > 51 and ML predicts open_t > close_{t - 1}
                    buy on close, sell on next open
            
            also if had a sell on next close from the last timestep and this timestep have a buy on close, 
            cancel the previous sell order and stay long to save on commissions.
    
        Calculate Profit, Sharpe Ratio, Win Rate, Bought lower Rate, Avg Profit per Trade, Best Trade, Worst Trade, Exposure, B&H profit
        across all stocks in test test and compare with equally weighted B&H profit.

        One thing to keep in mind is that in the backtesting we assume we can know the true close price and volume
        before placing a trade on the close. However in reality this is not the case since we trade e.g. 5 mins before close.
        Calculations for 15 stocks using yfinance querying 5 mins before close and then the day after shows
        an average price deviation of the price 5 mins before close and actual close is 0.07% and average volume increase is 18.35%.
        So in a real life application it's recomended to artificially raise the voume by 18%, the deviation in the close is acceptable.
        Once a broker is decided I would retest this with chosed broker data collection.

    - Future improvements:
        - There are 8 combinations of c_t > c_{t - 1}, c_t > o_t, o_t > c_{t - 1}, but only 6 of them are physically possible.
          For the ML model train a single multi-label classification model with logical constraints (see: https://arxiv.org/pdf/2103.13427.pdf)
          and use it if it performs better on the test set for the three predictions.
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

    if train_config["data_npz_file"]:
        # load data arrays which have previously been preprocessed
        datasets = np.load(train_config["data_npz_file"])

        train_dataset = StocksDatasetInMem(
            tickers=None, features_to_use=train_config["features_to_use"],
            vix_features_to_use=train_config["vix_features_to_use"],
            start_date=None, end_date=None, tp=None, tsl=None, 
            num_candles_to_stack=train_config["num_candles_to_stack"],
            candle_size=None, features=datasets["train_X"], labels=datasets["train_y"], 
            lengths=datasets["train_lengths"]
        )

        val_dataset = StocksDatasetInMem(
            tickers=None, features_to_use=train_config["features_to_use"],
            vix_features_to_use=train_config["vix_features_to_use"],
            start_date=None, end_date=None, tp=None, tsl=None, 
            num_candles_to_stack=train_config["num_candles_to_stack"],
            candle_size=None, features=datasets["val_X"], labels=datasets["val_y"], 
            lengths=datasets["val_lengths"]
        )

        test_dataset = StocksDatasetInMem(
            tickers=None, features_to_use=train_config["features_to_use"],
            vix_features_to_use=train_config["vix_features_to_use"],
            start_date=None, end_date=None, tp=None, tsl=None, 
            num_candles_to_stack=train_config["num_candles_to_stack"],
            candle_size=None, features=datasets["test_X"], labels=datasets["test_y"], 
            lengths=datasets["test_lengths"]
        )

    else:
        # load all data for tickers into file cache (this is only neccessary 
        # because of yfinance 2K requests rate limit per hour)
        # loading into file cache means 1 req per ticker instead of 3 (train, val, test)
        preprocessor = AssetPreprocessor(features_to_use=train_config["features_to_use"], 
                                         candle_size=train_config["candle_size"])
        adjusted_start_date = preprocessor.adjust_start_date(
            parse(train_config["train_start_date"]), train_config["num_candles_to_stack"]
        ).strftime("%Y-%m-%d")
        for ticker, exchange in train_config["tickers"]:
            get_historical_data(symbol=ticker, start_date=adjusted_start_date, end_date=train_config["test_end_date"],
                                candle_size=train_config["candle_size"], exchange=exchange)

        # load preprocessed datasets (train, val, test)
        logger.info("Loading training data")
        train_dataset = StocksDatasetInMem(
            tickers=train_config["tickers"], features_to_use=train_config["features_to_use"],
            vix_features_to_use=train_config["vix_features_to_use"],
            start_date=train_config["train_start_date"], 
            end_date=train_config["val_start_date"], tp=train_config["tp"], tsl=train_config["tsl"],
            num_candles_to_stack=train_config["num_candles_to_stack"], candle_size=train_config["candle_size"]
        )

        # save means and stds, these will be required at inference time
        means, stds = train_dataset.means, train_dataset.stds
        with open(local_storage_dir + "normalisation_info.json", "w") as fp:
            json.dump({"means": means, "stds": stds, "in_features": train_dataset.features.shape[-1]}, fp)

        logger.info("Loading validation data")
        val_dataset = StocksDatasetInMem(
            tickers=train_config["tickers"], features_to_use=train_config["features_to_use"],
            vix_features_to_use=train_config["vix_features_to_use"],
            start_date=train_config["val_start_date"], 
            end_date=train_config["test_start_date"], tp=train_config["tp"], tsl=train_config["tsl"],
            num_candles_to_stack=train_config["num_candles_to_stack"],
            means=means, stds=stds, candle_size=train_config["candle_size"]
        )

        logger.info("Loading test data")
        test_dataset = StocksDatasetInMem(
            tickers=train_config["tickers"], features_to_use=train_config["features_to_use"],
            vix_features_to_use=train_config["vix_features_to_use"],
            start_date=train_config["test_start_date"], 
            end_date=train_config["test_end_date"], tp=train_config["tp"], tsl=train_config["tsl"],
            num_candles_to_stack=train_config["num_candles_to_stack"],
            means=means, stds=stds, candle_size=train_config["candle_size"]
        )

        np.savez(local_storage_dir + "datasets.npz", 
                 train_X=train_dataset.features, train_y=train_dataset.labels, train_lengths=train_dataset.lengths,
                 val_X=val_dataset.features,     val_y=val_dataset.labels,     val_lengths=val_dataset.lengths,
                 test_X=test_dataset.features,   test_y=test_dataset.labels,   test_lengths=test_dataset.lengths)

    train_dataloader = DataLoader(train_dataset, batch_size=model_cfg["batch_size"], shuffle=True)
    val_dataloader   = DataLoader(val_dataset,   batch_size=model_cfg["batch_size"])
    test_dataloader  = DataLoader(test_dataset,  batch_size=model_cfg["batch_size"])

    # initialise model
    device = get_device()
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
        
    else:
        raise ValueError(f"model_type '{model_cfg['model_type']}' is not recognised.")
    summary(classifier, input_size=(model_cfg["batch_size"], train_config["num_candles_to_stack"], in_features), device=device)

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

    logger.info(f"Train dataset length = {len(train_dataloader.dataset)}, with label counts: {train_dataset.label_counts}")
    logger.info(f"Val dataset length = {len(val_dataloader.dataset)}, with label counts: {val_dataset.label_counts}")
    logger.info(f"Test dataset length = {len(test_dataloader.dataset)}, with label counts: {test_dataset.label_counts}")
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

    # load the best model
    classifier.load_state_dict(torch.load(local_storage_dir + "model.pth"))
    classifier.eval()
    eval_model(classifier=classifier, val_dataloader=val_dataloader, test_dataloader=test_dataloader, 
               val_roc_path=local_storage_dir + "val_ROC.pdf", test_roc_path=local_storage_dir + "test_ROC.pdf",
               classification_report_path=local_storage_dir + "classification_report.txt")

    logger.info(f"Finished training and evlauation in {time.time() - start_time}s. Find related files in {local_storage_dir}")


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
            "best_thresh_report:", best_thresh_report,

            f"\nsafe_thresh = {safe_thresh}\n",
            "safe_thresh_report:", safe_thresh_report,

            f"\nsafe_thresh2 = 0.75\n",
            "safe_thresh_report:", safe_thresh_report2,

            f"\nrisky_thresh = {risky_thresh}\n",
            "risky_thresh_report", risky_thresh_report,

            f"\nrisky_thresh2 = 0.25\n",
            "risky_thresh_report", risky_thresh_report2
        ])


def calc_optimal_threshold(data_loader: DataLoader, classifier: nn.Module, 
                           device: str, plot_fn: Optional[str]) -> Tuple[float, float]:
    """ Calculate threshold which maximizes TPR - FPR

    Args:
        data_loader (DataLoader):  dataset used for calculating best threshold
        classifier (nn.Module): torch model which has output (batch_size, 2) (binary classification)
        device (str): device ("cpu" or "cuda")
        plot_fn (Optional[str]): path to save ROC plot, if None won't plot.
    Returns:
        Tuple[float, float]: (optimal_thresh, safe_thresh, risky_thresh):
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
    parser.add_argument("-c", "--config", default="train/configs/supervised_mlp_base_config.json", 
                        help="train config json file")
    
    args = parser.parse_args()

    with open(args.config) as json_file:
        train_config = json.load(json_file)
    
    main(train_config)
