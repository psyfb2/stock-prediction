from typing import List, Tuple

import numpy as np

from data_collection.historical_data import get_historical_data, get_all_sp500_tickers

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
TICKERS = get_all_sp500_tickers() + [
    # some extra symbols outside the S&P 500
    "GOLD", "AEM", "WPM", "FNV", "GFI", "RGLD",
    "GLD", "AU", "KGC", "PAAS", "AGI", "B2G", "OR",
    "EQX", "CEY", "OGC", "CG", "AIG", "FSM", "ORA",

    "NHC", "PLS", "YAL", "ALK", "TER", "SJT", "PDN", "NRT",
    "EFR", "PLL", "NXE"
]



def load_dataset(tickers: List[str], start_date: str, end_date: str, 
                 candle_size="1d") -> Tuple[np.ndarray, np.ndarray]:
    """ load preprocessed dataset. Does not perform windowing to transform
    X shape from (N, D) to (N, T, D). This should be done at train time
    whenever a batch is loaded to save memory.

    Args:
        tickers (List[str]): ticker symbols to be included in the dataset
        start_date (str): start date for data in yyyy-mm-dd format
        end_date (str): end date for data in yyyy-mm-dd format (exclusive)
        num_candles_to_stack (int): number of time-steps for a single data-point
        candle_size (str): frequency of candles. "1d" or "1h"

    Returns:
        Tuple[np.ndarray, np.ndarray]: (X, y). X has shape (N, D), 
            y has shape (N, ) and contains binary labels.
    """
    for ticker in tickers:
        get_historical_data(symbol=ticker, start_date=start_date, end_date=end_date, candle_size=candle_size)

    return None, None


def main(tickers: List[str], train_start_date="2012-01-01", val_start_date="2023-05-01", 
         test_start_date="2023-08-01", test_end_date="2024-01-06", num_candles_to_stack=180, candle_size="1d"):

    X, y = load_dataset(tickers=tickers, start_date=train_start_date, end_date=test_end_date, 
                        num_candles_to_stack=num_candles_to_stack, candle_size=candle_size)


if __name__ == "__main__":
    main(TICKERS)
