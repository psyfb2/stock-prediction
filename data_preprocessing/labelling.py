import pandas as pd
import numpy as np


def binary_label_tp_tsl(df: pd.DataFrame, tp: float, tsl: float) -> pd.Series:
    """ Calculate series which will have value 1 at each index if take profit of tp 
    is hit before trailing stop loss of tsl, if the trailing stop loss is hit
    first then value of 0 is given. If value cannot be determined 
    (i.e. data ends before tp or tsl is hit), then will have np.nan.
    tp_tsl order on candle is assumed to take place relative to this candles close.

    Args:
        df (pd.DataFrame): df containing columns ["o", "c", "h", "l"]
        tp (float): take profit value (e.g. 0.05 for 5% take profit)
        tsl (float): trailing stop loss value (e.g. 0.05 for 5% trailing stop loss)
    """
    if len({"o", "c", "l", "h"}.difference(set(df.columns))) > 0:
        raise ValueError(f"Argument 'df' must contain columns ['o', 'c', 'h', 'l']. Got columns {list(df.columns)}")
    
    def row_func(r: pd.Series):
        if r.name + 1 >= len(df):
            return np.nan

        max_price = entry_price = df["c"].iloc[r.name]

        for idx, row in df.iloc[r.name + 1:].iterrows():
            if row["c"] > row["o"]:
                # green candle, low assumed to occur first then high
                max_price = max(max_price, row['o'])
                drawdown  = -perc_change(max_price,   row['l'])  # highest drawdown on this candle

                if drawdown >= tsl:
                    return 0
                
                max_price = max(max_price, row['h'])
                profit    = perc_change(entry_price, row['h'])  # highest profit on this candle

                if profit >= tp:
                    return 1
            else:
                # red candle, high assumed to occur first then low
                max_price = max(max_price, row['h'])

                profit   =  perc_change(entry_price, row['h'])  # highest profit on this candle
                drawdown = -perc_change(max_price,   row['l'])  # highest drawdown on this candle

                if profit >= tp:
                    return 1
                if drawdown >= tsl:
                    return 0

        return np.nan

    return df.apply(row_func, axis=1)


def next_close_higher(df: pd.DataFrame) -> pd.Series:
    """ Calculate series which will have value of
    1 if close_{t + 1} > close_{t}  (next close higher than close at current index)
    0 otherwise

    Args:
        df (pd.DataFrame): df containing columns ["o", "c", "h", "l"]
    Returns:
        pd.Series: series with same number of rows as df, last value will be NaN
    """
    if len({"o", "c", "l", "h"}.difference(set(df.columns))) > 0:
        raise ValueError(f"Argument 'df' must contain columns ['o', 'c', 'h', 'l']. Got columns {list(df.columns)}")
    
    labels = (df['c'] < df['c'].shift(periods=-1)).astype(int)
    labels.iloc[-1] = np.NaN

    return labels


def perc_change(a: float, b: float) -> float:
    """ calculate percentage change from a to b
    (e.g. percentage_increase(100, 110) => 0.1)

    Args:
        a (float): first number
        b (float): second number

    Returns:
        float: percentage change from a to b
    """
    return (b / a) - 1