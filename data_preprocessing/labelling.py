import pandas as pd
import numpy as np


def binary_label_tp_tsl(df: pd.DataFrame, tp: float, tsl: float) -> pd.Series:
    """ Calculate series which will have value 1 at each index if take profit of tp 
    is hit before trailing stop loss of tsl, if the trailing stop loss is hit
    first then value of 0 is given. If value cannot be determined 
    (i.e. data ends before tp or tsl is hit), then will have np.nan.
    tp_tsl order on candle is assumed to take place on next candles open.

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

        entry_price = df["o"].iloc[r.name + 1]
        max_price   = df['h'].iloc[r.name + 1]

        for idx, row in df.iloc[r.name + 1:].iterrows():
            max_price = max(max_price, row['h'])

            profit   =  perc_change(entry_price, row['h'])  # highest profit on this candle
            drawdown = -perc_change(max_price,   row['l'])  # highest drawdown on this candle

            if profit >= tp:
                return 1
            if drawdown >= tsl:
                return 0
    
        return np.nan
            
    return df.apply(row_func, axis=1)


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