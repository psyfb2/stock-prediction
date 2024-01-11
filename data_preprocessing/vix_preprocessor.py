from logging import getLogger

import pandas as pd
import pandas_ta as ta

from custom_ti.percentage_macd import percentage_macd
from data_preprocessing.base_preprocessor import BasePreprocessor


class VixPreprocessor(BasePreprocessor):
    def __init__(self):
        """ Create a preprocessor specifically for VIX data. """
        super().__init__(
            logger=getLogger(__name__), 
            bounded_cols={
                "percentile":        (0, 1),
                "percentile_126":    (0, 1),
                "percentile_30":     (0, 1),
                "rank":              (0, 1),
                "rank_126":          (0, 1),
                "rank_30":           (0, 1),
            },
            candle_size="1d"  # VIX is only available using daily candles
        )
    
    def calc_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """ Calculate vix unnormalised features.

        Args:
            df (pd.DataFrame): df containing ["t", "o", "c", "h", "l"] columns or just ["t", "c"] columns

        Returns:
            pd.DataFrame: df containing original unchanged columns + feature columns
        """
        price_cols = ["o", "c", "l", "h"]
        df = df.copy()
        
        # ------ apply TI's ------ #
        df['ema10']  = ta.ema(df['c'], length=10)

        df['percentile']     = df['c'].rolling(252).apply(lambda x: x[x <= x.iloc[-1]].count() / 252)
        df['percentile_126'] = df['c'].rolling(126).apply(lambda x: x[x <= x.iloc[-1]].count() / 126)
        df['percentile_30']  = df['c'].rolling(30 ).apply(lambda x:  x[x <= x.iloc[-1]].count() / 30)

        df['rank']     = df['c'].rolling(252).apply(lambda x: (x.iloc[-1] - x.min()) / (x.max() - x.min()))
        df['rank_126'] = df['c'].rolling(126).apply(lambda x: (x.iloc[-1] - x.min()) / (x.max() - x.min()))
        df['rank_30']  = df['c'].rolling(30 ).apply(lambda x: (x.iloc[-1] - x.min()) / (x.max() - x.min()))

        macd = percentage_macd(close=df['c'], fast=12, slow=26, signal=9)
        df['macd']        = macd['MACD_12_26_9']
        df['macd_signal'] = macd['MACDs_12_26_9']
        # ------ ------ #

        # ------ keep the original columns as VIX is stationary ----- #
        for col in price_cols:
            if col in df.columns:
                df[f"{col}_orig"] = df[col]
        # ------ ------ #

        return df

    