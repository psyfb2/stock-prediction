from logging import getLogger
from typing import List

import pandas as pd
import pandas_ta as ta

from custom_ti.percentage_macd import percentage_macd
from data_preprocessing.base_preprocessor import BasePreprocessor


class VixPreprocessor(BasePreprocessor):
    def __init__(self, features_to_use: List[str]):
        """ Create a preprocessor specifically for VIX data. 
        
        Args:
            features_to_use (List[str]): which features to include within preprocessed VIX df.
        """
        super().__init__(
            logger=getLogger(__name__), 
            bounded_cols={
                "vix_percentile":        (0, 1),
                "vix_percentile_126":    (0, 1),
                "vix_percentile_30":     (0, 1),
                "vix_rank":              (0, 1),
                "vix_rank_126":          (0, 1),
                "vix_rank_30":           (0, 1),
            },
            features_to_use=features_to_use,
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
        df['vix_ema10']  = ta.ema(df['c'], length=10)

        df['vix_percentile']     = df['c'].rolling(252).apply(lambda x: x[x <= x.iloc[-1]].count() / 252)
        df['vix_percentile_126'] = df['c'].rolling(126).apply(lambda x: x[x <= x.iloc[-1]].count() / 126)
        df['vix_percentile_30']  = df['c'].rolling(30 ).apply(lambda x:  x[x <= x.iloc[-1]].count() / 30)

        df['vix_rank']     = df['c'].rolling(252).apply(lambda x: (x.iloc[-1] - x.min()) / (x.max() - x.min()))
        df['vix_rank_126'] = df['c'].rolling(126).apply(lambda x: (x.iloc[-1] - x.min()) / (x.max() - x.min()))
        df['vix_rank_30']  = df['c'].rolling(30 ).apply(lambda x: (x.iloc[-1] - x.min()) / (x.max() - x.min()))

        macd = percentage_macd(close=df['c'], fast=12, slow=26, signal=9)
        df['vix_macd']        = macd['MACD_12_26_9']
        df['vix_macd_signal'] = macd['MACDs_12_26_9']
        # ------ ------ #

        # ------ keep the original columns as VIX is stationary ----- #
        for col in price_cols:
            if col in df.columns:
                df[f"vix_{col}_orig"] = df[col]
        # ------ ------ #

        return df

    