import pandas as pd
import pandas_ta as ta
import numpy as np

from logging import getLogger

from custom_ti import percentage_macd, normalised_return
from data_preprocessing.base_preprocessor import BasePreprocessor


class AssetPreprocessor(BasePreprocessor):
    def __init__(self, candle_size="1d"):
        """ Initialise a preprocessor for any asset with columns ['t', 'o', 'c', 'h', 'l', 'v']. 
        Args:
            candle_size (str): frequency of candles. Either "1d" or "1h".
        """
        super().__init__(
            logger=getLogger(__name__), 
            bounded_cols={
                "rsi":               (0, 100),
                "stoch_rsi":         (0, 100),
                "mfi":               (0, 100),
                "adx":               (0, 100),
                "dmp":               (0, 100),
                "dmn":               (0, 100),
                "stochk":            (0, 100),
                "stochd":            (0, 100),
                "aroond":            (0, 100),
                "aroonu":            (0, 100),
                "uo":                (0, 100),
                "percentile":        (0, 1),
                "percentile_126":    (0, 1),
                "percentile_30":     (0, 1),
                "rank":              (0, 1),
                "rank_126":          (0, 1),
                "rank_30":           (0, 1),
                "hour_of_day_sin":   (-1, 1),
                "hour_of_day_cos":   (-1, 1),
                "day_of_month_sin":  (-1, 1),
                "day_of_month_cos":  (-1, 1),
                "month_of_year_sin": (-1, 1),
                "month_of_year_cos": (-1, 1)
            }
        )

    def calc_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """ Calculate unnormalised features.

        Args:
            df (pd.DataFrame): df containing ["t", "o", "c", "h", "l", "v"] columns

        Returns:
            pd.DataFrame: df containing original unchanged columns + feature columns
        """
        price_cols = ["o", "c", "l", "h", "v"]

        if set(df.columns) != set(price_cols + ['t']):
            raise ValueError(f"df must have column ['t', 'o', 'c', 'l', 'h', 'v'], got the following columns instead:\n{df.columns}")

        df = df.copy()
        
        # ------ apply TI's ------ #
        df['rsi'] = ta.rsi(close=df['c'], length=14)

        stoch_rsi = ta.stochrsi(close=df['c'], length=14)
        df['stoch_rsi'] = stoch_rsi['STOCHRSId_14_14_3_3']

        macd = percentage_macd.percentage_macd(close=df['c'], fast=12, slow=26, signal=9)
        df['macd'] = macd['MACD_12_26_9']
        df['macd_signal'] = macd['MACDs_12_26_9']

        adx = ta.adx(high=df['h'], low=df['l'], close=df['c'], length=14) 
        df['adx'] = adx['ADX_14']
        df['dmp'] = adx['DMP_14']
        df['dmn'] = adx['DMN_14']

        stoch = ta.stoch(high=df['h'], low=df['l'], close=df['c'], k=14, d=3, smooth_k=3)  
        df['stochk'] = stoch['STOCHk_14_3_3']
        df['stochd'] = stoch['STOCHd_14_3_3']

        aroon = ta.aroon(high=df['h'], low=df['l'], length=25)  
        df['aroond'] = aroon['AROOND_25']
        df['aroonu'] = aroon['AROONU_25']

        df['mfi'] = ta.mfi(high=df['h'], low=df['l'], close=df['c'], volume=df['v'], length=14)

        df['uo'] = ta.uo(high=df['h'], low=df['l'], close=df['c'], fast=7, medium=14, slow=28)

        df['percentile']     = df['c'].rolling(252).apply(lambda x: x[x <= x.iloc[-1]].count() / 252)
        df['percentile_126'] = df['c'].rolling(126).apply(lambda x: x[x <= x.iloc[-1]].count() / 126)
        df['percentile_30']  = df['c'].rolling(30 ).apply(lambda x:  x[x <= x.iloc[-1]].count() / 30)

        df['rank']     = df['c'].rolling(252).apply(lambda x: (x.iloc[-1] - x.min()) / (x.max() - x.min()))
        df['rank_126'] = df['c'].rolling(126).apply(lambda x: (x.iloc[-1] - x.min()) / (x.max() - x.min()))
        df['rank_30']  = df['c'].rolling(30 ).apply(lambda x: (x.iloc[-1] - x.min()) / (x.max() - x.min()))

        df['massi'] = ta.massi(high=df['h'], low=df['l'], fast=9, slow=25)

        bbands = ta.bbands(close=df['c'], length=20, std=2) 
        df['bbl'] = (df['c'] / bbands['BBL_20_2.0']) - 1  # % increase of c from bbl
        df['bbm'] = (df['c'] / bbands['BBM_20_2.0']) - 1  # % increase of c from bbm
        df['bbu'] = (df['c'] / bbands['BBU_20_2.0']) - 1  # % increase of c from bbu

        df['ema10']  = (df['c'] / ta.ema(df['c'], length=10))  - 1  # % increase of c from ema10
        df['ema20']  = (df['c'] / ta.ema(df['c'], length=20))  - 1  # % increase of c from ema10
        df['ema50']  = (df['c'] / ta.ema(df['c'], length=50))  - 1  # % increase of c from ema50
        df['ema100'] = (df['c'] / ta.ema(df['c'], length=100)) - 1  # % increase of c from ema100
        df['ema200'] = (df['c'] / ta.ema(df['c'], length=200)) - 1  # % increase of c from ema200

        df['sma50']  = (df['c'] / ta.sma(df['c'], length=50))  - 1  # % increase of c from sma50
        df['sma100'] = (df['c'] / ta.sma(df['c'], length=100)) - 1  # % increase of c from sma100
        df['sma200'] = (df['c'] / ta.sma(df['c'], length=200)) - 1  # % increase of c from sma200

        df['atr'] = ta.atr(high=df['h'], low=df['l'], close=df['c'], length=14, percent=True)

        v14 = ta.sma(df['v'], length=14)
        v28 = ta.sma(df['v'], length=28)
        df['vo'] = (v14 / v28) - 1

        for col in price_cols:
            returns = normalised_return.normalised_returns(df[col])
            df[f'{col}_perc_increase'] = returns

            if col == 'c':
                df[f'{col}_perc_increase_ema10']  = ta.ema(returns, length=10)
                df[f'{col}_perc_increase_ema50']  = ta.ema(returns, length=50)
                df[f'{col}_perc_increase_ema100'] = ta.ema(returns, length=100)
                df[f'{col}_perc_increase_ema200'] = ta.ema(returns, length=200)
        
        df['l_o_perc_increase'] = (df['o'] / df['l']) - 1 # % increase from low to open
        df['o_c_perc_increase'] = (df['c'] / df['o']) - 1 # % increase from open to close
        df['c_h_perc_increase'] = (df['h'] / df['c']) - 1 # % increase from close to high
        # ------ ------ #

        # ------ Add Time Features ------ #
        if (df['t'].iloc[-1] - df['t'].iloc[0]).days < 1:
            raise ValueError(f"df must have atleast one days worth of data. df dates:\n{df['t']}")

        hours = df['t'].apply(lambda t: t.hour)
        min_hour = hours.min()
        max_hour = hours.max()
        self._logger.info(f"min_hour={min_hour}, max_hour={max_hour}")

        # if using daily data or above, hourly features should be excluded
        if min_hour != max_hour and self.candle_size != "1d":  
            hours_from_zero = df['t'].apply(lambda t: t.hour - min_hour)
            df["hour_of_day_sin"] = np.sin(hours_from_zero * (2 * np.pi / (max_hour - min_hour + 1)))
            df["hour_of_day_cos"] = np.cos(hours_from_zero * (2 * np.pi / (max_hour - min_hour + 1)))

        day_of_month = df['t'].apply(lambda t: t.day - 1)
        df["day_of_month_sin"] = np.sin(day_of_month * (2 * np.pi / 31))
        df["day_of_month_cos"] = np.cos(day_of_month * (2 * np.pi / 31))

        month_of_year = df['t'].apply(lambda t: t.month - 1)
        df["month_of_year_sin"] = np.sin(month_of_year * (2 * np.pi / 12))
        df["month_of_year_cos"] = np.cos(month_of_year * (2 * np.pi / 12))
        # ------ ------#

        return df

