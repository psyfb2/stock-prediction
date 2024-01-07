import pandas as pd

from logging import Logger
from dateutil.parser import parse
from datetime import timedelta
from warnings import warn
from typing import Dict, Tuple
from abc import ABC, abstractmethod


class BasePreprocessor(ABC):
    def __init__(self, logger: Logger, bounded_cols: Dict[str, Tuple[float, float]], candle_size="1d"):
        """ Create a preprocessor.
        Args:
            logger (Logger): logger object
            bounded_cols (Dict[str, Tuple[float, float]]): mapping for bounded features.
                all features are assumed to be stationary.
                the bounded ones are normalised using min-max scaling to be in [-1, 1]
                otherwise standardisation is used.
            candle_size (str): frequency of candles. "1d" or "1h" 
        """
        self._logger = logger
        self.bounded_cols = bounded_cols
        self.candle_size = candle_size

        if candle_size not in ("1d", "1h"):
            raise ValueError(f"candle_size of '{candle_size}' is not supported.")

        # get minimum number of rows required to preprocess a df
        date = parse("2008-01-01")
        if candle_size == "1h":
            dummy_df = pd.DataFrame([{'t': date + timedelta(hours=i), 'o': i, 'c': i, 'h': i, 'l': i, "v": i} for i in range(3000)])
        else:
            dummy_df = pd.DataFrame([{'t': date + timedelta(days=i), 'o': i, 'c': i, 'h': i, 'l': i, "v": i} for i in range(3000)])
        original_len = len(dummy_df.index)
        dummy_df = self.calc_features(dummy_df).dropna().reset_index(drop=True)
        self.min_rows_required = original_len - len(dummy_df.index)

    @abstractmethod
    def calc_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """ Calculate unnormalised features.

        Args:
            df (pd.DataFrame): df containing ["t", "o", "c", "h", "l", "v"]

        Returns:
            pd.DataFrame: df containing original unchanged columns + feature columns
        """
        pass
    
    def preprocess_ochl_df(self, df: pd.DataFrame) -> pd.DataFrame:
        """ Preprocess price dataframe by adding TI's (doesn't normalise features).
        Args:
            df (pd.DataFrame): DataFrame containing ["t", "o", "c", "h", "l", "v"] columns or just ["t", "c"] columns

        Returns:
            (pd.DataFrame): preprocessed df. Original columns in df are left unchanged
                within preprocessed df (i.e. will contain tochlv columns with unchanged values).
        """
        # ------ perform some checks on df ------ #
        if df.isnull().values.any():
            warn(f"df contains NaN values, rows with NaN values will be dropped! df:\n{df}")
            df = df.dropna().reset_index(drop=True)

        if len(df) < self.min_rows_required:
            raise ValueError(f"df must have atleast {self.min_rows_required} rows, got the following df instead:\n{df}")
        
        if set(df.columns) not in ({"t", "o", "c", "h", "l", "v"}, {"t", "c"}):
            raise ValueError(f'df must have columns ["t", "o", "c", "h", "l", "v"] or ["t", "c"]. df has columns:\n{df.columns}')

        self._logger.info(f"Initiating preprocesing of df:\n{df}")
        # ------ ------ #

        # apply TI's
        df = self.calc_features(df)
        
        # drop NaN values produced by TI's
        df = df.dropna().reset_index(drop=True)

        self._logger.info(f"Calculated pre-processed df:\n{df}")
        return df
    
    def normalise_df(self, df: pd.DataFrame, means: Dict[str, float], stds: Dict[str, float]) -> pd.DataFrame:
        """ Normalise dataframe. If a column is bounded will use min-max scaling otherwise will
        use standardisation. Also ["t", "o", "c", "h", "l", "v"] columns are left unchanged.

        Args:
            df (pd.DataFrame): preprocessed df. Contains un-normalised features from preprocess_df method.

        Returns:
            pd.DataFrame: normalised dataframe.
        """
        for col in df.columns:
            # original columns should be unchanged (unnormalised)
            if col in ["t", "o", "c", "h", "l", "v"]: continue

            if col in self.bounded_cols:
                # perform min-max scaling to put in range [-1, 1]
                low, high = self.bounded_cols[col]
                df[col] = 2 * ((df[col] - low) / (high - low)) - 1
            else:
                # standardise
                df[col] = (df[col] - means[col]) / stds[col]
