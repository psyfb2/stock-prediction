import logging
from typing import List, Dict, Tuple, Optional, Union
from dateutil.parser import parse
from datetime import timedelta
from multiprocessing import Pool

import pandas as pd
import numpy as np
from torch.utils.data import Dataset

from data_collection.historical_data import get_historical_data, get_vix_daily_data, validate_historical_data
from data_preprocessing.asset_preprocessor import AssetPreprocessor
from data_preprocessing.vix_preprocessor import VixPreprocessor
from data_preprocessing.labelling import binary_label_tp_tsl
from data_preprocessing.index_mapper import IndexMapper
from utils.date_range import daterange

logger = logging.getLogger(__name__)

# TODO: create a StocksDataset which subclasses IterableDataset. This should not load the whole dataset in memory
#       which is useful when the dataset is so huge it cannot fit in memory

class StocksDatasetInMem(Dataset):
    def __init__(self, tickers: List[List[str]], start_date: str, end_date: str, 
                 tp: float, tsl: float, num_candles_to_stack: int, 
                 means: Optional[Dict[str, float]] = None, stds: Optional[Dict[str, float]] = None,
                 candle_size: str ="1d", procs: Optional[int] = None):
        """ Initialise preprocessed dataset. Performs windowing at inference time to save memory
        (i.e. convert sample shape from (D, ) to (T, D) whenever getting the next data-point.

        Args:
            tickers (List[List[str]]): [[ticker, exchange], ...] for tickers to be included in the dataset
            start_date (str): start date for data in yyyy-mm-dd format
            end_date (str): end date for data in yyyy-mm-dd format (exclusive)
            tp (float): take profit value used for labelling (e.g. 0.05 for 5% take profit)
            tsl (float): trailing stop loss value used for labelling (e.g. 0.05 for 5% trailing stop loss)
            num_candles_to_stack (int): number of time-steps for a single data-point
            means (Optional[Dict[str, float]]): dict mapping each feature name to it's mean.
                Used for normalisation. If not specified will calculate this.
            stds: (Optional[Dict[str, float]]): dict mapping each feature name to it's std.
                Used for normalisation. If not specified will calculate this.
            candle_size (str): frequency of candles. "1d" or "1h"
            procs (Optional[int]): number of proccesses to use for data preprocessing (will perform map over tickers).
                If None, will use os.cpu_count()
        """
        super().__init__()
        data_df, lengths, means, stds = self.load_dataset_in_mem(
            tickers=tickers, start_date=start_date, end_date=end_date, 
            tp=tp, tsl=tsl, num_candles_to_stack=num_candles_to_stack, 
            means=means, stds=stds, candle_size=candle_size, procs=procs
        )

        self.features = data_df.drop(columns=list("toclhv") + ["labels"]).to_numpy(dtype=np.float32)  # (N, D)
        self.labels   = data_df["labels"].to_numpy(dtype=np.int64)  # (N, )
        self.lengths  = np.array(lengths)  # (len(tickers), )
        self.num_candles_to_stack = num_candles_to_stack
        self.means = means
        self.stds = stds
        self.idx_mapper = IndexMapper(lengths, num_candles_to_stack)

    def __getitem__(self, idx: int):
        true_idx = self.idx_mapper(idx)

        # perform windowing on element at index true_idx, x has shape (num_candles_to_stack, D)
        x = self.features[true_idx - self.num_candles_to_stack + 1: true_idx + 1, :]
        y = self.labels[true_idx]

        return x, y

    def __len__(self):
        return len(self.idx_mapper)

    @classmethod
    def load_dataset_in_mem(cls, tickers: List[List[str]], start_date: str, end_date: str, 
                            tp: float, tsl: float, num_candles_to_stack: int, 
                            means: Optional[Dict[str, float]] = None, 
                            stds: Optional[Dict[str, float]] = None,
                            candle_size="1d", procs: Optional[int] = None
                            ) -> Tuple[pd.DataFrame, List[int], Dict[str, float], Dict[str, float]]:
        """ Load whole preprocessed dataset. Does not perform windowing. This should
        be done at inference time to save memory.

        Args:
            tickers (List[List[str]]): [[ticker, exchange], ...] for tickers to be included in the dataset
            start_date (str): start date for data in yyyy-mm-dd format
            end_date (str): end date for data in yyyy-mm-dd format (exclusive)
            tp (float): take profit value used for labelling (e.g. 0.05 for 5% take profit)
            tsl (float): trailing stop loss value used for labelling (e.g. 0.05 for 5% trailing stop loss)
            num_candles_to_stack (int): number of time-steps for a single data-point
            means (Optional[Dict[str, float]]): dict mapping each feature name to it's mean.
                Used for normalisation. If not specified will calculate this.
            stds: (Optional[Dict[str, float]]): dict mapping each feature name to it's std.
                Used for standardisation. If not specified will calculate this.
            candle_size (str): frequency of candles. "1d" or "1h"
            procs (Optional[int]): number of proccesses to use for data preprocessing (will perform map over tickers).
                If None, will use os.cpu_count()
        Returns:
            (Tuple[pd.DataFrame, List[int], Dict[str, float], Dict[str, float]]):
                (data_df, lengths, means, stds).
                data_df: df containing original unchanged ['t','o','c','l','h','v'] columns 
                    "labels" columns which has the true labels, all other columns
                    represent features.
                lengths: number of rows each ticker has within data_df, in order.
                means: means used for standardisation.
                stds:  stds used for standardisation.
        """
        preprocessor = AssetPreprocessor(candle_size=candle_size)

        with Pool(procs) as p:
            results = p.starmap(cls._preprocess_ticker, ((ticker, exchange, start_date, end_date, preprocessor,
                                                         num_candles_to_stack, tp, tsl, candle_size) 
                                                        for ticker, exchange in tickers))
            
        all_dfs = []
        lengths = []
        for res in results:
            if isinstance(res, pd.DataFrame):
                all_dfs.append(res)
                lengths.append(len(res))
            else:
                logger.warning(res)
        
        data_df = pd.concat(all_dfs, ignore_index=True)
        logger.info(f"all data label counts:\n{data_df['labels'].value_counts(normalize=True)}")

        num_nans = data_df.isna().sum().sum()
        assert num_nans == 0, f"Expected Number of NaNs prior to loading VIX to be 0, but was {num_nans}. NaNs per col:\n{dict(data_df.isna().sum())}"

        # add VIX data as broad market sentiment indicators
        vix_preprocessor = VixPreprocessor()

        vix_df = get_vix_daily_data("VIX")
        vix_df = vix_preprocessor.preprocess_ochl_df(vix_df).drop(columns=["o", "c", "l", "h"])
        vix_df = vix_df.rename(columns={col : f"vix_{col}" for col in vix_df.columns if col != 't'})
        preprocessor.bounded_cols.update({f"vix_{k}": v for k, v in vix_preprocessor.bounded_cols.items()})

        # fill VIX df (might be missing some days, just fill with previous value)
        last_idx = 0
        vix_df_filled = pd.DataFrame(columns=list(vix_df.columns))
        for date in daterange(vix_df["t"].iloc[0], vix_df["t"].iloc[-1] + timedelta(days=1)):
            if date == vix_df["t"].iloc[last_idx]:
                vix_df_filled = pd.concat([vix_df_filled, vix_df.iloc[last_idx:last_idx + 1]], ignore_index=True)
                last_idx += 1
            else:
                row = vix_df.iloc[last_idx - 1:last_idx].copy()
                row["t"] = date
                vix_df_filled = pd.concat([vix_df_filled, row], ignore_index=True)
        vix_df = vix_df_filled

        unmerged_data_df = data_df
        data_df = data_df.merge(vix_df, on="t", how="left")
        assert all(unmerged_data_df["t"] == data_df["t"]), f"Merging has not preserved order of rows!"

        num_nans = data_df.isna().sum().sum()
        assert num_nans == 0, f"Expected Number of NaNs prior to normalisation to be 0, but was {num_nans}. NaNs per col:\n{dict(data_df.isna().sum())}"
        
        # normalise data
        if not means or not stds:
            means = data_df.mean(numeric_only=True)
            logger.info(f"Calulated means:\n{dict(means)}")

            stds  = data_df.std(numeric_only=True)
            logger.info(f"Calulated stds:\n{dict(stds)}")
            
        preprocessor.normalise_df(data_df, means, stds)

        assert len(data_df) == sum(lengths), f"length of data_df is {len(data_df)}, but expected it to be {sum(lengths)}"
        assert ( (data_df["labels"] == 0.0) | (data_df["labels"] == 1.0) ).all()

        num_nans = data_df.isna().sum().sum()
        assert num_nans == 0, f"Expected Number of NaNs in finalised data_df to be 0, but was {num_nans}. NaNs per col:\n{dict(data_df.isna().sum())}"

        return data_df, lengths, dict(means), dict(stds)
    
    @staticmethod
    def preprocess_ticker(ticker: str, exchange: str, start_date: str, end_date: str, 
                          preprocessor: AssetPreprocessor, num_candles_to_stack: int, 
                          tp: Optional[float] = None, tsl: Optional[float] = None, candle_size="1d"
                          ) -> pd.DataFrame:
        """ Load preprocessed data for a ticker.

        Args:
            ticker (str): ticker symbol to preprocess
            exchange (str): exchange for ticker. If empty string will use main exchange.
            start_date (str): start date for data in yyyy-mm-dd format
            end_date (str): end date for data in yyyy-mm-dd format (exclusive)
            preprocessor (AssetPreprocessor): preprocessor to use
            num_candles_to_stack (int): number of time-steps for a single data-point
            tp (Optional[float]): take profit value used for labelling (e.g. 0.05 for 5% take profit).
                If None, wont perform labelling ("labels" col wont be present in returned df). Defaults to None.
            tsl (Optional[float]): trailing stop loss value used for labelling (e.g. 0.05 for 5% trailing stop loss)
                If None, wont perform labelling ("labels" col wont be present in returned df). Defaults to None
            candle_size (str): frequency of candles. "1d" or "1h". Defaults to "1d"

        Returns:
            pd.DataFrame: preprocessed df
        """
        adjusted_start_date = preprocessor.adjust_start_date(
            parse(start_date), num_candles_to_stack
        ).strftime("%Y-%m-%d")

        df = get_historical_data(symbol=ticker, start_date=adjusted_start_date, 
                                 end_date=end_date, candle_size=candle_size, exchange=exchange)

        df = preprocessor.preprocess_ochl_df(df)

        # the first candle after performing stacking should have the first date after or including start_date
        df_after_start_date = df[df["t"] >= start_date]

        if len(df_after_start_date) == 0:
            raise ValueError(f"Not enough data, df has 0 rows after start_date '{start_date}', will skip this ticker")

        start_date_idx = df_after_start_date.index[0]
        if start_date_idx >= num_candles_to_stack - 1:
            # cut left part of df which goes before start_date even after stacking
            df = df.iloc[start_date_idx - num_candles_to_stack + 1:].reset_index(drop=True)
        else:
            logger.warning(f"Data is not complete. start_date_idx={start_date_idx}, "
                           f"but it should be atleast {num_candles_to_stack - 1}. Check data-collection.")

        # calculate labels
        if tp is not None and tsl is not None:
            df["labels"] = binary_label_tp_tsl(df, tp, tsl)
            logger.info(f"label counts:\n{df['labels'].value_counts(normalize=True)}")
            logger.info(f"labels has {(df['labels'].isna().sum() / len(df)) * 100}% NaNs, these will be removed.")

            nan_indicies = list(df.loc[pd.isna(df['labels']), :].index)
            if not all([nan_indicies[i + 1] == nan_indicies[i] + 1 for i in range(len(nan_indicies) - 1)]):
                logger.warning(f"NaN indicies not contigious: {nan_indicies}")

            df = df.dropna().reset_index(drop=True)

        if len(df) < num_candles_to_stack:
            raise ValueError(f"ticker '{ticker}' with start_date={start_date}, end_date={end_date} and candle_size={candle_size} "
                             f"only has {len(df)} candles after preprocessing. This is less than num_candles_to_stack={num_candles_to_stack}.")

        return df

    @classmethod
    def _preprocess_ticker(cls, ticker: str, exchange: str, start_date: str, 
                           end_date: str, preprocessor: AssetPreprocessor,
                           num_candles_to_stack: int, tp: Optional[float] = None, 
                           tsl: Optional[float] = None, candle_size="1d") -> Union[pd.DataFrame, str]:
        try:
            return cls.preprocess_ticker(ticker, exchange, start_date, end_date, preprocessor, 
                                         num_candles_to_stack, tp, tsl, candle_size)
        except ValueError as ex:
            return f"Error while preprocessing ticker '{ticker}': {ex}"