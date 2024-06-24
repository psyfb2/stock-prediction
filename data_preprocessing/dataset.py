import logging
from typing import List, Dict, Tuple, Optional, Union, Any
from dateutil.parser import parse
from datetime import timedelta
from multiprocessing import Pool

import pandas as pd
import numpy as np
from torch.utils.data import Dataset

from data_collection.historical_data import get_historical_data, get_vix_daily_data
from data_preprocessing.asset_preprocessor import AssetPreprocessor
from data_preprocessing.vix_preprocessor import VixPreprocessor
from data_preprocessing import labelling
from data_preprocessing.index_mapper import IndexMapper
from utils.date_range import daterange

logger = logging.getLogger(__name__)

# TODO: create a StocksDataset which subclasses IterableDataset. This should not load the whole dataset in memory
#       which is useful when the dataset is so huge it cannot fit in memory
#       on that note, each stock takes up around 2.46MB of RAM 
#       (assuming data from 1998-2024, 26 features, 4 candles to stack, flattened get_full_data_matrix)

class StocksDatasetInMem(Dataset):
    DATASET_FILENAME = "/dataset.csv"
    LENGTHS_FILENAME = "/lengths.npz"

    def __init__(self, tickers: List[List[str]], 
                 sectors: Optional[Dict[str, str]],
                 features_to_use: List[str], 
                 vix_features_to_use: List[str],
                 start_date: str, end_date: str, 
                 label_config: Dict[str, Any],
                 num_candles_to_stack: int,
                 means: Optional[Dict[str, Dict[str, float]]] = None,
                 stds: Optional[Dict[str, Dict[str, float]]] = None,
                 candle_size: str ="1d", procs: Optional[int] = None, 
                 save_path: Optional[str] = None, 
                 load_path: Optional[str] = None,
                 recalculate_labels: bool = False,
                 perform_normalisation: bool = True):
        """ Initialise preprocessed dataset. Performs windowing at inference time to save memory
        (i.e. convert sample shape from (D, ) to (T, D) whenever getting the next data-point.

        Args:
            tickers (List[List[str]]): [[ticker, exchange], ...] for tickers to be included in the dataset
            sectors (Dict[str, str]): maps ticker_exchange to sector. E.g. "AAPL_NASDAQ": "Technology".
                If None, wont add sectors as a feature.
            features_to_use (List[str]): feature names for asset preprocessor
            vix_features_to_use (List[str]): features to use for VIX preprocessor. If None or empty won't
                include VIX data.
            start_date (str): start date for data in yyyy-mm-dd format
            end_date (str): end date for data in yyyy-mm-dd format (exclusive)
            label_config (Dict[str, Any]): config for calculating labels
            num_candles_to_stack (int): number of time-steps for a single data-point
            means (Optional[Dict[str, Dict[str, float]]]): dict mapping each feature name to it's mean.
                Used for normalisation. If not specified will calculate this.
            stds: (Optional[Dict[str, Dict[str, float]]]): dict mapping each feature name to it's std.
                Used for normalisation. If not specified will calculate this.
            candle_size (str): frequency of candles. "1d" or "1h"
            procs (Optional[int]): number of proccesses to use for data preprocessing (will perform map over tickers).
                If None, will use os.cpu_count()
            save_path (Optional[str]): path for where to save files to load later on. If None, will not save files to disk
            load_path (Optional[str]): path for where to load previously saved files. If None, will perform preprocessing
                to load data. If not None, will just load previously preprocessed dataset from disk.
            recalculate_labels (bool): only applies when load_path is not None. Setting this true will ignore
                the loaded labels and re-calculate the labels. This is useful if you are using a different label config
                but the features are the same.
            labels (Optional[np.ndarray]): (N, ) array representing labels. If this is given will ignore
                tickers, start_date, end_date, label_config, means, stds, candle_size and procs, as will treat
                this as the dataset.
            perform_normalisation (bool): normalise the data as part of the preprocessing step?
        """
        super().__init__()
        if load_path is not None:
            data_df = pd.read_csv(load_path + self.DATASET_FILENAME)
            lengths = np.load(load_path + self.LENGTHS_FILENAME)["lengths"]

            logger.info(f"Loaded data_df:\n{data_df}")

            if recalculate_labels and label_config:
                def df_iter():
                    s = 0
                    for i in range(len(lengths)):
                        df = data_df.iloc[s : s + lengths[i]].copy(deep=True).reset_index(drop=True)
                        df = df.dropna().reset_index(drop=True)
                        s += lengths[i]
                        yield df, label_config
                
                with Pool(procs) as p:
                    results = p.starmap(self._calculate_labels, df_iter())
                
                new_dfs = []
                new_lengths = []

                for df in results:
                    df = df.dropna().reset_index(drop=True)

                    if len(df) < num_candles_to_stack:
                        continue
                    
                    new_dfs.append(df)
                    new_lengths.append(len(df))

                data_df = pd.concat(new_dfs, ignore_index=True)
                logger.info(f"all data label counts:\n{data_df['labels'].value_counts(normalize=True)}")
                logger.info(f"difference in lengths: {np.sum(lengths) - np.sum(new_lengths)}")
                lengths = np.array(new_lengths)

            features = data_df.drop(columns=list("toclhv") + ["labels"]).to_numpy(dtype=np.float32)  # (N, D)
            labels   = data_df["labels"].to_numpy(dtype=np.int64)  # (N, )

            # perform some checks on data
            if features.shape[0] != np.sum(lengths):
                raise ValueError(f"Length of features is {features.shape[0]}, but expected it to be {np.sum(lengths)}")
            
            if features.shape[0] != labels.shape[0]:
                raise ValueError(f"Features must have same length as labels ({features.shape[0]} != {labels.shape[0]})")
            
            num_nans = np.count_nonzero(np.isnan(features))
            if num_nans != 0:
                raise ValueError(f"features has {num_nans} NaN elements. This must be zero.")
    
            if not np.all( (labels == 0.0) | (labels == 1.0) ):
                raise ValueError(f"Labels may only contain elements 0 or 1, but it has elements outside this range.")

            self.features = features
            self.labels = labels
            self.lengths = lengths
            self.means = self.stds = None
        else:
            data_df, lengths, means, stds = self.load_dataset_in_mem(
                tickers=tickers, sectors=sectors,
                features_to_use=features_to_use, vix_features_to_use=vix_features_to_use,
                start_date=start_date, end_date=end_date, 
                label_config=label_config, num_candles_to_stack=num_candles_to_stack,
                means=means, stds=stds, candle_size=candle_size, procs=procs,
                perform_normalisation=perform_normalisation
            )
            lengths = np.array(lengths)

            if save_path is not None:
                data_df.to_csv(save_path + self.DATASET_FILENAME, index=False)
                np.savez(save_path + self.LENGTHS_FILENAME, lengths=lengths)

            self.features = data_df.drop(columns=list("toclhv") + ["labels"]).to_numpy(dtype=np.float32)  # (N, D)
            self.labels   = data_df["labels"].to_numpy(dtype=np.int64)  # (N, )
            self.lengths  = lengths  # (len(tickers), )
            self.means    = means
            self.stds     = stds

        unique, counts = np.unique(self.labels, return_counts=True)
        self.label_counts = {l: c / self.labels.shape[0] for l, c in zip(unique, counts)}

        self.num_candles_to_stack = num_candles_to_stack
        self.idx_mapper = IndexMapper(lengths, num_candles_to_stack)

    def __getitem__(self, idx: int):
        true_idx = self.idx_mapper(idx)

        # perform windowing on element at index true_idx, x has shape (num_candles_to_stack, D)
        x = self.features[true_idx - self.num_candles_to_stack + 1: true_idx + 1, :]
        y = self.labels[true_idx]

        return x, y

    def __len__(self):
        return len(self.idx_mapper)

    def get_full_data_matrix(self) -> Tuple[np.ndarray, np.ndarray]:
        """ The feature are stored in a (N, D) matrix.
        Whenever a batch is loaded, a (batch_size, num_candles_to_stack, D) is returned
        so the windowing happens dynamically whenever a batch is loaded. 
        This will save memory for Deep Learning models. 
        
        Some models (e.g. XGBoost) require the whole data matrix to be passed at the
        start of training. So convert (N, D) matrix to (N', num_candles_to_stack * D) matrix and return it,
        also return corrosponding labels.

        Returns:
            Tuple[np.ndarray, np.ndarray]: 
                (N', num_candles_to_stack * D) data matrix,
                (N', ) labels
        """
        Xs = []
        ys = []
        start_idx = 0

        for n in self.lengths:
            Xs.append(self.window_matrix(mat=self.features[start_idx : start_idx + n, :], window_size=self.num_candles_to_stack))
            ys.append(self.labels[start_idx + self.num_candles_to_stack - 1 : start_idx + n])
            start_idx += n
        
        return np.vstack(Xs), np.hstack(ys)
    
    @staticmethod
    def window_matrix(mat: np.ndarray, window_size: int) -> np.ndarray:
        """ Apply sliding window of window_size to mat to go from a (N, D) mat
        to a (N - window_size + 1, window_size * D) mat.

        Args:
            mat (np.ndarray): 2D matrix with shape (N, D)
            window_size (int): sliding window size. Must be less than or equal to N

        Returns:
            np.ndarray: 2D np array with shape (N - window_size + 1, window_size * D)
        """
        if window_size > mat.shape[0]:
            raise ValueError(f"Argument 'window_size' ({window_size}) must be less than or equal to mat.shape[0] ({mat.shape[0]})")
        
        return np.lib.stride_tricks.sliding_window_view(mat.ravel(), window_size * mat.shape[1])[::mat.shape[1]]

    @classmethod
    def load_dataset_in_mem(cls, tickers: List[List[str]], sectors: Optional[Dict[str, str]],
                            features_to_use: List[str],
                            vix_features_to_use: List[str], start_date: str, end_date: str, 
                            label_config: Dict[str, Any], num_candles_to_stack: int,
                            means: Optional[Dict[str, Dict[str, float]]] = None, 
                            stds: Optional[Dict[str, Dict[str, float]]] = None,
                            candle_size="1d", procs: Optional[int] = None,
                            perform_normalisation: bool = True
                            ) -> Tuple[pd.DataFrame, List[int], Dict[str, float], Dict[str, float]]:
        """ Load whole preprocessed dataset. Does not perform windowing. This should
        be done at inference time to save memory.

        Args:
            tickers (List[List[str]]): [[ticker, exchange], ...] for tickers to be included in the dataset
            sectors (Optional[Dict[str, str]]): maps ticker_exchange to sector. E.g. "AAPL_NASDAQ": "Technology".
                If None, wont add sectors as a feature.
            features_to_use (List[str]): feature names for asset preprocessor
            vix_features_to_use (List[str]): features to use for VIX preprocessor. If None or empty won't
                include VIX data.
            start_date (str): start date for data in yyyy-mm-dd format
            end_date (str): end date for data in yyyy-mm-dd format (exclusive)
            label_config (Dict[str, Any]): config for calculating labels
            num_candles_to_stack (int): number of time-steps for a single data-point
            means (Optional[Dict[str, Dict[str, float]]]): dict mapping each feature name to it's mean.
                Used for normalisation. If not specified will calculate this.
            stds: (Optional[Dict[str, Dict[str, float]]]): dict mapping each feature name to it's std.
                Used for standardisation. If not specified will calculate this.
            candle_size (str): frequency of candles. "1d" or "1h"
            procs (Optional[int]): number of proccesses to use for data preprocessing (will perform map over tickers).
                If None, will use os.cpu_count()
            perform_normalisation (bool): normalise the data as part of the preprocessing step?
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
        preprocessor = AssetPreprocessor(features_to_use=features_to_use, candle_size=candle_size)

        calc_means = means is None or stds is None
        if calc_means:
            means = {}
            stds = {}

        with Pool(procs) as p:
            results = p.starmap(cls._preprocess_ticker, ((ticker, exchange, sectors[f"{ticker}_{exchange}"] if sectors else None, 
                                                          start_date, end_date, preprocessor,
                                                          num_candles_to_stack, label_config, candle_size) 
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

        if calc_means:
            means["asset"] = dict(data_df.mean(numeric_only=True))
            stds["asset"]  = dict(data_df.std(numeric_only=True))
        
        if perform_normalisation:
            preprocessor.normalise_df(data_df, means=means["asset"], stds=stds["asset"])
        
        if sectors:
            if "sector_categorical" in features_to_use:
                # convert from string to categorical
                data_df["sector_categorical"], uniques = pd.factorize(data_df["sector"])
                logger.info(f"Added sector_categorical feature. Num sectors: {len(uniques)}, they are {uniques}")

            data_df = data_df.drop("sector", axis=1)

        num_nans = data_df.isna().sum().sum()
        assert num_nans == 0, f"Expected Number of NaNs prior to loading VIX to be 0, but was {num_nans}. NaNs per col:\n{dict(data_df.isna().sum())}"

        # add VIX data as broad market sentiment indicators
        if vix_features_to_use:
            vix_preprocessor = VixPreprocessor(features_to_use=vix_features_to_use)
            data_df = cls.merge_vix_data(
                vix_preprocessor=vix_preprocessor, 
                data_df=data_df,
                calc_means=calc_means,
                means=means,
                stds=stds,
                num_candles_to_stack=num_candles_to_stack,
                perform_normalisation=perform_normalisation
            )

        assert len(data_df) == sum(lengths), f"length of data_df is {len(data_df)}, but expected it to be {sum(lengths)}"
        assert ( (data_df["labels"] == 0.0) | (data_df["labels"] == 1.0) ).all()

        num_nans = data_df.isna().sum().sum()
        assert num_nans == 0, f"Expected Number of NaNs in finalised data_df to be 0, but was {num_nans}. NaNs per col:\n{dict(data_df.isna().sum())}"

        return data_df, lengths, means, stds
    
    @staticmethod
    def merge_vix_data(vix_preprocessor: VixPreprocessor, 
                       data_df: pd.DataFrame,
                       calc_means: bool, 
                       means: Dict[str, Dict[str, float]],
                       stds: Dict[str, Dict[str, float]],
                       num_candles_to_stack: int,
                       perform_normalisation: bool = True, 
                       ) -> pd.DataFrame:
        """ Merge preprocessed df with VIX data

        Args:
            vix_preprocessor (VixPreprocessor): vix preprocessor
            data_df (pd.DataFrame): preprocessed dataframe. must contain 't' column
                at a daily interval.
            calc_means (bool): calculate means? If False expects means["vix"]
                and stds["vix"] to contain means and stds. otherwise will calculate
                these and add key "vix" to means and stds.
            means (Dict[str, Dict[str, float]]): means dict
            stds (Dict[str, Dict[str, float]]): stds dict
            num_candles_to_stack (int): number of candles to stack. 
                Used to cut VIX df to only preprocess the required rows.
            perform_normalisation (bool): normalise VIX df?
        Returns:
            pd.DataFrame: VIX data merged with data_df. Columns of data_df
                are kept in order.
        """
        vix_df = get_vix_daily_data("VIX")

        adjusted_start_date = vix_preprocessor.adjust_start_date(
            data_df["t"].iloc[0].to_pydatetime(), num_candles_to_stack
        ).strftime("%Y-%m-%d")
        vix_df = vix_df[vix_df["t"] >= adjusted_start_date].reset_index(drop=True)

        vix_df = vix_preprocessor.preprocess_ochl_df(vix_df).drop(columns=["o", "c", "l", "h", "v"])

        if calc_means:
            means["vix"] = dict(vix_df.mean(numeric_only=True))
            stds["vix"]  = dict(vix_df.std(numeric_only=True))

        if perform_normalisation:
            vix_preprocessor.normalise_df(vix_df, means=means["vix"], stds=stds["vix"])

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

        return data_df
    
    @classmethod
    def preprocess_ticker(cls, ticker: str, exchange: str, start_date: str, end_date: str, 
                          preprocessor: AssetPreprocessor, num_candles_to_stack: int, 
                          label_config: Dict[str, Any], candle_size="1d",
                          raise_invalid_data_exception=False) -> pd.DataFrame:
        """ Load preprocessed data for a ticker. Will not merge VIX data or perform normalisation.

        Args:
            ticker (str): ticker symbol to preprocess
            exchange (str): exchange for ticker. If empty string will use main exchange.
            start_date (str): start date for data in yyyy-mm-dd format
            end_date (str): end date for data in yyyy-mm-dd format (exclusive)
            preprocessor (AssetPreprocessor): preprocessor to use
            num_candles_to_stack (int): number of time-steps for a single data-point
            label_config (Dict[str, Any]): config for calculating labels
            candle_size (str): frequency of candles. "1d" or "1h". Defaults to "1d"
            raise_invalid_data_exception (bool): when getting historical data from API, check that
                there are no missing candles between start_date and end_date. If there is then
                throw an exception if this arg is True. Otherwise just warn.
        Returns:
            pd.DataFrame: preprocessed df
        """
        adjusted_start_date = preprocessor.adjust_start_date(
            parse(start_date), num_candles_to_stack
        ).strftime("%Y-%m-%d")

        df = get_historical_data(symbol=ticker, start_date=adjusted_start_date, 
                                 end_date=end_date, candle_size=candle_size, 
                                 exchange=exchange, raise_invalid_data_exception=raise_invalid_data_exception)

        df = preprocessor.preprocess_ochl_df(df)

        # the first candle after performing stacking should have the first date after or including start_date
        df_after_start_date = df[df["t"] >= start_date]

        if len(df_after_start_date) == 0:
            raise ValueError(f"Not enough data, df has 0 rows after start_date '{start_date}', will skip this ticker")

        start_date_idx = df_after_start_date.index[0]
        if start_date_idx > num_candles_to_stack - 1:
            # cut left part of df which goes before start_date even after stacking
            df = df.iloc[start_date_idx - num_candles_to_stack + 1:].reset_index(drop=True)
        else:
            logger.warning(f"Ticker {ticker} data does not go back to {adjusted_start_date}. start_date_idx={start_date_idx}, "
                           f"but it should be atleast {num_candles_to_stack - 1}.")

        cls._calculate_labels(df=df, label_config=label_config)

        # drop any rows with NaN or inf 
        original_len = len(df)
        df = df.replace([np.inf, -np.inf], np.nan)
        df = df.dropna().reset_index(drop=True)
        if original_len != len(df):
            logger.warning(f"df had {original_len - len(df)} NaN or inf elements. These have been dropped.")

        if len(df) < num_candles_to_stack:
            raise ValueError(f"ticker '{ticker}' with start_date={start_date}, end_date={end_date} and candle_size={candle_size} "
                             f"only has {len(df)} candles after preprocessing. This is less than num_candles_to_stack={num_candles_to_stack}.")

        return df

    @staticmethod
    def _calculate_labels(df: pd.DataFrame, label_config: Optional[Dict[str, Any]]) -> pd.DataFrame:
        if label_config is not None:
            label_func = getattr(labelling, label_config["label_function"])
            df["labels"] = label_func(df, **label_config["kwargs"])

            logger.info(f"label counts:\n{df['labels'].value_counts(normalize=True)}")
            logger.info(f"labels has {(df['labels'].isna().sum() / len(df)) * 100}% NaNs, these will be removed.")
        
        return df

    @classmethod
    def _preprocess_ticker(cls, ticker: str, exchange: str, sector: Optional[str], 
                           start_date: str, end_date: str, preprocessor: AssetPreprocessor,
                           num_candles_to_stack: int, label_config: Dict[str, Any], candle_size="1d"
                           ) -> Union[pd.DataFrame, str]:
        try:
            df = cls.preprocess_ticker(
                ticker=ticker, 
                exchange=exchange, 
                start_date=start_date, 
                end_date=end_date, 
                preprocessor=preprocessor, 
                num_candles_to_stack=num_candles_to_stack, 
                label_config=label_config,
                candle_size=candle_size
            )
            if sector:
                df["sector"] = sector
            return df
        except Exception as ex:
            logger.exception(f"Error while preprocessing ticker '{ticker}'.")
            return f"Error while preprocessing ticker '{ticker}': {ex}"