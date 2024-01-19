import os
import logging
import requests
import json
from datetime import datetime, timedelta, date
from typing import List

import pandas as pd
import yfinance as yf
import pandas_market_calendars as mcal
import bs4 as bs
from dateutil.parser import parse

logger = logging.getLogger(__name__)
CACHE_DIR = os.path.join(os.path.dirname(__file__), "file_cache")


def _num_days_overlap(start1: datetime, end1: datetime, start2: datetime, end2: datetime) -> int:
    """ Calculate number of days overlap between two date ranges.

    Args:
        start1 (datetime): first start date
        end1 (datetime): first end date
        start2 (datetime): second start date
        end2 (datetime): second end date

    Returns:
        int: number of days overlap. This return 0 if there is no overlap.
    """
    latest_start = max(start1, start2)
    earliest_end = min(end1, end2)
    delta = (earliest_end - latest_start).days + 1
    return max(delta, 0)


def _args_to_filename(symbol: str, start_date: str, end_date: str, 
                      candle_size, exchange, outside_rth) -> str:
    if isinstance(start_date, datetime):
        start_date = start_date.strftime("%Y-%m-%d")
    else:
        start_date = start_date.replace("/", "-")
    
    if isinstance(end_date, datetime):
        end_date = end_date.strftime("%Y-%m-%d")
    else: 
        end_date = end_date.replace("/", "-")

    return f"{exchange}_{symbol}_{candle_size}_{start_date}_{end_date}_{outside_rth}.csv"


def _cached_file_with_most_overlap(symbol: str, start_date: str, end_date: str, 
                                   candle_size: str, exchange: str, outside_rth: bool) -> str:
    """ Get filename in file_cache folder which matches symbol, candle_size, exchange, outside_rth
    and has the highest date overlap.

    Args:
        symbol (str): symbol (e.g. AAPL)
        start_date (str): start date string
        end_date (str): end date string
        candle_size (str): candle size to use. Defaults to '1h'.
        exchange (str): exchange to use.
        outside_rth (bool): get outside regular trading hours data too?.

    Returns:
        str: filename with most overlap. None if no files overlap.
    """
    filename_most_overlap, most_overlap = None, 0
    start_date, end_date = parse(start_date), parse(end_date)

    for filename in os.listdir(CACHE_DIR):
        try:
            exchange2, symbol2, candle_size2, start_date2, end_date2, outside_rth2 = filename.split(".")[0].split("_")
        except (ValueError, IndexError) as ex:
            # file is not in the correct format, could be another file (e.g. .gitkeep)
            continue

        outside_rth2 = outside_rth2 == "True"

        if exchange != exchange2 or symbol != symbol2 or candle_size != candle_size2 or outside_rth != outside_rth2:
            continue

        start_date2, end_date2 = parse(start_date2), parse(end_date2) 
        days_overlap = _num_days_overlap(start_date, end_date, start_date2, end_date2)

        if days_overlap > most_overlap:
            most_overlap = days_overlap
            filename_most_overlap = filename
    return filename_most_overlap


def get_historical_data(symbol: str, start_date: str, end_date: str, 
                        candle_size='1d', exchange='', outside_rth=False,
                        raise_invalid_data_exception=False)  -> pd.DataFrame:
    """ Get historical data for given stock. Uses local file cache to reduce number of API calls.

    Args:
        symbol (str): symbol (e.g. AAPL)
        start_date (str): start date in yyyy-mm-dd format, inclusive
        end_date (str): end date in yyyy-mm-dd format, exclusive
        candle_size (str, optional): candle size to use. Defaults to '1d'.
        exchange (str, optional): exchange to use, by default will use primary exchange for symbol.
        outside_rth (bool, optional): get outside regular trading hours data too? Defaults to False.
        raise_invalid_data_exception (bool, optional): raise an exception if requested data contains
            missing rows? If False will just log a warning.
    Returns:
        pd.Dataframe: dataframe containing ['t', 'o', 'c', 'h', 'l', 'v'] columns. 't' column relative to UTC.
    """
    logger.info(f"Getting historical data for symbol='{symbol}', start_date='{start_date}', "
                f"end_date='{end_date}', candle_size='{candle_size}', exchange='{exchange}', "
                f"outside_rth={outside_rth}, raise_invalid_data_exception={raise_invalid_data_exception}")

    cached_file = _cached_file_with_most_overlap(symbol, start_date, end_date, candle_size, exchange, outside_rth)
    write_df_to_cache = True
    start_date_str, end_date_str = start_date, end_date

    if cached_file is None:
        df = _get_historical_data(symbol, start_date, end_date, candle_size, exchange, outside_rth)
    else:
        # the cached file has some overlap, now calculate left and right missing parts
        exchange2, symbol2, candle_size2, start_date2, end_date2, outside_rth2 = cached_file.split(".")[0].split("_")
        start_date, end_date   = parse(start_date), parse(end_date)
        start_date2, end_date2 = parse(start_date2), parse(end_date2)

        center_df = pd.read_csv(os.path.join(CACHE_DIR, cached_file))
        center_df["t"] = pd.to_datetime(center_df["t"])
        left_df, right_df = pd.DataFrame(), pd.DataFrame()

        if start_date < start_date2:
            left_df = _get_historical_data(symbol, start_date.strftime("%Y-%m-%d"), start_date2.strftime("%Y-%m-%d"), 
                                            candle_size, exchange, outside_rth)
        if end_date > end_date2:
            right_df = _get_historical_data(symbol, end_date2.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d"), 
                                            candle_size, exchange, outside_rth)
        
        if left_df.empty and right_df.empty:
            write_df_to_cache = False  # a request which fully encompasses this request already exists in cache
        
        df = pd.concat([left_df, center_df, right_df])
        df = df[(df["t"] >= start_date) & (df["t"] < end_date)].reset_index(drop=True)
    
    if write_df_to_cache:
        cached_file_path = os.path.join(
            CACHE_DIR,
            _args_to_filename(symbol, start_date, end_date, candle_size, exchange, outside_rth)
        )
        df.to_csv(cached_file_path, index=False)

    try:
        validate_historical_data(df, start_date_str, end_date_str, 
                                 candle_size, outside_rth, exchange if exchange else get_exchange(symbol))
    except AssertionError as ex:
        if raise_invalid_data_exception:
            raise ex
        logger.warning(f"Invalid data for ticker '{symbol}', with start_date='{start_date}', "
                       f"end_date='{end_date}', candle_size='{candle_size}', exception: {ex}")

    return df


def _get_historical_data(symbol: str, start_date: str, end_date: str, 
                         candle_size: str, exchange: str, outside_rth: bool
                        ) -> pd.DataFrame:
    """ Get open, close, high, low, volume, time data from API (does not use local file cache).

    Args:
        symbol (str): symbol (e.g. AAPL)
        start_date (str): start date in yyyy-mm-dd format
        end_date (str): end date in yyyy-mm-dd format, exclusive
        candle_size (str): candle size to use.
        exchange (str): exchange to use, by default will use primary exchange for symbol.
        outside_rth (bool): get outside regular trading hours data too? Defaults to False.
    Returns:
        (pd.DataFrame): dataframe containing ['t', 'o', 'c', 'h', 'l', 'v'] columns. 't' column relative to UTC.
    """
    logger.info(f"Making yfinance request with start='{start_date}', end='{end_date}', "
                f"prepost={outside_rth}, interval='{candle_size}' for ticker '{symbol}'")
    
    yf_exchange_map = {
        "LSE": "L",
        "ASX": "AX",
        "TSX": "TO",
        "ETR": "DE"
    }
    
    if exchange in yf_exchange_map and '.' not in symbol:
        # convert to yfinance ticker format 
        symbol = f"{symbol}.{yf_exchange_map[exchange]}"

    ticker = yf.Ticker(symbol)
    df = ticker.history(start=start_date, end=end_date, prepost=outside_rth, interval=candle_size).reset_index()
    df = df.rename(columns={"Date": "t", "Open": "o", "Close": "c", "Low": "l", "High": "h", "Volume": "v"})

    try:
        df['t'] = df['t'].dt.tz_convert(None)  # convert to UTC and make naive
    except TypeError:
        pass  # df['t'] is already naive

    if candle_size == "1d":
        df['t'] = df['t'].dt.normalize()  # all daily candles should start at 00:00:00 regardless of tz

    return df[["t", "o", "c", "h", "l", "v"]]


def get_exchange(symbol: str) -> str:
    """ Get main exchange for symbol

    Args:
        symbol (str): ticker symbol

    Returns:
        str: main exchange (e.g. "NYSE") for symbol
    """
    correct_exchange_map = {
        "NYQ": "NYSE",
        "NGM": "NASDAQ",
        "NMS": "NASDAQ",
        "BTS": "BATS",
        "TOR": "TSX",
        "AX":  "ASX",
        "DE":  "ETR"
    }

    logger.info(f"Making request to yfinance to get exchange for ticker '{symbol}'")
    exchange = yf.Ticker(symbol).info["exchange"]

    return correct_exchange_map.get(exchange, exchange)


def validate_historical_data(df: pd.DataFrame, start_date: str, end_date: str, candle_size: str, 
                             outside_rth: bool, market_calander_name="NASDAQ"):
    """ Raise assertion error if data coming from df is not valid 
    (i.e. is missing rows, dates are not correct, etc)

    Args:
        df (pd.DataFrame): df containing ['t', 'o', 'c', 'h', 'l', 'v'] columns.
        start_date (str): start_date string for historical data request
        end_date (str): end_date string for historicla data request
        candle_size (str): requested candle size
        outside_rth (bool): request for outside regular trading hours data?
        market_calander_name (str): the name of the exchange. e.g. "NYSE".
    """
    if candle_size == "1d":
        try:
            exchange = mcal.get_calendar(market_calander_name)
        except RuntimeError as ex:
            logger.warning(f"exchange '{market_calander_name}' not recognised, cannot validate data. Exact exception:\n{ex}")
            return

        inclusive_end_date = (parse(end_date) - timedelta(days=1)).strftime("%Y-%m-%d")
        dates = exchange.valid_days(start_date=start_date, end_date=inclusive_end_date)

        df_idx = 0
        for date in dates:
            if df_idx >= len(df):
                break

            # same yyyy-mm-dd?
            assert df['t'].iloc[df_idx].date() == date.date(), (f"Check df['t'].iloc[{df_idx}], "
                f"expected date part to be {date.date()}, but was {df['t'].iloc[df_idx].date()}")
                
            df_idx += 1
        
        assert len(df) == len(dates), (f"Expected df to have len {len(dates)}, "
            f"but got {len(df)}.\ndates:\n{dates}\ndf['t']:\n{df['t']}\nUsing exchange {market_calander_name}")
    else:
        logger.warning(f"validation of data df not currently implemented for candle size '{candle_size}'")


def get_last_full_trading_day(market_calander_name="NASDAQ") -> datetime:
    """ Get the most recent trading day for a given exchange. 
    if today is on a trading day, will only return today if
    current time is 15 minutes after the close.

    Args:
        market_calander_name (str, optional): name of exchange. Defaults to "NASDAQ".
    Returns:
        datetime: most recent trading day
    """
    exchange = mcal.get_calendar(market_calander_name)

    tz         = exchange.tz
    close_time = exchange.close_time  # datetime.time object with tz of exchange
    close_time = datetime.combine(date.today(), close_time) + timedelta(minutes=15)
    close_time = close_time.time()
    now        = datetime.now(tz)

    start_date = date.today() - timedelta(days=30)
    end_date = date.today()
    if now.time() < close_time:
        end_date = end_date - timedelta(days=1)
    
    valid_days = exchange.valid_days(
        start_date=start_date.strftime("%Y-%m-%d"), 
        end_date=end_date.strftime("%Y-%m-%d")
    )
    return valid_days[-1].to_pydatetime()


def get_all_sp500_tickers() -> List[str]:
    """ get all the ticker symbols currently in the S&P 500

    Returns:
        List[str]: all symbols in S&P 500
    """
    resp = requests.get('http://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
    soup = bs.BeautifulSoup(resp.text, 'lxml')
    table = soup.find('table', {'class': 'wikitable sortable'})

    tickers = []
    for row in table.findAll('tr')[1:]:
        ticker = row.findAll('td')[0].text.replace("\n", "")
        tickers.append(ticker)
    
    return tickers


def get_vix_daily_data(vix_ticker: str) -> pd.DataFrame:
    """ get daily vix data until the last trading day.

    Args:
        vix_ticker (str): one of:
            VIX: VIX (IV of S&P 500) data starting from 1990-01-02 with columns t, o, c, h, l
            VVIX: VVIX (IV of VIX) data starting from 2006-06-03 with columns t, c
            VXN: VXN (IV of nasdaq-100) data starting from 2009-09-14 with columns t, o, c, h, l
            GVZ: GVZ (IV of Gold ETF) data starting from 2009-09-18 with columns t, c
            OVX: OVX (IV of Crude Oil ETF) data starting from 2009-09-18 with columns t, c

    Returns:
        pd.DataFrame: dataframe containing vix data
    """
    vix_folder = CACHE_DIR + os.sep + "vix_data"
    if not os.path.exists(vix_folder):
        os.makedirs(vix_folder)
    file_prefix =  vix_folder + os.sep + vix_ticker

    try:
        df = pd.read_csv(file_prefix + datetime.today().strftime('%Y-%m-%d') + ".csv")
    except FileNotFoundError:
        logger.info(f"Making request to CBOE for {vix_ticker} data")
        df = pd.read_csv(f"https://cdn.cboe.com/api/global/us_indices/daily_prices/{vix_ticker}_History.csv")
        df = df.rename(columns={"DATE": "t", "OPEN": "o", "CLOSE": "c", "LOW": "l", "HIGH": "h", 
                                "VVIX": "c", "VXN": "c", "GVZ": "c", "OVX": "c"})
        df.to_csv(file_prefix + datetime.today().strftime('%Y-%m-%d') + ".csv", index=False)

    df["t"] = pd.to_datetime(df["t"])
    return df


if __name__ == "__main__":
    # test retrieving data with live API
    logging.basicConfig(level=logging.INFO)

    print("get 2021-01-01 to 2023-01-01 from API")
    df = get_historical_data("AAPL", "2021-01-01", "2023-01-01")
    print(df)

    print("shouldn't call API as historical data saved in file cache")
    df = get_historical_data("AAPL", "2021-01-01", "2023-01-01")
    print(df)

    print("shouldn't call API as historical data saved in file cache")
    df = get_historical_data("AAPL", "2021-05-01", "2022-06-01")
    print(df)

    print("get 2023-01-01 to 2023-02-01 from API (rest is in file cache)")
    df = get_historical_data("AAPL", "2021-12-01", "2023-02-01")
    print(df)

    print("get 2020-12-01 to 2021-01-01 and 2023-01-01 to 2023-03-01 from API (rest is in file cache)")
    df = get_historical_data("AAPL", "2020-12-01", "2023-03-01")
    print(df)
