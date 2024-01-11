import pandas as pd


def normalised_returns(close: pd.Series) -> pd.Series:
    """ Get normalised returns, i.e. (x_t / x{t - 1}) - 1

    Args:
        close (pd.Series): series of float values

    Returns:
        pd.Series: normalised returns. First element will be NaN
    """
    shifted = close.shift(periods=1)
    return (close - shifted) / shifted
