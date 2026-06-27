"""Module 2: Calculate returns and covariance matrices."""

import numpy as np
import pandas as pd


def compute_returns(prices: pd.DataFrame, method: str = "log") -> pd.DataFrame:
    """Compute daily returns from price series.

    Args:
        prices: DataFrame of asset prices indexed by date.
        method: Return type, either ``"log"`` or ``"simple"``.

    Returns:
        DataFrame of daily returns.
    """
    if(prices.any().any() <= 0):
        raise ValueError("Prices cannot be negative")
    
    if(method == "log"):
        return np.log(prices/prices.shift(1)).dropna()
    elif(method == "simple"):
        return prices.pct_change().dropna()
    raise ValueError(f"Unknown return method: {method!r}")

def compute_expected_returns(returns: pd.DataFrame) -> pd.Series:
    """Compute expected returns from return series.

    Args:
        returns: DataFrame of daily returns.

    Returns:
        Series of expected returns.
    """
    mu = returns.mean()*252
    return mu

    
def compute_covariance(returns: pd.DataFrame, annualize: bool = True) -> pd.DataFrame:
    """Compute the covariance matrix of asset returns.

    Args:
        returns: DataFrame of daily returns.
        annualize: If True, scale by 252 trading days.

    Returns:
        Covariance matrix as a DataFrame.
    """
    cov = returns.cov()
    if annualize:
        cov = cov * 252
    return cov
