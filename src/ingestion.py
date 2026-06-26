"""Module 1: Fetch and clean raw stock data."""

from pathlib import Path

import pandas as pd


def load_prices(tickers: list[str], data_dir: Path | str = "data") -> pd.DataFrame:
    tickers = ["AAPL", "MSFT", "GOOG"]
    stocks = yf.download(
        tickers=tickers,
        period = "1mo",
        auto_adjust=True,
        keepna=False
        
        
        )
    stocks.to_parquet("stocks.parquet")
    loaded = pd.read_parquet("stocks.parquet")
    return stocks

def clean_prices(prices: pd.DataFrame) -> pd.DataFrame:
    """Clean raw price data (forward-fill gaps, drop invalid rows).

    Args:
        prices: Raw price DataFrame.

    Returns:
        Cleaned price DataFrame.
    """
    cleaned = prices.ffill().dropna(how="all")
    return cleaned.dropna(axis=1, how="all")

