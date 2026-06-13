"""Module 1: Fetch and clean raw stock data."""

from pathlib import Path

import pandas as pd


def load_prices(tickers: list[str], data_dir: Path | str = "data") -> pd.DataFrame:
    """Load adjusted close prices for the given tickers from local Parquet files.

    Args:
        tickers: List of stock ticker symbols.
        data_dir: Directory containing downloaded Parquet files.

    Returns:
        DataFrame indexed by date with one column per ticker.

    Raises:
        NotImplementedError: Placeholder until data pipeline is implemented.
    """
    raise NotImplementedError("Data ingestion not yet implemented.")


def clean_prices(prices: pd.DataFrame) -> pd.DataFrame:
    """Clean raw price data (forward-fill gaps, drop invalid rows).

    Args:
        prices: Raw price DataFrame.

    Returns:
        Cleaned price DataFrame.
    """
    cleaned = prices.ffill().dropna(how="all")
    return cleaned.dropna(axis=1, how="all")
