"""Module 1: Fetch and clean raw stock data."""

from pathlib import Path
import pandas as pd
from abc import ABC, abstractmethod
from datetime import date
import time
import yfinance as yf

# Custom exceptions
class TickerNotCachedError(Exception):
    pass

# ----------------------------------
# abstract base class for data sources
class DataSource(ABC):
    @abstractmethod
    def fetch(self, tickers: list[str], start: date, end: date) -> pd.DataFrame:
        """Fetch raw price data for the given tickers and date range.

        Args:
            tickers: List of ticker symbols to fetch.
            start: Start date of the range to fetch (inclusive).
            end: End date of the range to fetch (exclusive, per yfinance convention).

        Returns:
            DataFrame indexed by date, with one column per ticker.
        """
        pass

# data sources
class YFinanceSource(DataSource):
   def fetch(self, tickers: list[str], start: date, end: date) -> pd.DataFrame:
        """Fetch adjusted close prices from Yahoo Finance, retry on failure.

        Retries up to 3 times with exponential backoff (1s, 2s, 4s) if the
        download raises an exception or returns an empty result.

        Args:
            tickers: List of ticker symbols to fetch.
            start: Start date of the range to fetch (inclusive).
            end: End date of the range to fetch (exclusive).

        Returns:
            DataFrame of adjusted close prices indexed by date, with one
            column per ticker.

        Raises:
            Exception: The underlying yfinance exception, re-raised if all
                3 attempts fail.
            ValueError: If yfinance returns an empty DataFrame on every attempt.
        """
        for attempt in range(3):
            try:
                raw = yf.download(
                    tickers,
                    start=start,
                    end=end,
                    auto_adjust=False,
                )
                if raw.empty:
                    raise ValueError("No data fetched.")

                # extract adjusted close prices
                if isinstance(raw.columns, pd.MultiIndex):
                    data = raw.xs("Adj Close", axis=1, level="Price")
                else:
                    data = raw[["Adj Close"]]
                
                return data
            except Exception as e:
                print(f"[YFinanceSource] attempt {attempt+1} failed: {e!r}")
                if attempt == 2:   # last attempt, re-raise exception
                    raise
                time.sleep(2 ** attempt)  # exponential backoff
                 

class AlphaVantageSource(DataSource):
    def fetch(self, tickers: list[str], start: date, end: date) -> pd.DataFrame:
        """Fetch price data from Alpha Vantage.

        Not yet implemented; currently returns None.

        Args:
            tickers: List of ticker symbols to fetch.
            start: Start date of the range to fetch (inclusive).
            end: End date of the range to fetch (exclusive).

        Returns:
            DataFrame indexed by date, with one column per ticker.
        """
        ...  # retry logic can live here, or be factored into a helper

# ------------ ---------------------------------------- --------------
# cleans data (source-agnostic)
def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Forward-fill gaps and drop rows/columns with no usable data.

    Forward-fills missing values so a ticker's last known price carries
    over non-trading days, then drops any row or column that is still
    entirely NaN afterward (e.g. a date with no data for any ticker, or
    a ticker with no data across the whole range).

    Args:
        df: DataFrame of raw prices indexed by date, one column per ticker.

    Returns:
        Cleaned DataFrame with no fully-empty rows or columns.
    """
    # remove rows with all NaN values 
    cleaned = df.ffill().dropna(how="all")

    #remove columns with all NaN values
    return cleaned.dropna(axis=1, how="all")

# ----------------------------------
# parquet functions
def save_to_parquet(df: pd.DataFrame, path: str):
    """Write a price DataFrame to disk as a Parquet file.

    Args:
        df: DataFrame to save.
        path: Destination file path for the Parquet file.

    Returns:
        None.
    """
    df.to_parquet(
         path, 
         engine="pyarrow", 
         compression="snappy",
         index=True)
    
# reads from parquet, returns requested tickers, raise if missing
def load_data(tickers: list[str], data_dir: Path | str = "data") -> pd.DataFrame:
    """Load cached price data for the requested tickers from Parquet.

    Args:
        tickers: List of ticker symbols to load.
        data_dir: Directory containing the cached ``prices.parquet`` file.

    Returns:
        DataFrame of prices indexed by date, limited to the requested tickers.

    Raises:
        FileNotFoundError: If no ``prices.parquet`` file exists in ``data_dir``.
        TickerNotCachedError: If one or more requested tickers are not
            present in the cached file.
    """
    data_dir = Path(data_dir)
    parquet_path = data_dir / "prices.parquet"

    # check the file exists at all
    if not parquet_path.exists():
        raise FileNotFoundError(
            f"No parquet file found at {parquet_path}. "
            f"Run run_ingestion() first to fetch and cache data."
        )

    df = pd.read_parquet(parquet_path)

    # check which requested tickers are missing from the file
    missing = [t for t in tickers if t not in df.columns]
    if missing:
        raise TickerNotCachedError(
            f"No cached data for: {sorted(missing)}. "
            f"Run run_ingestion() first, e.g. run_ingestion(YFinanceSource(), "
            f"{sorted(missing)}, start, end, '{data_dir}/')"
        )

    return df[tickers]

# ----------------------------------
# main ingestion function to refresh data and save to Parquet (called from app.py)
def run_ingestion(source: DataSource, tickers: list[str], start: date, end: date, out_path: str) -> pd.DataFrame:
    """Fetch, clean, and cache price data in one step.

    Args:
        source: DataSource implementation to fetch raw data from.
        tickers: List of ticker symbols to fetch.
        start: Start date of the range to fetch (inclusive).
        end: End date of the range to fetch (exclusive).
        out_path: File path to write the cleaned data to, as Parquet.

    Returns:
        Cleaned DataFrame of prices indexed by date, one column per ticker.
    """
    raw = source.fetch(tickers, start, end)
    cleaned = clean_data(raw)
    save_to_parquet(cleaned, out_path)
    return cleaned
