"""Module 1: Fetch and clean raw stock data."""

from pathlib import Path
import pandas as pd
from abc import ABC, abstractmethod
from datetime import date
import time
import yfinance as yf

# abstract base class for data sources
class DataSource(ABC):
    @abstractmethod
    def fetch(self, tickers: list[str], start: date, end: date) -> pd.DataFrame:
        """return dataframe with colummns: date, ticker, ..."""
        pass

# data sources
class YFinanceSource(DataSource):
    def fetch(self, tickers: list[str], start: date, end: date) -> pd.DataFrame:
        for attempt in range(3):
            try:
                data = yf.download(tickers, start=start, end=end, auto_adjust=False)["Adj Close"]
                if data.empty:
                     raise ValueError("No data fetched. Check tickers and date range.")
                return data
            except Exception:
                 if attempt == 2:   # last attempt, re-raise exception
                      raise
                 time.sleep(2 ** attempt)  # exponential backoff
                 


class AlphaVantageSource(DataSource):
    def fetch(self, tickers: list[str], start: date, end: date) -> pd.DataFrame:
                ...  # retry logic can live here, or be factored into a helper

# ------------ ---------------------------------------- --------------

# cleans data (source-agnostic)
def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    # remove rows with all NaN values 
    cleaned = df.ffill().dropna(how="all")

    #remove columns with all NaN values
    return cleaned.dropna(axis=1, how="all")

# parquet functions
def save_to_parquet(df: pd.DataFrame, path: str):
    df.to_parquet(
         path, 
         engine="pyarrow", 
         compression="snappy",
         index=True)
    
# reads from parquet 
def load_data(tickers: list[str], data_dir: Path | str = "data") -> pd.DataFrame:
    """reads cached Parquet files from data_dir, returns columns for requested tickers

    Args:
        tickers: List of stock ticker symbols.
        data_dir: Directory containing downloaded Parquet files.

    Returns:
        DataFrame indexed by date with one column per ticker.

    Raises:
        NotImplementedError: Placeholder until data pipeline is implemented.
    """
    # if missing:
    #      raise TickerNotCachedError(
    #           f"No cached data for: {sorted(missing)}."
    #           f"Run ingestion first to fetch and cache data, e.g. run_ingestion(YFinanceSource(), {sorted(missing)}, {start}, {end}, 'data/')"
    #      )
    raise NotImplementedError("Data ingestion not yet implemented.")




# main ingestion function to refresh data and save to Parquet (called from app.py)
def run_ingestion(source: DataSource, tickers: list[str], start: date, end: date, out_path: str) -> pd.DataFrame:
    raw = source.fetch(tickers, start, end)
    cleaned = clean_data(raw)
    save_to_parquet(cleaned, out_path)
    return cleaned
