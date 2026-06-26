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
    
# reads from parquet, returns requested tickers, raise if missing
def load_data(tickers: list[str], data_dir: Path | str = "data") -> pd.DataFrame:
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




# main ingestion function to refresh data and save to Parquet (called from app.py)
def run_ingestion(source: DataSource, tickers: list[str], start: date, end: date, out_path: str) -> pd.DataFrame:
    raw = source.fetch(tickers, start, end)
    cleaned = clean_data(raw)
    save_to_parquet(cleaned, out_path)
    return cleaned
