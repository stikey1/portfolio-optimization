"""Module 0: Provide the curated list of tickers available for selection in the UI."""

from pathlib import Path
import pandas as pd

# dashboard dropdown of available tickers
# (separate from cached parquet data)
def load_available_tickers(path: str | Path = "data/ticker_universe.csv") -> list[str]:
    path = Path(__file__).parent.parent / "data" / "tickers.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"No ticker universe file found at {path}. "
            f"Create it with a single 'ticker' column."
        )
    return pd.read_csv(path)["ticker"].tolist()