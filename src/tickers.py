"""Module 0: Provide the curated list of tickers available for selection in the UI."""

from pathlib import Path
import pandas as pd

# dashboard dropdown of available tickers
# (separate from cached parquet data)
def load_available_tickers(path: str | Path = "data/tickers.csv") -> list[str]:
    path = Path(__file__).parent.parent / path
    if not path.exists():
        raise FileNotFoundError(
            f"No ticker universe file found at {path}. "
            f"Create it with a single 'ticker' column."
        )
    tickers = (
        pd.read_csv(path)["ticker"]
        .dropna()
        .astype(str)
        .str.strip()
        .str.upper()
        .tolist()
    )

    if not tickers:
        raise ValueError(f"No tickers found in {path}.")

    return tickers