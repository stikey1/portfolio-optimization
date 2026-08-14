"""Module 0: Provide the curated list of tickers available for selection in the UI."""

from pathlib import Path
import pandas as pd

# dashboard dropdown of available tickers
# (separate from cached parquet data)
def load_available_tickers(path: str | Path = "data/tickers.csv") -> list[str]:
    """Load the curated list of available tickers from a CSV file.
    
    Reads a CSV file containing ticker symbols and returns them as a cleaned,
    uppercase list. Filters out NaN/null values and strips whitespace. Used to
    populate the ticker selection dropdown in the UI.
    
    Args:
        path (str | Path): Path to a CSV file with a 'ticker' column, relative
            to the project root (default 'data/tickers.csv').
    
    Returns:
        list[str]: List of unique ticker symbols, all uppercase and stripped
            of whitespace.
    
    Raises:
        FileNotFoundError: If the CSV file does not exist at the specified path.
        ValueError: If the CSV file contains no valid tickers after filtering.
    """
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