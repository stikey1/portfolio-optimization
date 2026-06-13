import pandas as pd

from src.ingestion import clean_prices


def test_clean_prices_forward_fills_and_drops_empty_columns():
    prices = pd.DataFrame(
        {
            "AAPL": [100.0, None, 102.0],
            "DEAD": [None, None, None],
        },
        index=pd.date_range("2024-01-01", periods=3, freq="D"),
    )

    cleaned = clean_prices(prices)

    assert "DEAD" not in cleaned.columns
    assert cleaned["AAPL"].iloc[1] == 100.0
