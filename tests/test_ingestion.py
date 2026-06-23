import pandas as pd
from src.ingestion import YFinanceSource, clean_data
from datetime import date

def test_yfinance_fetch_returns_dataframe():
    source = YFinanceSource()
    df = source.fetch(
        tickers=["AAPL", "MSFT"], 
        start=date(2024, 1, 1), 
        end=date(2024, 1, 10)
    )
    # returns something
    assert not df.empty 

    # returns right tickers
    assert "AAPL" in df.columns
    assert "MSFT" in df.columns

    # index has date
    assert hasattr(df.index, "date")

    # no completely empty column
    assert not df.isnull().all().any()


def test_clean_data_forward_fills_and_drops_empty_columns():
    prices = pd.DataFrame(
        {
            "AAPL": [100.0, None, 102.0],
            "DEAD": [None, None, None],
        },
        index=pd.date_range("2024-01-01", periods=3, freq="D"),
    )

    cleaned = clean_data(prices)

    assert "DEAD" not in cleaned.columns
    assert cleaned["AAPL"].iloc[1] == 100.0
