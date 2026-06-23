import pandas as pd
import pytest
from src.ingestion import YFinanceSource, clean_data, save_to_parquet
from datetime import date
from unittest.mock import patch


def make_mock_yfinance_data():
    """Plain helper returning mock yf.download response (multi-index columns)."""
    return pd.DataFrame(
        {
            ("Adj Close", "AAPL"): [150.0, 151.0, 152.0],
            ("Adj Close", "MSFT"): [400.0, 401.0, 402.0],
        },
        index=pd.date_range("2024-01-01", periods=3, freq="B"),
    )

# YFinanceSource fetch tests ----------------------------------
@patch("src.ingestion.yf.download")
def test_yfinance_fetch_returns_dataframe(mock_download):
    mock_download.return_value = make_mock_yfinance_data()

    source = YFinanceSource()
    df = source.fetch(
        tickers=["AAPL", "MSFT"],
        start=date(2024, 1, 1),
        end=date(2024, 1, 10),
    )

    assert not df.empty
    assert "AAPL" in df.columns
    assert "MSFT" in df.columns
    assert hasattr(df.index, "date")

    mock_download.assert_called_once_with(
        ["AAPL", "MSFT"], start=date(2024, 1, 1), end=date(2024, 1, 10), auto_adjust=False
    )


@patch("src.ingestion.time.sleep")
@patch("src.ingestion.yf.download")
def test_yfinance_retries_then_succeeds(mock_download, mock_sleep):
    mock_download.side_effect = [
        Exception("temporary failure"),
        Exception("temporary failure"),
        make_mock_yfinance_data(),
    ]

    source = YFinanceSource()
    df = source.fetch(["AAPL"], date(2024, 1, 1), date(2024, 1, 10))

    assert not df.empty
    assert mock_download.call_count == 3
    mock_sleep.assert_any_call(1)
    mock_sleep.assert_any_call(2)


@patch("src.ingestion.yf.download")
def test_yfinance_raises_after_three_failures(mock_download):
    mock_download.side_effect = Exception("API down")

    source = YFinanceSource()
    with pytest.raises(Exception, match="API down"):
        source.fetch(["AAPL"], date(2024, 1, 1), date(2024, 1, 10))

    assert mock_download.call_count == 3

# clean data test ----------------------------------
def test_clean_data_forward_fills_and_drops_empty_columns():
    prices = pd.DataFrame(
        {
            "AAPL": [100.0, None, 102.0],
            "DEAD": [None, None, None],
        },
        index=pd.date_range("2024-01-01", periods=3, freq="B"),
    )

    cleaned = clean_data(prices)

    assert "DEAD" not in cleaned.columns
    assert cleaned["AAPL"].iloc[1] == 100.0

# save data test ----------------------------------
def test_save_to_parquet_writes_readable_file(tmp_path):
    df = pd.DataFrame(
        {"AAPL": [150.0, 151.0], "MSFT": [400.0, 401.0]},
        index=pd.date_range("2024-01-01", periods=2, freq="B"),
    )
    out_path = tmp_path / "prices.parquet"

    save_to_parquet(df, str(out_path))

    # file actually exists on disk
    assert out_path.exists()

    # can be read back and matches original
    loaded = pd.read_parquet(out_path)
    pd.testing.assert_frame_equal(df, loaded, check_freq=False)