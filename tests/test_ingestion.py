import pandas as pd
import pytest
from src.ingestion import (
    YFinanceSource,
    clean_data,
    save_to_parquet,
    load_data,
    TickerNotCachedError,
    run_ingestion,
)
from datetime import date
from unittest.mock import patch, MagicMock
from pathlib import Path

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

# load data tests ----------------------------------
def test_load_data_raises_if_no_parquet_file(tmp_path):
    # tmp_path is an empty directory --> no prices.parquet inside
    with pytest.raises(FileNotFoundError, match="No parquet file found"):
        load_data(["AAPL"], data_dir=tmp_path)


def test_load_data_raises_if_ticker_not_in_file(tmp_path):
    # dummy parquet file, don't use save_to_parquet 
    df = pd.DataFrame(
        {"AAPL": [150.0, 151.0]},
        index=pd.date_range("2024-01-01", periods=2),
    )
    df.to_parquet(tmp_path / "prices.parquet")

    # ask for MSFT which isn't in the file
    with pytest.raises(TickerNotCachedError, match="MSFT"):
        load_data(["AAPL", "MSFT"], data_dir=tmp_path)

def test_load_data_returns_requested_tickers(tmp_path):
    # dummy parquet file with extra tickers
    df = pd.DataFrame(
        {"AAPL": [150.0, 151.0], "MSFT": [400.0, 401.0], "GOOG": [180.0, 181.0]},
        index=pd.date_range("2024-01-01", periods=2),
    )
    df.to_parquet(tmp_path / "prices.parquet")

    loaded = load_data(["AAPL", "MSFT"], data_dir=tmp_path)

    # only requested tickers come back
    assert list(loaded.columns) == ["AAPL", "MSFT"]
    assert "GOOG" not in loaded.columns
    assert not loaded.empty

# run ingestion test ----------------------------------

def test_run_ingestion_orchestrates_fetch_clean_save(tmp_path):
    # mock source fetching not fully cleaned df
    mock_source = MagicMock()
    mock_source.fetch.return_value = pd.DataFrame(
        {"AAPL": [150.0, None, 152.0], "MSFT": [400.0, 401.0, 402.0]},
        index=pd.date_range("2024-01-01", periods=3),
    )

    out_path = str(tmp_path / "prices.parquet")
    result = run_ingestion(
        mock_source,
        tickers=["AAPL", "MSFT"],
        start=date(2024, 1, 1),
        end=date(2024, 1, 3),
        out_path=out_path,
    )

    # fetch was called with the right arguments
    mock_source.fetch.assert_called_once_with(
        ["AAPL", "MSFT"], date(2024, 1, 1), date(2024, 1, 3)
    )

    # returned DataFrame is cleaned (no NaNs) and has right columns
    assert not result.isnull().values.any()
    assert "AAPL" in result.columns
    assert "MSFT" in result.columns

    # parquet file was actually written to disk
    assert Path(out_path).exists()