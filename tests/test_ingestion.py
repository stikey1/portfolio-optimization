"""Tests for Module 1 (ingestion.py): fetching
and cleaning raw stock data."""


import pandas as pd
import pytest
from src.ingestion import (
    YFinanceSource,
    AlphaVantageSource,
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
    columns = pd.MultiIndex.from_tuples(
        [("Adj Close", "AAPL"), ("Adj Close", "MSFT")],
        names=["Price", "Ticker"],
    )
    return pd.DataFrame(
        [[150.0, 400.0], [151.0, 401.0], [152.0, 402.0]],
        columns=columns,
        index=pd.date_range("2024-01-01", periods=3, freq="B"),
    )

# YFinanceSource fetch tests ----------------------------------
class TestYFinanceSource:
    """Tests for YFinanceSource.fetch, including retry/backoff behavior."""
 
    @patch("src.ingestion.yf.download")
    def test_yf_fetch_returns_dataframe_for_multiple_tickers(self, mock_download):
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
            ["AAPL", "MSFT"], 
            start=date(2024, 1, 1), 
            end=date(2024, 1, 10), 
            auto_adjust=False
        )

    @patch("src.ingestion.yf.download")
    def test_yf_fetch_single_ticker_uses_non_multiindex_branch(self, mock_download):
        """When yf.download isn't given multiple tickers, columns aren't a
        MultiIndex, so fetch() takes the `raw[["Adj Close"]]` path. This
        documents current behavior: the resulting column is literally named
        "Adj Close" rather than the ticker symbol."""
        mock_download.return_value = pd.DataFrame(
            {
                "Adj Close": [150.0, 151.0, 152.0],
                "Close": [151.0, 152.0, 153.0],
            },
            index=pd.date_range("2024-01-01", periods=3, freq="B"),
        )
 
        source = YFinanceSource()
        df = source.fetch(["AAPL"], date(2024, 1, 1), date(2024, 1, 10))
 
        assert not df.empty
        assert "Adj Close" in df.columns
    
    @patch("src.ingestion.time.sleep") 
    @patch("src.ingestion.yf.download")
    def test_yf_retries_then_succeeds(self, mock_download, mock_sleep):
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
    def test_yf_raises_after_three_consecutive_failures(self, mock_download):
        mock_download.side_effect = Exception("API down")

        source = YFinanceSource()
        with pytest.raises(Exception, match="API down"):
            source.fetch(["AAPL"], date(2024, 1, 1), date(2024, 1, 10))

        assert mock_download.call_count == 3
    
    @patch("src.ingestion.time.sleep")
    @patch("src.ingestion.yf.download")
    def test_yf_raises_value_error_after_repeated_empty_responses(self, mock_download, mock_sleep):
        """An empty DataFrame from yf.download is treated as a failure and
        retried; it should still raise after exhausting all attempts."""
        mock_download.return_value = pd.DataFrame()
 
        source = YFinanceSource()
        with pytest.raises(ValueError, match="No data fetched"):
            source.fetch(["AAPL"], date(2024, 1, 1), date(2024, 1, 10))
 
        assert mock_download.call_count == 3

# ----------------------------------
# AlphaVantage tests
class TestAlphaVantageSource:
    """Tests for AlphaVantageSource, currently an unimplemented stub."""
 
    def test_fetch_is_not_yet_implemented(self):
        """fetch() has no body beyond `...`, so it implicitly returns None.
        This test documents the current stub behavior so a future
        implementation is expected to break (and update) it."""
        source = AlphaVantageSource()
        result = source.fetch(["AAPL"], date(2024, 1, 1), date(2024, 1, 10))
        assert result is None

# ----------------------------------
# clean data test 
class TestCleanData:
    """Tests for clean_data (source-agnostic forward-fill + drop-empty)."""
    
    def test_forward_fills_and_drops_empty_columns(self):
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

    def test_drops_rows_that_are_entirely_nan(self):
        prices = pd.DataFrame(
            {
                "AAPL": [100.0, None, 102.0],
                "MSFT": [400.0, None, 402.0],
            },
            index=pd.date_range("2024-01-01", periods=3, freq="B"),
        )
 
        cleaned = clean_data(prices)
 
        # before ffill: row 1 is all-NaN, after: filled w/ row 0 values
        # so row 1 not dropped -- only rows still all-NaN after ffill are
        assert len(cleaned) == 3
 
    def test_leading_nan_is_not_backfilled(self):
        """ffill only propagates forward, so a NaN at the very start of a
        column (nothing earlier to fill from) should remain NaN."""
        prices = pd.DataFrame(
            {
                "AAPL": [None, 101.0, 102.0],
                "MSFT": [400.0, 401.0, 402.0],
            },
            index=pd.date_range("2024-01-01", periods=3, freq="B"),
        )
 
        cleaned = clean_data(prices)
 
        assert pd.isna(cleaned["AAPL"].iloc[0])
 
    def test_handles_empty_dataframe(self):
        cleaned = clean_data(pd.DataFrame())
        assert cleaned.empty

# ----------------------------------
# save data test 
class TestSaveToParquet:
    """Tests for save_to_parquet."""

    def test_writes_readable_file(self, tmp_path):
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

# ----------------------------------
# load data tests
class TestLoadData:
    """Tests for load_data, including missing-file and missing-ticker cases."""
    
    def test_raises_if_no_parquet_file(self, tmp_path):
        # tmp_path is an empty directory --> no prices.parquet inside
        with pytest.raises(FileNotFoundError, match="No parquet file found"):
            load_data(["AAPL"], data_dir=tmp_path)


    def test_raises_if_ticker_not_in_file(self, tmp_path):
        # dummy parquet file, don't use save_to_parquet 
        df = pd.DataFrame(
            {"AAPL": [150.0, 151.0]},
            index=pd.date_range("2024-01-01", periods=2),
        )
        df.to_parquet(tmp_path / "prices.parquet")

        # ask for MSFT which isn't in the file
        with pytest.raises(TickerNotCachedError, match="MSFT"):
            load_data(["AAPL", "MSFT"], data_dir=tmp_path)

    def test_returns_requested_tickers(self, tmp_path):
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
    
    def test_returns_empty_dataframe_for_empty_ticker_list(self, tmp_path):
        """No tickers requested means nothing is "missing", so load_data
        should return an empty-columns frame rather than raising."""
        df = pd.DataFrame(
            {"AAPL": [150.0, 151.0]},
            index=pd.date_range("2024-01-01", periods=2),
        )
        df.to_parquet(tmp_path / "prices.parquet")
 
        loaded = load_data([], data_dir=tmp_path)
 
        assert list(loaded.columns) == []
 
    def test_accepts_data_dir_as_string(self, tmp_path):
        """data_dir is typed as Path | str, so a plain string should work too."""
        df = pd.DataFrame(
            {"AAPL": [150.0, 151.0]},
            index=pd.date_range("2024-01-01", periods=2),
        )
        df.to_parquet(tmp_path / "prices.parquet")
 
        loaded = load_data(["AAPL"], data_dir=str(tmp_path))
 
        assert "AAPL" in loaded.columns

# ----------------------------------
# run ingestion tests
class TestRunIngestion:
    """Tests for run_ingestion, the fetch -> clean -> save orchestration."""

    def test_orchestrates_fetch_clean_save(self, tmp_path):
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
    
    def test_propagates_fetch_exception_without_writing_file(self, tmp_path):
        """If the source's fetch() raises, run_ingestion should propagate
        the error rather than swallowing it, and no output file should be
        written."""
        mock_source = MagicMock()
        mock_source.fetch.side_effect = Exception("network error")
 
        out_path = str(tmp_path / "prices.parquet")
 
        with pytest.raises(Exception, match="network error"):
            run_ingestion(
                mock_source,
                tickers=["AAPL"],
                start=date(2024, 1, 1),
                end=date(2024, 1, 3),
                out_path=out_path,
            )
 
        assert not Path(out_path).exists()