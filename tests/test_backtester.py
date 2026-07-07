import pandas as pd
import numpy as np
import pytest
from src.backtester import backtest

@pytest.fixture
def price_df():
    """5 assets x 504 trading days (~2 years), seeded GBM."""
    rng = np.random.default_rng(42)
    n_assets, n_days = 5, 504
    daily_returns = rng.normal(loc=0.0004, scale=0.012, size=(n_days, n_assets))
    prices = 100 * np.exp(np.cumsum(daily_returns, axis=0))
    tickers = ["AAPL", "MSFT", "GOOG", "AMZN", "TSLA"]
    dates = pd.bdate_range("2022-01-03", periods=n_days)
    return pd.DataFrame(prices, index=dates, columns=tickers)


@pytest.fixture
def short_price_df(price_df):
    """Fewer rows than lookback_days=252 — should produce no rebalances."""
    return price_df.iloc[:100]


@pytest.fixture
def single_asset_price_df(price_df):
    """Only one ticker — weight should always be 1.0."""
    return price_df.iloc[:, [0]]


@pytest.fixture
def backtest_result(price_df):
    """Run the backtest once and share the result across tests that don't mutate it."""
    return backtest(price_df)


def test_returns_has_required_keys(backtest_result):
    assert "returns" in backtest_result
    assert "cumulative_value" in backtest_result
    assert "weights_history" in backtest_result


def test_returns_is_series(backtest_result):
    assert isinstance(backtest_result["returns"], pd.Series)


def test_cumulative_value_is_series(backtest_result):
    assert isinstance(backtest_result["cumulative_value"], pd.Series)


def test_returns_index_is_datetime(backtest_result):
    assert isinstance(backtest_result["returns"].index, pd.DatetimeIndex)


def test_returns_index_sorted(backtest_result):
    idx = backtest_result["returns"].index
    assert (idx[1:] >= idx[:-1]).all()


def test_no_nan_in_returns(backtest_result):
    assert not backtest_result["returns"].isnull().any()



def test_weights_sum_to_one(backtest_result):
    weight_sums = backtest_result["weights_history"].sum(axis=1)
    assert np.allclose(weight_sums, 1.0, atol=1e-4)


def test_weights_within_bounds(backtest_result):
    weights = backtest_result["weights_history"]
    assert (weights >= -1e-6).all().all()
    assert (weights <= 1 + 1e-6).all().all()


def test_weights_columns_match_tickers(backtest_result, price_df):
    assert set(backtest_result["weights_history"].columns) == set(price_df.columns)

def test_insufficient_history_produces_no_returns(short_price_df):
    result = backtest(short_price_df, lookback_days=252)
    assert len(result["returns"]) == 0


def test_single_asset_weight_is_always_one(single_asset_price_df):
    result = backtest(single_asset_price_df)
    if len(result["weights_history"]) > 0:
        assert np.allclose(result["weights_history"].iloc[:, 0], 1.0, atol=1e-4)

def test_no_lookahead_bias(price_df):
    cutoff = price_df.index[300]

    result_original = backtest(price_df)

    corrupted = price_df.copy()
    corruption_start = price_df.index[price_df.index.get_loc(cutoff) + 1]
    corrupted.loc[corruption_start:] = corrupted.loc[corruption_start:] * 5.0
    result_corrupted = backtest(corrupted)

    pre_cutoff_original = result_original["returns"].loc[:cutoff]
    pre_cutoff_corrupted = result_corrupted["returns"].loc[:cutoff]

    pd.testing.assert_series_equal(pre_cutoff_original, pre_cutoff_corrupted)