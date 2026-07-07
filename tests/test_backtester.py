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
    """Run the backtest once at default params (max_weight=1.0, shrinkage=0.0)
    and share the result across tests that don't care about those knobs."""
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
    # Default max_weight=1.0 here, so bounds are the trivial [0,1] case.
    weights = backtest_result["weights_history"]
    assert (weights >= -1e-6).all().all()
    assert (weights <= 1 + 1e-6).all().all()


def test_weights_columns_match_tickers(backtest_result, price_df):
    assert set(backtest_result["weights_history"].columns) == set(price_df.columns)


def test_insufficient_history_produces_no_returns(short_price_df):
    result = backtest(short_price_df, lookback_days=252)
    assert len(result["returns"]) == 0


def test_single_asset_weight_is_always_one(single_asset_price_df):
    # max_weight must be explicitly 1.0 here -- a tighter cap makes the
    # sum(w)=1 constraint infeasible when there's only one asset to hold.
    result = backtest(single_asset_price_df, max_weight=1.0)
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


# --- New tests: bounds and shrinkage actually do something ---

def test_max_weight_is_respected(price_df):
    """No single position should ever exceed max_weight, given enough
    assets that the cap is actually feasible (5 assets, cap=0.4 -> feasible
    since 5*0.4=2.0 >= 1.0)."""
    result = backtest(price_df, max_weight=0.4)
    weights = result["weights_history"]
    assert (weights <= 0.4 + 1e-4).all().all()


def test_max_weight_below_feasibility_threshold_still_sums_near_one(price_df):
    """Sanity check: even with a tight cap, SLSQP should still find a
    feasible solution (5 assets, cap=0.3 -> max total = 1.5, still feasible)
    and weights should still sum to ~1."""
    result = backtest(price_df, max_weight=0.3)
    weight_sums = result["weights_history"].sum(axis=1)
    assert np.allclose(weight_sums, 1.0, atol=1e-3)


def test_full_shrinkage_reduces_weight_dispersion(price_df):
    """As shrinkage -> 1, every asset's expected return collapses toward
    the same grand-mean value, so the optimizer should rely increasingly
    on the covariance structure alone -- weights across rebalances should
    show less month-to-month variance than the unshrunk (shrinkage=0) case.
 
    max_weight is held constant across both runs and deliberately capped
    below 1.0 -- without a cap, near-tied mu estimates at high shrinkage
    let the optimizer still snap to fully concentrated corner solutions,
    with the "winner" decided by numerical noise rather than signal. That
    makes weights MORE erratic at high shrinkage, not less, and confounds
    what this test is actually trying to isolate.
    """
    result_raw = backtest(price_df, shrinkage=0.0, max_weight=0.4)
    result_shrunk = backtest(price_df, shrinkage=0.9, max_weight=0.4)
 
    raw_std = result_raw["weights_history"].std().mean()
    shrunk_std = result_shrunk["weights_history"].std().mean()
 
    assert shrunk_std <= raw_std
 