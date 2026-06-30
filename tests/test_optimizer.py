import pandas as pd
import numpy as np
import pytest
from src.optimizer import (
    portfolio_return, 
    portfolio_variance,
    sharpe_ratio,
    maximize_sharpe_ratio  
) 

# portfolio_return ----------------------------------
 
def test_portfolio_return_equal_weights():
    weights = np.array([0.5, 0.5])
    expected_returns = np.array([0.10, 0.20])
    result = portfolio_return(weights, expected_returns)
    assert np.isclose(result, 0.15)
 
 
def test_portfolio_return_single_asset():
    weights = np.array([1.0, 0.0, 0.0])
    expected_returns = np.array([0.08, 0.12, 0.05])
    result = portfolio_return(weights, expected_returns)
    assert np.isclose(result, 0.08)
 
 
def test_portfolio_return_zero_expected_returns():
    weights = np.array([0.3, 0.3, 0.4])
    expected_returns = np.zeros(3)
    result = portfolio_return(weights, expected_returns)
    assert np.isclose(result, 0.0)
 

# portfolio_variance ----------------------------------
 
def test_portfolio_variance_single_asset():
    weights = np.array([1.0])
    cov_matrix = np.array([[0.04]])  # variance = 0.04
    result = portfolio_variance(weights, cov_matrix)
    assert np.isclose(result, 0.04)
 
 
def test_portfolio_variance_uncorrelated_assets():
    weights = np.array([0.5, 0.5])
    cov_matrix = np.array([
        [0.04, 0.0],
        [0.0, 0.09],
    ])
    # variance = w1^2*var1 + w2^2*var2 (no covariance term)
    expected = (0.5 ** 2) * 0.04 + (0.5 ** 2) * 0.09
    result = portfolio_variance(weights, cov_matrix)
    assert np.isclose(result, expected)
 
 
def test_portfolio_variance_perfectly_correlated_assets():
    # Two identical assets, correlation = 1, equal weights.
    # Combined std dev should equal the individual std dev (no diversification).
    std = 0.2
    cov_matrix = np.array([
        [std ** 2, std ** 2],
        [std ** 2, std ** 2],
    ])
    weights = np.array([0.5, 0.5])
    result = portfolio_variance(weights, cov_matrix)
    assert np.isclose(np.sqrt(result), std)
 
# sharpe_ratio ----------------------------------
 
def test_sharpe_ratio_known_value():
    weights = np.array([1.0])
    expected_returns = np.array([0.10])
    cov_matrix = np.array([[0.04]])  # std dev = 0.2
    risk_free_rate = 0.02
    # (0.10 - 0.02) / 0.2 = 0.4
    result = sharpe_ratio(weights, expected_returns, cov_matrix, risk_free_rate)
    assert np.isclose(result, 0.4)
 
 
def test_sharpe_ratio_zero_volatility_returns_zero():
    weights = np.array([1.0])
    expected_returns = np.array([0.10])
    cov_matrix = np.array([[0.0]])  # zero variance -> zero std dev
    result = sharpe_ratio(weights, expected_returns, cov_matrix)
    assert result == 0.0
 
 
def test_sharpe_ratio_negative_when_below_risk_free_rate():
    weights = np.array([1.0])
    expected_returns = np.array([0.01])
    cov_matrix = np.array([[0.04]])
    risk_free_rate = 0.05
    result = sharpe_ratio(weights, expected_returns, cov_matrix, risk_free_rate)
    assert result < 0
 
 
# maximize_sharpe_ratio ----------------------------------
idx_abc = ["AAA", "BBB", "CCC"]

@pytest.fixture
def sample_weights():
    expected_returns = pd.Series([0.10, 0.15, 0.08], index=idx_abc)
    cov_matrix = pd.DataFrame(
        [[0.04, 0.01, 0.00],
         [0.01, 0.09, 0.02],
         [0.00, 0.02, 0.03]],
        index=idx_abc,
        columns=idx_abc,
    )
    weights = maximize_sharpe_ratio(expected_returns, cov_matrix)
    return weights

def test_maximize_sharpe_ratio_weights_sum_to_one(sample_weights):
    weights = sample_weights
    assert np.isclose(weights.sum(), 1.0, atol=1e-6)
 

def test_maximize_sharpe_ratio_weights_within_bounds(sample_weights):
    weights = sample_weights
    assert (weights >= -1e-6).all()
    assert (weights <= 1.0 + 1e-6).all()
 
 
def test_maximize_sharpe_ratio_returns_series_with_correct_index(sample_weights):
    weights = sample_weights
    assert isinstance(weights, pd.Series)
    assert list(weights.index) == idx_abc
    assert weights.name == "weight"
 
def test_maximize_sharpe_ratio_dominant_asset_gets_more_weight():
    # AAA has a much higher return with the same risk and no correlation,
    # so it should receive most/all of the allocation.
    tickers = ["AAA", "BBB"]
    expected_returns = pd.Series([0.30, 0.05], index=tickers)
    cov_matrix = pd.DataFrame(
        [[0.04, 0.00],
         [0.00, 0.04]],
        index=tickers,
        columns=tickers,
    )
    weights = maximize_sharpe_ratio(expected_returns, cov_matrix)
    assert weights["AAA"] > weights["BBB"]
 
 
def test_maximize_sharpe_ratio_identical_assets_split_evenly():
    # two identical, uncorrelated assets should receive roughly equal weight.
    tickers = ["AAA", "BBB"]
    expected_returns = pd.Series([0.10, 0.10], index=tickers)
    cov_matrix = pd.DataFrame(
        [[0.04, 0.00],
         [0.00, 0.04]],
        index=tickers,
        columns=tickers,
    )
    weights = maximize_sharpe_ratio(expected_returns, cov_matrix)
    assert np.isclose(weights["AAA"], weights["BBB"], atol=1e-3)
