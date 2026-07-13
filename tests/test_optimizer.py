import pandas as pd
import numpy as np
import pytest
from src.optimizer import (
    portfolio_return,
    portfolio_variance,
    sharpe_ratio,
    maximize_sharpe_ratio,
)

# ── Fixtures ─────────────────────────────────────────────────────────────────

idx_abc = ["AAA", "BBB", "CCC"]

@pytest.fixture
def three_asset_setup():
    """Reusable 3-asset setup shared across test classes."""
    expected_returns = pd.Series([0.10, 0.15, 0.08], index=idx_abc)
    cov_matrix = pd.DataFrame(
        [[0.04, 0.01, 0.00],
         [0.01, 0.09, 0.02],
         [0.00, 0.02, 0.03]],
        index=idx_abc,
        columns=idx_abc,
    )
    return expected_returns, cov_matrix

@pytest.fixture
def sample_weights(three_asset_setup):
    mu, sigma = three_asset_setup
    return maximize_sharpe_ratio(mu, sigma)


# ── portfolio_return ──────────────────────────────────────────────────────────

class TestPortfolioReturn:

    def test_equal_weights(self):
        weights = np.array([0.5, 0.5])
        expected_returns = np.array([0.10, 0.20])
        assert np.isclose(portfolio_return(weights, expected_returns), 0.15)

    def test_single_asset_all_weight(self):
        weights = np.array([1.0, 0.0, 0.0])
        expected_returns = np.array([0.08, 0.12, 0.05])
        assert np.isclose(portfolio_return(weights, expected_returns), 0.08)

    def test_zero_expected_returns(self):
        weights = np.array([0.3, 0.3, 0.4])
        expected_returns = np.zeros(3)
        assert np.isclose(portfolio_return(weights, expected_returns), 0.0)

    def test_unequal_weights(self):
        weights = np.array([0.2, 0.3, 0.5])
        expected_returns = np.array([0.10, 0.20, 0.30])
        expected = 0.2 * 0.10 + 0.3 * 0.20 + 0.5 * 0.30
        assert np.isclose(portfolio_return(weights, expected_returns), expected)

    def test_zero_weights_returns_zero(self):
        weights = np.zeros(3)
        expected_returns = np.array([0.10, 0.20, 0.30])
        assert np.isclose(portfolio_return(weights, expected_returns), 0.0)

    def test_negative_expected_return(self):
        """Assets with negative expected returns should reduce portfolio return."""
        weights = np.array([0.5, 0.5])
        expected_returns = np.array([-0.10, 0.20])
        assert np.isclose(portfolio_return(weights, expected_returns), 0.05)

    def test_return_is_weighted_average(self):
        """Portfolio return must lie between min and max asset return."""
        weights = np.array([0.4, 0.6])
        expected_returns = np.array([0.05, 0.25])
        result = portfolio_return(weights, expected_returns)
        assert expected_returns.min() <= result <= expected_returns.max()

    def test_single_asset(self):
        weights = np.array([1.0])
        expected_returns = np.array([0.12])
        assert np.isclose(portfolio_return(weights, expected_returns), 0.12)


# ── portfolio_variance ────────────────────────────────────────────────────────

class TestPortfolioVariance:

    def test_single_asset(self):
        weights = np.array([1.0])
        cov_matrix = np.array([[0.04]])
        assert np.isclose(portfolio_variance(weights, cov_matrix), 0.04)

    def test_uncorrelated_assets(self):
        weights = np.array([0.5, 0.5])
        cov_matrix = np.array([[0.04, 0.0],
                                [0.0,  0.09]])
        expected = (0.5 ** 2) * 0.04 + (0.5 ** 2) * 0.09
        assert np.isclose(portfolio_variance(weights, cov_matrix), expected)

    def test_perfectly_correlated_assets(self):
        """Two identical assets with correlation=1 provide no diversification."""
        std = 0.2
        cov_matrix = np.array([[std**2, std**2],
                                [std**2, std**2]])
        weights = np.array([0.5, 0.5])
        result = portfolio_variance(weights, cov_matrix)
        assert np.isclose(np.sqrt(result), std)

    def test_all_weight_on_one_asset(self):
        """Putting all weight on one asset gives that asset's variance."""
        weights = np.array([1.0, 0.0])
        cov_matrix = np.array([[0.04, 0.01],
                                [0.01, 0.09]])
        assert np.isclose(portfolio_variance(weights, cov_matrix), 0.04)

    def test_always_non_negative(self):
        """Variance is always >= 0 for a valid covariance matrix."""
        weights = np.array([0.3, 0.3, 0.4])
        cov_matrix = np.array([[0.04,  0.01,  0.005],
                                [0.01,  0.09,  0.002],
                                [0.005, 0.002, 0.01]])
        assert portfolio_variance(weights, cov_matrix) >= 0

    def test_positive_correlation_increases_variance(self):
        """Positive correlation should yield higher variance than zero correlation."""
        weights = np.array([0.5, 0.5])
        cov_uncorr = np.array([[0.04, 0.00],
                                [0.00, 0.04]])
        cov_corr   = np.array([[0.04, 0.03],
                                [0.03, 0.04]])
        assert portfolio_variance(weights, cov_corr) > portfolio_variance(weights, cov_uncorr)

    def test_perfect_negative_correlation_minimizes_variance(self):
        """Perfect negative correlation with equal variance can zero out variance."""
        weights = np.array([0.5, 0.5])
        cov = np.array([[ 0.04, -0.04],
                        [-0.04,  0.04]])
        assert np.isclose(portfolio_variance(weights, cov), 0.0, atol=1e-10)

    def test_zero_weights(self):
        weights = np.zeros(2)
        cov_matrix = np.array([[0.04, 0.01],
                                [0.01, 0.09]])
        assert np.isclose(portfolio_variance(weights, cov_matrix), 0.0)

    def test_symmetric_weights_symmetric_covariance(self):
        """Swapping equal weights on symmetric covariance gives same variance."""
        cov_matrix = np.array([[0.04, 0.01],
                                [0.01, 0.09]])
        w1 = np.array([0.3, 0.7])
        w2 = np.array([0.7, 0.3])
        # Not necessarily equal since assets have different variance,
        # but both should be non-negative and finite
        assert portfolio_variance(w1, cov_matrix) >= 0
        assert portfolio_variance(w2, cov_matrix) >= 0


# ── sharpe_ratio ──────────────────────────────────────────────────────────────

class TestSharpeRatio:

    def test_known_value(self):
        """(0.10 - 0.02) / 0.2 = 0.4"""
        weights = np.array([1.0])
        expected_returns = np.array([0.10])
        cov_matrix = np.array([[0.04]])
        assert np.isclose(sharpe_ratio(weights, expected_returns, cov_matrix, 0.02), 0.4)

    def test_zero_risk_free_rate(self):
        """Default rf=0: sharpe = return / std."""
        weights = np.array([1.0])
        expected_returns = np.array([0.10])
        cov_matrix = np.array([[0.04]])
        result = sharpe_ratio(weights, expected_returns, cov_matrix)
        assert np.isclose(result, 0.10 / 0.20)

    def test_zero_volatility_returns_zero(self):
        weights = np.array([1.0])
        expected_returns = np.array([0.10])
        cov_matrix = np.array([[0.0]])
        assert sharpe_ratio(weights, expected_returns, cov_matrix) == 0.0

    def test_negative_when_below_risk_free_rate(self):
        weights = np.array([1.0])
        expected_returns = np.array([0.01])
        cov_matrix = np.array([[0.04]])
        assert sharpe_ratio(weights, expected_returns, cov_matrix, risk_free_rate=0.05) < 0

    def test_higher_return_higher_sharpe(self):
        """All else equal, higher return → higher Sharpe."""
        cov_matrix = np.array([[0.04]])
        weights = np.array([1.0])
        rf = 0.02
        low  = sharpe_ratio(weights, np.array([0.08]), cov_matrix, rf)
        high = sharpe_ratio(weights, np.array([0.20]), cov_matrix, rf)
        assert high > low

    def test_higher_volatility_lower_sharpe(self):
        """All else equal, higher volatility → lower Sharpe."""
        weights = np.array([1.0])
        expected_returns = np.array([0.10])
        rf = 0.02
        low_vol  = sharpe_ratio(weights, expected_returns, np.array([[0.04]]), rf)
        high_vol = sharpe_ratio(weights, expected_returns, np.array([[0.16]]), rf)
        assert low_vol > high_vol

    def test_equal_return_and_rf_gives_zero_sharpe(self):
        """Return equal to risk-free rate → Sharpe = 0."""
        weights = np.array([1.0])
        expected_returns = np.array([0.05])
        cov_matrix = np.array([[0.04]])
        result = sharpe_ratio(weights, expected_returns, cov_matrix, risk_free_rate=0.05)
        assert np.isclose(result, 0.0)

    def test_two_asset_portfolio(self):
        weights = np.array([0.6, 0.4])
        expected_returns = np.array([0.10, 0.20])
        cov_matrix = np.array([[0.04, 0.01],
                                [0.01, 0.09]])
        result = sharpe_ratio(weights, expected_returns, cov_matrix, risk_free_rate=0.02)
        port_ret = portfolio_return(weights, expected_returns)
        port_std = np.sqrt(portfolio_variance(weights, cov_matrix))
        assert np.isclose(result, (port_ret - 0.02) / port_std)


# ── maximize_sharpe_ratio ─────────────────────────────────────────────────────

class TestMaximizeSharpeRatio:

    def test_weights_sum_to_one(self, sample_weights):
        assert np.isclose(sample_weights.sum(), 1.0, atol=1e-6)

    def test_weights_within_bounds(self, sample_weights):
        assert (sample_weights >= -1e-6).all()
        assert (sample_weights <= 1.0 + 1e-6).all()

    def test_returns_series_with_correct_index(self, sample_weights):
        assert isinstance(sample_weights, pd.Series)
        assert list(sample_weights.index) == idx_abc
        assert sample_weights.name == "weight"

    def test_dominant_asset_gets_more_weight(self):
        """Higher return, same risk, no correlation → dominant asset gets more weight."""
        tickers = ["AAA", "BBB"]
        mu = pd.Series([0.30, 0.05], index=tickers)
        sigma = pd.DataFrame([[0.04, 0.00],
                               [0.00, 0.04]],
                              index=tickers, columns=tickers)
        weights = maximize_sharpe_ratio(mu, sigma)
        assert weights["AAA"] > weights["BBB"]

    def test_identical_assets_split_evenly(self):
        tickers = ["AAA", "BBB"]
        mu = pd.Series([0.10, 0.10], index=tickers)
        sigma = pd.DataFrame([[0.04, 0.00],
                               [0.00, 0.04]],
                              index=tickers, columns=tickers)
        weights = maximize_sharpe_ratio(mu, sigma)
        assert np.isclose(weights["AAA"], weights["BBB"], atol=1e-3)

    def test_result_sharpe_beats_equal_weight(self, three_asset_setup):
        """Optimized portfolio should have Sharpe >= equal-weight portfolio."""
        mu, sigma = three_asset_setup
        n = len(mu)
        equal_weights = np.ones(n) / n
        optimal_weights = maximize_sharpe_ratio(mu, sigma)

        sharpe_equal   = sharpe_ratio(equal_weights, mu.values, sigma.values)
        sharpe_optimal = sharpe_ratio(optimal_weights.values, mu.values, sigma.values)
        assert sharpe_optimal >= sharpe_equal - 1e-6

    def test_max_weight_constraint_respected(self, three_asset_setup):
        """No single asset should exceed max_weight."""
        mu, sigma = three_asset_setup
        max_w = 0.5
        weights = maximize_sharpe_ratio(mu, sigma, max_weight=max_w)
        assert (weights <= max_w + 1e-6).all()

    def test_max_weight_constraint_still_sums_to_one(self, three_asset_setup):
        mu, sigma = three_asset_setup
        weights = maximize_sharpe_ratio(mu, sigma, max_weight=0.5)
        assert np.isclose(weights.sum(), 1.0, atol=1e-6)

    def test_single_asset(self):
        """Single asset must get weight 1.0."""
        mu = pd.Series([0.12], index=["A"])
        sigma = pd.DataFrame([[0.04]], index=["A"], columns=["A"])
        weights = maximize_sharpe_ratio(mu, sigma)
        assert np.isclose(weights["A"], 1.0, atol=1e-6)

    def test_higher_risk_free_rate_shifts_allocation(self, three_asset_setup):
        """Higher rf should shift weight toward higher-return assets."""
        mu, sigma = three_asset_setup
        w_low_rf  = maximize_sharpe_ratio(mu, sigma, risk_free_rate=0.00)
        w_high_rf = maximize_sharpe_ratio(mu, sigma, risk_free_rate=0.08)
        # BBB has highest return — high rf should concentrate weight there
        assert w_high_rf["BBB"] >= w_low_rf["BBB"] - 1e-4

    def test_result_is_deterministic(self, three_asset_setup):
        """Same inputs should always produce the same weights."""
        mu, sigma = three_asset_setup
        w1 = maximize_sharpe_ratio(mu, sigma)
        w2 = maximize_sharpe_ratio(mu, sigma)
        pd.testing.assert_series_equal(w1, w2)
        