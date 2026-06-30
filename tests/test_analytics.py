import pandas as pd
import numpy as np
import pytest
from src.analytics import (
    cumulative_returns,
    annualized_sharpe_ratio,
    max_drawdown,
    annualized_volatility
) 
# cumulative_returns ----------------------------------
class TestCumulativeReturns:
    def test_cumulative_returns_positive_returns(self):
        returns = pd.Series([0.01, 0.02, 0.03])
        expected = pd.Series([0.01, 0.0302, 0.061106])
        result = cumulative_returns(returns)
        np.testing.assert_allclose(result, expected)

    def test_cumulative_returns_negative_returns(self):
        returns = pd.Series([-0.01, -0.02, -0.03])
        expected = pd.Series([-0.01, -0.0298, -0.058906])
        result = cumulative_returns(returns)
        np.testing.assert_allclose(result, expected)

    def test_cumulative_returns_mixed_returns(self):
        returns = pd.Series([0.01, -0.02, 0.03])
        expected = pd.Series([0.01, -0.0102, 0.019494])
        result = cumulative_returns(returns)
        np.testing.assert_allclose(result, expected)

    def test_cumulative_returns_nan_value(self):
        returns = pd.Series([0.01, np.nan, 0.03])
        expected = pd.Series([0.01, 0.01, 0.0403])
        result = cumulative_returns(returns)
        np.testing.assert_allclose(result, expected)


    def test_cumulative_returns_empty_series(self):
        returns = pd.Series([], dtype=float)
        result = cumulative_returns(returns)
        assert result.empty
    
    def test_cumulative_returns_single_value(self):
        returns = pd.Series([0.05])
        result = cumulative_returns(returns)
        np.testing.assert_allclose(result, [0.05])
    
    def test_cumulative_returns_all_nan(self):
        returns = pd.Series([np.nan, np.nan, np.nan])
        result = cumulative_returns(returns)
        np.testing.assert_allclose(result, [0.0, 0.0, 0.0])
    
    def test_cumulative_returns_total_loss(self):
        # -100% return wipes out the portfolio; cumulative should be -1
        returns = pd.Series([-1.0])
        result = cumulative_returns(returns)
        np.testing.assert_allclose(result, [-1.0])
 
# annualized_sharpe_ratio ----------------------------------
 
class TestAnnualizedSR:
    def test_sharpe_ratio_zero_volatility_returns_nan_or_inf(self):
        # constant returns -> std = 0 -> division by zero
        returns = pd.Series([0.01, 0.01, 0.01])
        result = annualized_sharpe_ratio(returns)
        assert result == 0
    
    def test_sharpe_ratio_empty_series_returns_nan(self):
        returns = pd.Series([], dtype=float)
        result = annualized_sharpe_ratio(returns)
        assert np.isnan(result)
    
    
    def test_sharpe_ratio_single_value_returns_nan(self):
        # std of a single value is NaN (ddof=1 default)
        returns = pd.Series([0.02])
        result = annualized_sharpe_ratio(returns)
        assert np.isnan(result)
    
    
    def test_sharpe_ratio_higher_rf_lowers_ratio(self):
        returns = pd.Series([0.01, 0.02, -0.01, 0.015, 0.005])
        low_rf = annualized_sharpe_ratio(returns, risk_free_rate=0.0)
        high_rf = annualized_sharpe_ratio(returns, risk_free_rate=0.10)
        assert high_rf < low_rf
    
    
    def test_sharpe_ratio_contains_nan_in_returns(self):
        returns = pd.Series([0.01, np.nan, 0.02])
        result = annualized_sharpe_ratio(returns)
        expected = annualized_sharpe_ratio(returns.dropna())
        assert not np.isnan(result)
        np.testing.assert_allclose(result, expected)
 
# max_drawdown ----------------------------------

class TestMDD:
    def test_max_drawdown_all_positive_returns_is_zero(self):
        # portfolio only goes up -> no drawdown at all
        returns = pd.Series([0.01, 0.02, 0.01, 0.03])
        result = max_drawdown(returns)
        np.testing.assert_allclose(result, 0.0)
    
    
    def test_max_drawdown_known_decline(self):
        # value goes 1.0 -> 1.10 -> 0.99 (peak 1.10, trough 0.99)
        returns = pd.Series([0.10, -0.10])
        result = max_drawdown(returns)
        expected = (0.99 - 1.10) / 1.10
        np.testing.assert_allclose(result, expected)
    
    
    def test_max_drawdown_full_recovery_still_shows_worst_point(self):
        # drop then full recovery -- max_drawdown should reflect the worst dip,
        # not the final (recovered) value
        returns = pd.Series([-0.50, 1.0])  # down 50%, then back up 100%
        result = max_drawdown(returns)
        assert result < 0
        np.testing.assert_allclose(result, -0.5)
    
    
    def test_max_drawdown_empty_series(self):
        returns = pd.Series([], dtype=float)
        result = max_drawdown(returns)
        assert pd.isna(result)
    
    
    def test_max_drawdown_total_wipeout(self):
        returns = pd.Series([-1.0])
        result = max_drawdown(returns)
        np.testing.assert_allclose(result, -1.0)
 
 
# annualized_volatility ----------------------------------

class TestAnnualizedVolatility:
    def test_annualized_volatility_zero_for_constant_returns(self):
        returns = pd.Series([0.01, 0.01, 0.01])
        result = annualized_volatility(returns)
        np.testing.assert_allclose(result, 0.0)
    
    
    def test_annualized_volatility_scales_with_periods_per_year(self):
        returns = pd.Series([0.01, -0.02, 0.015, -0.005, 0.02])
        daily = annualized_volatility(returns, periods_per_year=252)
        monthly = annualized_volatility(returns, periods_per_year=12)
        # same input std, different annualization factor -> different results
        assert daily > monthly
    
    
    def test_annualized_volatility_single_value_returns_nan(self):
        returns = pd.Series([0.02])
        result = annualized_volatility(returns)
        assert np.isnan(result)
    
    
    def test_annualized_volatility_empty_series_returns_nan(self):
        returns = pd.Series([], dtype=float)
        result = annualized_volatility(returns)
        assert np.isnan(result)
    
    
    def test_annualized_volatility_matches_manual_calculation(self):
        returns = pd.Series([0.01, -0.02, 0.03, 0.005])
        result = annualized_volatility(returns, periods_per_year=252)
        expected = returns.std() * np.sqrt(252)
        np.testing.assert_allclose(result, expected)