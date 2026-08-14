"""Module 4: Simulate historical performance."""

import pandas as pd
from src.math_engine import compute_returns, compute_expected_returns, compute_covariance
from src.optimizer import maximize_sharpe_ratio, shrink_expected_returns


def backtest(
    prices: pd.DataFrame,
    risk_free_rate: float = 0.0,
    lookback_days: int = 252,
    max_weight: float = 1.0,
    shrinkage: float = 0.0,
):
    """Simulate historical portfolio performance using a rolling optimization strategy.
    
    Rebalances the portfolio at month-end dates using the maximum Sharpe ratio strategy.
    For each rebalancing date, computes optimal weights from historical returns (lookback
    window), then applies those weights to the next month's returns. This simulates a
    realistic dynamic trading strategy where allocations respond to changing market conditions.
    
    Args:
        prices (pd.DataFrame): Adjusted close prices indexed by date, one column per ticker.
        risk_free_rate (float): Annualized risk-free rate for Sharpe ratio calculation
            (default 0.0).
        lookback_days (int): Number of historical trading days to use for each optimization;
            use 252 for approximately one year of daily data (default 252).
        max_weight (float): Maximum weight allowed for any single asset in the portfolio;
            1.0 means no concentration limit, 0.1 means each asset capped at 10%
            (default 1.0).
        shrinkage (float): Shrinkage intensity for return estimates (0 = no shrinkage,
            1 = all assets get equal return estimate); helps stabilize optimization
            with limited data (default 0.0).
    
    Returns:
        dict: A dictionary with the following keys:
            - 'returns' (pd.Series): Monthly portfolio returns indexed by rebalance date.
            - 'cumulative_value' (pd.Series): Cumulative portfolio value (starting at 1.0).
            - 'weights_history' (pd.DataFrame): Portfolio weights at each rebalance date,
              indexed by rebalance date with one column per ticker.
              Returns an empty dict if insufficient data for any rebalance window.
    """
    log_returns = compute_returns(prices, method="log")
    simple_returns = compute_returns(prices, method="simple")

    month_end_dates = log_returns.resample("ME").last().index
    portfolio_returns = []
    weights_history = {}

    for i in range(len(month_end_dates) - 1):
        rebal_date = month_end_dates[i]
        next_rebal_date = month_end_dates[i + 1]

        window = log_returns.loc[:rebal_date].tail(lookback_days)
        if len(window) < lookback_days:
            continue

        exp_returns = shrink_expected_returns(compute_expected_returns(window), shrinkage=shrinkage)
        cov_matrix = compute_covariance(window)
        weights = maximize_sharpe_ratio(exp_returns, cov_matrix, risk_free_rate, max_weight=max_weight)
        weights_history[rebal_date] = weights

        next_month_returns = simple_returns.loc[rebal_date:next_rebal_date].iloc[1:]
        port_returns_next_month = next_month_returns @ weights
        portfolio_returns.append(port_returns_next_month)

    if not portfolio_returns:
        return {"returns": pd.Series(dtype=float), "cumulative_value": pd.Series(dtype=float)}
    portfolio_returns = pd.concat(portfolio_returns).sort_index()
    cumulative_value = (1 + portfolio_returns).cumprod()

    return {
        "returns": portfolio_returns,
        "cumulative_value": cumulative_value,
        "weights_history": pd.DataFrame(weights_history).T,
    }