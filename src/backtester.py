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