"""Module 5: Calculate risk metrics (Sharpe, Max Drawdown)."""

import numpy as np
import pandas as pd

# Args:
#   returns: per period rate of change --> SAME FOR ALL FUNCTIONS
#   risk_free_rate: annualized risk-free rate, default 0.0
#   periods_per_year: 252 for daily, 12 for monthly


# raw, bottom-line outcome (over time)
def cumulative_returns(returns: pd.Series) -> pd.Series:
    # fill NaN with 0
    clean_returns = returns.fillna(0.0)

    return (1 + clean_returns).cumprod() - 1

# return earned per unit of risk, annualized
def annualized_sharpe_ratio(returns: pd.Series, risk_free_rate: float = 0.0,
                            periods_per_year: int = 252,) -> float:
    rf_per_period = (1 + risk_free_rate) ** (1 / periods_per_year) - 1
    excess_returns = returns.dropna() - rf_per_period

    # not enough data to compute variance --> return NaN
    if len(excess_returns) < 2:
        return np.nan
    
    std_dev = excess_returns.std()
    if std_dev == 0:
        return 0
    return (excess_returns.mean() / std_dev) * np.sqrt(periods_per_year)

# worst case loss
# or return as tuple[float, pd.Timestamp, pd.Timestamp] for specific historical period?
def max_drawdown(returns: pd.Series) -> float:
    if returns.empty:
        return np.nan
    
    # add baseline
    cumulated = pd.Series([1.0] + (1 + returns.fillna(0.0)).cumprod().tolist())
    peak = cumulated.cummax()
    drawdown = (cumulated - peak) / peak
    return drawdown.min()

# how much portfolio value fluctuates, annualized
def annualized_volatility(returns: pd.Series, periods_per_year: int = 252) -> float:
    return returns.dropna().std() * np.sqrt(periods_per_year)