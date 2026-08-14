"""Module 5: Calculate risk metrics (Sharpe, Max Drawdown)."""

import numpy as np
import pandas as pd

# Args:
#   returns: per period rate of change --> SAME FOR ALL FUNCTIONS
#   risk_free_rate: annualized risk-free rate, default 0.0
#   periods_per_year: 252 for daily, 12 for monthly


def cumulative_returns(returns: pd.Series) -> pd.Series:
    """Calculate cumulative returns over time from periodic returns.
    
    Computes the running cumulative product of (1 + returns), representing
    the growth of a $1 investment from the start of the series. Missing values
    are treated as 0% return for that period.
    
    Args:
        returns (pd.Series): Periodic returns (e.g., daily or monthly changes).
    
    Returns:
        pd.Series: Cumulative returns indexed by date, where 0.10 means 10% total gain.

    """
    # fill NaN with 0
    clean_returns = returns.fillna(0.0)

    return (1 + clean_returns).cumprod() - 1

def annualized_sharpe_ratio(returns: pd.Series, risk_free_rate: float = 0.0,
                            periods_per_year: int = 252,) -> float:
    """Calculate the Sharpe ratio: excess return per unit of risk, annualized.
    
    Measures how much excess return is earned for each unit of volatility taken.
    A higher Sharpe ratio indicates better risk-adjusted performance.
    
    Args:
        returns (pd.Series): Periodic returns (e.g., daily or monthly).
        risk_free_rate (float): Annualized risk-free rate (default 0.0). Used to
            calculate excess returns.
        periods_per_year (int): Number of periods per year; 252 for daily data,
            12 for monthly, 52 for weekly (default 252).
    
    Returns:
        float: Annualized Sharpe ratio. Returns 0 if volatility is zero, NaN if
            fewer than 2 data points.

    """
    rf_per_period = (1 + risk_free_rate) ** (1 / periods_per_year) - 1
    excess_returns = returns.dropna() - rf_per_period

    # not enough data to compute variance --> return NaN
    if len(excess_returns) < 2:
        return np.nan
    
    std_dev = excess_returns.std()
    if std_dev == 0:
        return 0
    return (excess_returns.mean() / std_dev) * np.sqrt(periods_per_year)

def max_drawdown(returns: pd.Series) -> float:
    """Calculate the maximum drawdown: largest peak-to-trough decline.
    
    Measures the worst-case loss (as a percentage) from any peak in cumulative
    value to the subsequent trough. A drawdown of -0.20 means the portfolio lost
    20% from its peak before recovering.
    
    Args:
        returns (pd.Series): Periodic returns (e.g., daily or monthly).
    
    Returns:
        float: Maximum drawdown as a decimal (negative), e.g., -0.25 for 25% loss.
            Returns NaN if returns are empty.

    """
    if returns.empty:
        return np.nan
    
    # add baseline
    cumulated = pd.Series([1.0] + (1 + returns.fillna(0.0)).cumprod().tolist())
    peak = cumulated.cummax()
    drawdown = (cumulated - peak) / peak
    return drawdown.min()

def annualized_volatility(returns: pd.Series, periods_per_year: int = 252) -> float:
    """Calculate annualized volatility: standard deviation of returns, scaled to annual.
    
    Volatility (standard deviation of returns) is a measure of portfolio risk.
    Higher volatility indicates more fluctuation in value. This function annualizes
    the periodic volatility by scaling it appropriately.
    
    Args:
        returns (pd.Series): Periodic returns (e.g., daily or monthly).
        periods_per_year (int): Number of periods per year; 252 for daily data,
            12 for monthly, 52 for weekly (default 252).
    
    Returns:
        float: Annualized volatility as a decimal, e.g., 0.15 for 15% annual volatility.

    """
    return returns.dropna().std() * np.sqrt(periods_per_year)