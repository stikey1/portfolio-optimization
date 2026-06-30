"""Module 3: Compute optimal portfolio weights."""

import numpy as np
import pandas as pd
from scipy.optimize import minimize

# Args:
#   expected_returns (mu): Expected return per asset.
#   cov_matrix (Sigma): Covariance matrix of returns.
#   risk_free_rate (rf): Annual risk-free rate.
#   sharpe_ratio = (portfolio_return - rf) / portfolio_std_dev

# goal: minimize the neg sharpe_ratio, subject to sum(weight) = 1 and weight >= 0

def portfolio_return(weights: np.ndarray, expected_returns: np.ndarray) -> float:
    # expected returns = mu, weights = w
    return weights @ expected_returns

def portfolio_variance(weights: np.ndarray, cov_matrix: np.ndarray) -> float:
    # covariance matrix = Sigma, weights = w
    return weights.T @ cov_matrix @ weights

def sharpe_ratio(weights : np.ndarray, expected_returns: np.ndarray, cov_matrix: np.ndarray, risk_free_rate: float = 0.0) -> float:
    numerator = portfolio_return(weights, expected_returns) - risk_free_rate
    denominator = np.sqrt(portfolio_variance(weights, cov_matrix))
    return numerator / denominator if denominator != 0 else 0.0

def maximize_sharpe_ratio( expected_returns: pd.Series,cov_matrix: pd.DataFrame,
                            risk_free_rate: float = 0.0) -> pd.Series:
    n = len(expected_returns)
    tickers = expected_returns.index

    # lambda fucntion to minimize negative Sharpe ratio
    neg_sharpe = lambda w: -sharpe_ratio(w, expected_returns.values, cov_matrix.values, risk_free_rate)

    # sum of weights must equal 1, weights must be between 0 and 1
    constraints = {"type": "eq", "fun": lambda w: np.sum(w) - 1.0}
    bounds = [(0.0, 1.0)] * n
    x0 = np.ones(n) / n

    # minimize with sequantial least squares programming for constrained optimization
    result = minimize(neg_sharpe, x0, method="SLSQP", bounds=bounds, constraints=constraints)
    return pd.Series(result.x, index=tickers, name="weight")
