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

def maximize_sharpe_ratio(
    expected_returns: pd.Series,
    cov_matrix: pd.DataFrame,
    risk_free_rate: float = 0.0,
    max_weight: float = 1.0,
) -> pd.Series:
    n = len(expected_returns)
    tickers = expected_returns.index

    def neg_sharpe(w):
        return -sharpe_ratio(w, expected_returns.values, cov_matrix.values, risk_free_rate)

    constraints = {"type": "eq", "fun": lambda w: np.sum(w) - 1.0}
    bounds = [(0.0, max_weight)] * n
    x0 = np.ones(n) / n

    result = minimize(neg_sharpe, x0, method="SLSQP", bounds=bounds, constraints=constraints)
    return pd.Series(result.x, index=tickers, name="weight")


def minimize_variance(expected_returns: pd.Series, cov_matrix: pd.DataFrame,
                       target_return: float | None = None) -> pd.Series:
    """Find portfolio weights that minimize variance.

    If target_return is None, this is the Global Minimum Variance (GMV)
    portfolio — the leftmost point of the efficient frontier.
    If target_return is set, this finds the min-variance portfolio that
    achieves exactly that return — used to trace the frontier itself.
    """
    n = len(expected_returns)
    tickers = expected_returns.index

    def variance(w):
        return portfolio_variance(w, cov_matrix.values)

    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
    if target_return is not None:
        constraints.append({
            "type": "eq",
            "fun": lambda w: portfolio_return(w, expected_returns.values) - target_return
        })

    bounds = [(0.0, 1.0)] * n
    x0 = np.ones(n) / n

    result = minimize(variance, x0, method="SLSQP", bounds=bounds, constraints=constraints)
    return pd.Series(result.x, index=tickers, name="weight")





def efficient_frontier(expected_returns: pd.Series, cov_matrix: pd.DataFrame,
                        n_points: int = 50) -> pd.DataFrame:
    """Trace the efficient frontier by minimizing variance across a range
    of target returns.

    Returns:
        DataFrame with columns ['target_return', 'volatility'], one row
        per point on the frontier, plus a 'weights' column (Series per row)
        for handing off to the allocation chart if needed.
    """
    min_ret = expected_returns.min()
    max_ret = expected_returns.max()
    target_returns = np.linspace(min_ret, max_ret, n_points)

    records = []
    for target in target_returns:
        weights = minimize_variance(expected_returns, cov_matrix, target_return=target)
        vol = np.sqrt(portfolio_variance(weights.values, cov_matrix.values))
        records.append({
            "target_return": target,
            "volatility": vol,
            "weights": weights,
        })

    return pd.DataFrame(records)

def shrink_expected_returns(expected_returns: pd.Series, shrinkage: float = 0.5) -> pd.Series:
    """Pull noisy per-asset return estimates toward the average.
    shrinkage=0 -> no shrinkage (raw estimates). shrinkage=1 -> full shrinkage
    (every asset gets the same return estimate, i.e. mu has zero effect).
    """
    grand_mean = expected_returns.mean()
    return expected_returns * (1 - shrinkage) + grand_mean * shrinkage