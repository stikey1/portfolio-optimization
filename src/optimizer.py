"""Module 3: Compute optimal portfolio weights."""

import numpy as np
import pandas as pd
from scipy.optimize import minimize


def max_sharpe_weights(
    expected_returns: pd.Series,
    cov_matrix: pd.DataFrame,
    risk_free_rate: float = 0.0,
) -> pd.Series:
    """Find weights that maximize the Sharpe ratio.

    Args:
        expected_returns: Expected return per asset.
        cov_matrix: Covariance matrix of returns.
        risk_free_rate: Annual risk-free rate.

    Returns:
        Optimal portfolio weights summing to 1.
    """
    n = len(expected_returns)
    tickers = expected_returns.index

    def neg_sharpe(weights: np.ndarray) -> float:
        port_return = weights @ expected_returns.values
        port_vol = np.sqrt(weights @ cov_matrix.values @ weights)
        if port_vol == 0:
            return 0.0
        return -(port_return - risk_free_rate) / port_vol

    constraints = {"type": "eq", "fun": lambda w: np.sum(w) - 1.0}
    bounds = [(0.0, 1.0)] * n
    x0 = np.ones(n) / n

    result = minimize(neg_sharpe, x0, method="SLSQP", bounds=bounds, constraints=constraints)
    return pd.Series(result.x, index=tickers, name="weight")
