"""Module 3: Compute optimal portfolio weights."""

import numpy as np
import pandas as pd
from scipy.optimize import minimize


def portfolio_return(weights: np.ndarray, expected_returns: np.ndarray) -> float:
    return 0.0

def portfolio_variance(weights: np.ndarray, cov_matrix: np.ndarray) -> float:

    return 0.0

def sharpe_ratio(weights : np.ndarray, expected_returns: np.ndarray, cov_matrix: np.ndarray, risk_free_rate: float = 0.0) -> float:
    return 0.0
def maximize_sharpe_ratio(expected_returns: np.ndarray, cov_matrix: np.ndarray, risk_free_rate: float = 0.0, asset_names: list[str] = None) -> np.ndarray:
    return 0.0