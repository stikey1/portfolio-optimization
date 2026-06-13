"""Module 5: Calculate risk metrics (Sharpe, Max Drawdown)."""

import numpy as np
import pandas as pd


def sharpe_ratio(
    returns: pd.Series,
    risk_free_rate: float = 0.0,
    periods_per_year: int = 252,
) -> float:
    """Annualized Sharpe ratio for a return series."""
    excess = returns - risk_free_rate / periods_per_year
    if excess.std() == 0:
        return 0.0
    return float(np.sqrt(periods_per_year) * excess.mean() / excess.std())


def max_drawdown(values: pd.Series) -> float:
    """Maximum peak-to-trough drawdown as a positive fraction."""
    rolling_max = values.cummax()
    drawdown = (rolling_max - values) / rolling_max
    return float(drawdown.max())
