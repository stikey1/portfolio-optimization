"""Module 4: Simulate historical performance."""

import pandas as pd


def simulate_portfolio(
    returns: pd.DataFrame,
    weights: pd.Series,
    initial_value: float = 10_000.0,
) -> pd.Series:
    """Simulate portfolio value over time using fixed weights.

    Args:
        returns: Daily returns for each asset.
        weights: Portfolio weights (must align with return columns).
        initial_value: Starting portfolio value.

    Returns:
        Series of portfolio values indexed by date.
    """
    aligned_weights = weights.reindex(returns.columns).fillna(0.0)
    portfolio_returns = (returns * aligned_weights).sum(axis=1)
    return initial_value * (1 + portfolio_returns).cumprod()
