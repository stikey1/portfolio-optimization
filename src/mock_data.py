"""Mock price data generator -- Geometric Brownian Motion, same approach as
the pytest fixtures in tests/test_backtester.py, so the numbers you see in
the plots are consistent with what your test suite already validates against.

GBM model: dS = mu * S * dt + sigma * S * dW
Discretized:  S_t = S_{t-1} * exp((mu - 0.5*sigma^2)*dt + sigma*sqrt(dt)*Z)
where Z ~ N(0, 1).
"""

import numpy as np
import pandas as pd


def generate_mock_prices(
    tickers: list[str] = None,
    n_days: int = 504,
    start_date: str = "2022-01-03",
    start_price: float = 100.0,
    annual_drift: float = 0.08,
    annual_vol: float = 0.25,
    correlation: float = 0.3,
    seed: int = 9,
) -> pd.DataFrame:
    """Generate correlated GBM price paths for multiple tickers.

    Args:
        tickers: asset names. Defaults to a 5-asset universe.
        n_days: number of trading days (252 = 1 year).
        start_date: first date in the series.
        start_price: starting price for every asset.
        annual_drift: mu, annualized expected return.
        annual_vol: sigma, annualized volatility.
        correlation: pairwise correlation between assets' random shocks.
            Real equities aren't independent -- ignoring this would make
            the covariance matrix in your optimizer artificially diagonal,
            which defeats the purpose of testing diversification.
        seed: for reproducibility across runs.

    Returns:
        DataFrame of prices, index=dates, columns=tickers.
    """
    if tickers is None:
        tickers = ["AAPL", "MSFT", "GOOG", "AMZN", "TSLA"]

    rng = np.random.default_rng(seed)
    n_assets = len(tickers)
    dt = 1 / 252

    # Correlated shocks: build a correlation matrix (uniform pairwise
    # correlation off-diagonal), then use its Cholesky factor to turn
    # independent standard normals into correlated ones.
    corr_matrix = np.full((n_assets, n_assets), correlation)
    np.fill_diagonal(corr_matrix, 1.0)
    chol = np.linalg.cholesky(corr_matrix)

    independent_shocks = rng.standard_normal((n_days, n_assets))
    correlated_shocks = independent_shocks @ chol.T  # shape (n_days, n_assets)

    # Give each asset a slightly different drift/vol so the frontier isn't
    # degenerate (identical assets would collapse GMV and Max Sharpe
    # toward similar corners rather than showing a meaningful spread).
    drifts = annual_drift + rng.uniform(-0.04, 0.04, n_assets)
    vols = annual_vol + rng.uniform(-0.08, 0.08, n_assets)

    log_returns = (drifts - 0.5 * vols**2) * dt + vols * np.sqrt(dt) * correlated_shocks

    log_prices = np.log(start_price) + np.cumsum(log_returns, axis=0)
    prices = np.exp(log_prices)

    dates = pd.bdate_range(start=start_date, periods=n_days)
    return pd.DataFrame(prices, index=dates, columns=tickers)


if __name__ == "__main__":
    prices = generate_mock_prices()
    print(prices.head())
    print(f"\n{len(prices)} trading days, {len(prices.columns)} assets")
    print(f"Date range: {prices.index[0].date()} to {prices.index[-1].date()}")