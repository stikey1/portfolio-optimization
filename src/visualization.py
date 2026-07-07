"""Module 5: Interactive visualizations for the portfolio optimizer."""

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from src.optimizer import (
    portfolio_return,
    portfolio_variance,
    maximize_sharpe_ratio,
    minimize_variance,
    efficient_frontier,
)


def plot_efficient_frontier(
    expected_returns: pd.Series,
    cov_matrix: pd.DataFrame,
    risk_free_rate: float = 0.0,
    n_points: int = 50,
) -> go.Figure:
    # The actual frontier boundary, traced point-by-point via constrained
    # minimize_variance(target_return=...) -- not an approximation.
    frontier = efficient_frontier(expected_returns, cov_matrix, n_points=n_points)

    # GMV: target_return=None case -- leftmost point on that same curve.
    gmv_weights = minimize_variance(expected_returns, cov_matrix)
    gmv_ret = portfolio_return(gmv_weights.values, expected_returns.values)
    gmv_vol = np.sqrt(portfolio_variance(gmv_weights.values, cov_matrix.values))

    # Max Sharpe: the tangency point, found by the *other* optimizer.
    ms_weights = maximize_sharpe_ratio(expected_returns, cov_matrix, risk_free_rate)
    ms_ret = portfolio_return(ms_weights.values, expected_returns.values)
    ms_vol = np.sqrt(portfolio_variance(ms_weights.values, cov_matrix.values))

    fig = go.Figure()

    # 1. The frontier curve itself -- a LINE, since it's an ordered sequence
    #    of points along one boundary, not a scatter of independent samples.
    fig.add_trace(go.Scatter(
        x=frontier["volatility"],
        y=frontier["target_return"],
        mode="lines",
        line=dict(color="black", width=2),
        name="Efficient Frontier",
        hovertemplate="Vol: %{x:.2%}<br>Return: %{y:.2%}<extra></extra>",
    ))

    # 2. Capital Market Line: (0, rf) through the tangency portfolio.
    #    Extend it a bit past the tangency point to show leverage territory.
    cml_x = [0, ms_vol * 1.3]
    slope = (ms_ret - risk_free_rate) / ms_vol
    cml_y = [risk_free_rate, risk_free_rate + slope * ms_vol * 1.3]
    fig.add_trace(go.Scatter(
        x=cml_x, y=cml_y, mode="lines",
        line=dict(color="gray", dash="dash", width=1.5),
        name="Capital Market Line",
    ))

    # 3. GMV -- single highlighted point, mode="markers" not "lines" since
    #    it's one specific portfolio, not a path.
    fig.add_trace(go.Scatter(
        x=[gmv_vol], y=[gmv_ret],
        mode="markers",
        marker=dict(size=16, color="blue", symbol="star"),
        name="Global Min Variance",
        hovertemplate="GMV<br>Vol: %{x:.2%}<br>Return: %{y:.2%}<extra></extra>",
    ))

    # 4. Max Sharpe -- same idea, different point.
    fig.add_trace(go.Scatter(
        x=[ms_vol], y=[ms_ret],
        mode="markers",
        marker=dict(size=16, color="red", symbol="star"),
        name="Max Sharpe (Tangency)",
        hovertemplate="Max Sharpe<br>Vol: %{x:.2%}<br>Return: %{y:.2%}<extra></extra>",
    ))

    fig.update_layout(
        title="Efficient Frontier",
        xaxis_title="Volatility (annualized σ)",
        yaxis_title="Expected Return (annualized μ)",
        xaxis_tickformat=".0%",
        yaxis_tickformat=".0%",
        template="plotly_white",
        hovermode="closest",
    )
    return fig


def plot_equity_curve(
    strategy_cumulative: pd.Series,
    benchmarks: dict[str, pd.Series],
    title: str = "Backtest: Strategy vs. Benchmark",
) -> go.Figure:
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=strategy_cumulative.index,
        y=strategy_cumulative.values,
        mode="lines",
        line=dict(color="crimson", width=2.5),
        name="Max Sharpe Strategy",
        hovertemplate="%{x|%Y-%m-%d}<br>Value: %{y:.2f}<extra></extra>",
    ))

    colors = ["steelblue", "gray", "seagreen"]
    for (label, series), color in zip(benchmarks.items(), colors):
        fig.add_trace(go.Scatter(
            x=series.index,
            y=series.values,
            mode="lines",
            line=dict(color=color, width=1.5, dash="dot"),
            name=label,
            hovertemplate="%{x|%Y-%m-%d}<br>Value: %{y:.2f}<extra></extra>",
        ))

    fig.update_layout(
        title=title,
        xaxis_title="Date",
        yaxis_title="Growth of $1",
        template="plotly_white",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


if __name__ == "__main__":
    from src.mock_data import generate_mock_prices
    from src.math_engine import compute_returns, compute_expected_returns, compute_covariance
    from src.backtester import backtest

    # Swap this line for src.ingestion.load_data(...) once you're ready
    # to test against real market data instead of synthetic GBM paths.
    prices = generate_mock_prices()
    log_returns = compute_returns(prices, method="log")
    mu = compute_expected_returns(log_returns)
    sigma = compute_covariance(log_returns)

    fig1 = plot_efficient_frontier(mu, sigma, risk_free_rate=0.04)
    fig1.show()

    result = backtest(prices)

    strategy_start = result["cumulative_value"].index[0]
    equal_weight_returns = compute_returns(prices, method="simple").mean(axis=1)
    equal_weight_cum_full = (1 + equal_weight_returns).cumprod()

    # Rebase so both lines start at 1.0 on the same date — fixes the
    # artificial gap caused by the strategy starting its clock later
    # (once the 252-day lookback window is filled) than equal-weight,
    # which starts from day one of the full price history.
    equal_weight_cum = equal_weight_cum_full / equal_weight_cum_full.loc[strategy_start]
    equal_weight_cum = equal_weight_cum.loc[strategy_start:]

    fig2 = plot_equity_curve(result["cumulative_value"], {"Equal-Weight": equal_weight_cum})
    fig2.show()