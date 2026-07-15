"""Streamlit dashboard for portfolio optimization."""
from pathlib import Path
import streamlit as st
# import yfinance as yf
import pandas as pd
from datetime import date, datetime
from zoneinfo import ZoneInfo
import exchange_calendars as xcals
from src.tickers import load_available_tickers

from src.ingestion import YFinanceSource, run_ingestion, load_data, TickerNotCachedError
from src.math_engine import compute_returns, compute_expected_returns, compute_covariance
from src.optimizer import maximize_sharpe_ratio
from src.backtester import backtest
from src.visualization import plot_efficient_frontier, plot_equity_curve

st.set_page_config(page_title="Portfolio Optimizer", layout="wide")

st.title("Portfolio Optimizer")
st.markdown(
    """
    Interactive dashboard for mean-variance portfolio optimization.

    **Modules**
    - **Ingestion** — fetch and clean stock data
    - **Math Engine** — returns and covariance
    - **Optimizer** — optimal weights
    - **Backtester** — historical simulation
    - **Analytics** — Sharpe ratio, max drawdown
    """
)
# ----------------------------------
# sidebar inputs 
with st.sidebar.form("portfolio_inputs"):
    st.subheader("Tickers")
    available_tickers = load_available_tickers()
    dropdown_selection = st.multiselect(
        "Choose common tickers",
        options=available_tickers,
        default=["AAPL", "MSFT", "GOOGL"],
    )
    custom_input = st.text_input(
        "Add other tickers (comma-separated)",
        placeholder="e.g. SNOW, ARM",
    )
    custom_tickers = [t.strip().upper() for t in custom_input.split(",") if t.strip()]

    # Merge, dedupe, preserve order
    selected_tickers = list(dict.fromkeys(dropdown_selection + custom_tickers))

    st.subheader("Lookback window (Trading Days)")
    lookback_years = st.slider("select # of years", 1, 10, 3)

    st.subheader("Risk-free rate")
    risk_free_rate = st.number_input("default is 2%", value=0.02, step=0.005)
    submitted = st.form_submit_button("Run Optimization")

# ----------------------------------
# fetch data (and cache for performance)  
# Check cache coverage before deciding to prompt for ingestion
ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"

@st.cache_data(ttl=3600, show_spinner="Fetching price data...")
def get_price_data(tickers: tuple[str, ...], start: date, end: date) -> pd.DataFrame:
    tickers = list(tickers)
    try:
        return load_data(tickers, data_dir=DATA_DIR)
    except (FileNotFoundError, TickerNotCachedError):
        return run_ingestion(
            YFinanceSource(), tickers, start, end, f"{DATA_DIR}/prices.parquet"
        )

# refresh cached data if user clicks button
if st.sidebar.button("Refresh data"):
    get_price_data.clear()  # clears this function's Streamlit cache
    st.rerun()

# ----------------------------------
# start/end day calculations 
# NYSE Calendar
xnys = xcals.get_calendar("XNYS")
now = datetime.now(ZoneInfo("America/New_York"))

# Market closes at 4:00 PM ET
if now.hour < 16:
    end = xnys.date_to_session(now.date(), direction="previous")
else:
    end = xnys.date_to_session(now.date(), direction="none")

lookback_days = lookback_years * 252
start = xnys.sessions_window(end, -lookback_days + 1)[0]
# ----------------------------------
# main app logic
if submitted:
    print("Tickers:", selected_tickers)
    print("Start:", start)
    print("End:", end)
    # pass selected_tickers as tuple, st.cache_data needs hashable args
    prices = get_price_data(tuple(selected_tickers), start, end)
    
    # --- full period stats ---
    simple_returns = compute_returns(prices, method="simple")
    mu = compute_expected_returns(simple_returns)
    sigma = compute_covariance(simple_returns)
    weights = maximize_sharpe_ratio(mu, sigma, risk_free_rate)  # whatever your optimizer's entrypoint is
    
    # --- backtest ---
    rolling_window_years = max(1, lookback_years - 2)
    backtest_result = backtest(prices, risk_free_rate, rolling_window_years * 252)
    print("calculations done")

    # --- benchmark (equal-weight, rebased to strategy's start date) ---
    strategy_cum = backtest_result["cumulative_value"]
    if not strategy_cum.empty:
        strategy_start = strategy_cum.index[0]
        equal_weight_returns = simple_returns.mean(axis=1)
        equal_weight_cum_full = (1 + equal_weight_returns).cumprod()
        equal_weight_cum = equal_weight_cum_full / equal_weight_cum_full.loc[strategy_start]
        equal_weight_cum = equal_weight_cum.loc[strategy_start:]
        benchmarks = {"Equal-Weight": equal_weight_cum}
    else:
        benchmarks = {}
        st.warning("Not enough price history for the selected lookback window.")

    st.session_state["results"] = {
        "mu": mu,
        "sigma": sigma,
        "weights": weights,
        "backtest": backtest_result,
        "benchmarks": benchmarks,
        "risk_free_rate": risk_free_rate,
    }

    if "results" in st.session_state:
        r = st.session_state["results"]

        # --- visualizations ---
        st.header("Efficient Frontier")
        st.plotly_chart(
            plot_efficient_frontier(r["mu"], r["sigma"], risk_free_rate=risk_free_rate),
            use_container_width=True,
        )

        st.divider()

        st.subheader("Backtest: Strategy vs. Benchmark")
        st.plotly_chart(
            plot_equity_curve(r["backtest"]["cumulative_value"], r["benchmarks"]),
            use_container_width=True,
        )