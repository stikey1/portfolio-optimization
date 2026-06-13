"""Streamlit dashboard for portfolio optimization."""

import streamlit as st

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

st.info("Connect modules in `src/` to enable full workflow.")
