# Portfolio Optimizer

Mean-variance portfolio optimization toolkit with a Streamlit dashboard. Download stock data, estimate risk/return, compute optimal weights, backtest, and analyze performance.

## Architecture

```
portfolio-optimization/
├── .github/workflows/   # CI/CD (tests + lint)
├── data/                # Local Parquet storage
├── src/                 # Core modules
│   ├── ingestion.py     # Fetch and clean raw stock data
│   ├── math_engine.py   # Returns and covariance matrices
│   ├── optimizer.py     # Optimal portfolio weights
│   ├── backtester.py    # Historical performance simulation
│   └── analytics.py     # Sharpe ratio, max drawdown
├── tests/               # Unit tests
├── app.py               # Streamlit dashboard
└── requirements.txt
```

### Data flow

1. **Ingestion** — download or load price data into `data/` as Parquet.
2. **Math Engine** — compute daily returns and a covariance matrix.
3. **Optimizer** — solve for maximum-Sharpe (or other) weights via `scipy`.
4. **Backtester** — simulate portfolio value under fixed weights.
5. **Analytics** — report Sharpe ratio and maximum drawdown.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Usage

Run the dashboard:

```bash
streamlit run app.py
```

Run tests:

```bash
pytest tests/ -v
```

## CI

GitHub Actions runs on push/PR to `main`: `ruff` lint on `src/`, `tests/`, and `app.py`, then `pytest`.
