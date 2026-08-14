# Portfolio Optimizer 📈

Mean-variance portfolio optimization toolkit with a Streamlit dashboard. This project allows users to download stock data, estimate risk and return, compute optimal portfolio weights, backtest strategies, and analyze performance.

[![Build Status](https://img.shields.io/travis/com/stikey1/portfolio-optimization/main.svg?style=flat-square)](https://travis-ci.com/stikey1/portfolio-optimization)
[![Version](https://img.shields.io/github/package/v/stikey1/portfolio-optimization/main?style=flat-square)](https://github.com/stikey1/portfolio-optimization/releases)
[![License](https://img.shields.io/github/license/stikey1/portfolio-optimization?style=flat-square)](LICENSE)
[![Stars](https://img.shields.io/github/stars/stikey1/portfolio-optimization?style=flat-square)](https://github.com/stikey1/portfolio-optimization/stargazers)
[![Forks](https://img.shields.io/github/forks/stikey1/portfolio-optimization?style=flat-square)](https://github.com/stikey1/portfolio-optimization/forks)

## Table of Contents 📚

* [Features](#features)
* [Tech Stack](#tech-stack)
* [Architecture](#architecture)
* [Data Flow](#data-flow)
* [Installation](#installation) 🛠️
* [Usage](#usage) 🚀
* [Project Structure](#project-structure) 📂
* [Contributing](#contributing) 🤝
* [License](#license) 📄
* [Important Links](#important-links) 🔗

## Features ✨

* **Interactive Streamlit Dashboard:** A user-friendly web interface for portfolio optimization tasks.
* **Stock Data Ingestion:** Downloads and cleans historical stock data using Yahoo Finance.
* **Mean-Variance Optimization:** Computes optimal portfolio weights based on expected returns and covariance matrices.
* **Performance Metrics:** Calculates key risk and return metrics like Sharpe ratio and maximum drawdown.
* **Backtesting:** Simulates historical portfolio performance under various strategies.
* **Efficient Frontier Visualization:** Plots the optimal risk-return tradeoff curve.
* **Unit Tested:** Comprehensive unit tests for core modules.
* **CI/CD Integration:** GitHub Actions for linting and testing.

## Tech Stack 💻

* **Languages:** Python
* **Frameworks/Libraries:** Streamlit, Pandas, NumPy, SciPy, Plotly, yfinance, PyArrow, Ruff, Pytest

## Architecture 🏗️

```
portfolio-optimization/
├── .github/workflows/   # CI/CD (tests + lint)
├── data/                # Local Parquet storage
├── src/
│   ├── __init__.py      # Initializes the src package
│   ├── ingestion.py     # Fetch and clean raw stock data
│   ├── math_engine.py   # Returns and covariance matrices
│   ├── optimizer.py     # Optimal portfolio weights
│   ├── backtester.py    # Historical performance simulation
│   ├── analytics.py     # Sharpe ratio, max drawdown
│   ├── mock_data.py     # Generates synthetic price data for testing
│   ├── tickers.py       # Manages the list of available tickers for the UI
│   └── visualization.py # Creates interactive plots (e.g., Efficient Frontier)
├── tests/               # Unit tests
├── app.py               # Streamlit dashboard
└── pyproject.toml       # Project metadata and dependencies
```

### Data Flow 🌊

1. **Ingestion** — Download or load price data into `data/` as Parquet.
2. **Math Engine** — Compute daily returns and a covariance matrix.
3. **Optimizer** — Solve for maximum-Sharpe (or other) weights via `scipy`.
4. **Backtester** — Simulate portfolio value under fixed weights.
5. **Analytics** — Report Sharpe ratio and maximum drawdown.


## Installation 🛠️
 
1. **Clone the repository:**
```bash
   git clone https://github.com/stikey1/portfolio-optimization.git
   cd portfolio-optimization
```
 
2. **Create and activate a virtual environment:**
```bash
   python -m venv .venv
   source .venv/bin/activate   # Windows: .venv\Scripts\activate
```
 
3. **Install dependencies:**
```bash
   pip install .
```
   * **For development (including testing and linting):**
```bash
     pip install ".[dev]"
```
 
## Usage 🚀
 
1. **Launch the dashboard:**
```bash
   streamlit run app.py
```
2. **Select tickers** using the sidebar multiselect or text input.
3. **Set parameters** — lookback window (in years) and risk-free rate.
4. **Run optimization** by clicking "Run Optimization."
5. **Analyze results** in the **Efficient Frontier** plot and **Backtest** comparison against benchmarks like the Equal-Weight portfolio.
### Running Tests
 
```bash
pytest tests/ -v
```

## Contributing 🤝

Contributions are welcome! Please follow these steps:

1. Fork the repository.
2. Create a new branch (`git checkout -b feature/your-feature-name`).
3. Make your changes.
4. Commit your changes (`git commit -am 'Add some feature'`).
5. Push to the branch (`git push origin feature/your-feature-name`).
6. Open a Pull Request.

Please ensure all tests pass and follow the code style guidelines (enforced by Ruff).

## License 📄

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Links 🔗

* **Project Repository:** [https://github.com/stikey1/portfolio-optimization](https://github.com/stikey1/portfolio-optimization)
* **Live Demo:** (Not available, but the Streamlit app can be run locally)
* **Author Profile:** [stikey1 on GitHub](https://github.com/stikey1) & [sliu247 on GitHub](https://github.com/sliu247)
---

© 2026 [portfolio optimizer]. All rights reserved. | Fork me on [GitHub](https://github.com/stikey1/portfolio-optimization) ⭐ | Like this project? Give it a star! 👍 | Report issues 🐞
