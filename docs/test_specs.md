# Testing Specification — Modular Portfolio Optimization Engine

> **Stack:** `pytest` · `pytest-cov` · `ruff` · GitHub Actions  
> **Coverage gate:** ≥ 85%  
> **Branch policy:** All PRs into `dev` or `main` must pass the full CI pipeline.

---

## Table of Contents

1. [Project Layout](#1-project-layout)
2. [conftest.py — Shared Fixtures](#2-conftestpy--shared-fixtures)
3. [test_math_engine.py](#3-test_math_enginepy)
4. [test_optimizer_core.py](#4-test_optimizer_corepy)
5. [test_analytics_tier.py](#5-test_analytics_tierpy)
6. [test_ingest_engine.py — Strategy Pattern](#6-test_ingest_enginepy--strategy-pattern)
7. [pyproject.toml Configuration](#7-pyprojecttoml-configuration)
8. [GitHub Actions CI Workflow](#8-github-actions-ci-workflow)
9. [Implementation Notes](#9-implementation-notes)

---

## 1. Project Layout

Your test directory should mirror your source tree exactly:

```
portfolio_engine/
├── src/
│   ├── math_engine.py
│   ├── optimizer_core.py
│   ├── backtester_sim.py
│   ├── analytics_tier.py
│   └── ingest_engine.py
├── tests/
│   ├── conftest.py           # shared fixtures live here
│   ├── test_math_engine.py
│   ├── test_optimizer_core.py
│   ├── test_backtester_sim.py
│   ├── test_analytics_tier.py
│   └── test_ingest_engine.py
├── pyproject.toml
└── .github/workflows/ci.yml
```

---

## 2. `conftest.py` — Shared Fixtures

This is the most important file in your test suite. All mock price matrices, return vectors, and covariance matrices live here so every test module can reuse them without redefining anything.

```python
# tests/conftest.py
import numpy as np
import pandas as pd
import pytest

# Deterministic seed so every test run is identical
RNG = np.random.default_rng(42)

@pytest.fixture
def price_df() -> pd.DataFrame:
    """5 assets × 504 trading days (~2 years).
    Prices are seeded GBM so log-return stats are predictable.
    """
    n_assets, n_days = 5, 504
    daily_returns = RNG.normal(loc=0.0004, scale=0.012, size=(n_days, n_assets))
    prices = 100 * np.exp(np.cumsum(daily_returns, axis=0))
    tickers = ["AAPL", "MSFT", "GOOG", "AMZN", "TSLA"]
    dates = pd.bdate_range("2022-01-03", periods=n_days)
    return pd.DataFrame(prices, index=dates, columns=tickers)

@pytest.fixture
def log_returns(price_df) -> pd.DataFrame:
    return np.log(price_df / price_df.shift(1)).dropna()

@pytest.fixture
def expected_returns(log_returns) -> pd.Series:
    """Annualized mean log returns (μ)."""
    return log_returns.mean() * 252

@pytest.fixture
def cov_matrix(log_returns) -> pd.DataFrame:
    """Annualized covariance matrix (Σ)."""
    return log_returns.cov() * 252

@pytest.fixture
def equal_weights(price_df) -> np.ndarray:
    n = price_df.shape[1]
    return np.full(n, 1.0 / n)

@pytest.fixture
def risk_free_rate() -> float:
    return 0.05  # 5% annualized
```

---

## 3. `test_math_engine.py`

These tests enforce the mathematical contracts of your return and covariance calculations. If someone accidentally uses arithmetic returns instead of log returns, or forgets to annualize by 252, a test here will catch it immediately.

```python
# tests/test_math_engine.py
import numpy as np
import pandas as pd
import pytest
from src.math_engine import compute_log_returns, compute_expected_returns, compute_covariance

class TestLogReturns:
    def test_output_shape(self, price_df, log_returns):
        result = compute_log_returns(price_df)
        # One row is lost to the diff — shape should be (n_days - 1, n_assets)
        assert result.shape == (price_df.shape[0] - 1, price_df.shape[1])

    def test_no_nan_values(self, price_df):
        result = compute_log_returns(price_df)
        assert not result.isnull().any().any()

    def test_log_return_formula(self, price_df):
        """Spot-check one value against the raw formula ln(P_t / P_{t-1})."""
        result = compute_log_returns(price_df)
        expected_first = np.log(price_df.iloc[1, 0] / price_df.iloc[0, 0])
        assert result.iloc[0, 0] == pytest.approx(expected_first, rel=1e-9)

    def test_columns_preserved(self, price_df):
        result = compute_log_returns(price_df)
        assert list(result.columns) == list(price_df.columns)

    def test_raises_on_missing_prices(self):
        bad_df = pd.DataFrame({"A": [100.0, np.nan, 102.0]})
        with pytest.raises(ValueError, match="missing"):
            compute_log_returns(bad_df)

    def test_raises_on_zero_price(self):
        bad_df = pd.DataFrame({"A": [100.0, 0.0, 102.0]})
        with pytest.raises(ValueError):
            compute_log_returns(bad_df)


class TestExpectedReturns:
    def test_annualization(self, log_returns):
        result = compute_expected_returns(log_returns, trading_days=252)
        manual = log_returns.mean() * 252
        pd.testing.assert_series_equal(result, manual)

    def test_output_index_matches_tickers(self, log_returns, price_df):
        result = compute_expected_returns(log_returns)
        assert list(result.index) == list(price_df.columns)

    def test_custom_trading_days(self, log_returns):
        r252 = compute_expected_returns(log_returns, trading_days=252)
        r365 = compute_expected_returns(log_returns, trading_days=365)
        # 365/252 scaling relationship must hold exactly
        ratio = (r365 / r252).round(9)
        expected_ratio = round(365 / 252, 9)
        assert (ratio == expected_ratio).all()


class TestCovarianceMatrix:
    def test_shape_is_square(self, log_returns):
        result = compute_covariance(log_returns)
        n = log_returns.shape[1]
        assert result.shape == (n, n)

    def test_symmetric(self, cov_matrix):
        assert np.allclose(cov_matrix.values, cov_matrix.values.T)

    def test_positive_semidefinite(self, cov_matrix):
        eigenvalues = np.linalg.eigvalsh(cov_matrix.values)
        assert np.all(eigenvalues >= -1e-10)  # tolerance for floating point

    def test_diagonal_is_positive(self, cov_matrix):
        assert (np.diag(cov_matrix.values) > 0).all()

    def test_annualization_factor(self, log_returns):
        daily_cov = log_returns.cov()
        annual_cov = compute_covariance(log_returns, trading_days=252)
        assert np.allclose(annual_cov.values, daily_cov.values * 252)
```

---

## 4. `test_optimizer_core.py`

This is the most critical test module. These tests verify that the weight vectors produced by your `scipy`/`cvxpy` solvers actually satisfy the mathematical constraints that define GMV and Max Sharpe portfolios.

```python
# tests/test_optimizer_core.py
import numpy as np
import pytest
from src.optimizer_core import (
    optimize_gmv,
    optimize_max_sharpe,
    portfolio_variance,
    portfolio_return,
    sharpe_ratio,
)

TOLERANCE = 1e-6  # acceptable floating point delta


class TestWeightConstraints:
    """Every optimizer output must satisfy the same hard constraints."""

    def _assert_valid_weights(self, weights: np.ndarray, n_assets: int):
        assert weights.shape == (n_assets,), "Wrong number of weights"
        assert np.isclose(weights.sum(), 1.0, atol=TOLERANCE), "Weights must sum to 1"
        assert np.all(weights >= -TOLERANCE), "Long-only: no negative weights"
        assert np.all(weights <= 1.0 + TOLERANCE), "No single position > 100%"

    def test_gmv_weight_constraints(self, expected_returns, cov_matrix):
        weights = optimize_gmv(cov_matrix)
        self._assert_valid_weights(weights, len(expected_returns))

    def test_max_sharpe_weight_constraints(self, expected_returns, cov_matrix, risk_free_rate):
        weights = optimize_max_sharpe(expected_returns, cov_matrix, risk_free_rate)
        self._assert_valid_weights(weights, len(expected_returns))


class TestGMVOptimizer:
    def test_gmv_minimizes_variance(self, expected_returns, cov_matrix):
        """GMV weights must produce lower variance than an equal-weight portfolio."""
        gmv_weights = optimize_gmv(cov_matrix)
        equal_weights = np.full(len(expected_returns), 1.0 / len(expected_returns))

        gmv_var = portfolio_variance(gmv_weights, cov_matrix)
        eq_var = portfolio_variance(equal_weights, cov_matrix)

        assert gmv_var <= eq_var, "GMV portfolio should have lower variance than 1/N"

    def test_gmv_analytical_two_asset(self):
        """
        For a 2-asset portfolio with known covariance, verify against
        the closed-form GMV solution:
            w1* = (σ2² - σ12) / (σ1² + σ2² - 2σ12)
        """
        sigma1, sigma2, rho = 0.2, 0.3, 0.1
        cov = np.array([
            [sigma1**2,             rho * sigma1 * sigma2],
            [rho * sigma1 * sigma2, sigma2**2            ]
        ])
        import pandas as pd
        cov_df = pd.DataFrame(cov, columns=["A", "B"], index=["A", "B"])

        w_analytical = (sigma2**2 - rho * sigma1 * sigma2) / (
            sigma1**2 + sigma2**2 - 2 * rho * sigma1 * sigma2
        )
        weights = optimize_gmv(cov_df)
        assert weights[0] == pytest.approx(w_analytical, abs=TOLERANCE)

    def test_gmv_ignores_returns(self, cov_matrix):
        """GMV should not change when expected returns change — it only uses Σ."""
        import pandas as pd
        returns_high = pd.Series([0.3] * 5, index=cov_matrix.columns)
        returns_low  = pd.Series([0.01] * 5, index=cov_matrix.columns)

        w_high = optimize_gmv(cov_matrix)
        w_low  = optimize_gmv(cov_matrix)
        assert np.allclose(w_high, w_low)


class TestMaxSharpeOptimizer:
    def test_max_sharpe_beats_equal_weight(self, expected_returns, cov_matrix, risk_free_rate):
        weights = optimize_max_sharpe(expected_returns, cov_matrix, risk_free_rate)
        eq_weights = np.full(len(expected_returns), 1.0 / len(expected_returns))

        opt_sharpe = sharpe_ratio(weights, expected_returns, cov_matrix, risk_free_rate)
        eq_sharpe  = sharpe_ratio(eq_weights, expected_returns, cov_matrix, risk_free_rate)

        assert opt_sharpe >= eq_sharpe - TOLERANCE

    def test_sharpe_sensitivity_to_rf(self, expected_returns, cov_matrix):
        """Higher risk-free rate must lower the Sharpe ratio, all else equal."""
        weights = optimize_max_sharpe(expected_returns, cov_matrix, rf=0.02)
        sr_low_rf  = sharpe_ratio(weights, expected_returns, cov_matrix, rf=0.02)
        sr_high_rf = sharpe_ratio(weights, expected_returns, cov_matrix, rf=0.08)
        assert sr_low_rf > sr_high_rf

    def test_raises_on_negative_excess_returns(self, cov_matrix):
        """If all assets return less than rf, optimizer should raise or return None."""
        import pandas as pd
        below_rf_returns = pd.Series([-0.1] * 5, index=cov_matrix.columns)
        with pytest.raises((ValueError, RuntimeError)):
            optimize_max_sharpe(below_rf_returns, cov_matrix, rf=0.05)
```

---

## 5. `test_analytics_tier.py`

These tests verify the statistical diagnostics — compounding, Sharpe, and drawdown calculations — against hand-computed ground truth.

```python
# tests/test_analytics_tier.py
import numpy as np
import pandas as pd
import pytest
from src.analytics_tier import (
    compute_cumulative_returns,
    compute_annualized_sharpe,
    compute_max_drawdown,
    compute_annualized_volatility,
)

@pytest.fixture
def flat_returns():
    """Daily returns of exactly 1% every day for 10 days."""
    return pd.Series([0.01] * 10)

@pytest.fixture
def drawdown_returns():
    """Returns that produce a known max drawdown: up 10%, down 20%, recover partially."""
    return pd.Series([0.10, -0.20, 0.05])


class TestCumulativeReturns:
    def test_flat_returns_compound_correctly(self, flat_returns):
        result = compute_cumulative_returns(flat_returns)
        # (1.01)^10 - 1, not simply 10%
        expected_final = (1.01 ** 10) - 1
        assert result.iloc[-1] == pytest.approx(expected_final, rel=1e-9)

    def test_starts_at_zero(self, flat_returns):
        result = compute_cumulative_returns(flat_returns)
        assert result.iloc[0] == pytest.approx(0.0, abs=1e-10)

    def test_zero_returns_produce_zero_cumulative(self):
        returns = pd.Series([0.0] * 5)
        result = compute_cumulative_returns(returns)
        assert (result == 0.0).all()


class TestAnnualizedSharpe:
    def test_sharpe_of_constant_positive_returns(self):
        """Constant daily return → std = 0 → Sharpe should be inf or very large."""
        constant = pd.Series([0.001] * 252)
        result = compute_annualized_sharpe(constant, risk_free=0.0)
        assert np.isinf(result) or result > 100

    def test_sharpe_scaling_by_sqrt252(self):
        """Daily Sharpe × sqrt(252) must equal annualized Sharpe."""
        rng = np.random.default_rng(42)
        daily = pd.Series(rng.normal(0.0005, 0.01, 252))
        daily_sharpe = daily.mean() / daily.std()
        annualized = compute_annualized_sharpe(daily, risk_free=0.0)
        assert annualized == pytest.approx(daily_sharpe * np.sqrt(252), rel=1e-6)

    def test_negative_sharpe_on_losing_strategy(self):
        rng = np.random.default_rng(42)
        losing = pd.Series(rng.normal(-0.002, 0.01, 252))
        assert compute_annualized_sharpe(losing, risk_free=0.0) < 0


class TestMaxDrawdown:
    def test_known_drawdown(self, drawdown_returns):
        """
        After +10% then -20%:
            peak  = 1.10
            trough = 1.10 × 0.80 = 0.88
            max drawdown = (0.88 - 1.10) / 1.10 ≈ -0.2000
        """
        result = compute_max_drawdown(drawdown_returns)
        assert result == pytest.approx(-0.2, abs=1e-6)

    def test_no_drawdown_on_monotone_up(self):
        always_up = pd.Series([0.01, 0.02, 0.01, 0.03])
        assert compute_max_drawdown(always_up) == pytest.approx(0.0, abs=1e-9)

    def test_drawdown_is_negative_or_zero(self):
        rng = np.random.default_rng(99)
        returns = pd.Series(rng.normal(0.0003, 0.015, 500))
        assert compute_max_drawdown(returns) <= 0


class TestAnnualizedVolatility:
    def test_scaling_by_sqrt252(self, log_returns):
        daily_vol = log_returns.std()
        result = compute_annualized_volatility(log_returns)
        pd.testing.assert_series_equal(result, daily_vol * np.sqrt(252))
```

---

## 6. `test_ingest_engine.py` — Strategy Pattern

This is where the abstract base ingestor pays off for testing. Because the data contract is decoupled from `yfinance`, all downstream logic can be tested using a `MockIngestor` that returns deterministic data with zero network calls. When you eventually swap `yfinance` for a Bloomberg or Polygon REST adapter, the abstract contract tests stay green throughout.

```python
# tests/test_ingest_engine.py
import pandas as pd
import numpy as np
import pytest
from unittest.mock import patch
from src.ingest_engine import AbstractIngestor, YFinanceIngestor


class MockIngestor(AbstractIngestor):
    """Deterministic stub: returns a clean 252-row price DataFrame every time."""

    def fetch(self, tickers: list[str], start: str, end: str) -> pd.DataFrame:
        rng = np.random.default_rng(0)
        dates = pd.bdate_range(start, periods=252)
        prices = 100 * np.exp(
            np.cumsum(rng.normal(0.0004, 0.01, (252, len(tickers))), axis=0)
        )
        return pd.DataFrame(prices, index=dates, columns=tickers)


@pytest.fixture
def mock_ingestor():
    return MockIngestor()


class TestAbstractIngestorContract:
    def test_mock_returns_dataframe(self, mock_ingestor):
        df = mock_ingestor.fetch(["AAPL", "MSFT"], "2023-01-01", "2024-01-01")
        assert isinstance(df, pd.DataFrame)

    def test_mock_columns_match_tickers(self, mock_ingestor):
        tickers = ["AAPL", "MSFT", "GOOG"]
        df = mock_ingestor.fetch(tickers, "2023-01-01", "2024-01-01")
        assert list(df.columns) == tickers

    def test_mock_no_missing_values(self, mock_ingestor):
        df = mock_ingestor.fetch(["AAPL"], "2023-01-01", "2024-01-01")
        assert not df.isnull().any().any()

    def test_mock_positive_prices(self, mock_ingestor):
        df = mock_ingestor.fetch(["AAPL", "TSLA"], "2023-01-01", "2024-01-01")
        assert (df > 0).all().all()


class TestYFinanceIngestorUnit:
    """Patch yfinance so these tests never hit the network."""

    def test_throttle_retry_on_empty_response(self):
        with patch("yfinance.download") as mock_dl:
            mock_dl.return_value = pd.DataFrame()  # simulate empty/bad response
            ingestor = YFinanceIngestor()
            with pytest.raises(ValueError, match="No data"):
                ingestor.fetch(["AAPL"], "2023-01-01", "2024-01-01")

    def test_sanitizes_missing_data(self):
        """Forward-fill then back-fill NaNs — no NaNs should survive."""
        dirty = pd.DataFrame({
            "AAPL": [100.0, np.nan, 102.0, np.nan],
            "MSFT": [200.0, 201.0, np.nan, 203.0],
        })
        with patch("yfinance.download", return_value=dirty):
            ingestor = YFinanceIngestor()
            result = ingestor.fetch(["AAPL", "MSFT"], "2023-01-01", "2024-01-01")
            assert not result.isnull().any().any()
```

---

## 7. `pyproject.toml` Configuration

Wire everything together so `pytest`, the coverage gate, and `ruff` are all configured in one place:

```toml
[tool.pytest.ini_options]
testpaths = ["tests"]
addopts = "--tb=short -q"

[tool.coverage.run]
source = ["src"]
omit = ["tests/*", "src/app.py"]  # exclude Streamlit UI from coverage gate

[tool.coverage.report]
fail_under = 85
show_missing = true

[tool.ruff]
line-length = 88
select = ["E", "F", "W", "I"]    # pycodestyle + pyflakes + isort
```

---

## 8. GitHub Actions CI Workflow

```yaml
# .github/workflows/ci.yml
name: CI

on:
  pull_request:
    branches: [main, dev]

jobs:
  lint-and-test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - uses: actions/setup-python@v5
        with:
          python-version: "3.11"

      - name: Install dependencies
        run: pip install -e ".[dev]"

      - name: Ruff lint
        run: ruff check src/ tests/

      - name: Run tests with coverage
        run: pytest --cov=src --cov-report=term-missing --cov-fail-under=85

      - name: Upload coverage report
        uses: actions/upload-artifact@v4
        with:
          name: coverage-report
          path: .coverage
```

---

## 9. Implementation Notes

**The two-asset GMV analytical test** (`test_gmv_analytical_two_asset`) is the single most important test in the suite. It verifies your solver against a closed-form solution that doesn't depend on your own code at all — if your optimizer is broken, this test catches it even if everything else looks internally consistent.

**The `MockIngestor` pattern** means Developer A can run the full test suite offline with no internet, and the CI runner never burns API rate limits. All downstream math tests are completely decoupled from live data.

**The coverage gate excludes `app.py`** intentionally. Streamlit UI code is difficult to unit test meaningfully and is better covered by manual or end-to-end testing. The 85% gate applies only to the mathematical core.

**Running locally:**

```bash
# Run all tests
pytest

# Run with coverage report
pytest --cov=src --cov-report=term-missing

# Run a single module
pytest tests/test_optimizer_core.py -v

# Run ruff linter
ruff check src/ tests/
```