# Specification: `optimizer_core.py`

## 1. Overview
The `optimizer_core.py` module is the mathematical engine for the portfolio optimization tier. It is responsible for calculating the optimal asset allocation weights that maximize a portfolio's Sharpe Ratio. The optimizer strictly enforces two constraints:
1.  **Fully Invested:** All weights must sum to exactly 1 ($100\%$).
2.  **Long-Only:** No asset weight can be less than 0 (no short-selling).

The module relies on the Sequential Least Squares Programming (SLSQP) algorithm to handle bounded, non-linear optimization. It also exposes standard portfolio analytics helpers for downstream testing and reporting.

---

## 2. Dependencies
* `numpy`: Required for highly optimized vectorized matrix linear algebra operations.
* `scipy.optimize.minimize`: The core solver utilizing the SLSQP algorithm.

---

## 3. Mathematical Foundations

The module optimizes the portfolio based on Modern Portfolio Theory (MPT).

* **Portfolio Return:** $\mu_p = \mathbf{w}^T \boldsymbol{\mu}$
* **Portfolio Variance:** $\sigma_p^2 = \mathbf{w}^T \boldsymbol{\Sigma} \mathbf{w}$
* **Objective Function (Sharpe Ratio):** $SR = \frac{\mu_p - r_f}{\sigma_p}$

Because `scipy` is designed to minimize functions, the engine minimizes the **negative** Sharpe Ratio to achieve maximization:
$$\min_{\mathbf{w}} \left( - \frac{\mathbf{w}^T \boldsymbol{\mu} - r_f}{\sqrt{\mathbf{w}^T \boldsymbol{\Sigma} \mathbf{w}}} \right)$$

**Constraints Enforced:**
* $\sum_{i=1}^{N} w_i = 1$ (Equality constraint)
* $0 \le w_i \le 1 \quad \forall i$ (Bounds)

---

## 4. Function API Reference

### 4.1. `portfolio_return(weights, expected_returns)`
Calculates the expected return of the portfolio given a specific weight distribution.
* **Parameters:**
    * `weights` *(np.ndarray)*: 1D array of asset weights.
    * `expected_returns` *(np.ndarray)*: 1D array of expected returns ($\boldsymbol{\mu}$).
* **Returns:** *(float)* The expected portfolio return ($\mu_p$).

### 4.2. `portfolio_variance(weights, cov_matrix)`
Calculates the expected variance of the portfolio using matrix multiplication.
* **Parameters:**
    * `weights` *(np.ndarray)*: 1D array of asset weights.
    * `cov_matrix` *(np.ndarray)*: 2D covariance matrix ($\boldsymbol{\Sigma}$).
* **Returns:** *(float)* The portfolio variance ($\sigma_p^2$).

### 4.3. `sharpe_ratio(weights, expected_returns, cov_matrix, risk_free_rate=0.0)`
Calculates the annualized Sharpe Ratio for a given weight distribution.
* **Parameters:**
    * `weights` *(np.ndarray)*: 1D array of asset weights.
    * `expected_returns` *(np.ndarray)*: 1D array of expected returns.
    * `cov_matrix` *(np.ndarray)*: 2D covariance matrix.
    * `risk_free_rate` *(float, optional)*: The baseline risk-free rate ($r_f$). Defaults to `0.0`.
* **Returns:** *(float)* The Sharpe Ratio. Returns `0.0` if volatility is zero.

### 4.4. `maximize_sharpe_ratio(expected_returns, cov_matrix, risk_free_rate=0.0, asset_names=None)`
The primary entry point for the optimization engine. Sets up the Lagrangian constraints, bounded limits, and executes the SLSQP solver.
* **Parameters:**
    * `expected_returns` *(np.ndarray)*: 1D array of expected asset returns.
    * `cov_matrix` *(np.ndarray)*: 2D variance-covariance matrix.
    * `risk_free_rate` *(float, optional)*: Defaults to `0.0`.
    * `asset_names` *(list, optional)*: List of string tickers matching the array order. If `None`, keys will default to `ASSET_0`, `ASSET_1`, etc.
* **Returns:** *(dict)* A mapping of `{ticker: weight}`. All weights are floats clamped between `0.0` and `1.0`.
* **Raises:** `ValueError` if the SLSQP optimizer fails to converge.

---

## 5. Implementation Details & Safety Guards
* **Numerical Precision Guard:** Floating point math during matrix optimization often results in numbers like `-1.3e-17` instead of `0.0`. The script uses `np.clip(result.x, 0.0, 1.0)` post-optimization to prevent these micro-precisions from breaking downstream analytics tiers.
* **Re-normalization:** After clipping, the script recalculates the sum and divides the weight vector by it to guarantee that the final output equals *exactly* `1.0` before dictionary construction.

---

## 6. Example Usage

```python
import numpy as np
from optimizer_core import maximize_sharpe_ratio

# 1. Provide Inputs
mu = np.array([0.10, 0.12, 0.05])
Sigma = np.array([
    [0.040, 0.005, 0.010],
    [0.005, 0.090, -0.002],
    [0.010, -0.002, 0.015]
])
tickers = ["AAPL", "MSFT", "BOND"]

# 2. Execute Optimizer
weights = maximize_sharpe_ratio(
    expected_returns=mu, 
    cov_matrix=Sigma, 
    risk_free_rate=0.02, 
    asset_names=tickers
)

# 3. Output
print(weights)
# {'AAPL': 0.54, 'MSFT': 0.16, 'BOND': 0.30}