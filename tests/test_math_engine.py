import numpy as np
import pandas as pd
import pytest

from src.math_engine import compute_covariance, compute_expected_returns, compute_returns


@pytest.fixture
def prices() -> pd.DataFrame:
    return pd.DataFrame(
        {"AAPL": [100.0, 110.0, 121.0], "MSFT": [200.0, 220.0, 242.0]},
        index=pd.date_range("2024-01-01", periods=3),
    )


class TestComputeReturns:
    def test_log_return_formula(self, prices: pd.DataFrame) -> None:
        returns = compute_returns(prices, method="log")

        assert returns.shape == (2, 2)
        assert returns.iloc[0, 0] == pytest.approx(np.log(110 / 100))
        assert returns.iloc[1, 0] == pytest.approx(np.log(121 / 110))

    def test_simple_return_formula(self, prices: pd.DataFrame) -> None:
        returns = compute_returns(prices, method="simple")

        assert returns.shape == (2, 2)
        assert returns.iloc[0, 0] == pytest.approx(0.10)
        assert returns.iloc[1, 0] == pytest.approx(0.10)

    def test_preserves_columns(self, prices: pd.DataFrame) -> None:
        returns = compute_returns(prices)

        assert list(returns.columns) == list(prices.columns)

    def test_no_nan_values(self, prices: pd.DataFrame) -> None:
        returns = compute_returns(prices)

        assert not returns.isnull().any().any()

    def test_raises_on_unknown_method(self, prices: pd.DataFrame) -> None:
        with pytest.raises(ValueError, match="Unknown return method"):
            compute_returns(prices, method="bad")


class TestComputeExpectedReturns:
    def test_annualizes_by_252(self) -> None:
        returns = pd.DataFrame(
            {"A": [0.01, 0.01, 0.01]},
            index=pd.date_range("2024-01-01", periods=3),
        )

        result = compute_expected_returns(returns)

        pd.testing.assert_series_equal(result, pd.Series({"A": 0.01 * 252}))

    def test_matches_manual_calculation(self, prices: pd.DataFrame) -> None:
        returns = compute_returns(prices)
        result = compute_expected_returns(returns)

        for ticker in returns.columns:
            assert result[ticker] == pytest.approx(returns[ticker].mean() * 252)

    def test_index_matches_columns(self, prices: pd.DataFrame) -> None:
        returns = compute_returns(prices)
        result = compute_expected_returns(returns)

        assert list(result.index) == list(returns.columns)


class TestComputeCovariance:
    @pytest.fixture
    def returns(self) -> pd.DataFrame:
        return pd.DataFrame(
            {"A": [0.01, -0.02, 0.03], "B": [0.02, 0.01, -0.01]},
            index=pd.date_range("2024-01-01", periods=3),
        )

    def test_shape_and_symmetry(self, returns: pd.DataFrame) -> None:
        cov = compute_covariance(returns)

        assert cov.shape == (2, 2)
        assert list(cov.index) == list(returns.columns)
        assert np.allclose(cov.values, cov.values.T)

    def test_annualization_factor(self, returns: pd.DataFrame) -> None:
        annual = compute_covariance(returns, annualize=True)
        daily = compute_covariance(returns, annualize=False)

        assert np.allclose(annual.values, daily.values * 252)

    def test_diagonal_is_positive(self, returns: pd.DataFrame) -> None:
        cov = compute_covariance(returns)

        assert (np.diag(cov.values) > 0).all()
