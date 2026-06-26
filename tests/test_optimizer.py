# import pandas as pd

# from src.optimizer import max_sharpe_weights


# def test_max_sharpe_weights_sum_to_one():
#     expected_returns = pd.Series({"A": 0.10, "B": 0.05, "C": 0.08})
#     cov = pd.DataFrame(
#         [[0.04, 0.01, 0.02], [0.01, 0.02, 0.01], [0.02, 0.01, 0.03]],
#         index=["A", "B", "C"],
#         columns=["A", "B", "C"],
#     )

#     weights = max_sharpe_weights(expected_returns, cov)

#     assert abs(weights.sum() - 1.0) < 1e-6
#     assert (weights >= 0).all()
