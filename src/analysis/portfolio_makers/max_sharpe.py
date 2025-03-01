import json
from datetime import date
from functools import partial
from typing import List, Dict, Tuple, Any

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from analysis.core.portfolio_optimizer import PortfolioOptimizer
from analysis.core.time_utils import Bounds
from analysis.core.utils import attach_usdt_to_returns
from src.analysis.core.currency import Currency
from src.analysis.core.utils import load_data_from_currencies, compute_log_returns


# Define constraints
def weight_constraint(weights: np.ndarray) -> float:
    """Constraint for the portfolio weights such that we can't buy more than we have"""
    return weights.sum() - 1


class MaxSharpePortfolio(PortfolioOptimizer):

    def target_function(self, W: np.ndarray, R: np.ndarray, cov_matrix: np.ndarray) -> float:
        """Returns portfolio return over variance which should be maximized"""
        sigma_p: float = np.sqrt(W @ cov_matrix @ W.T)  # find variance of the portfolio
        return -np.dot(W, R) / sigma_p

    def get_constraints(self) -> List[Dict[str, Any]]:
        """Returns a List of dictionaries with constraints"""
        return [
            {"type": "eq", "fun": weight_constraint},
        ]

    def find_portfolio(self, df_returns: pd.DataFrame, selected_currencies: List[Currency]) -> Dict[Currency, float]:
        """Returns optimal portfolio as a dictionary Dict[Currency, float]"""
        assert all(currency.name in df_returns.columns for currency in selected_currencies), \
            "Not all currencies are in df_returns"

        n_assets: int = len(selected_currencies)
        cols: List[str] = [currency.name for currency in selected_currencies]

        bounds: List[Tuple[float, float]] = [(0, 1) for _ in range(n_assets)]
        x0: np.ndarray = np.array([1 / n_assets] * n_assets)  # initial guess

        R: np.ndarray = df_returns[cols].mean().to_numpy()  # returns vector
        cov_matrix: np.ndarray = df_returns[cols].cov().to_numpy()  # var-covariance matrix

        res = minimize(
            fun=partial(self.target_function, R=R, cov_matrix=cov_matrix),
            x0=x0,
            bounds=bounds,
            constraints=self.get_constraints(),  # type:ignore
        )

        assert res.success, "Solver didn't finish with success status"
        return dict(zip(selected_currencies, res.x))


if __name__ == "__main__":
    # Select date range we would like to perform portfolio optimization for
    start_date: date = date(2013, 1, 1)
    end_date: date = date(2025, 2, 28)

    # Select the set of currencies used for optimization
    currencies: List[Currency] = [
        Currency.BTC,
        Currency.ETH,
        Currency.USDT,
        Currency.XRP,
        Currency.LINK
    ]

    bounds: Bounds = Bounds.for_days(start_date, end_date)

    df_prices: pd.DataFrame = load_data_from_currencies(bounds=bounds, currencies=currencies)
    df_returns: pd.DataFrame = compute_log_returns(df_prices=df_prices)
    attach_usdt_to_returns(df_returns=df_returns)

    optimizer: MaxSharpePortfolio = MaxSharpePortfolio()
    portfolio: Dict[Currency, float] = optimizer.find_portfolio(
        df_returns=df_returns, selected_currencies=currencies,
    )

    print(json.dumps({currency.name: np.round(weight, 5) for currency, weight in portfolio.items()}, indent=4))
