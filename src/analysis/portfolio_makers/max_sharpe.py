import json
from datetime import date
from functools import partial
from typing import List, Dict, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from analysis.core.time_utils import Bounds
from analysis.core.utils import attach_usdt_to_returns, display_cov_matrix
from src.analysis.core.currency import Currency
from src.analysis.core.utils import load_data_from_currencies, compute_log_returns


def max_sharpe_objective(weights: np.ndarray, vec_expected_returns: np.ndarray, cov_matrix: np.ndarray) -> float:
    """Returns portfolio return over variance which should be maximized"""
    sigma_p: float = np.sqrt(weights @ cov_matrix @ weights.T)  # find variance of the portfolio
    return -np.dot(weights, vec_expected_returns) / sigma_p


# Define constraints
def weight_constraint(weights: np.ndarray) -> float:
    """Constraint for the portfolio weights such that we can't buy more than we have"""
    return weights.sum() - 1


def hh_constraint(weights: np.ndarray) -> float:
    """HH constraint for portfolio concentration in one asset"""
    return -(np.sum(weights ** 2) - 0.5)


def get_max_sharpe_portfolio(df_returns: pd.DataFrame, currencies: List[Currency]) -> Dict[Currency, float]:
    """Maximizes Sharpe ratio and returns the weights of the portfolio."""
    assert all(currency.name in df_returns.columns for currency in currencies), "Not all currencies are in df_returns"
    constraints = [{"type": "eq", "fun": weight_constraint}]
    n_assets: int = len(currencies)
    # Create bounds for the weights, we only allow long positions, therefore each weight is within (0, 1)
    bounds: List[Tuple[float, float]] = [(0, 1) for _ in range(n_assets)]
    x0: np.ndarray = np.array([1 / n_assets] * n_assets)  # initial guess

    vec_expected_returns: np.ndarray = df_returns.mean().to_numpy()
    cov_matrix: np.ndarray = df_returns.cov().to_numpy()

    display_cov_matrix(cov_matrix=cov_matrix, currencies=df_returns.columns)

    res = minimize(
        fun=partial(
            max_sharpe_objective, vec_expected_returns=vec_expected_returns, cov_matrix=cov_matrix
        ),
        x0=x0,
        bounds=bounds,
        constraints=constraints,  # type:ignore
    )

    assert res.success, "Solver didn't finish with success status"
    return dict(zip(currencies, res.x))


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

    portfolio: Dict[Currency, float] = get_max_sharpe_portfolio(
        df_returns=df_returns, currencies=currencies
    )

    print(json.dumps({currency.name: np.round(weight, 5) for currency, weight in portfolio.items()}, indent=4))
