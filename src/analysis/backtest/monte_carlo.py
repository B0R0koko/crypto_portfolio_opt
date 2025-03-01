from typing import List

import numpy as np
from tqdm import tqdm

from analysis.core.currency import Currency


def _simulate_correlated_paths(
        S0: np.ndarray,
        currencies: List[Currency],
        num_steps: int,
        corr_matrix: np.ndarray,
        stddevs: np.ndarray,  # Standard deviation of returns for each currency
        r: float = 0.05
) -> np.ndarray:
    num_series = len(currencies)
    dt = 1 / 365

    # Generate correlated Brownian motion
    W = np.random.normal(size=(num_series, num_steps)) * np.sqrt(dt)
    L = np.linalg.cholesky(corr_matrix)
    W_corr = (L @ W).T  # Shape: (num_steps, num_series)

    # Precompute drift and volatility terms
    drift = (r - 0.5 * stddevs ** 2) * dt
    diffusion = stddevs * W_corr

    # Initialize the price paths
    St = np.zeros((num_steps + 1, num_series))
    St[0] = S0

    # Vectorized GBM calculation
    St[1:] = S0 * np.exp(np.cumsum(drift + diffusion, axis=0))

    return St


def generate_correlated_gbm(
        S0: np.ndarray,
        currencies: List[Currency],
        num_steps: int,
        corr_matrix: np.ndarray,
        stddevs: np.ndarray,
        r: float = 0.05,
        num_paths: int = 250
) -> np.ndarray:
    """
    Returns a 3D tensor where the first dimension corresponds to the number of simulated path,
    second is the asset simulated, third is the simulated value
    """
    paths: List[np.ndarray] = [
        _simulate_correlated_paths(S0, currencies, num_steps, corr_matrix, stddevs, r)
        for _ in tqdm(range(num_paths))
    ]

    St: np.ndarray = np.stack(paths)
    St = np.nan_to_num(St, nan=1.0)

    return St


def get_portfolio_returns_paths(St: np.ndarray, weights: np.ndarray) -> np.ndarray:
    assert St.ndim == 3
    portfolio_prices: np.ndarray = np.sum(St * weights, axis=2).T
    return portfolio_prices / portfolio_prices[0][0]
