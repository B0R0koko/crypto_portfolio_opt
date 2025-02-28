from datetime import date
from typing import Dict, List

import numpy as np
import pandas as pd

from analysis.core.currency import Currency
from analysis.core.time_utils import Bounds
from analysis.core.utils import load_data_from_currencies, attach_usdt_to_returns, compute_log_returns

existing_portfolio: Dict[Currency, float] = {
    Currency.BTC: 0.6086,
    Currency.ETH: 0.105,
    Currency.XRP: 0.049,
    Currency.SOL: 0.0278,
    Currency.TRX: 0.0066,
    Currency.TON: 0.003,
    Currency.LINK: 0.0535,
    Currency.AAVE: 0.0175,
    Currency.ONDO: 0.0171,
    Currency.ENA: 0.0061,
    Currency.MOVE: 0.0057,
    Currency.HYPE: 0.0232,
    Currency.UNI: 0.0162,
    Currency.TAO: 0.0082,
    Currency.MKR: 0.0024,
    Currency.USDT: 0.0167 * 3
}


def compute_return_of_portfolio(df_returns: pd.DataFrame, portfolio: Dict[Currency, float]) -> float:
    currencies: List[str] = list(map(lambda x: x.name, portfolio.keys()))
    weights: np.ndarray = np.array(portfolio.values())

    return ((df_returns[currencies] * weights).sum(axis=1) + 1).prod()


if __name__ == '__main__':
    start_date: date = date(2024, 1, 1)
    end_date: date = date(2025, 2, 28)
    bounds: Bounds = Bounds.for_days(start_date, end_date)

    currencies: List[Currency] = list(existing_portfolio.keys())

    df_prices: pd.DataFrame = load_data_from_currencies(bounds=bounds, currencies=currencies)
    df_returns: pd.DataFrame = compute_log_returns(df_prices=df_prices)
    attach_usdt_to_returns(df_returns=df_returns)

    print(compute_return_of_portfolio(df_returns=df_returns, portfolio=existing_portfolio))