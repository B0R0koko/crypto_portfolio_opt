from datetime import date
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd

from analysis.core.currency import Currency
from analysis.core.paths import DATA_DIR
from analysis.core.time_utils import Bounds


def load_data_from_currencies(currencies: List[Currency], bounds: Optional[Bounds] = None) -> pd.DataFrame:
    """
    Returns DataFrame like this
    currency          AAVE          BTC       ENA  ...       TRX       UNI       XRP
    date                                           ...
    2013-04-28         NaN    135.30000       NaN  ...       NaN       NaN       NaN
    2013-04-29         NaN    141.96000       NaN  ...       NaN       NaN       NaN
    2013-04-30         NaN    135.30000       NaN  ...       NaN       NaN       NaN
    2013-05-01         NaN    117.00000       NaN  ...       NaN       NaN       NaN
    2013-05-02         NaN    103.43000       NaN  ...       NaN       NaN       NaN
    """
    dfs: List[pd.DataFrame] = []

    for currency in currencies:
        file_path: Path = DATA_DIR.joinpath(f"{currency.name}_data.csv")
        if file_path.exists():
            df: pd.DataFrame = pd.read_csv(file_path)
            df.columns = ["date", "price"]
            df["date"] = pd.to_datetime(df["date"]).dt.date
            df["currency"] = currency.name

            dfs.append(df)

    # Concatenate all dataframes
    df_merged: pd.DataFrame = pd.concat(dfs)
    df_aggregated = df_merged.groupby(["date", "currency"], as_index=False).mean()

    # Pivot to get wide format with currencies as columns
    df_prices: pd.DataFrame = df_aggregated.pivot(index="date", columns="currency", values="price")

    if bounds is not None:
        df_prices = df_prices.loc[bounds.day0:bounds.day1]

    return df_prices


def display_cov_matrix(cov_matrix: np.ndarray, currencies: List[str]) -> None:
    print(pd.DataFrame(data=cov_matrix, index=currencies, columns=currencies))


def compute_log_returns(df_prices: pd.DataFrame) -> pd.DataFrame:
    """Compute log returns for all columns in df_prices"""
    df_returns: pd.DataFrame = pd.DataFrame()
    for currency in df_prices.columns:
        df_returns[currency] = np.log(df_prices[currency].shift(-1) / df_prices[currency])
    return df_returns


def attach_usdt_to_returns(df_returns: pd.DataFrame) -> None:
    df_returns[Currency.USDT.name] = 0


if __name__ == '__main__':
    bounds: Bounds = Bounds.for_days(
        date(2025, 1, 1), date(2025, 2, 28)
    )

    df_prices: pd.DataFrame = load_data_from_currencies(bounds=bounds)  # load data
    df_returns: pd.DataFrame = compute_log_returns(df_prices=df_prices)  # preprocess it
    attach_usdt_to_returns(df_returns=df_returns)  # attach stablecoins

    print(df_returns)
