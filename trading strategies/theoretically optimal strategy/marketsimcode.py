#!/usr/bin/env python3
# -*- coding: utf-8 -*-
		  		  		    	 		 		   		 		  
import numpy as np
import pandas as pd

from util import *


def author():
    return 'dnkwantabisa3'


def compute_portvals(
        orders_df,
        start_val=100_000,
        commission=0.0,
        impact=0.0,
):
    """
    Computes the portfolio values.

    :param orders_df: DataFrame of trades
    :type orders_df: pd.DataFrame
    :param start_val: The starting value of the portfolio
    :type start_val: int
    :param commission: The fixed amount in dollars charged for each transaction (both entry and exit)
    :type commission: float
    :param impact: The amount the price moves against the trader compared to the historical data at each transaction
    :type impact: float
    :return: the result (portvals) as a single-column dataframe, containing the value of the portfolio for each trading day in the first column from start_date to end_date, inclusive.
    :rtype: pandas.DataFrame
    """

    orders_df["Date"] = pd.to_datetime(orders_df.Date)

    symbols = orders_df.Symbol.unique().tolist()
    all_dates = pd.date_range(orders_df.Date.min(), orders_df.Date.max())
    # we need only trading days
    prices = get_data(symbols=symbols, dates=all_dates).drop("SPY", axis="columns")
    dates = prices.index

    # Pivoting Transactions
    trans_logs = orders_df.copy()

    trans_shares = trans_logs.pivot(columns="Symbol", values="Shares").fillna(0).astype(int)
    trans_shares = trans_shares[symbols]
    trans_shares.columns.name = None
    trans_shares.index = trans_logs.Date

    # mapping stock shares to $$$
    # we multiply by -1 because buying stocks means spending cash
    trans_cash = prices.loc[trans_shares.index][symbols] * trans_shares[symbols] * -1

    # Factoring in Market Impact
    trans_cash -= trans_cash.abs() * impact

    # Calculating Cash left
    trans_cash = trans_cash.assign(Cash=trans_cash[symbols].sum(axis=1) - commission)

    trans_cash = trans_cash.groupby("Date").sum()
    trans_shares = trans_shares.groupby("Date").sum()
    trans_df = trans_shares.assign(Cash=trans_cash.Cash)

    # Resampling Dates
    trans_df = pd.DataFrame({"Date": dates}).merge(trans_df, right_index=True, left_on="Date", how="left").fillna(0)
    trans_df = trans_df.set_index("Date")

    # Holdings
    holdings = trans_df.copy()
    holdings.iloc[0, -1] += start_val
    holdings = holdings.rolling(min_periods=1, window=len(trans_df)).sum().astype(float)

    prices['Cash'] = 1.0
    port_val = (prices * holdings[prices.columns]).sum(axis="columns").to_frame().rename(columns={0: "PortVal"})
    
    return port_val
