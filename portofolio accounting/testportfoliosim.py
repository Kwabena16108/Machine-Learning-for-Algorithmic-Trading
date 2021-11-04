
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from util import *


def compute_portvals(
        orders_file="./orders/orders-01.csv",
        start_val=1000000,
        commission=9.95,
        impact=0.005,
):
    """
    Computes the portfolio values.

    :param orders_file: Path of the order file or the file object
    :type orders_file: str or file object
    :param start_val: The starting value of the portfolio
    :type start_val: int
    :param commission: The fixed amount in dollars charged for each transaction (both entry and exit)
    :type commission: float
    :param impact: The amount the price moves against the trader compared to the historical data at each transaction
    :type impact: float
    :return: the result (portvals) as a single-column dataframe, containing the value of the portfolio for each trading day in the first column from start_date to end_date, inclusive.
    :rtype: pandas.DataFrame
    """
    if isinstance(orders_file, str):
        orders_df = pd.read_csv(orders_file, parse_dates=True, na_values=["nan"])
    else:  # orders_file is a file object
        header = orders_file.readline().strip().split(",")
        rows = [line.strip().split(",") for line in orders_file]
        orders_df = pd.DataFrame(rows, columns=header)
        orders_df["Shares"] = orders_df.Shares.astype(int)

    orders_df["Date"] = pd.to_datetime(orders_df.Date)

    symbols = orders_df.Symbol.unique().tolist()
    all_dates = pd.date_range(orders_df.Date.min(), orders_df.Date.max())
    # we need only trading days
    prices = get_data(symbols=symbols, dates=all_dates).drop("SPY", axis="columns")
    dates = prices.index

    # Pivoting Transactions
    trans_logs = orders_df.copy()
    trans_logs["Order"] = trans_logs.Order.replace({"BUY": 1, "SELL": -1})
    trans_logs["Shares"] = trans_logs.Shares * trans_logs.Order
    trans_logs = trans_logs.drop("Order", axis="columns")

    trans_shares = trans_logs.pivot(columns="Symbol", values="Shares").fillna(0).astype(int)
    trans_shares = trans_shares[symbols]
    trans_shares.columns.name = None
    trans_shares.index = trans_logs.Date

    # mapping stock shares to $$$
    # we multiply by -1 because buying stocks means spending cash
    trans_cash = prices.loc[trans_shares.index][symbols] * trans_shares[symbols] * -1

    # Factoring in Market Impact
    trans_cash -= trans_cash.abs() * impact

    # Calculating Cash
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
    port_val = (prices * holdings[prices.columns]).sum(axis="columns").to_frame()
    
    return port_val


def test_code():
    """
    Helper function to test code
    """
    # this is a helper function you can use to test your code
    # note that during autograding his function will not be called.
    # Define input parameters

    of = "./orders/orders2.csv"
    sv = 1_000_000
    rfr = 0.0
    sample_freq = 252.0

    # Process orders
    portvals = compute_portvals(orders_file=of, start_val=sv, commission=0.0, impact=0.0)
    if isinstance(portvals, pd.DataFrame):
        portvals = portvals[
            portvals.columns[0]]  # just get the first column
    else:
        "warning, code did not return a DataFrame"

    start_date = portvals.index[0]
    end_date = portvals.index[-1]
    dates = portvals.index
    # get $SPY stats
    order = pd.read_csv(of, parse_dates=True, na_values=["nan"])
    symbols = order.Symbol.unique()
    prices_SPY = get_data(symbols, dates)["SPY"]

    daily_rets = (prices_SPY / prices_SPY.shift(1)) - 1
    daily_rets = daily_rets[1:]

    cr_SPY = (prices_SPY[-1] / prices_SPY[0]) - 1
    adr_SPY = daily_rets.mean()
    sddr_SPY = daily_rets.std()
    sr_SPY = np.sqrt(sample_freq) * np.mean(daily_rets - rfr) / daily_rets.std()

    # get portfolio stats
    daily_rets = (portvals / portvals.shift(1)) - 1
    daily_rets = daily_rets[1:]

    cr = (portvals[-1] / portvals[0]) - 1
    adr = daily_rets.mean()
    sddr = daily_rets.std()
    sr = np.sqrt(sample_freq) * np.mean(daily_rets - rfr) / daily_rets.std()

    # plot portfolio returns vs. $SPY
    df_temp = pd.concat(
        [portvals, prices_SPY], keys=['Portfolio', 'SPY'], axis=1)
    df_normed = df_temp / df_temp.iloc[0, :]

    plt.figure(figsize=(8, 6))
    plt.title('Daily Portfolio Value and SPY')
    plt.plot(df_normed.iloc[:, 0], linewidth=2.0, label='Portfolio')
    plt.plot(df_normed.iloc[:, 1], linewidth=2.0, label='SPY')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend(loc='upper right')
    plt.grid(color='black', linestyle='--', linewidth=0.5)
    plt.savefig('figure1.png')

    # Compare portfolio against $SPX
    print(f"Date Range: {start_date} to {end_date}")
    print()
    print(f"Sharpe Ratio of Fund: {sr}")
    print(f"Sharpe Ratio of SPY : {sr_SPY}")
    print()
    print(f"Cumulative Return of Fund: {cr}")
    print(f"Cumulative Return of SPY : {cr_SPY}")
    print()
    print(f"Standard Deviation of Fund: {sddr}")
    print(f"Standard Deviation of SPY : {sddr_SPY}")
    print()
    print(f"Average Daily Return of Fund: {adr}")
    print(f"Average Daily Return of SPY : {adr_SPY}")
    print()
    print(f"Final Portfolio Value: {portvals[-1]}")


if __name__ == "__main__":
    test_code()
