#!/usr/bin/env python3
# -*- coding: utf-8 -*-
 
import datetime as dt
from util import get_data
from marketsimcode import compute_portvals
import TheoreticallyOptimalStrategy as tos
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import indicators as ind


def testproject():
    symbol="JPM"
    sd=dt.datetime(2008, 1, 1)
    ed=dt.datetime(2009, 12, 31)
    sv=100_000
    rfr = 0.0
    sample_freq = 252.0
    
    # protfolio
    trades_df = tos.testPolicy(symbol=symbol, sd=sd, ed=ed)
    orders_df = tos.trades_df_to_marketsim(trades_df, symbol)
    portvals = compute_portvals(orders_df=orders_df, commission=0.0, impact=0.0, start_val=sv)
    portfolio_rets = (portvals / portvals.shift(1)) - 1
    portfolio_rets = portfolio_rets[1:]

    portfolio_cr = (portvals.iloc[-1] / portvals.iloc[0]) - 1
    portfolio_adr = portfolio_rets.mean()
    portfolio_sddr = portfolio_rets.std()
    portfolio_sr = np.sqrt(sample_freq) * np.mean(portfolio_rets - rfr) / portfolio_rets.std()
    
    # benchmark
    benchmark_trades = tos.benchmark(symbol=symbol,sd=sd,ed=ed)
    orders_df = tos.trades_df_to_marketsim(benchmark_trades, symbol)
    benchmark_portvals = compute_portvals(orders_df=orders_df, commission=0.0, impact=0.0, start_val=sv)
    bnchmrk_rets = (benchmark_portvals / benchmark_portvals.shift(1)) - 1
    bnchmrk_rets = bnchmrk_rets[1:]

    bnchmrk_cr = (benchmark_portvals.iloc[-1] / benchmark_portvals.iloc[0]) - 1
    bnchmrk_adr = bnchmrk_rets.mean()
    bnchmrk_sddr = bnchmrk_rets.std()
    bnchmrk_sr = np.sqrt(sample_freq) * np.mean(bnchmrk_rets - rfr) / bnchmrk_rets.std()
  
    # plot portfolio returns vs. $SPY
    df_temp = pd.concat(
        [portvals, benchmark_portvals], keys=['Theoretically Optimal Portfolio', 'Benchmark'], axis=1)
    df_normed = df_temp / df_temp.iloc[0, :]

    plt.figure(figsize=(10, 6))
    plt.title('Daily Portfolio Value (Theoretically Optimal Portfolio vs. Benchmark)')
    plt.plot(df_normed.iloc[:, 0], linewidth=2.0,color="red",label='Theoretically Optimal Portfolio')
    plt.plot(df_normed.iloc[:, 1], linewidth=2.0,color="green", label='Benchmark')
    plt.xlabel('Date')
    plt.ylabel('Normallized Portfolio Value')
    plt.legend(loc='upper right')
    plt.grid(color='black', linestyle='--', linewidth=0.5)
    plt.legend(loc="upper left")
    plt.savefig('OptimalStrategy.png')

    # Compare portfolio against $SPX
    print(f"Sharpe Ratio of Theoretically Optimal Strategy : {portfolio_sr}")
    print(f"Sharpe Ratio of Benchmark : {bnchmrk_sr}")
    print(f"Cumulative Return of Theoretically Optimal Strategy : {portfolio_cr}")
    print(f"Cumulative Return of Benchmark : {bnchmrk_cr}")
    print(f"Standard Deviation of Theoretically Optimal Strategy : {portfolio_sddr}")
    print(f"Standard Deviation of Benchmark : {bnchmrk_sddr}")
    print(f"Average Daily Return of Theoretically Optimal Strategy : {portfolio_adr}")
    print(f"Average Daily Return of Benchmark : {bnchmrk_adr}")
    print(f"Final Portfolio Value: {portvals.iloc[-1]}")


    

if __name__ == "__main__":
    testproject()
    ind.indicators()
    
