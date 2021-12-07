#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on Sun Nov  7 17:33:08 2021

@author: dicksonnkwantabisa
"""

import numpy as np
import pandas as pd
import datetime as dt
from util import get_data
from marketsimcode import *
from indicators import *
import matplotlib.pyplot as plt
import ManualStrategy as ms
import experiment1 as ep1
import experiment2 as ep2
import StrategyLearner as sl
import datetime as dt

# brute force optimization on lookback window
# from itertools import product

# sma1 = range(10, 31, 2)
# sma2 = range(30, 51, 2)

# results = pd.DataFrame()
# date=pd.date_range(dt.datetime(2009, 6, 1),dt.datetime(2009, 12, 1))

# for sma1, sma2 in product(sma1, sma2):
#     data = get_data(symbols=["JPM"], dates=date).drop("SPY", axis=1).fillna(0)
#     data['Returns'] = np.log(data["JPM"] / data["JPM"].shift(1))
#     data['sma1'] = data['JPM'].rolling(sma1).mean()
#     data['sma2'] = data['JPM'].rolling(sma2).mean()
#     data.dropna(inplace=True)
#     data['Position'] = np.where(data['sma1'] > data['sma2'], 1, -1)
#     data['Strategy'] = data['Position'].shift(1) * data['Returns']
#     data.dropna(inplace=True)
#     MKT = np.exp(data[['Returns', 'Strategy']].sum())
#     results = results.append(pd.DataFrame(
#         {'SMA1':sma1, 'SMA2':sma2, 
#          'MARKET':MKT['Returns'],
#          'STRATEGY':MKT['Strategy'],
#          'OUT':MKT['Strategy'] - MKT['Returns']},
#         index=[0]), ignore_index=True)

# results.sort_values('OUT', ascending=False).head(10)



def author():
    return "dnkwantabisa3"

def trades_df_to_marketsim(trades_df, symbol):
    return trades_df.assign(Symbol=symbol).reset_index()


def benchmark(symbol="JPM",
              sd=dt.datetime(2008, 1, 1),
              ed=dt.datetime(2009, 12, 31),
              sv=100_000):
    dates = pd.date_range(sd, ed)
    all_prices = get_data([symbol], dates).drop("SPY", axis="columns").fillna(
        method="ffill").fillna(method="bfill")
    trades_df = pd.DataFrame(data=all_prices.index).rename(columns={0: "Date"})
    trades_df = trades_df.set_index("Date")
    trades_df['Shares'] = 0
    trades_df.iloc[0, 0] = 1000    
    return trades_df


def port_stats(portvals):
    rfr = 0.0
    sample_freq = 252.0
    pv = portvals.values
    cr = (pv[-1] / pv[0]) - 1
    rets = (pv[1:] / pv[:-1]) - 1
    adr = rets.mean()
    sddr = rets.std()
    sr = np.sqrt(sample_freq) * np.mean(rets - rfr) / rets.std()
    port_vals = pv[-1]
    return cr, adr, sddr, sr, port_vals
    

def test_manual_strategy():
    #############
    # in sample #
    #############
    symbol="JPM";     
    sd="2008-1-1"; 
    ed="2009-12-31"; 

    sv=100_000
    orders = ms.testPolicy(symbol, sd, ed, sv)
    # print(orders)
    port_values = compute_portvals(trades_df_to_marketsim(orders, "JPM"),
                                  impact=0.005, commission=9.95, start_val=sv)
    
    orders_bnchmk = benchmark(symbol=symbol,sd=sd,ed=ed)
    benchmark_portvals = compute_portvals(trades_df_to_marketsim(orders_bnchmk,"JPM"),
                                          impact=0.005, commission=9.95, start_val=sv)
    
    # get portfolio stats
    cr, adr, sddr, sr, portvals = port_stats(port_values)
    
    print("-" * 50 + 'In-sample')
    print(f"Cumulative Return of Manual Strategy: {cr}")
    print(f"Average Daily Return of Manual Strategy : {adr}")
    print(f"Standard Deviation of Manual Strategy : {sddr}")
    print(f"Sharpe Ratio of Manual Strategy : {sr}")
    print(f"Final Portfolio Value of Manual Strategy: {portvals}")
    print(f"# of trades for Manual Strategy: {sum(orders.Shares !=0)}")

    print("-" * 50)

    cr, adr, sddr, sr, portvals = port_stats(benchmark_portvals)
        
    print(f"Cumulative Return of Benchmark Strategy: {cr}")
    print(f"Average Daily Return of Benchmark Strategy : {adr}")
    print(f"Standard Deviation of Benchmark Strategy : {sddr}")
    print(f"Sharpe Ratio of Benchmark Strategy : {sr}")
    print(f"Final Portfolio Value of Benchmark Strategy: {portvals}")
    print(f"# of trades for Benchmark Strategy: {sum(orders_bnchmk.Shares !=0)}")
    
    # plot portfolio returns vs. Benchmark
    
    orders=orders[orders.Shares !=0]
    positions = pd.DataFrame(np.where(orders['Shares'] > 0, 1, -1))
    positions.index = orders.index
    
    df_temp = pd.concat(
        [port_values.PortVal, benchmark_portvals.PortVal], keys=['Manual_Strategy', 'Benchmark_Strategy'], axis=1)
    df_normed = df_temp / df_temp.iloc[0, :]
    df_normed['Position'] = positions


    fig, axl = plt.subplots()
    fig.set_size_inches(10,6)    
    plt.rcParams['lines.linewidth'] = 2
    axl.plot(df_normed.index, df_normed.Manual_Strategy,'-', color='red', label='Manual Strategy')
    axl.plot(df_normed.index, df_normed.Benchmark_Strategy,'-', color='green', label='Benchmark Strategy')
    axl.vlines(x=df_normed.loc[df_normed['Position']==1, 'Position'].index, ymin=0.8, ymax=1.4,
               color='blue', label='LONG')
    axl.vlines(x=df_normed.loc[df_normed['Position']==-1, 'Position'].index, ymin=0.8, ymax=1.4,
           color='black', label='SHORT')
    axl.legend(frameon=False)
    axl.legend(loc='upper left')
    axl.set_xlabel('Date')
    axl.set_ylabel('Normalized Portfolio Vlaue')
    axl.grid(color='grey', linestyle='--', linewidth=0.5)
    fig.savefig('ManualStrategy_in-sample.png')
    
    
    #################
    # out-of sample #
    #################
   
    sd="2010-1-1"; 
    ed="2011-12-31"; 
    
    sv=100_000
    orders = ms.testPolicy(symbol, sd, ed, sv)
    # print(orders)
    port_values = compute_portvals(trades_df_to_marketsim(orders, "JPM"),
                                  impact=0.005, commission=9.95,start_val=sv)
    
    orders_bnchmk = benchmark(symbol=symbol,sd=sd,ed=ed)
    benchmark_portvals = compute_portvals(trades_df_to_marketsim(orders_bnchmk,"JPM"),
                                          impact=0.005, commission=9.95,start_val=sv)

    # get portfolio stats
    cr, adr, sddr, sr, portvals = port_stats(port_values)
    
    print("-" * 50 + 'Out-of-sample')
    print(f"Cumulative Return of Manual Strategy: {cr}")
    print(f"Average Daily Return of Manual Strategy : {adr}")
    print(f"Standard Deviation of Manual Strategy : {sddr}")
    print(f"Sharpe Ratio of Manual Strategy : {sr}")
    print(f"Final Portfolio Value of Manual Strategy: {portvals}")
    print(f"# of trades for Manual Strategy: {sum(orders.Shares !=0)}")

    print("-" * 50)

    cr, adr, sddr, sr, portvals = port_stats(benchmark_portvals)
        
    print(f"Cumulative Return of Benchmark Strategy: {cr}")
    print(f"Average Daily Return of Benchmark Strategy : {adr}")
    print(f"Standard Deviation of Benchmark Strategy : {sddr}")
    print(f"Sharpe Ratio of Benchmark Strategy : {sr}")
    print(f"Final Portfolio Value of Benchmark Strategy: {portvals}")
    print(f"# of trades for Benchmark Strategy: {sum(orders_bnchmk.Shares !=0)}")
    
    
    # plot portfolio returns vs. Benchmark
    
    orders=orders[orders.Shares !=0]
    positions = pd.DataFrame(np.where(orders['Shares'] > 0, 1, -1))
    positions.index = orders.index
    
    df_temp = pd.concat(
        [port_values.PortVal, benchmark_portvals.PortVal], keys=['Manual_Strategy', 'Benchmark_Strategy'], axis=1)
    df_normed = df_temp / df_temp.iloc[0, :]
    df_normed['Position'] = positions


    fig, axl = plt.subplots()
    fig.set_size_inches(10,6)    
    plt.rcParams['lines.linewidth'] = 2
    axl.plot(df_normed.index, df_normed.Manual_Strategy,'-', color='red', label='Manual Strategy')
    axl.plot(df_normed.index, df_normed.Benchmark_Strategy,'-', color='green', label='Benchmark Strategy')
    axl.vlines(x=df_normed.loc[df_normed['Position']==1, 'Position'].index, ymin=0.8, ymax=1.4,
               color='blue', label='LONG')
    axl.vlines(x=df_normed.loc[df_normed['Position']==-1, 'Position'].index, ymin=0.8, ymax=1.4,
           color='black', label='SHORT')
    axl.legend(frameon=False)
    axl.legend(loc='upper left')
    axl.set_xlabel('Date')
    axl.set_ylabel('Normalized Portfolio Vlaue')
    axl.grid(color='grey', linestyle='--', linewidth=0.5)
    fig.savefig('ManualStrategy_out-of-sample.png')


if __name__ == "__main__":
    test_manual_strategy()
    ep1.experiment1()
    ep2.experiment2()


