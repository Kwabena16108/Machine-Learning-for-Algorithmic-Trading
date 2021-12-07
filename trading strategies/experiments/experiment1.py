#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 25 21:53:32 2021

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
import StrategyLearner as sl


def gtid():
    """
    :return: a random seed
    """
    return 903658462


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


def experiment1():
    symbol="JPM"
    sv=100_000
    sd=dt.datetime(2008, 1, 1)  		  	   		   	 		  		  		    	 		 		   		 		  
    ed=dt.datetime(2009, 12, 31) 		  	   		   	 		  		  		    	 		 		   		 		  

    np.random.seed(gtid())  
    learner = sl.StrategyLearner()
    c1=learner.add_evidence(symbol="JPM", sd=sd, ed=ed, sv=sv) 
    c2=learner.testPolicy(symbol="JPM", sd=sd, ed=ed, sv=sv) 
    port_values = compute_portvals(trades_df_to_marketsim(c2, "JPM"),
                                  impact=0.005, commission=9.95)

    orders_bnchmk = benchmark(symbol=symbol,sd=sd,ed=ed)
    benchmark_portvals = compute_portvals(trades_df_to_marketsim(orders_bnchmk,"JPM"),
                                          impact=0.005, commission=9.95, start_val=sv)
    
    orders_manual = ms.testPolicy(symbol, sd, ed, sv)
    manual_portvals = compute_portvals(trades_df_to_marketsim(orders_manual,"JPM"),
                                          impact=0.005, commission=9.95, start_val=sv)
    
    # get portfolio stats
    cr, adr, sddr, sr, portvals = port_stats(port_values)
    
    print("-" * 50 + 'In-sample')
    print(f"Cumulative Return of Strategy Learner: {cr}")
    print(f"Average Daily Return of Strategy Learner : {adr}")
    print(f"Standard Deviation of Strategy Learner : {sddr}")
    print(f"Sharpe Ratio of Strategy Learner : {sr}")
    print(f"Final Portfolio Value of Strategy Learner: {portvals}")
    print(f"# of trades for Strategy Learner: {sum(c2.Shares !=0)}")

    print("-" * 50)

    cr, adr, sddr, sr, portvals = port_stats(benchmark_portvals)
        
    print(f"Cumulative Return of Benchmark Strategy: {cr}")
    print(f"Average Daily Return of Benchmark Strategy : {adr}")
    print(f"Standard Deviation of Benchmark Strategy : {sddr}")
    print(f"Sharpe Ratio of Benchmark Strategy : {sr}")
    print(f"Final Portfolio Value of Benchmark Strategy: {portvals}")
    print(f"# of trades for Benchmark Strategy: {sum(orders_bnchmk.Shares !=0)}")


    print("-" * 50)

    cr, adr, sddr, sr, portvals = port_stats(manual_portvals)
        
    print(f"Cumulative Return of Manual Strategy: {cr}")
    print(f"Average Daily Return of Manual Strategy : {adr}")
    print(f"Standard Deviation of Manual Strategy : {sddr}")
    print(f"Sharpe Ratio of Manual Strategy : {sr}")
    print(f"Final Portfolio Value of Manual Strategy: {portvals}")
    print(f"# of trades for Manual Strategy: {sum(orders_manual.Shares !=0)}")

    # plot portfolio returns vs. Benchmark vs. Strategy Learner
    
    df_temp = pd.concat(
        [port_values.PortVal, benchmark_portvals.PortVal,manual_portvals.PortVal],
        keys=['Strategy_Learner', 'Benchmark_Strategy','Manual_Strategy'], axis=1)
    df_normed = df_temp / df_temp.iloc[0, :]


    fig, axl = plt.subplots()
    fig.set_size_inches(10,6)    
    plt.rcParams['lines.linewidth'] = 2
    axl.plot(df_normed.index, df_normed.Manual_Strategy,'-', color='red', label='Manual Strategy')
    axl.plot(df_normed.index, df_normed.Benchmark_Strategy,'-', color='black', label='Benchmark Strategy')
    axl.plot(df_normed.index, df_normed.Strategy_Learner,'-', color='green', label='Strategy Learner')
    axl.legend(frameon=False)
    axl.legend(loc='upper left')
    axl.set_xlabel('Date')
    axl.set_ylabel('Normalized Portfolio Vlaue')
    axl.grid(color='grey', linestyle='--', linewidth=0.5)
    fig.savefig('experiment1.png')

		
if __name__ == '__main__':
    experiment1()
    
