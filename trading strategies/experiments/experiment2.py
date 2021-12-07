#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 26 12:18:08 2021

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
    :return: The GT ID of the student
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


def experiment2():
    symbol="JPM"
    sv=100_000
    sd=dt.datetime(2008, 1, 1)  		  	   		   	 		  		  		    	 		 		   		 		  
    ed=dt.datetime(2009, 12, 31)
    # impact = [0.05, 0.005, 0.0005]
    impact = [0.025, 0.0125, 0.00625]
    syms = [symbol]  		  	   		   	 		  		  		    	 		 		   		 		  
    dates = pd.date_range(sd, ed)  		  	   		   	 		  		  		    	 		 		   		 		  
    prices_all = get_data(syms, dates)  	  	   		   	 		  		  		    	 		 		   		 		  
    prices = prices_all[syms] 	  	   		   	 		  		  		    	 		 		   		 		  
    np.random.seed(gtid())
    port_values = []
    
    for i in impact:
        learner = sl.StrategyLearner(impact= i)
        c1=learner.add_evidence(symbol="JPM", sd=sd, ed=ed, sv=sv) 
        c2=learner.testPolicy(symbol="JPM", sd=sd, ed=ed, sv=sv) 
        c3=compute_portvals(trades_df_to_marketsim(c2, "JPM"),
                                  impact=i, commission=0.00)
        port_values.append(c3)
        cr, adr, sddr, sr, portvals = port_stats(c3)
        print("-" * 50 + 'In-sample')
        print(f"impact : {i}")
        print(f"Cumulative Return of Strategy Learner: {cr}")
        print(f"Average Daily Return of Strategy Learner : {adr}")
        print(f"Standard Deviation of Strategy Learner : {sddr}")
        print(f"Sharpe Ratio of Strategy Learner : {sr}")
        print(f"Final Portfolio Value of Strategy Learner: {portvals}")
        print(f"# of trades for Strategy Learner: {sum(c2.Shares !=0)}")

    pv = pd.concat([port_values[0],port_values[1],port_values[2]], axis=1)
    pv.columns=['impact_0.025','impact_0.0125','impact_0.00625']

    df_normed = pv / pv.iloc[0, :]

    fig, axl = plt.subplots()
    fig.set_size_inches(10,6)    
    plt.rcParams['lines.linewidth'] = 2
    axl.plot(df_normed.index, df_normed['impact_0.025'],'-', color='orange', label='impact = 0.025')
    axl.plot(df_normed.index, df_normed['impact_0.0125'],'-', color='blue', label='impact = 0.0125')
    axl.plot(df_normed.index, df_normed['impact_0.00625'],'-', color='purple', label='impact = 0.00625')
    axl.legend(frameon=False)
    axl.legend(loc='upper left')
    axl.set_xlabel('Date')
    axl.set_ylabel('Normalized Portfolio Vlaue')
    axl.grid(color='grey', linestyle='--', linewidth=0.5)
    fig.savefig('experiment2.png')



if __name__ == '__main__':
    experiment2()

































