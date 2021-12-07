#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd  
import datetime as dt
from util import get_data
from marketsimcode import compute_portvals


def author():
    return 'dnkwantabisa3'


def testPolicy(symbol="AAPL",
               sd=dt.datetime(2008, 1, 1),
               ed=dt.datetime(2009, 12, 31),
               sv=100000):
    dates = pd.date_range(sd, ed)
    all_prices = get_data([symbol], dates).drop("SPY", axis="columns").fillna(method="ffill").fillna(method="bfill")
    now = all_prices.rename(columns={symbol: "now"})
    nxt = all_prices.shift(periods=-1).rename(columns={symbol: "nxt"})
    prices = pd.concat([now, nxt], axis="columns")
    prices.index.name = "Date"

    holdings = 0

    def trade_decision(row):
        nonlocal holdings  # point to the "holdings" at line 28
        now = row["now"]
        nxt = row["nxt"]

        if nxt > now:  # the market is gonna rise (BUY NOW)
            new_holdings = 1000
        elif nxt < now:  # the market is gonna fall (SELL NOW)
            new_holdings = -1000
        else:
            new_holdings = holdings

        shares = new_holdings - holdings
        holdings = new_holdings
        
        return shares

    trades_df = prices.apply(trade_decision, axis="columns").to_frame().rename(columns={0: "Shares"})
    
    return trades_df


def trades_df_to_marketsim(trades_df, symbol):
    return trades_df.assign(Symbol=symbol).reset_index()



def benchmark(symbol="AAPL",
              sd=dt.datetime(2008, 1, 1),
              ed=dt.datetime(2009, 12, 31),
              sv=100000):
    dates = pd.date_range(sd, ed)
    all_prices = get_data([symbol], dates).drop("SPY", axis="columns").fillna(method="ffill").fillna(method="bfill")
    Date = pd.date_range(all_prices.index.min(), all_prices.index.max())
    
    trades_df = pd.DataFrame(data=Date).rename(columns={0: "Date"})
    trades_df = trades_df.set_index("Date")
    trades_df['Shares'] = 0
    trades_df['Symbol'] = symbol
    trades_df.iloc[0, 0] = 1000    
    
    return trades_df
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
