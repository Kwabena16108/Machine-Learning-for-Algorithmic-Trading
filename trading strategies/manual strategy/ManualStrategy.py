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
from indicators import *



def author():
    return "dnkwantabisa3"


def testPolicy(symbol,sd,ed,sv):
    dates = pd.date_range(sd, ed)
    prices = get_data([symbol], dates).drop("SPY", axis="columns")
    ndays=15
    # get predictors
    bb = get_bb(prices, symbol=[symbol], window=ndays)
    psma = get_psma(prices, symbol=[symbol], window=ndays)
    momentum = get_momentum(prices, symbol=[symbol], window=ndays)
    
    holdings = 0
    new_holdings = 0
    trades = np.zeros(prices.shape[0])
    for i in range(prices.shape[0]-1):
        if ((momentum.iloc[i] < -1.0) or (psma.iloc[i] > 1.0)) and bb.iloc[i] > 2.0:
            new_holdings = -1000 # SELL
        elif ((momentum.iloc[i] > 1.0) or (psma.iloc[i] < -1.0)) and bb.iloc[i] < -2.0:
            new_holdings = 1000 # BUY
        else:
            new_holdings = holdings
        
        shares = new_holdings - holdings
        holdings = new_holdings
        trades[i] = shares
            
    trades[-1] = -holdings
    trades = pd.DataFrame({'Shares':trades}, index=prices.index)
    trades.index.name = "Date"
    return trades

