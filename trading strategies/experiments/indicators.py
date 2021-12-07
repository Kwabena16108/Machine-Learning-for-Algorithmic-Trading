#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  2 21:23:33 2021

@author: dicksonnkwantabisa
"""


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from util import get_data


def author():
    return 'dnkwantabisa3'


def get_bb(prices, symbol, window):
    df = prices[symbol]
    df = df.fillna(method="ffill").fillna(method="bfill")
    df = df / df.iloc[0]
    df.columns = ['Prices']

    p = df.Prices

    rm = p.rolling(window).mean()
    rstd = p.rolling(window).std()
    bbp = (p - rm) / 2 * rstd
    return (bbp - bbp.mean()) / bbp.std()


def get_psma(prices, symbol: list, window):
    df = prices[symbol]
    df = df.fillna(method="ffill").fillna(method="bfill")
    df = df / df.iloc[0]
    df.columns = ['Prices']

    p = df.Prices
    rm = p.rolling(window).mean()
    psma = p.divide(rm, axis=0) - 1
    return (psma - psma.mean()) / psma.std()


def get_pema(prices, symbol: list, window):
    df = prices[symbol]
    df = df.fillna(method="ffill").fillna(method="bfill")
    df = df / df.iloc[0]
    df.columns = ['Prices']

    p = df.Prices

    ema = p.ewm(window).mean()
    pema = p.divide(ema, axis=0) - 1
    return (pema - pema.mean()) / pema.std()


def get_momentum(prices, symbol: list, window):
    df = prices[symbol]
    df = df.fillna(method="ffill").fillna(method="bfill")
    df = df / df.iloc[0]
    df.columns = ['Prices']

    p = df.Prices
    momentum = p / p.shift(window) - 1
    return (momentum - momentum.mean()) / momentum.std()


def get_natr(symbol, sd, ed, window):
    dates = pd.date_range(sd, ed)
    low = get_data(symbols=[symbol], dates=dates,
                   colname="Low").drop('SPY', axis=1)
    high = get_data(symbols=[symbol], dates=dates,
                    colname="High").drop('SPY', axis=1)
    close = get_data(symbols=[symbol], dates=dates,
                     colname="Close").drop('SPY', axis=1)

    high_low = high - low
    high_cp = high - close.shift(1)
    low_cp = low - close.shift(1)
    ranges = pd.concat([high_low, high_cp, low_cp], axis=1)
    true_range = np.max(ranges, axis=1)
    atr = pd.DataFrame({'ATR': true_range.rolling(window).mean()})
    natr = pd.DataFrame(atr.values / close.values * 100, index=true_range.index)
    return (natr - natr.mean()) / natr.std()











