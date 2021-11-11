#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 17 14:42:19 2021

@author: dicksonnkwantabisa
"""


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from util import get_data


def author():
    return 'dnkwantabisa3'



def get_bollinger_bands(rm, rstd):
    upper_band = rm + rstd*2.
    lower_band = rm - rstd*2.
    return upper_band, lower_band



def indicators():
    date_range = pd.date_range("2008-1-1","2009-12-31")
    df = get_data(symbols=["JPM"], dates=date_range).drop("SPY", axis=1)
    df = df.fillna(method="ffill").fillna(method="bfill")
    df = df / df.iloc[0]
    df.columns = ['Prices']
    
    # indicator 1: Bollinger bands
    rm_JPM = df.rolling(14).mean()
    rstd_JPM = df.rolling(14).std()
    upper_band, lower_band = get_bollinger_bands(rm_JPM, rstd_JPM)
    bbp = (df - rm_JPM) / 2*rstd_JPM
    BB_df = pd.concat(
        [df, rm_JPM,upper_band,lower_band, bbp], axis=1)
    BB_df.columns = ["Prices", "SMA", "Upper Band", "Lower Band", "BB%"]
    bbp_z_score = ((BB_df.iloc[:, 4] - BB_df.iloc[:, 4].mean()) / BB_df.iloc[:, 4].std())

    fig = plt.figure(figsize=(10,6))
    ax1 = plt.subplot(2,1,1)
    ax2 = plt.subplot(2,1,2,sharex=ax1)
    ax1.plot(BB_df.iloc[:, 0], linewidth = 2.0,color='blue', label='Prices')
    ax1.plot(BB_df.iloc[:, 1], linewidth = 2.0,color='red', label='SMA')
    ax1.plot(BB_df.iloc[:, 2], linewidth = 1.0, label='Upper band')
    ax1.plot(BB_df.iloc[:, 3], linewidth = 1.0, label='Lower band')
    ax1.set_ylabel('Normalized Prices')
    ax1.legend(frameon=True)
    ax1.grid(color='black', linestyle='--', linewidth=0.5)
    
    ax2.plot(bbp_z_score,color='magenta', linewidth = 2.0, label='BB%')
    ax2.set_ylabel('Z-Score of BB%')
    ax2.set_xlabel('Dates')
    ax2.legend(frameon=True)
    ax2.grid(color='black', linestyle='--', linewidth=0.5)
    fig.savefig('figure1.1.png')
    
    
    # indicator 2: Price/EMA
    ema_JPM = df.ewm(14).mean()
    EMA_df = df.copy()
    EMA_df['EMA'] = ema_JPM
    EMA_df['Price/EMA'] = (EMA_df.Prices / ema_JPM.Prices) - 1
    EMA_df['z_score_EMA'] = EMA_df.iloc[:, 2]- EMA_df.iloc[:, 2].mean()/EMA_df.iloc[:, 2].std()
    
    fig = plt.figure(figsize=(10,6))
    ax1 = plt.subplot(2,1,1)
    ax2 = plt.subplot(2,1,2,sharex=ax1)
    ax1.plot(EMA_df.iloc[:, 0], linewidth = 2.0,color='blue', label='Prices')
    ax1.plot(EMA_df.iloc[:, 1], linewidth = 2.0,color='red', label='EMA')
    ax1.set_ylabel('Normalized Prices')
    ax1.legend(frameon=True)
    ax1.grid(color='black', linestyle='--', linewidth=0.5)
    
    ax2.plot(EMA_df.iloc[:, 3],color='green', linewidth = 2.0, label='Prices/EMA')
    ax2.set_ylabel('Z-Score of Prices/EMA')
    ax2.set_xlabel('Dates')
    ax2.legend(frameon=True)
    ax2.grid(color='black', linestyle='--', linewidth=0.5)
    fig.savefig('figure2.1.png')

    
    # indicator 3: Price/SMA
    SMA_df = df.copy()
    SMA_df['SMA'] = rm_JPM
    SMA_df['Price/SMA'] = (SMA_df.Prices / rm_JPM.Prices) - 1
    SMA_df['z_score_SMA'] = SMA_df.iloc[:, 2]- SMA_df.iloc[:, 2].mean()/SMA_df.iloc[:, 2].std()
        
    fig = plt.figure(figsize=(10,6))
    ax1 = plt.subplot(2,1,1)
    ax2 = plt.subplot(2,1,2,sharex=ax1)
    ax1.plot(SMA_df.iloc[:, 0], linewidth = 2.0,color='blue', label='Prices')
    ax1.plot(SMA_df.iloc[:, 1], linewidth = 2.0,color='red', label='SMA')
    ax1.set_ylabel('Normalized Prices')
    ax1.legend(frameon=True)
    ax1.grid(color='black', linestyle='--', linewidth=0.5)
    
    ax2.plot(SMA_df.iloc[:, 3],color='green', linewidth = 2.0, label='Price/SMA')
    ax2.set_ylabel('Z-Score of price/SMA')
    ax2.set_xlabel('Dates')
    ax2.legend(frameon=True)
    ax2.grid(color='black', linestyle='--', linewidth=0.5)
    fig.savefig('figure3.1.png')

    # indicator 4: Momentum
    MOM_df = df.copy()
    MOM_df['momentum'] = (df.Prices / df.Prices.shift(14)) - 1
    MOM_df['z_score_Momentum'] = MOM_df.iloc[:, 1]- MOM_df.iloc[:, 1].mean()/MOM_df.iloc[:, 1].std()
        
    fig = plt.figure(figsize=(10,6))
    ax1 = plt.subplot(2,1,1)
    ax2 = plt.subplot(2,1,2,sharex=ax1)
    ax1.plot(MOM_df.iloc[:, 0], linewidth = 2.0,color='blue', label='Prices')
    ax1.set_ylabel('Normalized Prices')
    ax1.legend(frameon=True)
    ax1.grid(color='black', linestyle='--', linewidth=0.5)
    
    ax2.plot(MOM_df.iloc[:, 2],color='magenta', linewidth = 2.0, label="Momentum")
    ax2.set_ylabel('z_score Momentum')
    ax2.set_xlabel('Dates')
    ax2.legend(frameon=True)
    ax2.grid(color='black', linestyle='--', linewidth=0.5)
    fig.savefig('figure4.1.png')
    
    # indicator 5: NATR
    low = get_data(symbols=['JPM'], dates=date_range, 
                      colname="Low").drop('SPY', axis=1)
    high = get_data(symbols=['JPM'], dates=date_range, 
                      colname="High").drop('SPY', axis=1)
    close = get_data(symbols=['JPM'], dates=date_range, 
                      colname="Close").drop('SPY', axis=1)
    
    high_low = high - low
    high_cp = high - close.shift(1)
    low_cp = low - close.shift(1)
    ranges = pd.concat([high_low, high_cp, low_cp], axis=1)
    true_range = np.max(ranges, axis=1)
    atr = pd.DataFrame({'ATR':true_range.rolling(14).mean()})
    natr = pd.DataFrame(atr.values / close.values * 100, index=true_range.index)
    natr.columns = ['NATR']
    natr['z_score_NATR'] = natr.iloc[:, 0]- natr.iloc[:, 0].mean()/natr.iloc[:, 0].std()
    
    fig = plt.figure(figsize=(10,6))
    ax1 = plt.subplot(2,1,1)
    ax2 = plt.subplot(2,1,2,sharex=ax1)
    ax1.plot(MOM_df.iloc[:, 0], linewidth = 2.0, color='blue', label='Prices')
    ax1.set_ylabel('Normalized Prices')
    ax1.legend(frameon=True)
    ax1.grid(color='black', linestyle='--', linewidth=0.5)
    
    ax2.plot(natr.iloc[:, 1],color='magenta', linewidth = 2.0, label='NATR')
    ax2.set_ylabel('Z-score NATR')
    ax2.set_xlabel('Dates')
    ax2.legend(frameon=True)
    ax2.grid(color='black', linestyle='--', linewidth=0.5)
    fig.savefig('figure5.1.png')

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
