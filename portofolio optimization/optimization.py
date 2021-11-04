  		   	 		  		  		    	 		 		   		 		  
 	 		  		  		    	 		 		   		 		  
  		  	   		   	 		  		  		    	 		 		   		 		  
import datetime as dt  		  	   		   	 		  		  		    	 		 		   		 		   		  	   		   	 		  		  		    	 		 		   		 		  
import numpy as np  		  	   		   	 		  		  		    	 		 		   		 		  
import scipy.optimize as spo 		  	   		   	 		  		  		    	 		 		   		 		   		  	   		   	 		  		  		    	 		 		   		 		  
import matplotlib.pyplot as plt  		  	   		   	 		  		  		    	 		 		   		 		  
import pandas as pd  		  	   		   	 		  		  		    	 		 		   		 		  
from util import get_data, plot_data, symbol_to_path
  		  	   		   	 		  		  		    	 		 		   		 		  
#%% Objective function

def negative_sharpe_ratio(allocs, prices, start_val, rfr, sample_freq):
    normed = prices / prices.iloc[0, :]
    alloced = normed * allocs
    pos_vals = alloced * start_val
    port_val = pos_vals.sum(axis=1)
        
    ret = port_val / port_val.shift(1) - 1
    sd = ret.std()
    
    return - np.sqrt(sample_freq) * np.mean(ret - rfr) / sd


#%%  		  	   		   	 		  		  		    	 		 		   		 		  
def optimize_portfolio(  		  	   		   	 		  		  		    	 		 		   		 		  
    sd = dt.datetime(2008,6,1),
    ed = dt.datetime(2009,6,1),
    syms = ["IBM", "X", "GLD", "JPM"],
    gen_plot=False,	  	   		   	 		  		  		    	 		 		   		 		  
):
    rfr = 0.0 ####
    sample_freq = 252.0 ###
    start_val = 1e6 ####
    
    dates = pd.date_range(sd, ed)
    
    prices_all = get_data(syms, dates)
    #fill missing data 
    prices_all.fillna(method='ffill', inplace=True)
    prices_all.fillna(method='bfill', inplace=True)
    
    prices = prices_all[syms]
    prices_SPY = prices_all['SPY']
    
    
    # find the allocations for the optimal portfolio 
    
    n_stocks = len(syms)
    allocs_guess = np.ones(shape=n_stocks, dtype=np.float)/n_stocks
    
    alloc_constraint = {'type': 'eq',
                        'fun': lambda x: 1 - np.sum(np.abs(x))}
    
    result = spo.minimize(negative_sharpe_ratio, allocs_guess, 
                      args=(prices,start_val, rfr, sample_freq), method='SLSQP',
                      constraints=alloc_constraint,
                      bounds=((0, 1),) * n_stocks,
                      options={'disp': True, 'ftol':1e-10, 'maxiter':1e4})
    
   # Optimized allocation weights
    opt_alloc = result['x']
      
    # add code here to compute stats 
    normed = prices / prices.iloc[0, :]     
    alloced_opt = normed * opt_alloc
    pos_vals = alloced_opt * start_val
    port_val = pos_vals.sum(axis=1)
    
    daily_rets = (port_val / port_val.shift(1))- 1
    daily_rets = daily_rets[1:]
    
    cr = (port_val[-1]/port_val[0]) - 1
    adr = daily_rets.mean()
    sddr = daily_rets.std()
    sr = np.sqrt(sample_freq) * np.mean(daily_rets - rfr) / daily_rets.std()
    
    # Compare daily portfolio value with SPY using a normalized plot
    
    if gen_plot:
                
        df_temp = pd.concat(
            [port_val, prices_SPY], keys=['Portfolio', 'SPY'], axis=1)
        df_normed = df_temp / df_temp.iloc[0, :]
        
        plt.figure(figsize=(8,6))
        plt.title('Daily Portfolio Value and SPY')
        plt.plot(df_normed.iloc[:, 0], linewidth = 2.0, label='Portfolio')
        plt.plot(df_normed.iloc[:, 1], linewidth = 2.0, label='SPY')
        plt.xlabel('Date')
        plt.ylabel('Price')
        plt.legend(loc='upper right')
        plt.grid(color='black', linestyle='--', linewidth=0.5)
        plt.savefig('figure1.png')
        
        pass
    
    return opt_alloc, cr, adr, sddr, sr

  	  	   		   	 		  		  		    	 		 		   		 		  	  	   		   	 		  		  		    	 		 		   		 		  
 def test_code():
 
     allocations, cr, adr, sddr, sr = optimize_portfolio(gen_plot=True,)
