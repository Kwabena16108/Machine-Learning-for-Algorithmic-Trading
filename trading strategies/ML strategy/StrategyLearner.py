   		   	 		  		  		    	 		 		   		 		  		  	   		   	 		  		  		    	 		 		   		 		  
  		  	   		   	 		  		  		    	 		 		   		 		  
import datetime as dt  		  	   		   	 		  		  		    	 		 		   		 		  
import random
import numpy as np 		  	   		   	 		  		  		    	 		 		   		 		  	  	   		   	 		  		  		    	 		 		   		 		  
import pandas as pd  		  	   		   	 		  		  		    	 		 		   		 		  
import util as ut
import BagLearner as bl
from indicators import *
from marketsimcode import *


 		  	   		   	 		  		  		    	 		 		   		 		  
class StrategyLearner(object):  		  	   		   	 		  		  		    	 		 		   		 		  
    """  		  	   		   	 		  		  		    	 		 		   		 		  
    A strategy learner that can learn a trading policy using the same indicators used in ManualStrategy.  		  	   		   	 		  		  		    	 		 		   		 		  
  		  	   		   	 		  		  		    	 		 		   		 		  
    :param verbose: If “verbose” is True, your code can print out information for debugging.  		  	   		   	 		  		  		    	 		 		   		 		  
        If verbose = False your code should not generate ANY output.  		  	   		   	 		  		  		    	 		 		   		 		  
    :type verbose: bool  		  	   		   	 		  		  		    	 		 		   		 		  
    :param impact: The market impact of each transaction, defaults to 0.0  		  	   		   	 		  		  		    	 		 		   		 		  
    :type impact: float  		  	   		   	 		  		  		    	 		 		   		 		  
    :param commission: The commission amount charged, defaults to 0.0  		  	   		   	 		  		  		    	 		 		   		 		  
    :type commission: float  		  	   		   	 		  		  		    	 		 		   		 		  
    """  	
    def author(self):
        return 'dnkwantabisa3'
    	  	   		   	 		  		  		    	 		 		   		 		  
    # constructor 
 		  	   		   	 		  		  		    	 		 		   		 		  
    def __init__(self, verbose=False, impact=0.005, commission=0.0):  		  	   		   	 		  		  		    	 		 		   		 		  
        """  		  	   		   	 		  		  		    	 		 		   		 		  
        Constructor method  		  	   		   	 		  		  		    	 		 		   		 		  
        """  		  	   		   	 		  		  		    	 		 		   		 		  
        self.verbose = verbose  		  	   		   	 		  		  		    	 		 		   		 		  
        self.impact = impact
        self.ndays = 10  		  	   		   	 		  		  		    	 		 		   		 		  
        self.commission = commission  	
        self.learner = bl.BagLearner(kwargs={"leaf_size":5}, bags=15, boost=False, verbose=False)

    def get_X(self, prices, symbol):
        pema = get_psma(prices, [symbol], window=self.ndays)
        bb = get_bb(prices, [symbol], window=self.ndays)
        momentum = get_momentum(prices, [symbol], window=self.ndays)
        
        x = pd.concat([pema, bb, momentum], axis=1)
        x.fillna(0, inplace=True)
        return x
	  	   		   	 		  		  		    	 		 		   		 		  
  		  	   		   	 		  		  		    	 		 		   		 		  
    def add_evidence(self, symbol="JPM",
        sd=dt.datetime(2008, 1, 1),  		  	   		   	 		  		  		    	 		 		   		 		  
        ed=dt.datetime(2009, 12, 31),  	
        sv=100_000):
        
        syms = [symbol]  		  	   		   	 		  		  		    	 		 		   		 		  
        dates = pd.date_range(sd, ed)  	   		   	 		  		  		    	 		 		   		 		  
        prices_all = ut.get_data(syms, dates)  	  	   		   	 		  		  		    	 		 		   		 		  
        prices = prices_all[syms]  # only portfolio symbols  		  	   		   	 		  		  		    	 		 		   		 		  
        if self.verbose:  		  	   		   	 		  		  		    	 		 		   		 		  
            print(prices)  		  	   		   	 		  		  		    	 		 		   		 		  
  		
        # get predictors        
        X_train = self.get_X(prices, symbol)
        X_train = X_train[:-self.ndays]
        
        # set threshold on returns
        returns = (prices.values[self.ndays:]/prices.values[:-self.ndays]) - 1
        buy_threshold = 0.02 + self.impact
        sell_threshold = -0.02 - self.impact
        
        long = (returns > buy_threshold).astype(int) # buy
        short = (returns < sell_threshold).astype(int) # sell
        # categorical target series
        y_train = np.array(long - short)
        
        self.learner.add_evidence(X_train.values, y_train)
  		  	   		   	 		  		  		    	 		 		   		 		  
    # this method should use the existing policy and test it against new data  		  	   		   	 		  		  		    	 		 		   		 		  
    def testPolicy(self,symbol="JPM", 
        # validation set
        sd=dt.datetime(2008, 1, 1),  		  	   		   	 		  		  		    	 		 		   		 		  
        ed=dt.datetime(2009, 12, 31),   		  	   		   	 		  		  		    	 		 		   		 		    		  	   		   	 		  		  		    	 		 		   		 		  
        sv=100_000):

        syms = [symbol]  		  	   		   	 		  		  		    	 		 		   		 		  
        dates = pd.date_range(sd, ed)  		  	   		   	 		  		  		    	 		 		   		 		  
        prices_all = ut.get_data(syms, dates)  	  	   		   	 		  		  		    	 		 		   		 		  
        prices = prices_all[syms] 	  	   		   	 		  		  		    	 		 		   		 		  
        if self.verbose:  		  	   		   	 		  		  		    	 		 		   		 		  
            print(prices)  		  	   		   	 		  		  		    	 		 		   		 		  
  		
        # get predictors
        X_test = self.get_X(prices, symbol)
        # prediction
        y_test = self.learner.query(X_test.values) 
        
        # get trades based on prediction
        trades = np.zeros(y_test.shape[0])
        holdings = 0
        for i in range(y_test.shape[0]-1):
            # if y_test == 1 (BUY) so we shuld end up with +1000 (LONG)
            # if y_test == -1 (SELL) we should end up with -1000 (SHORT)
            # no. of shares to trade depends on how much we are holding
            # so if we are buying, new holding should be +1000 (LONG)
            # and the shares we need to trade to reach that are (+1000 - current holdings)
            # similarly for selling, (-1000 - current holdings)
            if y_test[i] != 0:
                new_holdings = 1000 if y_test[i] == 1 else -1000
                trades[i] = new_holdings - holdings
                holdings = new_holdings
                
        trades[-1] = -holdings # meaning no trade on last day
        trades = pd.DataFrame({"Shares":trades}, index=prices.index)
        trades.index.name = "Date"
        return trades  		  	   		   	 		  		  		    	 		 		   		 		  
  		  	   		   	 		  		  		    	 		 		   		 		  
  		  	   		   	 		  		  		    	 		 		   		 		  

		













  	   		   	 		  		  		    	 		 		   		 		  
