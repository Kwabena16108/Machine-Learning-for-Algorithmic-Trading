""""""
"""  		  	   		   	 		  		  		    	 		 		   		 		  
Template for implementing StrategyLearner  (c) 2016 Tucker Balch  		  	   		   	 		  		  		    	 		 		   		 		  
  		  	   		   	 		  		  		    	 		 		   		 		  
Copyright 2018, Georgia Institute of Technology (Georgia Tech)  		  	   		   	 		  		  		    	 		 		   		 		  
Atlanta, Georgia 30332  		  	   		   	 		  		  		    	 		 		   		 		  
All Rights Reserved  		  	   		   	 		  		  		    	 		 		   		 		  
  		  	   		   	 		  		  		    	 		 		   		 		  
Template code for CS 4646/7646  		  	   		   	 		  		  		    	 		 		   		 		  
  		  	   		   	 		  		  		    	 		 		   		 		  
Georgia Tech asserts copyright ownership of this template and all derivative  		  	   		   	 		  		  		    	 		 		   		 		  
works, including solutions to the projects assigned in this course. Students  		  	   		   	 		  		  		    	 		 		   		 		  
and other users of this template code are advised not to share it with others  		  	   		   	 		  		  		    	 		 		   		 		  
or to make it available on publicly viewable websites including repositories  		  	   		   	 		  		  		    	 		 		   		 		  
such as github and gitlab.  This copyright statement should not be removed  		  	   		   	 		  		  		    	 		 		   		 		  
or edited.  		  	   		   	 		  		  		    	 		 		   		 		  
  		  	   		   	 		  		  		    	 		 		   		 		  
We do grant permission to share solutions privately with non-students such  		  	   		   	 		  		  		    	 		 		   		 		  
as potential employers. However, sharing with other current or future  		  	   		   	 		  		  		    	 		 		   		 		  
students of CS 7646 is prohibited and subject to being investigated as a  		  	   		   	 		  		  		    	 		 		   		 		  
GT honor code violation.  		  	   		   	 		  		  		    	 		 		   		 		  
  		  	   		   	 		  		  		    	 		 		   		 		  
-----do not edit anything above this line---  		  	   		   	 		  		  		    	 		 		   		 		  
  		  	   		   	 		  		  		    	 		 		   		 		  
Student Name: Tucker Balch (replace with your name)  		  	   		   	 		  		  		    	 		 		   		 		  
GT User ID: tb34 (replace with your User ID)  		  	   		   	 		  		  		    	 		 		   		 		  
GT ID: 900897987 (replace with your GT ID)  		  	   		   	 		  		  		    	 		 		   		 		  
"""

import datetime as dt

import util as ut
from QLearner import QLearner
from indicator_func import *
from itertools import product
from tqdm import tqdm

MAX_ITERATIONS = 100


def get_state_indexes(labels, indicators_count):
    return {state: i for i, state in enumerate(product(labels, repeat=indicators_count))}
    # for i, state in enumerate(product(labels, repeat=indicators_count)):
    #     state2idx[state] = i


DEFAULT_BINS = np.arange(-1.8, 1.8, 0.4)


class StrategyLearner(object):
    def author(self):
        return 'tpasumarthi3'

        # constructor

    def __init__(self, verbose=False, impact=0.0):
        self.verbose = verbose
        self.impact = impact
        self.ndays = 30

        self.bins_pema = DEFAULT_BINS
        self.bins_bb = DEFAULT_BINS
        self.bins_momentum = DEFAULT_BINS

        self.labels = np.arange(1, len(self.bins_bb), dtype=int)
        self.state2idx = get_state_indexes(labels=np.arange(0, len(self.bins_bb)),
                                           indicators_count=3)  # we include 0 because of the nans

        self.learner = QLearner(num_states=len(self.state2idx), num_actions=3, dyna=10)

        # this method should create a QLearner, and train it for trading

    def get_X(self, prices, symbol, sd=None, ed=None):
        pema = get_pema(prices, [symbol], window=self.ndays)
        bb = get_bb(prices, [symbol], window=self.ndays)
        momentum = get_momentum(prices, [symbol], window=self.ndays)

        x = pd.concat((pema, bb, momentum), axis=1)
        x.columns = ["PEMA", "BB", "MOMENTUM"]
        x.fillna(0, inplace=True)
        return x

    def add_evidence(self, symbol="IBM",
                     sd=dt.datetime(2008, 1, 1),
                     ed=dt.datetime(2009, 1, 1),
                     sv=10000):

        # example usage of the old backward compatible util function
        syms = [symbol]
        dates = pd.date_range(sd, ed)
        prices_all = ut.get_data(syms, dates)  # automatically adds SPY
        prices = prices_all[syms]  # only portfolio symbols

        normalized_prices = normalize_prices(prices)
        daily_price_diff = compute_daily_returns(normalized_prices)

        if self.verbose:
            print(prices)

        x_train = self.get_X(prices, symbol, sd, ed)
        x_train = self.discretize(x_train)

        # let's initialize our QLearner with the first state
        first_state = x_train.state.iloc[0]
        self.learner.querysetstate(first_state)
        # 0 NOTHING
        # 1 BUY
        # 2 SELL

        holdings = 0

        old_strategy = pd.DataFrame()  # you have a magic diary of trading strategy from older generations
        for _ in tqdm(range(MAX_ITERATIONS)):
            trades = pd.DataFrame({"Shares": 0},
                                  index=x_train.index
                                  )
            # go over each trading day in the training set
            for index in x_train.index[:-1]:
                # you woke up in the morning and observed the price change based on the actions you took yesterday
                # here is your reward
                price_diff = daily_price_diff.loc[index][symbol]
                reward = holdings * price_diff * (1 - self.impact)

                # feed this observation to the QLearner and let it suggest the next action
                # it will update its strategy according to that reward
                state = x_train.loc[index].state
                a = self.learner.query(state, reward)
                # apply the action suggested by the QLearner
                if a == 1 and holdings < 1000:
                    trades.loc[index]["Shares"] = 1000 - holdings
                    holdings = 1000
                elif a == 2 and holdings > -1000:
                    trades.loc[index]["Shares"] = -1000 - holdings
                    holdings = -1000
                # now your holdings are updated, you can go to sleep and wait to see what happens tomorrow

            # check if your strategy is the same as the one left from your ancestors
            if trades.equals(old_strategy):  # if it is the same then you have found karma
                print("Model has converged")
                return
            # if not, then replace this stupid magic diary with your own strategy
            old_strategy = trades.copy()

            # you got reincarnated and sent back to day 1
            # and now you have the magic diary from your old self as your Q table
        print("Max iterations reached. Model has not converged.")

    def testPolicy(self, symbol="IBM",
                   sd=dt.datetime(2009, 1, 1), ed=dt.datetime(2010, 1, 1),
                   sv=10000):

        # example usage of the old backward compatible util function
        syms = [symbol]
        dates = pd.date_range(sd, ed)
        prices_all = ut.get_data(syms, dates)  # automatically adds SPY
        prices = prices_all[syms]  # only portfolio symbol

        if self.verbose:
            print(prices)

        x_test = self.get_X(prices, symbol, sd, ed)
        x_test = self.discretize(x_test)

        first_state = x_test.state.iloc[0]
        self.learner.querysetstate(int(float(first_state)))

        holdings = 0
        trades = pd.DataFrame({"Shares": 0}, index=x_test.index)
        for index in x_test.index[:-1]:
            state = x_test.loc[index].state
            a = self.learner.querysetstate(state)
            if a == 1 and holdings < 1000:
                trades.loc[index]["Shares"] = 1000 - holdings
                holdings = 1000
            elif a == 2 and holdings > -1000:
                trades.loc[index]["Shares"] = -1000 - holdings
                holdings = -1000

        trades.iloc[-1]["Shares"] = 0 - holdings
        return trades[["Shares"]]

    def discretize(self, data):
        # number of states = labels**3
        # why 3? because we have 3 indicators
        data["PEMA_state"] = pd.cut(data.PEMA, self.bins_pema, labels=self.labels).astype(float)
        data["BB_state"] = pd.cut(data.BB, self.bins_bb, labels=self.labels).astype(float)
        data["MOMENTUM_state"] = pd.cut(data.MOMENTUM, self.bins_momentum, labels=self.labels).astype(float)
        data = data.fillna(0)
        cols = ["PEMA_state", "BB_state", "MOMENTUM_state"]
        data[cols] = data[cols].astype(int)

        def represent_state(row):
            p = row["PEMA_state"]
            b = row["BB_state"]
            m = row["MOMENTUM_state"]
            return p, b, m

        data["state_repr"] = data.apply(represent_state, axis="columns")
        data["state"] = data.state_repr.apply(self.state2idx.get)

        data = data.drop(["PEMA", "BB", "MOMENTUM"] + cols, axis="columns")

        return data


if __name__ == "__main__":
    sl = StrategyLearner()
    sl.add_evidence()
    t = sl.testPolicy()
    print("One does not simply think up a strategy")
