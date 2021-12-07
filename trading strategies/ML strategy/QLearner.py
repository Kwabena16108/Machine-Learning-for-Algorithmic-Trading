
"""  		  	   		   	 		  		  		    	 		 		   		 		  
The following code isthe implementation of a QLearner algorithm	  	   		   	 		  		  		    	 		 		   		 		  
"""

import random as rand
import numpy as np


class QLearner(object):
    """  		  	   		   	 		  		  		    	 		 		   		 		  
    This is a Q learner object.  		  	   		   	 		  		  		    	 		 		   		 		  
  		  	   		   	 		  		  		    	 		 		   		 		  
    :param num_states: The number of states to consider.  		  	   		   	 		  		  		    	 		 		   		 		  
    :type num_states: int  		  	   		   	 		  		  		    	 		 		   		 		  
    :param num_actions: The number of actions available..  		  	   		   	 		  		  		    	 		 		   		 		  
    :type num_actions: int  		  	   		   	 		  		  		    	 		 		   		 		  
    :param alpha: The learning rate used in the update rule. Should range between 0.0 and 1.0 with 0.2 as a typical value.  		  	   		   	 		  		  		    	 		 		   		 		  
    :type alpha: float  		  	   		   	 		  		  		    	 		 		   		 		  
    :param gamma: The discount rate used in the update rule. Should range between 0.0 and 1.0 with 0.9 as a typical value.  		  	   		   	 		  		  		    	 		 		   		 		  
    :type gamma: float  		  	   		   	 		  		  		    	 		 		   		 		  
    :param rar: Random action rate: the probability of selecting a random action at each step. Should range between 0.0 (no random actions) to 1.0 (always random action) with 0.5 as a typical value.  		  	   		   	 		  		  		    	 		 		   		 		  
    :type rar: float  		  	   		   	 		  		  		    	 		 		   		 		  
    :param radr: Random action decay rate, after each update, rar = rar * radr. Ranges between 0.0 (immediate decay to 0) and 1.0 (no decay). Typically 0.99.  		  	   		   	 		  		  		    	 		 		   		 		  
    :type radr: float  		  	   		   	 		  		  		    	 		 		   		 		  
    :param dyna: The number of dyna updates for each regular update. When Dyna is used, 200 is a typical value.  		  	   		   	 		  		  		    	 		 		   		 		  
    :type dyna: int  		  	   		   	 		  		  		    	 		 		   		 		  
    :param verbose: If “verbose” is True, your code can print out information for debugging.  		  	   		   	 		  		  		    	 		 		   		 		  
    :type verbose: bool  		  	   		   	 		  		  		    	 		 		   		 		  
    """

    def __init__(
            self,
            num_states=100,
            num_actions=4,
            alpha=0.2,
            gamma=0.9,
            rar=0.5,
            radr=0.99,
            dyna=0,
            verbose=False,
    ):
        """  		  	   		   	 		  		  		    	 		 		   		 		  
        Constructor method  		  	   		   	 		  		  		    	 		 		   		 		  
        """
        self.verbose = verbose
        self.num_actions = num_actions
        self.s = 0
        self.a = 0

        self.num_states = num_states
        self.alpha = alpha
        self.gamma = gamma
        self.rar = rar
        self.radr = radr
        self.dyna = dyna

        self.state_space = np.arange(self.num_states)
        self.action_space = np.arange(self.num_actions)

        self.Q = np.zeros((num_states, num_actions))

        # experience
        self.exp = []

    def querysetstate(self, s):
        """  		  	   		   	 		  		  		    	 		 		   		 		  
        Update the state without updating the Q-table  		  	   		   	 		  		  		    	 		 		   		 		  
  		  	   		   	 		  		  		    	 		 		   		 		  
        :param s: The new state  		  	   		   	 		  		  		    	 		 		   		 		  
        :type s: int  		  	   		   	 		  		  		    	 		 		   		 		  
        :return: The selected action  		  	   		   	 		  		  		    	 		 		   		 		  
        :rtype: int  		  	   		   	 		  		  		    	 		 		   		 		  
        """
        self.s = s
        if rand.random() < self.rar:  # explore
            action = rand.choice(self.action_space)  # random action
        else:  # exploit
            action = np.argmax(self.Q[s])  # get the action with highest Q value for state s

        if self.verbose:
            print(f"s = {s}, a = {action}")
        return action

    def update_Q(self, s, a, s_prime, r):
        # current value of Q(S, a)
        Q_s_a = self.Q[s, a]
        # picking best action from state S'
        best_a_prime = np.argmax(self.Q[s_prime])
        # getting maxQ(S', a')
        max_q_prime = self.Q[s_prime, best_a_prime]
        # updating Q(S,a)
        self.Q[s, a] = Q_s_a + self.alpha * (r + self.gamma * max_q_prime - Q_s_a)

    def query(self, s_prime, r):
        """  		  	   		   	 		  		  		    	 		 		   		 		  
        Update the Q table and return an action  		  	   		   	 		  		  		    	 		 		   		 		  
  		  	   		   	 		  		  		    	 		 		   		 		  
        :param s_prime: The new state  		  	   		   	 		  		  		    	 		 		   		 		  
        :type s_prime: int  		  	   		   	 		  		  		    	 		 		   		 		  
        :param r: The immediate reward  		  	   		   	 		  		  		    	 		 		   		 		  
        :type r: float  		  	   		   	 		  		  		    	 		 		   		 		  
        :return: The selected action  		  	   		   	 		  		  		    	 		 		   		 		  
        :rtype: int  		  	   		   	 		  		  		    	 		 		   		 		  
        """
        self.update_Q(self.s, self.a, s_prime, r)
        self.exp.append((self.s, self.a, s_prime, r))

        # hallucinate from experience
        for i in range(min(self.dyna, len(self.exp))):
            s, a, sp, dr = rand.choice(self.exp)
            self.update_Q(s, a, sp, dr)

        # picking the action for S'
        action = self.querysetstate(s_prime)
        self.a = action

        # update (decaying) random action rate, so we explore less and exploit more next time
        self.rar *= self.radr

        if self.verbose:
            print(f"s = {s_prime}, a = {action}, r={r}")
        return action

    def author(self):
        return "Dixon"


