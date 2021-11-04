
import math

import numpy as np


# this function should return a dataset (X and Y) that will work  		  	   		   	 		  		  		    	 		 		   		 		  
# better for linear regression than decision trees

def best_4_lin_reg(seed=1489683273):
    """  		  	   		   	 		  		  		    	 		 		   		 		  		  	   		   	 		  		  		    	 		 		   		 		  
    :param seed: The random seed for your data generation.  		  	   		   	 		  		  		    	 		 		   		 		  
    :type seed: int  		  	   		   	 		  		  		    	 		 		   		 		  
    :return: Returns data that performs significantly better with LinRegLearner than DTLearner.  		  	   		   	 		  		  		    	 		 		   		 		  
    :rtype: numpy.ndarray  		  	   		   	 		  		  		    	 		 		   		 		  
    """
    np.random.seed(seed)
    n_features = np.random.randint(2, 10)
    n_samples = np.random.randint(10, 1000)
    betas = np.random.random(n_features)
    features = []
    for i in range(n_features):
        mean = np.random.randint(1, 30)
        std = mean / 3
        column = np.random.normal(loc=mean, scale=std, size=n_samples)
        e = np.random.normal(loc=0, scale=std)
        column += e
        features.append(column)

    x = np.array(features).T
    y = np.matmul(x, betas)
    e = np.random.normal(loc=0, scale=np.std(y))
    y += e

    return x, y


def best_4_dt(seed=1489683273):
    """  		  	   		   	 		  		  		    	 		 		   		 		   		  	   		   	 		  		  		    	 		 		   		 		  
    :param seed: The random seed for your data generation.  		  	   		   	 		  		  		    	 		 		   		 		  
    :type seed: int  		  	   		   	 		  		  		    	 		 		   		 		  
    :return: Returns data that performs significantly better with DTLearner than LinRegLearner.  		  	   		   	 		  		  		    	 		 		   		 		  
    :rtype: numpy.ndarray  		  	   		   	 		  		  		    	 		 		   		 		  
    """
    np.random.seed(seed)
    n_features = np.random.randint(2, 10)
    n_samples = np.random.randint(10, 1000)

    x = np.random.normal(loc=0, scale=10, size=(n_samples, n_features))
    y = x[:, 0]**3 + x[:, 1]*x[:, -1] + x[:, 4]

    return x, y
