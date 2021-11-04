import numpy as np


class BagLearner(object):
    def __init__(self, learner, kwargs, bags, boost=False, verbose=False):
        self.learner = learner
        self.kwargs = kwargs
        self.bags = bags
        self.boost = boost
        self.verbose = verbose
    
        self.learners = []
    def author(self):  		  	   		   	 		  		  		    	 		 		   		 		  
        """  		  	   		   	 		  		  		    	 		 		   		 		  
        :return: The GT username of the student  		  	   		   	 		  		  		    	 		 		   		 		  
        :rtype: str  		  	   		   	 		  		  		    	 		 		   		 		  
        """  		  	   		   	 		  		  		    	 		 		   		 		  
        return "dnkwantabisa3" 

    def add_evidence(self, trainX, trainY):
        n = len(trainX)  # number of samples
        index = np.arange(0, n)
        if self.verbose:
            print(f"Training on {n} samples with {trainX.shape[1]} features")
        for i in range(self.bags):
            idx = np.random.choice(index, size=n, replace=True)
            learner = self.learner(**self.kwargs)
            learner.add_evidence(trainX[idx], trainY[idx])
            self.learners.append(learner)

    def query(self, data_x):
        predictions = []
        for learner in self.learners:
            bag_predictions = learner.query(data_x)
            predictions.append(bag_predictions)
        return np.mean(predictions, axis=0)
