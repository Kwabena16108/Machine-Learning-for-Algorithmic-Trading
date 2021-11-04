import BagLearner as bl
import LinRegLearner as lrl
import numpy as np
class InsaneLearner(object):
    def __init__(self, verbose=False):
        self.verbose = verbose
        self.bag_learners = []
    def author(self):  		  	   		   	 		  		  		    	 		 		   		 		  
        return "dnkwantabisa3" 
    def add_evidence(self, trainX, trainY):
        for i in range(20):
            bag_learner = bl.BagLearner(learner=lrl.LinRegLearner, kwargs={}, bags=20)
            bag_learner.add_evidence(trainX, trainY)
            self.bag_learners.append(bag_learner)
    def query(self, data_x):
        predictions = []
        for learner in self.bag_learners:
            bag_predictions = learner.query(data_x)
            predictions.append(bag_predictions)
        return np.mean(predictions, axis=0)
