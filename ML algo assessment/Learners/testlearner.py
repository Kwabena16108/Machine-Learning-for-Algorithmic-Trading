		   	 		  		  		    	 		 		   		 		    		  	   		   	 		  		  		    	 		 		   		 		  
import math  		  	   		   	 		  		  		    	 		 		   		 		  
import sys  	
import os	  	   		   	 		  		  		    	 		 		   		 		  
  		  	   		   	 		  		  		    	 		 		   		 		  
import numpy as np
import matplotlib.pyplot as plt  	   		   	 		  		  		    	 		 		   		 		  
  		  	   		   	 		  		  		    	 		 		   		 		  
import LinRegLearner as lrl  		
import DTLearner as dt
import RTLearner as rt
import BagLearner as bl
import InsaneLearner as it	


def pull_data(datafile):
   path = open(
       os.path.join(
           os.environ.get("LEARNER_DATA_DIR", "Data/"),
           datafile), "r")
   
   alldata = np.genfromtxt(path, delimiter=",")
   
   if datafile == "Istanbul.csv":
       alldata = alldata[1:, 1:]
       
   return alldata

def mae(actual, pred):
    sum_error = 0.0
    for i in range(len(actual)):
        sum_error+=abs(pred[i]-actual[i])
    return sum_error / float(len(actual))

def mbe(actual,pred):
    sum_error = 0.0
    for i in range(len(actual)):
        sum_error+=(pred[i]-actual[i])
    return sum_error / float(len(actual))
		   	 		  		  		    	 	
def msle(actual,pred):
    log_sum_error = 0.0
    for i in range(len(actual)):
        log_pred_error = np.log(pred[i]+1)-np.log(actual[i]+1)
        log_sum_error+= (log_pred_error ** 2)
    return log_sum_error / float(len(actual))
    
     
    

def experiment_one():
    
    """
    Overfitting as observed in with DTLearner
    data: Istanbul.csv
    """
    np.random.seed(gtid())  
    
    data = pull_data('Istanbul.csv')
    n =len(data)
    index = np.arange(0, n)
    idx = np.random.choice(index, size=n, replace=False)
    rand_df = data[idx,:]  		   	 		  		  		    	 		 		   		 		  
    train_rows = int(0.6 * data.shape[0])  		  	   		   	 		  		  		    	 		 		   		 		  
    test_rows = data.shape[0] - train_rows  		  	   		   	 		  		  		    	 		 		   		 		  
  		  	   		   	 		  		  		    	 		 		   		 		  
    # separate out training and testing data  		  	   		   	 		  		  		    	 		 		   		 		  
    train_x = rand_df[:train_rows, 0:-1]  		  	   		   	 		  		  		    	 		 		   		 		  
    train_y = rand_df[:train_rows, -1]  		  	   		   	 		  		  		    	 		 		   		 		  
    test_x = rand_df[train_rows:, 0:-1]  		  	   		   	 		  		  		    	 		 		   		 		  
    test_y = rand_df[train_rows:, -1]  		  	   		   	 		  		  		    	 		 		   		 		  
  		  	   		   	 		  		  		    	 		 		   		 		  
    print(f"{test_x.shape}")  		  	   		   	 		  		  		    	 		 		   		 		  
    print(f"{test_y.shape}")  		  	   		   	 		  		  		    	 		 		   		 		  

    
    leaf_size = [i for i in range(1,101)]
    insam_rmse_score, outsam_rmse_score = [], [] 
    
    for i in leaf_size:
        # create an instance of DTLearner
        learner = dt.DTLearner(leaf_size=i, verbose=False)
        # fit DT on training data
        learner.add_evidence(train_x, train_y)
        # query train data and evaluate score
        insample_pred_y = learner.query(train_x)
        insam_rmse = math.sqrt(((train_y-insample_pred_y)**2).sum()/ train_y.shape[0])
        insam_rmse_score.append(insam_rmse)
        # query test data and evaluate score
        outsample_pred_y = learner.query(test_x)
        outsam_rmse = math.sqrt(((test_y-outsample_pred_y)**2).sum()/ test_y.shape[0])
        outsam_rmse_score.append(outsam_rmse)
        
        # summarize
        print(">%d, insample: %.3f, outsample: %.3f" % (i, insam_rmse, outsam_rmse))
        
    plt.figure(figsize=(8, 6))
    plt.plot(leaf_size, insam_rmse_score, '-o', label='In-sample')
    plt.plot(leaf_size, outsam_rmse_score, '-ro', label='Out-of-sample')
    plt.ylabel('RMSE')
    plt.xlabel('Leaf Size')
    plt.title('In-sample vs Out-of-sample Performance \n (Decision Tree)')
    plt.legend()
    plt.savefig('experiment1.png')
    

def experiment_two():
    
    """
    Overfitting as observed in with DTLearner
    data: Istanbul.csv
    """
    np.random.seed(gtid())  
    
    data = pull_data('Istanbul.csv')
    n =len(data)
    index = np.arange(0, n)
    idx = np.random.choice(index, size=n, replace=False)
    rand_df = data[idx,:]  		   	 		  		  		    	 		 		   		 		  
    train_rows = int(0.6 * data.shape[0])  		  	   		   	 		  		  		    	 		 		   		 		  
    test_rows = data.shape[0] - train_rows  		  	   		   	 		  		  		    	 		 		   		 		  
  		  	   		   	 		  		  		    	 		 		   		 		  
    # separate out training and testing data  		  	   		   	 		  		  		    	 		 		   		 		  
    train_x = rand_df[:train_rows, 0:-1]  		  	   		   	 		  		  		    	 		 		   		 		  
    train_y = rand_df[:train_rows, -1]  		  	   		   	 		  		  		    	 		 		   		 		  
    test_x = rand_df[train_rows:, 0:-1]  		  	   		   	 		  		  		    	 		 		   		 		  
    test_y = rand_df[train_rows:, -1]  		  	   		   	 		  		  		    	 		 		   		 		  
  		  	   		   	 		  		  		    	 		 		   		 		  
    print(f"{test_x.shape}")  		  	   		   	 		  		  		    	 		 		   		 		  
    print(f"{test_y.shape}")  		  	   		   	 		  		  		    	 		 		   		 		  

    
    leaf_size = [i for i in range(1,101)]
    insam_rmse_score, outsam_rmse_score = [], [] 
    
    for i in leaf_size:
        # create an instance of BagLearner
        learner = bl.BagLearner(learner=dt.DTLearner, kwargs={"leaf_size":i}, bags=10, boost=False,verbose=False) 
        # fit BagLearner on training data
        learner.add_evidence(train_x, train_y)
        # query train data and evaluate score
        insample_pred_y = learner.query(train_x)
        insam_rmse = math.sqrt(((train_y-insample_pred_y)**2).sum()/ train_y.shape[0])
        insam_rmse_score.append(insam_rmse)
        # query test data and evaluate score
        outsample_pred_y = learner.query(test_x)
        outsam_rmse = math.sqrt(((test_y-outsample_pred_y)**2).sum()/ test_y.shape[0])
        outsam_rmse_score.append(outsam_rmse)
        
        # summarize
        print(">%d, insample: %.3f, outsample: %.3f" % (i, insam_rmse, outsam_rmse))
        
    plt.figure(figsize=(8, 6))
    plt.plot(leaf_size, insam_rmse_score, '-o', label='In-sample')
    plt.plot(leaf_size, outsam_rmse_score, '-ro', label='Out-of-sample')
    plt.ylabel('RMSE')
    plt.xlabel('Leaf Size')
    plt.title('In-sample vs Out-of-sample Performance \n (Bagged Decision Trees)')
    plt.legend()
    plt.savefig('experiment2.png')


def experiment_three():
    
    """
    Overfitting as observed in with DTLearner
    data: Istanbul.csv
    """
    np.random.seed(gtid())  
    
    data = pull_data('Istanbul.csv')
    n =len(data)
    index = np.arange(0, n)
    idx = np.random.choice(index, size=n, replace=False)
    rand_df = data[idx,:]  		   	 		  		  		    	 		 		   		 		  
    train_rows = int(0.6 * data.shape[0])  		  	   		   	 		  		  		    	 		 		   		 		  
    test_rows = data.shape[0] - train_rows  		  	   		   	 		  		  		    	 		 		   		 		  
  		  	   		   	 		  		  		    	 		 		   		 		  
    # separate out training and testing data  		  	   		   	 		  		  		    	 		 		   		 		  
    train_x = rand_df[:train_rows, 0:-1]  		  	   		   	 		  		  		    	 		 		   		 		  
    train_y = rand_df[:train_rows, -1]  		  	   		   	 		  		  		    	 		 		   		 		  
    test_x = rand_df[train_rows:, 0:-1]  		  	   		   	 		  		  		    	 		 		   		 		  
    test_y = rand_df[train_rows:, -1]  		  	   		   	 		  		  		    	 		 		   		 		  
  		  	   		   	 		  		  		    	 		 		   		 		  
    print(f"{test_x.shape}")  		  	   		   	 		  		  		    	 		 		   		 		  
    print(f"{test_y.shape}")  	

    # create a learner and train it  		  	   		   	 		  		  		    	 		 		   		 		  
    dt_learner = dt.DTLearner(leaf_size=1, verbose=False)  	   	 		  		  		    	 		 		   		 		  
    rt_learner = rt.RTLearner(leaf_size=1, verbose=False) 		  	   		   	 		  		  		    	 		 		   		 		  
   		   	 		  		  		    	 		 		   		 		  
    dt_learner.add_evidence(train_x, train_y)
    rt_learner.add_evidence(train_x, train_y)		  	   		   	 		  		  		    	 		 		   		 		  
  		  	   		   	 		  		  		    	 		 		   		 		  
    # evaluate in sample  		  	   		   	 		  		  		    	 		 		   		 		  
    dt_pred_y = dt_learner.query(train_x)  # get the predictions 
    rt_pred_y = rt_learner.query(train_x) 		  	   		   	 		  		  		    	 		 		   		 		  
    dt_mae = mae(train_y, dt_pred_y)	
    rt_mae = mae(train_y, rt_pred_y)
    dt_mbe = mbe(train_y, dt_pred_y)	
    rt_mbe = mbe(train_y, rt_pred_y)
    dt_msle = msle(train_y, dt_pred_y)	
    rt_msle = msle(train_y, rt_pred_y)
	   		   	 		  		  		    	 		 		   		 		  
    print()  		  	   		   	 		  		  		    	 		 		   		 		  
    print("In sample results")  		  	   		   	 		  		  		    	 		 		   		 		  
    print(f"MAE for DTLearner: {dt_mae}")
    print(f"MAE for RTLearner: {rt_mae}") 
    print(f"MBE for DTLearner: {dt_mbe}")
    print(f"MBE for RTLearner: {rt_mbe}")  
    print(f"MSLE for DTLearner: {dt_msle}")
    print(f"MSLE for RTLearner: {rt_msle}")    		  	   		   	 		  		  		    	 		 		   		 		  
  		  	   		   	 		  		  		    	 		 		   		 		  
    # evaluate out of sample  		  	   		   	 		  		  		    	 		 		   		 		  
    dt_pred_y = dt_learner.query(test_x)  # get the predictions  
    rt_pred_y = rt_learner.query(test_x) 		  	   		   	 		  		  		    	 		 		   		 		  
     		  	   		   	 		  	
    dt_mae = mae(test_y, dt_pred_y)	
    rt_mae = mae(test_y, rt_pred_y)
    dt_mbe = mbe(test_y, dt_pred_y)	
    rt_mbe = mbe(test_y, rt_pred_y)
    dt_msle = msle(test_y, dt_pred_y)	
    rt_msle = msle(test_y, rt_pred_y)
	   		   	 		  		  		    	 		 		   		 		  
    print()  		  	   		   	 		  		  		    	 		 		   		 		  
    print("Out of sample results")  		  	   		   	 		  		  		    	 		 		   		 		  
    print(f"MAE for DTLearner: {dt_mae}")
    print(f"MAE for RTLearner: {rt_mae}") 
    print(f"MBE for DTLearner: {dt_mbe}")
    print(f"MBE for RTLearner: {rt_mbe}")
    print(f"MSLE for DTLearner: {dt_msle}")
    print(f"MSLE for RTLearner: {rt_msle}")    		  	   		   	 		  		  		    	 		 		   		 		  
    		  	   		   	 		  		  		    	 		 		   		 		  
	  		    	 		 		   		 		  
    fig, axe = plt.subplots()	
    fig.set_size_inches(10,6)
    axe.plot(range(len(test_y)), test_y, 'b-', label='Actual')
    axe.plot(range(len(dt_pred_y)), dt_pred_y, 'r--', label='DT Model fit')
    plt.ylabel('Test set')
    plt.xlabel('')
    plt.title('Out-of-sample Model fit for Decision Tree Learner')
    plt.legend()
    plt.savefig('experiment3_1.png')
    
    fig, axe = plt.subplots()	
    fig.set_size_inches(10,6)
    axe.plot(range(len(test_y)), test_y, 'b-', label='Actual')
    axe.plot(range(len(rt_pred_y)), rt_pred_y, 'g--', label='RT Model fit')
    plt.ylabel('Test set')
    plt.xlabel('')
    plt.title('Out-of-sample Model fit for Random Tree Learner')
    plt.legend()
    plt.savefig('experiment3_2.png')
    	 		 		   		 		  



if __name__ == '__main__':
    experiment_one()
    experiment_two()
    experiment_three()
   
    
    
    
    

    
    
    
    
    
    
    
