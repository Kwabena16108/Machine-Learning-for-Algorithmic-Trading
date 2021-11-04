#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 16 21:39:36 2021

@author: dicksonnkwantabisa
"""

# Helper functions

import numpy as np

def variance(y):
    if len(y) == 1:
        return 0
    return np.var(y)

def calc_info_gain(y,mask):
    a = sum(mask)
    b = mask.shape[0] - a
    
    if a == 0 or b == 0:
        return 0
    return variance(y) - (a / (a + b) * variance(y[mask])) - (b / (a +b) * variance(y[~mask]))


def get_best_split_val(x ,y):
    best_ig = float('-inf')
    best_val = None
    options = sorted(set(x))[1:]
    for val in options:
        mask = x <= val
        val_ig = calc_info_gain(y, mask)
        if val_ig > best_ig:
            best_ig = val_ig
            best_val = val
    return best_ig, best_val


def get_best_split_feature(data_x, y):
    n_samples, n_features = data_x.shape
    best_feature = None
    max_info_gain = float('-inf')
    
    for i in range(n_features):
        x = data_x[:, i]
        split_val = np.median(x)
        mask = x <= split_val
        ig = calc_info_gain(y, mask)
        if ig > max_info_gain:
            max_info_gain = ig
            best_feature = i
    return best_feature, np.mean(data_x[:, best_feature]), max_info_gain



def build_tree(data_x, data_y, leaf_size):
    n_samples, n_features = data_x.shape
    if n_samples <= leaf_size or len(set(data_y)) == 1:
        return [["leaf", data_y.mean(), None, None]]
    
    best_feature = np.random.choice(np.arange(n_features))
    split_val = np.median(data_x[:, best_feature])
    
    mask = data_x[:, best_feature] <= split_val
    max_info_gain = calc_info_gain(data_y, mask)

    if max_info_gain == 0:  # no info gain by splitting, so consider this a leaf
        return [["leaf", data_y.mean(), None, None]]

    left_idx = data_x[:, best_feature] <= split_val
    left_tree = build_tree(data_x[left_idx], data_y[left_idx], leaf_size=leaf_size)

    right_idx = data_x[:, best_feature] > split_val
    right_tree = build_tree(data_x[right_idx], data_y[right_idx], leaf_size=leaf_size)

    root = [[best_feature, split_val, 1, len(left_tree) + 1]]

    return root + left_tree + right_tree




# RTLearner class

class RTLearner:
    def __init__(self, leaf_size, verbose=False):
        self.leaf_size = leaf_size
        self.verbose = verbose
        self.tree = None
    
    def author(self):  		  	   		   	 		  		  		    	 		 		   		 		  
        """  		  	   		   	 		  		  		    	 		 		   		 		  
        :return: The GT username of the student  		  	   		   	 		  		  		    	 		 		   		 		  
        :rtype: str  		  	   		   	 		  		  		    	 		 		   		 		  
        """  		  	   		   	 		  		  		    	 		 		   		 		  
        return "dnkwantabisa3" 

    def add_evidence(self, data_x, data_y):
        """
        :param data_x: A set of feature values used to train the learner
        :type data_x: numpy.ndarray
        :param data_y: The value we are attempting to predict given the X data
        :type data_y: numpy.ndarray
        """
        if self.verbose:
            print(f"Training data has {data_x.shape[0]} instances, and {data_x.shape[1]} features")
        self.tree = build_tree(data_x, data_y, self.leaf_size)

    def query(self, data_x):
        y_pred = np.array([])
        for row in data_x:
            idx = 0
            node = self.tree[idx]
            while True:
                factor, split_val, left, right = node
                if factor == "leaf":
                    y_pred = np.append(y_pred, split_val)
                    break
                x = row[factor]
                if x <= split_val:
                    idx += left
                else:
                    idx += right
                node = self.tree[idx]
        return y_pred