import numpy as np


def mode(array):
    vals, counts = np.unique(array, return_counts=True)
    return vals[np.argmax(counts)]


def build_tree(data_x, data_y, leaf_size):
    n_samples, n_features = data_x.shape
    # if we have less samples than min leaf size
    # or we only have samples from one class in that node
    if n_samples <= leaf_size or len(set(data_y[:,0])) == 1: # np.array([1],[-1],[0]) unhashable
        return [["leaf", mode(data_y), None, None]]
    best_feature = np.random.choice(np.arange(n_features))
    split_val = np.median(data_x[:, best_feature], axis=0)

    if split_val == np.max(data_x[:, best_feature]):
        max_x_idx = np.argmax(data_x[:, best_feature])
        return [["leaf", data_y[max_x_idx], None, None]]

    left_idx = data_x[:, best_feature] <= split_val
    right_idx = data_x[:, best_feature] > split_val

    left_tree = build_tree(data_x[left_idx], data_y[left_idx], leaf_size)
    right_tree = build_tree(data_x[right_idx], data_y[right_idx], leaf_size)

    root = [[best_feature, split_val, 1, len(left_tree) + 1]]

    return root + left_tree + right_tree


class RTLearner:
    def __init__(self, leaf_size, verbose=False):
        self.leaf_size = leaf_size
        self.verbose = verbose
        self.tree = None

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
        y_pred = []
        for row in data_x:
            idx = 0
            node = self.tree[idx]
            while True:
                factor, split_val, left, right = node
                if factor == "leaf":
                    y_pred.append(split_val)
                    break
                x = row[factor]
                if x <= split_val:
                    idx += left
                else:
                    idx += right
                node = self.tree[idx]
        return np.array(y_pred, dtype="object")
        
    def author(self):
        return "dnkwantabisa3"


class BagLearner:
    def __init__(self, kwargs, bags, boost=False, verbose=False):
        self.learner = RTLearner
        self.kwargs = kwargs
        self.bags = bags
        self.boost = boost
        self.verbose = verbose

        self.learners = []

    def add_evidence(self, data_x, data_y):
        self.learners = []
        n = len(data_x)  # number of samples
        index = np.arange(0, n)
        if self.verbose:
            print(f"Training on {n} samples with {data_x.shape[1]} features")
        for i in range(self.bags):
            idx = np.random.choice(index, size=n, replace=True)
            learner = self.learner(**self.kwargs)
            learner.add_evidence(data_x[idx], data_y[idx])
            self.learners.append(learner)

    def query(self, data_x):
        predictions = []
        for learner in self.learners:
            bag_predictions = learner.query(data_x)
            predictions.append(bag_predictions)
        return np.array([mode(i) for i in np.array(predictions).T], dtype="object")

    def author(self):
        return "dnkwantabisa3"
