from utils import AbstractClassifier, csv2xy
from ID3 import ID3Node
import random
from sklearn.model_selection import train_test_split, KFold
import argparse
import numpy as np


def get_centroid(data):
    centroid = data.copy()
    centroid = centroid.mean(axis=0)
    return centroid


def distance(v1, v2):
    return np.linalg.norm(v1 - v2)


class KNNForestClassifier(AbstractClassifier):
    def __init__(self, N=25, k=11, p=None):
        super().__init__()
        self.p = p
        self.prob_range = 0.3, 0.7
        self.N = N
        self.k = k
        self.forest = []
        self.centroids = []

    def split_data(self, x, y):
        data_splits = []
        if self.p is None:
            test_sizes = [1 - random.uniform(self.prob_range[0], self.prob_range[1]) for i in range(self.N)]
        else:
            test_sizes = [1 - self.p for i in range(self.N)]
        for size in test_sizes:
            X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=size)
            train_x = X_train.copy()
            train_x["diagnosis"] = y_train
            centroid = get_centroid(train_x)
            tup = train_x, centroid
            data_splits.append(tup)
        return data_splits

    def fit(self, x, y):
        data_splits = self.split_data(x, y)
        self.forest = [(ID3Node(data=data), centroid) for (data, centroid) in data_splits]

    def predict(self, x, y):
        data = x.copy()
        data["diagnosis"] = y
        test_centroid = get_centroid(data)
        knn_trees = [(tree, distance(test_centroid, centroid)) for (tree, centroid) in self.forest]
        knn_trees.sort(key=lambda val: val[1])
        knn_trees = [tree for (tree, dist) in knn_trees]
        knn_trees = knn_trees[:self.k]
        right_predictions, false_positive, false_negative = 0, 0, 0
        for row in range(len(data.index)):
            predictions = [self.walk_the_tree(tree, row, data) for tree in knn_trees]
            final_prediction = max(set(predictions), key=predictions.count)
            if final_prediction == data["diagnosis"].iloc[row]:
                right_predictions += 1
            elif final_prediction == "M":
                false_positive += 1
            else:
                false_negative += 1
        num_of_samples = len(data.index)
        acc = right_predictions / num_of_samples
        loss = (0.1 * false_positive + false_negative) / num_of_samples
        return acc, loss


def find_hyperparameters_for_forest(X, y, splits=5, n_values=None, k_values=None, p_values=None):
    # assign default test values
    if n_values is None:
        n_values = [x for x in range(10, 25)]
    if k_values is None:
        k_values = [x for x in range(7, 17, 2)]
    if p_values is None:
        p_values = [0.1, 0.2, 0.3, 0.4]

    best_hypers = []
    best_acc = float('-inf')
    print(f"---------------- starting a test to find hyper params -------------------")
    print(f"N values={n_values}")
    print(f"k values={k_values}")
    print(f"p values={p_values}")
    for n in n_values:
        for k in k_values:
            if k >= n:
                break
            for p in p_values:
                accuracies = []
                classifier = KNNForestClassifier(N=n, k=k, p=p)
                kf = KFold(n_splits=splits, random_state=307965806, shuffle=True)
                for train_index, test_index in kf.split(X):
                    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
                    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

                    classifier.fit(X_train, y_train)
                    accuracy, _ = classifier.predict(X_test, y_test)

                    accuracies.append(accuracy)
                avg = sum(accuracies) / len(accuracies)
                print(f"N={n}, k={k}, p={p}: accuracy={avg}")
                if avg == best_acc:
                    value = (n, k, p)
                    best_hypers.append(value)
                if avg > best_acc:
                    best_hypers = []
                    value = (n, k, p)
                    best_hypers.append(value)
                    best_acc = avg
        print(f"Best values for N={n} are {best_hypers} with accuracy={best_acc}")
    print(f"------------ test results -------------")
    print(f"The best hyper params are {best_hypers} with accuracy of {best_acc}")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('-v', '-verbose', dest="verbose", action='store_true', help="Show more information")
    parser.add_argument('-find_hyper', dest="find_hyper",
                        action='store_true', help="Running kfold test to find hyper params")

    args = parser.parse_args()

    # retrieving the data from the csv files
    train_x, train_y = csv2xy("train.csv")
    test_x, test_y = csv2xy("test.csv")
    # creating a classifier instance
    classifier = KNNForestClassifier()
    # fitting the classifier
    classifier.fit(train_x, train_y)
    # predicting on the test data set
    accuracy, loss = classifier.predict(test_x, test_y)

    if args.verbose:
        print(f"The accuracy for KNNForest={accuracy}")
        print(f"The loss for KNNForest={loss}")
    else:
        print(accuracy)

    # Todo - to run a kfold experiment to find hyper params run ImprovedKNNForest.py with -find_hyper flag.
    if args.find_hyper:
        find_hyperparameters_for_forest(train_x, train_y, splits=5)
