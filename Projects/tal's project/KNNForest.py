from utils import AbstractClassifier, csv2xy
from ID3 import ID3Node
import random
from sklearn.model_selection import train_test_split
import argparse
import numpy as np


def get_centroid(data):
    centroid = data.copy()
    centroid = centroid.mean(axis=0)
    return centroid


def distance(v1, v2):
    return np.linalg.norm(v1 - v2)


class KNNForestClassifier(AbstractClassifier):
    def __init__(self, N=25, k=11):
        super().__init__()
        self.prob_range = 0.3, 0.7
        self.N = N
        self.k = k
        self.forest = []
        self.centroids = []

    def split_data(self, x, y):
        data_splits = []
        test_sizes = [random.uniform(self.prob_range[0], self.prob_range[1]) for i in range(self.N)]
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


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('-v', '-verbose', dest="verbose", action='store_true', help="Show more information")
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
