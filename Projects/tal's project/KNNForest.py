from utils import AbstractClassifier, csv2xy
from ID3 import ID3Node
import random
from sklearn.model_selection import train_test_split
import argparse
import numpy as np


class KNNForestClassifier(AbstractClassifier):
    def __init__(self, N=15, k=9):
        super().__init__()
        self.prob_range = 0.3, 0.7
        self.N = N
        self.k = k
        self.forest = []
        self.centroids = []

    def fit(self, x, y):
        # clear those lists every time we fit!
        self.forest = []
        self.centroids = []
        n = len(x.index)
        for i in range(self.N):
            start, stop = self.prob_range
            portion = random.uniform(start, stop)
            test_size = 1 - ((n * portion) / n)
            X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=test_size)
            train_data = X_train.copy()
            train_data["diagnosis"] = y_train
            self.forest.append(ID3Node(data=train_data))
            centroid = train_data.copy()
            centroid = centroid.mean(axis=0)
            self.centroids.append(centroid)

    def predict(self, x, y):
        test_centroid = x.copy()
        test_centroid = test_centroid.mean(axis=0)
        distances = []
        for tree, centroid in zip(self.forest, self.centroids):
            dist = (np.linalg.norm(centroid - test_centroid))
            value = tree, dist
            distances.append(value)
        distances.sort(key=lambda val: val[1])
        distances = distances[:self.k]
        data = x.copy()
        data["diagnosis"] = y
        right_predictions, false_positive, false_negative = 0, 0, 0
        num_of_samples = len(data.index)
        for row in range(len(data.index)):
            predictions = []
            for tree, _ in distances:
                prediction = self.walk_the_tree(tree, row, data)
                predictions.append(prediction)
            most_common = max(set(predictions), key=predictions.count)
            if most_common == data["diagnosis"].iloc[row]:
                right_predictions += 1
            else:
                if most_common == "M":
                    false_positive += 1
                else:
                    false_negative += 1
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
