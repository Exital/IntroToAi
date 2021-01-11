from utils import AbstractClassifier, csv2xy
from ID3 import ID3Node
import random
from sklearn.model_selection import train_test_split
import argparse


class KNNForestClassifier(AbstractClassifier):
    def __init__(self, N, k):
        super().__init__()
        self.prob_range = 0.3, 0.7
        self.N = N
        self.k = k
        self.forest = []
        self.centroids = []

    def fit(self, x, y):
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


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('-v', '-verbose', dest="verbose", action='store_true', help="Show more information")
    args = parser.parse_args()

    # retrieving the data from the csv files
    train_x, train_y = csv2xy("train.csv")
    test_x, test_y = csv2xy("test.csv")
    # creating a classifier instance
    classifier = KNNForestClassifier(N=3, k=5)
    # fitting the classifier
    classifier.fit(train_x, train_y)
    # predicting on the test data set
    # accuracy, loss = classifier.predict(test_x, test_y)
    # if args.verbose:
    #     print(f"loss with cost optimizing={loss}")
    # else:
    #     print(loss)
