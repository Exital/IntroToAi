from utils import csv2xy, log, AbstractClassifier
from ID3 import ID3Node, ID3Classifier
import argparse
import pandas as pd
from KNNForest import KNNForestClassifier
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
import random
import numpy as np


class ImprovedKNNForestClassifier:
    def __init__(self, N=20, k=7):
        self.knn_classifier = KNNForestClassifier(N=N, k=k)
        self.scaling_consts = []

    def scalar(self, series, mean, deviation):
        for index, value in series.items():
            series.loc[index] = (value - mean) / deviation
        return series

    def fit_scaling(self, x):
        self.scaling_consts = []
        features = x.keys().tolist()
        x_scaled = x.copy()
        for feature in features:
            mean_val, std_val = x_scaled[feature].mean(), x_scaled[feature].std()
            scaling_const = feature, mean_val, std_val
            self.scaling_consts.append(scaling_const)
            x_scaled[feature] = self.scalar(x_scaled[feature], mean_val, std_val)
        return x_scaled

    def predict_scaling(self, x):
        x_scaled = x.copy()
        for feature, mean_val, std_val in self.scaling_consts:
            x_scaled[feature] = self.scalar(x_scaled[feature], mean_val, std_val)
        return x_scaled

    def fit(self, x, y):
        scaled_x = self.fit_scaling(x)
        self.knn_classifier.fit(scaled_x, y)

    def predict(self, x, y):
        scaled_x = self.predict_scaling(x)
        return self.knn_classifier.predict(scaled_x, y)


def experiment(train_x, train_y, test_x, test_y, splits=5, N=20, k=7, verbose=False):
        accuracy = []
        improved_accuracy = []
        classifier = KNNForestClassifier(N=N, k=k)
        improved_classifier = ImprovedKNNForestClassifier(N=N, k=k)

        for i in range(splits):
            classifier.fit(train_x, train_y)
            acc, loss = classifier.predict(test_x, test_y)
            accuracy.append(acc)
            improved_classifier.fit(train_x, train_y)
            acc, loss = improved_classifier.predict(test_x, test_y)
            improved_accuracy.append(acc)
        if args.verbose:
            print(f"accuracy is {accuracy} with average of {sum(accuracy)/len(accuracy)}")
            print(f"improved accuracy is {improved_accuracy} with average of {sum(improved_accuracy)/len(improved_accuracy)}")

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('-v', '-verbose', dest="verbose", action='store_true', help="Show more information")
    args = parser.parse_args()

    # retrieving the data from the csv files
    train_x, train_y = csv2xy("train.csv")
    test_x, test_y = csv2xy("test.csv")
    # classifier = ImprovedKNNForestClassifier(N=20, k=7)
    # classifier.fit(train_x, train_y)
    # acc, loss = classifier.predict(test_x, test_y)
    # print(acc)

    experiment(train_x, train_y, test_x, test_y, verbose=args.verbose, N=20, k=7, splits=50)
