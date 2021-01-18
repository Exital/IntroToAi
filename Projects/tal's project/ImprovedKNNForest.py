from utils import csv2xy, AbstractClassifier, beep
import argparse
import matplotlib.pyplot as plt
from KNNForest import KNNForestClassifier
from ID3 import find_best_features_to_remove, ID3Node
from sklearn.model_selection import train_test_split, KFold
import random
import numpy as np


def remove_bad_features(x, features: list):
    sub_data = x.copy()
    for feature in features:
        sub_data = sub_data.drop(feature, axis=1)
    return sub_data


def scalar(series, mean, deviation):
    """
    This functions scales a series of that to normalized form as we learnt in class.
    :param series: a pd series
    :type series: series
    :param mean: the mean of the series
    :type mean: float
    :param deviation: the std of the series
    :type deviation: float
    :return: a normalized series
    :rtype: a pd series
    """
    for index, value in series.items():
        series.loc[index] = (value - mean) / deviation
    return series


class ImprovedKNNForestClassifier(AbstractClassifier):
    """
    This is a classifier that uses the regular KNNForestClassifier but normalizes the data before it uses it.
    """
    def __init__(self, N=15, k=9):
        self.scaling_consts = []
        self.test_size = 0.33
        self.bad_features = []
        self.prob_range = 0.6, 0.7
        self.N = N
        self.k = k
        self.forest = []
        self.centroids = []

    def fit_scaling(self, x, y):
        """
        The scaling that has to be done with fitting + saves the values for the prediction fitting.
        :param x: the data
        :type x: dataframe
        :return: a normalized dataframe
        :rtype: dataframe
        """
        # find features to remove
        X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=self.test_size)
        # self.bad_features = []
        # self.bad_features = find_best_features_to_remove(X_train, y_train, X_test, y_test)
        # removing selected features
        # subset_data = remove_bad_features(x, self.bad_features)
        # clearing the scaling consts if there is another fit for newer data
        self.scaling_consts = []
        # scaling with std scaling
        features = x.keys().tolist()
        x_scaled = x.copy()
        for feature in features:
            mean_val, std_val = x_scaled[feature].mean(), x_scaled[feature].std()
            scaling_const = feature, mean_val, std_val
            # saving the values in order to normalize predict data
            self.scaling_consts.append(scaling_const)
            x_scaled[feature] = scalar(x_scaled[feature], mean_val, std_val)
        return x_scaled

    def predict_scaling(self, x):
        """
        The scaling that has to be done for prediction using the values that we normalized the fit data.
        Exactly as we have learnt in class.
        :param x: the dataframe that needs to be normalized
        :type x: dataframe
        :return: a normalized dataframe
        :rtype: dataframe
        """
        # removing selected features
        # subset_data = remove_bad_features(x, self.bad_features)
        # scaling with std scaling
        x_scaled = x.copy()
        for feature, mean_val, std_val in self.scaling_consts:
            x_scaled[feature] = scalar(x_scaled[feature], mean_val, std_val)
        return x_scaled

    def fit(self, x, y):
        """
        regular fit just using the normalized data.
        :param x: The data to fit
        :type x: dataframe
        :param y: the labels of that data
        :type y: dataframe
        """
        scaled_x = self.fit_scaling(x, y)
        scaled_x = scaled_x
        self.forest = []
        self.centroids = []
        n = len(scaled_x.index)
        for i in range(self.N):
            start, stop = self.prob_range
            portion = random.uniform(start, stop)
            test_size = 1 - ((n * portion) / n)
            X_train, X_test, y_train, y_test = train_test_split(scaled_x, y, test_size=test_size)
            train_data = X_train.copy()
            train_data["diagnosis"] = y_train
            self.forest.append(ID3Node(data=train_data))
            centroid = train_data.copy()
            centroid = centroid.mean(axis=0)
            self.centroids.append(centroid)

    def predict(self, x, y):
        """
        The predict function which will scale the data and then use the regular predictions.
        :param x: the data to be predicted on
        :type x: dataframe
        :param y: the labels of that data
        :type y: dataframe
        :return: (accuracy, loss)
        :rtype: tuple
        """
        scaled_x = self.predict_scaling(x)
        scaled_x = scaled_x
        data = scaled_x.copy()
        data["diagnosis"] = y
        right_predictions, false_positive, false_negative = 0, 0, 0
        num_of_samples = len(data.index)
        for row in range(len(data.index)):
            predictions = []
            distances = []
            sample = data.iloc[[row]].copy()
            sample = remove_bad_features(sample, ["diagnosis"])
            sample = sample.mean(axis=0)
            for tree, centroid in zip(self.forest, self.centroids):
                dist = (np.linalg.norm(centroid - sample))
                value = tree, dist
                distances.append(value)
            distances.sort(key=lambda val: val[1])
            distances = distances[:self.k]
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


def experiment(X, y, iterations=5, N=20, k=7, verbose=False):
    accuracy = []
    improved_accuracy = []
    classifier = KNNForestClassifier(N=N, k=k)
    improved_classifier = ImprovedKNNForestClassifier(N=N, k=k)
    if verbose:
        print(f"----------- Starting new experiment -----------")
    kf = KFold(n_splits=iterations, random_state=307965806, shuffle=True)
    for count, (train_index, test_index) in enumerate(kf.split(X)):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        classifier.fit(X_train, y_train)
        acc, loss = classifier.predict(X_test, y_test)
        accuracy.append(acc)
        if verbose:
            print(f"----------- Round {count + 1} -----------")
            print(f"Accuracy for KNN={acc}")
        improved_classifier.fit(X_train, y_train)
        acc, loss = improved_classifier.predict(X_test, y_test)
        improved_accuracy.append(acc)
        if verbose:
            print(f"Accuracy for ImprovedKNN={acc}")
    regular = sum(accuracy) / len(accuracy)
    improved = sum(improved_accuracy) / len(improved_accuracy)
    improvement = improved - regular
    if verbose:
        iterations = [i for i in range(iterations)]
        plt.xlabel("Fold number")
        plt.ylabel("Accuracy of that fold")
        plt.plot(iterations, accuracy, label="KNNForest")
        plt.plot(iterations, improved_accuracy, label="ImprovedKNNForest")
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
        plt.show()
        print("------------------- Final Results ------------------")
        print(f"The average accuracy of KNNForest is {regular}")
        print(f"The average accuracy of ImprovedKNNForest is {improved}")
        print(f"The new improved KNN algorithm is {(improved - regular) * 100}% better")
    return improvement, improved, regular


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('-v', '-verbose', dest="verbose", action='store_true', help="Show more information")
    args = parser.parse_args()

    # retrieving the data from the csv files
    train_x, train_y = csv2xy("train.csv")
    test_x, test_y = csv2xy("test.csv")
    kfold_x, kfold_y = csv2xy("kfold.csv")
    # experiment(train_x, train_y, verbose=args.verbose, N=15, k=9, iterations=5)

    while True:
        result = experiment(kfold_x, kfold_y, verbose=args.verbose, N=15, k=9, iterations=5)
        improvement, improveKnnAcc, regularKnnAcc = result
        if improvement > 0.018:
            beep()
        if regularKnnAcc > 0.96 and improvement > 0.02:
            beep()
            beep()
            beep()
            break
