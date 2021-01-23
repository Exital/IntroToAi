from utils import csv2xy, AbstractClassifier, beep, WEIGHTS
import argparse
import matplotlib.pyplot as plt
from KNNForest import KNNForestClassifier
from ID3 import ID3Node, ID3Classifier
from sklearn.model_selection import train_test_split, KFold
import random
import numpy as np
from math import e


def remove_feature(x, features: list):
    """
    This function removes a column of the dataset
    :param x: the dataset
    :type x: dataframe
    :param features: list of features to remove
    :type features: list
    :return: a copy of the dataframe without the features.
    :rtype: dataframe
    """
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


def get_weight(feature):
    """
    Gets the weight of a feature from the global weights that I computed earlier
    :param feature: the feature name
    :type feature: str
    :return: a weight
    :rtype: float
    """
    for f, w in WEIGHTS:
        if f == feature:
            return w
        else:
            return 0


def distance(v1, v2, weighted=False):
    """
    computes euclidean distance, can be weighted or not.
    :param v1: series 1
    :type v1: series
    :param v2: series 2
    :type v2: series
    :param weighted: True for weighted
    :type weighted: bool
    :return: distance between the series
    :rtype: float
    """
    if not weighted:
        return np.linalg.norm(v1 - v2)
    else:
        v1_weighted = v1.copy()
        v2_weighted = v2.copy()
        for index, value in v1_weighted.items():
            weight = get_weight(index)
            v1_weighted.loc[index] = value * weight
        for index, value in v2_weighted.items():
            weight = get_weight(index)
            v2_weighted.loc[index] = value * weight
        return np.linalg.norm(v1_weighted - v2_weighted)


class ImprovedKNNForestClassifier(AbstractClassifier):
    """
    This is a classifier that uses the regular KNNForestClassifier but normalizes the data before it uses it.
    """
    def __init__(self, N=25, k=11, p=None):
        self.scaling_consts = []
        self.test_size = 0.33
        self.bad_features = []
        self.prob_range = 0.3, 0.32
        self.N = N
        self.k = k
        self.p = p
        self.forest = []
        self.centroids = []

    def fit_scaling(self, x):
        """
        The scaling that has to be done with fitting + saves the values for the prediction fitting.
        :param x: the data
        :type x: dataframe
        :return: a normalized dataframe
        :rtype: dataframe
        """
        self.scaling_consts = []
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
        scaled_x = self.fit_scaling(x)
        scaled_x = scaled_x
        self.forest = []
        self.centroids = []
        n = len(scaled_x.index)
        for i in range(self.N):
            if self.p is None:
                start, stop = self.prob_range
                portion = random.uniform(start, stop)
            else:
                portion = self.p
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
            distances = []
            sample = data.iloc[[row]].copy()
            sample = remove_feature(sample, ["diagnosis"])
            sample = sample.mean(axis=0)
            for tree, centroid in zip(self.forest, self.centroids):
                dist = distance(centroid, sample, weighted=True)
                value = tree, dist
                distances.append(value)
            distances.sort(key=lambda val: val[1])
            distances = distances[:self.k]
            sick, healthy = 0, 0
            for tree, dist in distances:
                prediction = self.walk_the_tree(tree, row, data)
                if prediction == "M":
                    sick += e ** -dist
                else:
                    healthy += e ** -dist
            prediction = "M" if sick >= healthy else "B"
            if prediction == data["diagnosis"].iloc[row]:
                right_predictions += 1
            else:
                if prediction == "M":
                    false_positive += 1
                else:
                    false_negative += 1
        acc = right_predictions / num_of_samples
        loss = (0.1 * false_positive + false_negative) / num_of_samples
        return acc, loss


def kfold_experiment(X, y, iterations=5, verbose=False):
    accuracy = []
    improved_accuracy = []
    classifier = KNNForestClassifier()
    improved_classifier = ImprovedKNNForestClassifier()
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


def iteration_experiment(x_train, y_train, x_test, y_test, iterations=25, verbose=False):
    if verbose:
        print(f"----------- Starting experiment with {iterations} iterations -----------")
    classifier = KNNForestClassifier()
    improved_classifier = ImprovedKNNForestClassifier()
    accuracy = []
    improved_accuracy = []
    for iter in range(iterations):
        if verbose:
            print(f"----------- Round {iter + 1} -----------")
        classifier.fit(x_train, y_train)
        acc, _ = classifier.predict(x_test, y_test)
        if verbose:
            print(f"Accuracy for KNN={acc}")
        improved_classifier.fit(x_train, y_train)
        improved_acc, _ = improved_classifier.predict(x_test, y_test)
        if verbose:
            print(f"Accuracy for ImprovedKNN={improved_acc}")
        accuracy.append(acc)
        improved_accuracy.append(improved_acc)
    regular = sum(accuracy) / len(accuracy)
    improved = sum(improved_accuracy) / len(improved_accuracy)
    improvement = improved - regular
    if verbose:
        iterations = [i for i in range(iterations)]
        plt.xlabel("Iteration number")
        plt.ylabel("Accuracy of that iteration")
        plt.plot(iterations, accuracy, label="KNNForest")
        plt.plot(iterations, improved_accuracy, label="ImprovedKNNForest")
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
        plt.show()
        print("------------------- Final Results ------------------")
        print(f"The average accuracy of KNNForest is {regular}")
        print(f"The average accuracy of ImprovedKNNForest is {improved}")
        print(f"The new improved KNN algorithm is {(improved - regular) * 100}% better")
    return improvement, improved, regular


def permute_feature(data, feature):
    """
    That function permutes the values on the column of the feature in order to check its importance to accuracy.
    :param data: the dataset
    :type data: dataframe
    :param feature: the feature to permute
    :type feature: str
    :return: the permuted dataset
    :rtype: dataframe
    """
    permutation = data.copy()
    permutation[feature] = np.random.permutation(permutation[feature])
    return permutation


def compute_feature_importance(X, y, splits=5, verbose=False):
    """
    This function will use kfold cross validation in order to compute weights for the features.
    :param X: dataset
    :type X: dataframe
    :param y: labels
    :type y: datafram
    :param splits: number of splits for the kfold
    :type splits: int
    :param verbose: printing outputs
    :type verbose: bool
    :return: the weight list
    :rtype: list[Tuple]
    """
    weights = []
    features = X.keys().tolist()
    print(f"---------------- starting features weights computation ---------------")
    for feature in features:
        errors = []
        kf = KFold(n_splits=splits, random_state=307965806, shuffle=True)
        for train_index, test_index in kf.split(X):
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]

            classifier = ID3Classifier()
            classifier.fit(X_train, y_train)
            original_acc, _ = classifier.predict(X_test, y_test)

            permuted_data = permute_feature(X_train, feature)
            classifier.fit(permuted_data, y_train)
            new_acc, _ = classifier.predict(X_test, y_test)

            error = abs(original_acc - new_acc)
            errors.append(error)
        avg_error = sum(errors) / len(errors)
        weight = feature, avg_error
        weights.append(weight)
    sorted_weights = sorted(weights, key=lambda x: x[1], reverse=True)
    max_weight = sorted_weights[0]
    max_error = max_weight[1]
    normalized_weights = [(feature, error / max_error) for feature, error in weights]
    if verbose:
        print("------------ Feature's weights -------------")
        print(normalized_weights)
    return normalized_weights


def find_hyperparameters_for_forest(X, y, splits=5, n_values=None, k_values=None, p_values=None):
    # assign default test values
    if n_values is None:
        n_values = [x for x in range(3, 40)]
    if k_values is None:
        k_values = [x for x in range(1,30)]
    if p_values is None:
        p_values = [0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8]

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
                classifier = ImprovedKNNForestClassifier(N=n, k=k, p=p)
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
    print(f"------------ test results -------------")
    print(f"The best hyper params are {best_hypers} with accuracy of {best_acc}")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('-v', '-verbose', dest="verbose", action='store_true', help="Show more information")
    parser.add_argument('-feature_weights', dest="feature_weights",
                        action='store_true', help="Running a test to determine feature's weights")
    parser.add_argument('-kfold_experiment', dest="kfold_experiment",
                        action='store_true', help="Running a kfold test to show improvement")
    parser.add_argument('-iteration_experiment', dest="iteration_experiment",
                        action='store_true', help="Running an iteration test to show improvement")
    parser.add_argument('-find_hyper', dest="find_hyper",
                        action='store_true', help="Running kfold test to find hyper params")

    args = parser.parse_args()

    # retrieving the data from the csv files
    train_x, train_y = csv2xy("train.csv")
    test_x, test_y = csv2xy("test.csv")

    # Todo - in order to compute feature weights run ImprovedKNNForest.py with -feature_weights flag.
    if args.feature_weights:
        weights = compute_feature_importance(train_x, train_y, verbose=True)

    # Todo - in order to run a kfold experiment to see improvement run ImprovedKNNForest.py with -kfold_experiment flag.
    if args.kfold_experiment:
        kfold_experiment(train_x, train_y, verbose=True, iterations=5)

    # Todo - to run an iteration experiment to see improvement run ImprovedKNNForest.py with -iteration_experiment flag.
    if args.iteration_experiment:
        iteration_experiment(train_x, train_y, test_x, test_y, verbose=True, iterations=5)

    # Todo - to run a kfold experiment to find hyper params run ImprovedKNNForest.py with -find_hyper flag.
    if args.find_hyper:
        find_hyperparameters_for_forest(train_x, train_y, splits=5)
