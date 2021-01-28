from utils import csv2xy
import argparse
import matplotlib.pyplot as plt
from KNNForest import KNNForestClassifier, get_centroid
from ID3 import ID3Classifier
from sklearn.model_selection import KFold
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


class ImprovedKNNForestClassifier(KNNForestClassifier):
    """
    This is a classifier that uses the regular KNNForestClassifier but normalizes the data before it uses it.
    """
    def __init__(self, N=25, k=11, p=0.3):
        super().__init__(N, k, p)
        self.scaling_consts = []
        # weights for the features were explored by function "compute_feature_importance" from ImprovedKNNForest.py
        self.weights = {'radius_mean': 0.0, 'texture_mean': 0.10117302052785897, 'perimeter_mean': 0.0,
                        'area_mean': 0.0, 'smoothness_mean': 0.0, 'compactness_mean': 0.0, 'concavity_mean': 0.0,
                        'concave points_mean': 0.0, 'symmetry_mean': 0.0, 'fractal_dimension_mean': 0.0,
                        'radius_se': 0.0, 'texture_se': 0.0, 'perimeter_se': 0.0, 'area_se': 0.0, 'smoothness_se': 0.0,
                        'compactness_se': 0.0, 'concavity_se': 0.0, 'concave points_se': 0.0, 'symmetry_se': 0.0,
                        'fractal_dimension_se': 0.0, 'radius_worst': 0.0, 'texture_worst': 1.0,
                        'perimeter_worst': 0.6041055718475081, 'area_worst': 0.1994134897360707,
                        'smoothness_worst': 0.3020527859237537, 'compactness_worst': 0.0,
                        'concavity_worst': 0.1994134897360707, 'concave points_worst': 0.9032258064516138,
                        'symmetry_worst': 0.40175953079178944, 'fractal_dimension_worst': 0.1994134897360707}

    def get_weight(self, feature):
        if feature in self.weights:
            return self.weights[feature]
        else:
            return 1

    def distance(self, v1, v2, weighted=False):
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
                weight = self.get_weight(index)
                v1_weighted.loc[index] = value * weight
            for index, value in v2_weighted.items():
                weight = self.get_weight(index)
                v2_weighted.loc[index] = value * weight
            return np.linalg.norm(v1_weighted - v2_weighted)

    def fit_scaling(self, x):
        """
        The scaling that has to be done with fitting + saves the values for the prediction fitting.
        :param x: the data
        :type x: dataframe
        :return: a normalized dataframe
        :rtype: dataframe
        """
        features = x.keys().tolist()
        x_scaled = x.copy()
        # saving values for scaling
        self.scaling_consts = [(feature, x_scaled[feature].mean(), x_scaled[feature].std()) for feature in features]
        for feature, mean, std in self.scaling_consts:
            x_scaled[feature] = scalar(x_scaled[feature], mean, std)
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
        super().fit(scaled_x, y)

    def predict(self, x, y):
        """
        The predict function which will scale the data and then predict the test samples using distances and weights.
        :param x: the data to be predicted on
        :type x: dataframe
        :param y: the labels of that data
        :type y: dataframe
        :return: (accuracy, loss)
        :rtype: tuple
        """
        scaled_x = self.predict_scaling(x)
        data = scaled_x.copy()
        data["diagnosis"] = y
        right_predictions, false_positive, false_negative = 0, 0, 0
        for row in range(len(data.index)):
            row_centroid = get_centroid(data.iloc[[row]].copy())
            knn_trees = [(tree, self.distance(row_centroid, centroid, weighted=True)) for (tree, centroid) in self.forest]
            knn_trees.sort(key=lambda val: val[1])
            knn_trees = knn_trees[:self.k]
            predictions = [(self.walk_the_tree(tree, row, data), e ** -dist) for (tree, dist) in knn_trees]
            m_grade, b_grade = 0, 0
            for diag, grade in predictions:
                if diag == "M":
                    m_grade += grade
                else:
                    b_grade += grade
            prediction = "M" if m_grade > b_grade else "B"
            if prediction == data["diagnosis"].iloc[row]:
                right_predictions += 1
            else:
                if prediction == "M":
                    false_positive += 1
                else:
                    false_negative += 1
        num_of_samples = len(data.index)
        acc = right_predictions / num_of_samples
        loss = (0.1 * false_positive + false_negative) / num_of_samples
        return acc, loss


def kfold_experiment(X, y, splits=5):
    accuracy = []
    improved_accuracy = []
    classifier = KNNForestClassifier()
    improved_classifier = ImprovedKNNForestClassifier()
    print(f"----------- Starting a kfold experiment with {splits} splits -----------")
    kf = KFold(n_splits=splits, random_state=307965806, shuffle=True)
    for count, (train_index, test_index) in enumerate(kf.split(X)):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        classifier.fit(X_train, y_train)
        acc, loss = classifier.predict(X_test, y_test)
        accuracy.append(acc)
        print(f"----------- Round {count + 1} -----------")
        print(f"Accuracy for KNN={acc}")
        improved_classifier.fit(X_train, y_train)
        acc, loss = improved_classifier.predict(X_test, y_test)
        improved_accuracy.append(acc)
        print(f"Accuracy for ImprovedKNN={acc}")
    regular = sum(accuracy) / len(accuracy)
    improved = sum(improved_accuracy) / len(improved_accuracy)
    improvement = improved - regular
    iterations = [i for i in range(splits)]
    print("------------------- Final Results ------------------")
    print(f"The average accuracy of KNNForest is {regular}")
    print(f"The average accuracy of ImprovedKNNForest is {improved}")
    print(f"The new improved KNN algorithm is {(improved - regular) * 100}% better")
    plt.xlabel("Fold number")
    plt.ylabel("Accuracy of that fold")
    plt.plot(iterations, accuracy, label="KNNForest")
    plt.plot(iterations, improved_accuracy, label="ImprovedKNNForest")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    plt.show()
    return improvement, improved, regular


def iteration_experiment(x_train, y_train, x_test, y_test, iterations=25):
    print(f"----------- Starting experiment with {iterations} iterations -----------")
    classifier = KNNForestClassifier()
    improved_classifier = ImprovedKNNForestClassifier()
    accuracy = []
    improved_accuracy = []
    for iter in range(iterations):
        print(f"----------- Round {iter + 1} -----------")
        classifier.fit(x_train, y_train)
        acc, _ = classifier.predict(x_test, y_test)
        print(f"Accuracy for KNN={acc}")
        improved_classifier.fit(x_train, y_train)
        improved_acc, _ = improved_classifier.predict(x_test, y_test)
        print(f"Accuracy for ImprovedKNN={improved_acc}")
        accuracy.append(acc)
        improved_accuracy.append(improved_acc)
    regular = sum(accuracy) / len(accuracy)
    improved = sum(improved_accuracy) / len(improved_accuracy)
    improvement = improved - regular
    iterations = [i for i in range(iterations)]
    print("------------------- Final Results ------------------")
    print(f"The average accuracy of KNNForest is {regular}")
    print(f"The average accuracy of ImprovedKNNForest is {improved}")
    print(f"The new improved KNN algorithm is {(improved - regular) * 100}% better")
    plt.xlabel("Iteration number")
    plt.ylabel("Accuracy of that iteration")
    plt.plot(iterations, accuracy, label="KNNForest")
    plt.plot(iterations, improved_accuracy, label="ImprovedKNNForest")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    plt.show()
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


def compute_feature_importance(X, y, splits=5):
    """
    This function will use kfold cross validation in order to compute weights for the features.
    :param X: dataset
    :type X: dataframe
    :param y: labels
    :type y: datafram
    :param splits: number of splits for the kfold
    :type splits: int
    :return: the weight list
    :rtype: list[Tuple]
    """
    weights = []
    features = X.keys().tolist()
    print(f"---------------- starting features weights computation ---------------")
    for round, feature in enumerate(features, 1):
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
        print(f"Feature={feature}, weight={avg_error} [{round}/{len(features)}]")
    sorted_weights = sorted(weights, key=lambda x: x[1], reverse=True)
    max_weight = sorted_weights[0]
    max_error = max_weight[1]
    normalized_weights = dict([(feature, error / max_error) for feature, error in weights])
    print("------------ Feature's weights -------------")
    print(normalized_weights)
    return normalized_weights


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('-v', '-verbose', dest="verbose", action='store_true', help="Show more information")
    parser.add_argument('-feature_weights', dest="feature_weights",
                        action='store_true', help="Running a test to determine feature's weights")
    parser.add_argument('-kfold_experiment', dest="kfold_experiment",
                        action='store_true', help="Running a kfold test to show improvement")
    parser.add_argument('-iteration_experiment', dest="iteration_experiment",
                        action='store_true', help="Running an iteration test to show improvement")

    args = parser.parse_args()

    # retrieving the data from the csv files
    train_x, train_y = csv2xy("train.csv")
    test_x, test_y = csv2xy("test.csv")
    # creating a classifier instance
    classifier = ImprovedKNNForestClassifier()
    # fitting the classifier
    classifier.fit(train_x, train_y)
    # predicting on the test data set
    accuracy, loss = classifier.predict(test_x, test_y)

    if args.verbose:
        print(f"The accuracy for ImprovedKNNForest={accuracy}")
        print(f"The loss for ImprovedKNNForest={loss}")
    else:
        print(accuracy)

    # Todo - in order to compute feature weights run ImprovedKNNForest.py with -feature_weights flag.
    if args.feature_weights:
        weights = compute_feature_importance(train_x, train_y)

    # Todo - in order to run a kfold experiment to see improvement run ImprovedKNNForest.py with -kfold_experiment flag.
    if args.kfold_experiment:
        kfold_experiment(train_x, train_y, splits=5)

    # Todo - to run an iteration experiment to see improvement run ImprovedKNNForest.py with -iteration_experiment flag.
    if args.iteration_experiment:
        iteration_experiment(train_x, train_y, test_x, test_y, iterations=5)
