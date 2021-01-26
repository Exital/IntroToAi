from ID3 import build_id3_tree, ID3Classifier, get_data_from_csv
from CostSensitiveID3 import tour_tree
from KNNForest import slice_data, get_centroid, distance_between_vectors, KNNForestClassifier
from sklearn.model_selection import KFold
import random
import matplotlib.pyplot as plt
import numpy as np
from math import pi


def check_max(list_values):
    """

    :param list_values:
    :return:
    """
    sum_M = 0
    sum_B = 0
    n = len(list_values)
    for i in range(n):
        temp = list_values[i]
        if temp[0] == "M":
            sum_M += temp[1]
        else:
            sum_B += temp[1]
    funM_pi = pi ** -sum_M
    funB_pi = pi ** -sum_B
    return "M" if funM_pi < funB_pi else "B"



class ImprovedKNNForestClassifier:
    """

    """
    def __init__(self, N=25, k=11):
        self.forest = []
        self.centroids = []
        self.N = N
        self.k = k
        self.first = 0.3
        self.last = 0.31
        self.size_test = 0.33
        self.normalization_values = []
        self.weights = [('radius_mean', 0.0), ('texture_mean', 0.0), ('perimeter_mean', 0.0), ('area_mean', 0.0),
           ('smoothness_mean', 0.0), ('compactness_mean', 0.0), ('concavity_mean', 0.0026470588235294108),
           ('concave points_mean', 0.0), ('symmetry_mean', 0.0), ('fractal_dimension_mean', 0.0), ('radius_se', 0.0),
           ('texture_se', 0.0), ('perimeter_se', 0.0), ('area_se', 0.0), ('smoothness_se', 0.0),
           ('compactness_se', 0.0), ('concavity_se', 0.005797101449275362),
           ('concave points_se', 0.0002941176470588239), ('symmetry_se', 0.0), ('fractal_dimension_se', 0.0),
           ('radius_worst', 0.0), ('texture_worst', 0.0005839727195225922), ('perimeter_worst', 0.01130861040068201),
           ('area_worst', 0.0), ('smoothness_worst', 0.0), ('compactness_worst', 0.0),
           ('concavity_worst', 0.00823529411764706), ('concave points_worst', 0.013989769820971864),
           ('symmetry_worst', 0.0034782608695652175), ('fractal_dimension_worst', 0.002941176470588233)]

    def fit(self, x, y):
        """

        :param x:
        :param y:
        :return:
        """
        # minmax normalization
        self.centroids = []
        self.forest = []
        self.normalization_values = []
        features = x.keys().tolist()
        normalize_x = x.copy()
        for feature in features:
            min_val = min(set(normalize_x[feature]))
            max_val = max(set(normalize_x[feature]))
            data_col = normalize_x[feature]
            predict_value = feature, min_val, max_val

            self.normalization_values.append(predict_value)
            for index, val in data_col.items():
                data_col.loc[index] = (val - min_val) / (max_val - min_val)
            normalize_x[feature] = data_col
        # Training group size
        for i in range(self.N):  # to check if self.N or param in function
            fraction = random.uniform(self.first, self.last)
            sliced_x, sliced_y = slice_data(normalize_x, y, fraction)
            sliced_x["diagnosis"] = sliced_y
            tree_node = build_id3_tree(sliced_x)
            self.forest.append(tree_node)
            centroid = get_centroid(sliced_x)
            self.centroids.append(centroid)

    def predict(self, x, y):
        # minmax - normalization
        normalize_x = x.copy()
        for feature, min_val, max_val in self.normalization_values:
            data_list = normalize_x[feature]
            for index, val in data_list.items():
                data_list.loc[index] = (val - min_val) / (max_val - min_val)
            normalize_x[feature] = data_list
        # using the normalized data
        val_data = normalize_x
        val_data["diagnosis"] = y
        correct_predict = 0
        # check each row in data
        for cur_row in range(len(val_data.index)):
            all_dist = []
            exam = val_data.iloc[[cur_row]].copy()
            copy_exam = exam.copy()
            copy_exam = copy_exam.drop("diagnosis", axis=1)
            update_data = copy_exam
            update_data = update_data.mean(axis=0)
            for i_centroid, i_tree in zip(self.centroids, self.forest):
                val_destination = self.weighted_distance(i_centroid, update_data)
                val = i_tree, val_destination
                all_dist.append(val)
            all_dist.sort(key=lambda x: x[1])
            list_al_dist = []
            for i in range(self.k):
                list_al_dist.append(all_dist[i])
            m_dist = 0
            b_dist = 0
            list_values = []
            for i_tree, dist in all_dist:
                pred_to_add = tour_tree(i_tree, cur_row, val_data)
                val_to_add = (pred_to_add, dist)
                list_values.append(val_to_add)
                #
                # if pred_to_add == "M":
                #     m_dist += 1 / pi ** -dist
                # else:
                #     b_dist += 1 / pi ** -dist
            # max_val = "M" if m_dist > b_dist else "B"
            max_val = check_max(list_values)
            val_data_r = val_data["diagnosis"].iloc[cur_row]
            if max_val == val_data_r:
                correct_predict += 1

        accuracy = correct_predict / len(val_data.index)
        return accuracy

    def calculate_significance_features(self, x, y, splits=5):
        """

        :param y:
        :param splits:
        :return:
        """
        weights = []
        features = x.keys().tolist()
        for f in features:
            kf = KFold(n_splits=splits, random_state=204512396, shuffle=True)
            mistakes = []
            for i_train, test_index in kf.split(x):
                train_y, test_y = y.iloc[i_train], y.iloc[test_index]
                train_x, test_x = x.iloc[i_train], x.iloc[test_index]
                classifier = ID3Classifier()
                classifier.fit(train_x, train_y)
                # calculate original accuracy
                acc_first, _ = classifier.predict(test_x, test_y)
                train_x_data = train_x.copy()
                train_x_data[f] = np.random.permutation(train_x_data[f])
                data_permute = train_x_data
                classifier.fit(data_permute, train_y)
                # calculate new accuracy
                second_acc, _ = classifier.predict(test_x, test_y)

                error = abs(acc_first - second_acc)
                mistakes.append(error)
            avg_m = sum(mistakes) / len(mistakes)
            w = f, avg_m
            weights.append(w)
        return weights

    def weighted_distance(self, centroid1, centroid2):
        """
        function to calculate euclidean distance with weights
        :param centroid1:
        :param centroid2:
        :return:
        """
        centroid1 = centroid1.copy()
        centroid2 = centroid2.copy()
        for feature_key, val in centroid1.items():
            weight_to_cal = 1
            for feature_val, weight_val in self.weights:
                if feature_val == feature_key:
                    weight_to_cal = weight_val
            centroid1.loc[feature_key] = weight_to_cal * val
        for feature_key, val in centroid2.items():
            weight_to_cal = 1
            for feature_val, weight_val in self.weights:
                if feature_val == feature_key:
                    weight_to_cal = weight_val
            centroid2.loc[feature_key] = weight_to_cal * val
        distance = distance_between_vectors(centroid1, centroid2)
        return distance


def experiment(X, y, iterations=5, N=20, k=7, verbose=True):
    accuracy = []
    improved_accuracy = []
    classifier = KNNForestClassifier(N=15, k=7)
    improved_classifier = ImprovedKNNForestClassifier(N=25, k=11)
    if verbose:
        print(f"----------- Starting new experiment -----------")
    # kf = KFold(n_splits=iterations, random_state=204512396, shuffle=True)
    # for count, (train_index, test_index) in enumerate(kf.split(X)):
    #     X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    #     y_train, y_test = y.iloc[train_index], y.iloc[test_index]
    X_train, y_train = get_data_from_csv("train.csv")
    X_test, y_test = get_data_from_csv("test.csv")
    for i in range(5):
        classifier.fit(X_train, y_train)
        acc = classifier.predict(X_test, y_test)
        accuracy.append(acc)
        if verbose:
            print(f"----------- Round {i + 1} -----------")
            print(f"Accuracy for KNN={acc}")
        improved_classifier.fit(X_train, y_train)
        acc = improved_classifier.predict(X_test, y_test)
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

    train_x, train_y = get_data_from_csv("train.csv")
    test_x, test_y = get_data_from_csv("test.csv")
    kfold_x, kfold_y = get_data_from_csv("kfold.csv")
    experiment(train_x, train_y, verbose=True, N=15, k=9, iterations=5)

    # empty