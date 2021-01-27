from ID3 import build_id3_tree, ID3Classifier, get_data_from_csv
from CostSensitiveID3 import tour_tree
from KNNForest import slice_data, get_centroid, distance_between_vectors, KNNForestClassifier
from sklearn.model_selection import KFold
import random
import numpy as np
from math import pi


def check_max(list_values):
    """
    Auxiliary function: checks what is maximum between M and B distance values
    use function pi^(-sum of dist) as a function decreases
    :param list_values: list of tuples (prediction=M/B, distance value)
    :return: M OR B
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
    Classifier for improved KNN Forest
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
                        ('smoothness_mean', 0.0), ('compactness_mean', 0.0), ('concavity_mean', 0.0),
                        ('concave points_mean', 0.0), ('symmetry_mean', 0.0), ('fractal_dimension_mean', 0.0),
                        ('radius_se', 0.0), ('texture_se', 0.0), ('perimeter_se', 0.0), ('area_se', 0.0),
                        ('smoothness_se', 0.0), ('compactness_se', 0.002941176470588225),
                        ('concavity_se', 0.011679454390451838), ('concave points_se', 0.002941176470588247),
                        ('symmetry_se', 0.0), ('fractal_dimension_se', 0.0), ('radius_worst', 0.0),
                        ('texture_worst', 0.0058397271952258965), ('perimeter_worst', 0.0203324808184143),
                        ('area_worst', 0.0), ('smoothness_worst', 0.0), ('compactness_worst', 0.0),
                        ('concavity_worst', 0.002941176470588225), ('concave points_worst', 0.008695652173913038),
                        ('symmetry_worst', 0.00869565217391306), ('fractal_dimension_worst', 0.002941176470588247)]

    def fit(self, x, y):
        """
        This function fits the classifier and creates a decision tree
        also in this function doing normalization of MinMax in the data
        :param x: data without diagnosis
        :type x: Dataframe
        :param y: diagnosis
        :type y: Dataframe
        """
        # MinMax normalization
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
        for i in range(self.N):
            fraction = random.uniform(self.first, self.last)
            sliced_x, sliced_y = slice_data(normalize_x, y, fraction)
            sliced_x["diagnosis"] = sliced_y
            tree_node = build_id3_tree(sliced_x)
            self.forest.append(tree_node)
            centroid = get_centroid(sliced_x)
            self.centroids.append(centroid)

    def predict(self, x, y):
        """
        predicts new samples with the decision tree made by fit
        means using the values that we normalized the fit data
        :param x: isolated data without diagnosis, DataFrame
        :param y: diagnosis
        :return: accuracy value between [0,1], float number
        """
        # MinMax - normalization
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
            list_values = []
            list_dist = []
            for i_tree, dist in all_dist:
                pred_to_add = tour_tree(i_tree, cur_row, val_data)
                val_to_add = (pred_to_add, dist)
                list_values.append(val_to_add)
                list_dist.append(val_to_add[1])
            max_val = check_max(list_values)
            val_data_r = val_data["diagnosis"].iloc[cur_row]
            if max_val == val_data_r:
                correct_predict += 1

        accuracy = correct_predict / len(val_data.index)
        return accuracy

    def calculate_significance_features(self, x, y, splits=5):
        """
        This function calculates the values of weights to evaluate each feature
        :param x: isolated data without diagnosis, DataFrame
        :param y: diagnosis
        :param splits: number of splits
        :return: The weights
        """
        weights_list = []
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
                _, acc_first = classifier.predict(test_x, test_y)
                train_x_data = train_x.copy()
                train_x_data[f] = np.random.permutation(train_x_data[f])
                data_permute = train_x_data
                classifier.fit(data_permute, train_y)
                # calculate new accuracy
                _, second_acc = classifier.predict(test_x, test_y)
                error = abs(acc_first - second_acc)
                mistakes.append(error)
            avg_m = sum(mistakes) / len(mistakes)
            w = f, avg_m
            weights_list.append(w)
        return weights_list

    def weighted_distance(self, centroid1, centroid2):
        """
        function to calculate euclidean distance with weights
        :param centroid1: first vector
        :param centroid2: second vector
        :return: euclidean distance with weights between the two vectors
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


if __name__ == "__main__":
    # in the main part, first we get tha data from csv files, after that we create a ImprovedKNNForestClassifier,
    # then fit the classifier. in the end we predict in test data set and printing the accuracy of this classifier
    train_x, train_y = get_data_from_csv("train.csv")
    test_x, test_y = get_data_from_csv("test.csv")
    classifier = ImprovedKNNForestClassifier()
    classifier.fit(train_x, train_y)
    res_accuracy = classifier.predict(test_x, test_y)
    print(res_accuracy)

    # TODO to get values of weights pleas uncomment the next 3 lines
    # list = []
    # list = classifier.calculate_significance_features(train_x, train_y)
    # print(list)
