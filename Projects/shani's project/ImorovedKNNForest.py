from ID3 import build_id3_tree, ID3Classifier
from sklearn.model_selection import train_test_split
import random
from KNNForest import slice_data, get_centroid
from sklearn.model_selection import KFold
import random
import numpy as np

class ImprovedKNNForestClassifier:# to get something?
    """

    """
    def __init__(self, N=10, k=1):
        self.forest = []
        self.centroids = []
        self.N = N
        self.k = k
        self.first = 0.6
        self.last = 0.7
        self.size_test = 0.33
        self.normalization_values = []

    def normalization_rang(self, x, y):  # fit_scaling
        """

        :param x:
        :param y:
        :return:
        """

        train_x, test_x, train_y, test_y = train_test_split(x, y, size_test=self.size_test)
        self.normalization_values = []
        features = x.keys().tolist()
        normalize_x = x.copy()
        for feature in features:
            min_val = min(set(normalize_x), key=normalize_x[feature])
            max_val = max(set(normalize_x), key=normalize_x[feature])
            data_list = normalize_x[feature].items()
            predict_value = feature, min_val, max_val

            self.normalization_values.append(predict_value)
            for index, val in data_list:
                data_list.loc[index] = (val - min_val) / (max_val - min_val)
            normalize_x[feature] = data_list

        return normalize_x

    def predict_normalization(self, x):  # predict_scaling
        """

        :param x:
        :return:
        """
        normalize_x = x.copy()
        for feature, min_val, max_val in self.normalization_values:
            data_list = normalize_x[feature].items()
            for index, val in data_list:
                data_list.loc[index] = (val - min_val) / (max_val - min_val)
            normalize_x[feature] = data_list

        return normalize_x

    def fit(self, x, y):
        """

        :param x:
        :param y:
        :return:
        """
        normalize_x = self.normalization_values(x, y)
        # normalize_x = normalize_x

        self.centroids = []
        self.forest = []
        # Training group size
        n = len(normalize_x.index)
        for i in range(self.N):  # to check if self.N or param in function
            fraction = random.uniform(self.first, self.last)
            sliced_x, sliced_y = slice_data(x, y, fraction)
            sliced_x["diagnosis"] = sliced_y
            tree_node = build_id3_tree(sliced_x)
            self.forest.append(tree_node)
            centroid = get_centroid(sliced_x)
            self.centroids.append(centroid)

    def predict(self, x, y):  # TODO
        """

        :param x:
        :param y:
        :return:
        """

        centroid_check = x.copy()
        centroid_check = centroid_check.mean(axis=0)
        all_dist = []
        for i_centroid, i_tree in zip(self.centroids, self.forest):
            val_destination = distance_between_vectors(i_centroid, centroid_check)
            val = i_tree, val_destination
            all_dist.append(val)
        all_dist.sort(key=lambda x: x[1])
        list_al_dist =[]
        for i in range(self.k):
            list_al_dist.append(all_dist[i])
        val_data = x.copy()
        val_data["diagnosis"] = y
        correct_predict = 0
        # check each row in data
        for cur_row in range(len(val_data.index)):
            results_predicts = []
            for i_tree, _ in all_dist:
                pred_to_add = tour_tree(i_tree, cur_row, val_data)
                results_predicts.append(pred_to_add)

            max_val = max(set(results_predicts), key=results_predicts.count) # to check if can change
            val_data_r = val_data["diagnosis"].iloc[cur_row]
            if max_val == val_data_r:
                correct_predict += 1

        accuracy = correct_predict / len(val_data.index)
        return accuracy

    def calculate_significance_features(x, y, splits=5):
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
                train_x_data[f] = np.random.train_x_data(train_x_data[f])
                data_permute = train_x_data
                classifier.fit(data_permute, train_y)
                # calculate new accuracy
                second_acc, _ =classifier.predict(test_x, test_y)

                error = abs(acc_first - second_acc)
                mistakes.append(error)
            avg_m = sum(mistakes) / len(mistakes)
            w = f, avg_m
            weights.append(w)
        return weights
