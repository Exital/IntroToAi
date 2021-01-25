import random
import numpy as np
from ID3 import get_data_from_csv, build_id3_tree
from CostSensitiveID3 import tour_tree
from sklearn.model_selection import train_test_split

def slice_data(x, y, fraction):
    """
    slices the data and return a fraction of it
    :param data: a dataframe
    :type data: dataframe
    :param fraction: number between [0,1]
    :type fraction: float
    :return: a fraction of the data
    :rtype: dataframe
    """
    sliced_data = None
    # -------- Your code -------
    # TODO write a function that take the dataframe and randomly return a fraction of it.
    # TODO lets say the data has 100 sample and fraction is 0.5 then you return 50 samples randomly.
    # --------------------------
    test_size = 1 - fraction
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size)
    sliced_data = x_train.copy(), y_train.copy()

    return sliced_data


def get_centroid(data):
    """
    That function gets a dataframe and returns the vector of mean.
    :param data: a dataframe
    :type data: dataframe
    :return: a mean vector
    :rtype: series
    """
    centroid = None
    # -------- Your code -------
    # TODO create a function that takes the mean out of the whole data to make a mean vector.
    # --------------------------
    centroid = data.copy()
    centroid = centroid.mean(axis=0)

    return centroid


def distance_between_vectors(v1, v2):
    """
    this function get 2 vectors and return euclidean distance between them
    :param v1: series vector
    :type v1: series
    :param v2: series vector
    :type v2: series
    :return: a distance between the 2 vectors
    :rtype: float
    """
    distance = None
    # -------- Your code -------
    # TODO calculate euclidean distance of the 2 vectors given.
    # --------------------------
    distance = np.linalg.norm(v1 - v2)
    return distance


class KNNForestClassifier:
    """
    This is the classifier for the forest
    """
    def __init__(self, N=20, k=13):
        self.forest = []
        self.centroids = []
        self.N = N
        self.k = k
        self.first = 0.3
        self.last = 0.7

    def fit(self, x, y):
        """
        fits the classifier and creates a decision tree
        :param x: data without diagnosis
        :type x: Dataframe
        :param y: diagnosis
        :type y: Dataframe
        """

        self.centroids = []
        self.forest = []
        # Training group size
        for i in range(self.N):
            fraction = random.uniform(self.first, self.last)
            sliced_x, sliced_y = slice_data(x, y, fraction)
            sliced_x["diagnosis"] = sliced_y
            tree_node = build_id3_tree(sliced_x)
            self.forest.append(tree_node)
            centroid = get_centroid(sliced_x)
            self.centroids.append(centroid)


    def predict(self, x, y):

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



if __name__ == "__main__":
    # get tha data from csv files
    train_x, train_y = get_data_from_csv("train.csv")
    test_x, test_y = get_data_from_csv("test.csv")

    # create a ID3Classifier
    classifier = KNNForestClassifier()
    # fit the classifier
    classifier.fit(train_x, train_y)
    # predict in test data set
    res_accuracy = classifier.predict(test_x, test_y)
    print(res_accuracy)
    # TODO print the Maximum accuracy value
    # print(f"Maximum accuracy is {res_accuracy}")
