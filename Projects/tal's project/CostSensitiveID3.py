from ID3 import ID3Node, ID3Classifier
import argparse
from utils import csv2xy


class CostSensitiveID3Node(ID3Node):
    def __init__(self, data):

        self.costTP = 1
        self.costFP = 10
        self.costTN = 1
        self.costFN = 100

        self.feature = None
        self.slicing_val = None
        self.left = None
        self.right = None
        self.data = data
        self.diag = None

        # check if that is a leaf
        leaf, diag = self._all_data_has_same_diag()
        if leaf:
            self.diag = diag
        else:
            # this is not a leaf so we update feature for slicing and slicing val
            self.feature, _, self.slicing_val = self._choose_feature()
            # slicing the dataframe
            data_left = self.data[self.data[self.feature] <= self.slicing_val]
            data_right = self.data[self.data[self.feature] > self.slicing_val]
            # recursively creating more ID3Nodes
            self.left = CostSensitiveID3Node(data=data_left)
            self.right = CostSensitiveID3Node(data=data_right)

    def _IG_for_feature(self, feature):
        """
        Using the same API of ID3Node and utilizing it for cost efficiency.
        This function calculate costs and return which separator has the minimum cost.
        :param feature: feature name
        :type feature: str
        :return: (minimum cost, separator)
        :rtype: tuple
        """
        values = self.data[feature]
        diagnosis = self.data["diagnosis"]
        values_list = values.tolist()
        # creating the separator's list
        sorted_values = sorted(values_list, key=lambda x: x)
        separators_list = [(x + y) / 2 for x, y in zip(sorted_values, sorted_values[1:])]
        lowest_cost = (float("inf"), None)

        for separator in separators_list:
            smaller_true_positive, smaller_false_positive, larger_true_positive, larger_false_positive = 0, 0, 0, 0
            for val, diag in zip(values, diagnosis):
                if val <= separator:
                    if diag == "M":
                        smaller_true_positive += 1
                    else:
                        smaller_false_positive += 1
                else:
                    if diag == "M":
                        larger_true_positive += 1
                    else:
                        smaller_false_positive += 1

            # left son costs
            left_cost_for_all_positive = smaller_true_positive * self.costTP + smaller_false_positive * self.costFP
            left_cost_for_all_negative = smaller_false_positive * self.costTN + smaller_true_positive * self.costFN
            if left_cost_for_all_negative == 0 and left_cost_for_all_positive == 0:
                left_entropy = 0
            else:
                left_entropy = (left_cost_for_all_positive * left_cost_for_all_negative) / (left_cost_for_all_positive + left_cost_for_all_negative)
            # right son costs
            right_cost_for_all_positive = larger_true_positive * self.costTP + larger_false_positive * self.costFP
            right_cost_for_all_negative = larger_false_positive * self.costTN + larger_true_positive * self.costFN
            if right_cost_for_all_negative == 0 and right_cost_for_all_positive == 0:
                right_entropy = 0
            else:
                right_entropy = (right_cost_for_all_positive * right_cost_for_all_negative) / (right_cost_for_all_positive + right_cost_for_all_negative)

            total_cost = 2 * (left_entropy + right_entropy)

            if total_cost < lowest_cost[0]:
                lowest_cost = total_cost, separator

        if lowest_cost[1] is None:
            raise ValueError("separator not found!")
        return lowest_cost

    def _choose_feature(self):
        """
        Using the same API of ID3Node and utilizing it for cost efficiency.
        Will choose a feature to slice for that node according to all feature's lowest cost.
        :return: (feature name, feature's cost, separator value)
        :rtype: tuple
        """
        features = self.data.keys().tolist()
        features = features[:-1]

        lowest_cost = None, float("inf"), None

        for feature in features:
            cost, separator = self._IG_for_feature(feature)
            if cost < lowest_cost[1]:
                lowest_cost = feature, cost, separator
        if lowest_cost[0] is None:
            raise ValueError("feature to separate not found!")
        return lowest_cost


class ID3CostSensitiveClassifier(ID3Classifier):
    def fit(self, x, y):
        data = x.copy()
        data["diagnosis"] = y
        self.id3tree = CostSensitiveID3Node(data=data)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('-v', '-verbose', dest="verbose", action='store_true', help="Show more information")
    args = parser.parse_args()

    # retrieving the data from the csv files
    train_x, train_y = csv2xy("data/train.csv")
    test_x, test_y = csv2xy("data/test.csv")
    # creating a classifier instance
    classifier = ID3CostSensitiveClassifier()
    # fitting the classifier
    classifier.fit(train_x, train_y)
    # predicting on the test data set
    accuracy, loss = classifier.predict(test_x, test_y)
    print(accuracy)
    if args.verbose:
        print(f"loss with cost optimizing={loss}")
