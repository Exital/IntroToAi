from ID3 import ID3Node, ID3Classifier
import argparse
from utils import csv2xy, log, DEFAULT_CLASSIFICATION, graphPlotAndShow
from sklearn.model_selection import KFold


class CostSensitiveID3Node(ID3Node):
    def __init__(self, data, costFP=1, costFN=10):

        self.costTP = 1
        self.costFP = costFP
        self.costTN = 1
        self.costFN = costFN

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
        This function will check what is the best separator value.
        :return: (Best IG for this feature, separator value)
        :rtype: tuple
        """
        values = self.data[feature]
        diagnosis = self.data["diagnosis"]
        values_list = values.tolist()
        # creating the separator's list
        sorted_values = sorted(values_list, key=lambda x: x)
        separators_list = [(x + y) / 2 for x, y in zip(sorted_values, sorted_values[1:])]
        best_ig = (float("-inf"), None)

        for separator in separators_list:
            size_smaller, smaller_positive, size_larger, larger_positive = 0, 0, 0, 0
            for val, diag in zip(values, diagnosis):
                if val <= separator:
                    size_smaller += 1
                    if diag == "M":
                        smaller_positive += 1
                else:
                    size_larger += 1
                    if diag == "M":
                        larger_positive += 1
            false_positive = self.costFP
            false_negative = self.costFN
            # calculate the root's IG
            fraction = (larger_positive+smaller_positive) / len(values)
            entropy_root = -fraction * log(fraction) * false_positive - ((1 - fraction) * log(1 - fraction)) * false_negative
            # calculate the left son's IG
            fraction = smaller_positive / size_smaller if size_smaller != 0 else (larger_positive+smaller_positive) / len(values)
            entropy_left = -fraction * log(fraction) * false_positive - ((1 - fraction) * log(1 - fraction)) * false_negative
            # calculate the right son's IG
            fraction = larger_positive / size_larger if size_larger != 0 else (larger_positive+smaller_positive) / len(values)
            entropy_right = -fraction * log(fraction) * false_positive - ((1 - fraction) * log(1 - fraction)) * false_negative

            ig = entropy_root - entropy_left * size_smaller / len(values) - entropy_right*size_larger / len(values)
            if ig >= best_ig[0]:
                best_ig = ig, separator

        if best_ig[1] is None:
            raise ValueError("separator not found!")
        return best_ig


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
