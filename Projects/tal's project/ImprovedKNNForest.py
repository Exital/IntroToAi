from utils import csv2xy, log, AbstractClassifier
from ID3 import ID3Node
import argparse
import numpy as np
DEFAULT_CLASSIFICATION = 0.5

# def build_gradient_boosting_tree(x, y):
#     labels = y.copy()
#     labels = labels.replace(['M'], 1)
#     labels = labels.replace(['B'], 0)
#     data = x.copy()
#     data["diagnosis"] = labels
#     count = data["diagnosis"].value_counts()
#     m_count = count["M"] if "M" in count.index else 0
#     b_count = count["B"] if "B" in count.index else 0
#     fraction = m_count / b_count if m_count > b_count else b_count / m_count
#     log_of_odds0 = log(fraction)
#     probability0 = np.exp(log_of_odds0) / (np.exp(log_of_odds0) + 1)
#     prediction0 = 1 if probability0 >= 0.5 else 0


class GBTNode(ID3Node):
    def __init__(self, data):
        self.feature = None
        self.slicing_val = None
        self.left = None
        self.right = None
        self.diag = None
        self.data = data

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
            self.left = GBTNode(data=data_left)
            self.right = GBTNode(data=data_right)

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
                    if diag >= 0.5:
                        smaller_positive += 1
                else:
                    size_larger += 1
                    if diag >= 0.5:
                        larger_positive += 1

            # calculate the root's IG
            fraction = (larger_positive+smaller_positive) / len(values)
            entropy_root = -fraction * log(fraction) - ((1 - fraction) * log(1 - fraction))
            # calculate the left son's IG
            fraction = smaller_positive / size_smaller if size_smaller != 0 else (larger_positive+smaller_positive) / len(values)
            entropy_left = -fraction * log(fraction) - ((1 - fraction) * log(1 - fraction))
            # calculate the right son's IG
            fraction = larger_positive / size_larger if size_larger != 0 else (larger_positive+smaller_positive) / len(values)
            entropy_right = -fraction * log(fraction) - ((1 - fraction) * log(1 - fraction))

            ig = entropy_root - entropy_left * size_smaller / len(values) - entropy_right*size_larger / len(values)
            if ig >= best_ig[0]:
                best_ig = ig, separator

        if best_ig[1] is None:
            raise ValueError("separator not found!")
        return best_ig

    def _all_data_has_same_diag(self):
        """
        That function checks whether a node has reached a place where all of his data is either "M" or "B".
        :return: (True, diagnosis) or (False, None)
        :rtype: tuple
        """
        if len(self.data.index) == 0:
            return True, DEFAULT_CLASSIFICATION
        if len(self.data.diagnosis.unique()) == 1:
            result = (True, self.data["diagnosis"].iloc[0])
        else:
            result = (False, None)
        return result


class GBTTree:
    def __init__(self, train_data, k=10, lr=0.1):
        self.k = k
        self.lr = lr
        self.forest = []
        self.data = None
        self.ground_truth = None
        data_copy = train_data.copy()
        labels = data_copy["diagnosis"]
        labels = labels.replace(['M'], 1)
        labels = labels.replace(['B'], 0)
        data_copy["diagnosis"] = labels

        self.ground_truth = labels.copy()
        self.data = data_copy
        self.avg = data_copy["diagnosis"].mean()

        # build first tree
        tree_data = self.data.copy()
        new_diagnosis = self.ground_truth - self.avg
        tree_data["diagnosis"] = new_diagnosis
        tree = GBTNode(tree_data)
        self.forest.append(tree)

    def walk_the_tree(self, node, row, data):
        """
        This function is a recursive function that walks the tree till it reaches a leaf.
        :param node: A node that has diag, slicing_val members and is_leaf method.
        :type node: ID3Node
        :param row: row number on the dataframe
        :type row: int
        :param data: the dataset
        :type data: dataframe
        :return: diagnosis
        """
        if node.is_leaf():
            return node.diag
        else:
            feature = node.feature
            value = data[feature].iloc[row]
            if value <= node.slicing_val:
                return self.walk_the_tree(node.left, row, data)
            else:
                return self.walk_the_tree(node.right, row, data)

    def calc_prediction_for_row(self, row, data):
        result = self.avg
        for tree in self.forest:
            raw_prediction = self.walk_the_tree(tree, row, data)
            result += self.lr * raw_prediction
        return result

    def calc_residual(self, data):
        residuals = self.ground_truth.copy()
        for row in range(len(data.index)):
            prediction = self.calc_prediction_for_row(row, data)
            residual = self.ground_truth.iloc[row] - prediction
            residuals.iloc[row] = residual
        return residuals

    def add_another_layer(self):
        residual = self.calc_residual(self.data)
        new_data = self.data.copy()
        new_data["diagnosis"] = residual
        new_tree = GBTNode(data=new_data)
        self.forest.append(new_tree)

    def boost(self):
        for i in range(self.k):
            self.add_another_layer()


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('-v', '-verbose', dest="verbose", action='store_true', help="Show more information")
    args = parser.parse_args()

    # retrieving the data from the csv files
    train_x, train_y = csv2xy("train.csv")
    test_x, test_y = csv2xy("test.csv")
    train_x["diagnosis"] = train_y
    tree = GBTTree(train_x)
    tree.boost()

