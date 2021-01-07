from utils import csv2xy, AbstractClassifier
from numpy import log2


def log(x):
    return x if x == 0 else log2(x)


class ID3Node:
    """
    The ID3 class that creates the whole ID3Tree.
    """
    def __init__(self, data=None):
        """
        The init function will generate the whole id3tree
        :param data: The whole pandas data frame with the diagnosis column
        :type data: panda's Dataframe
        """
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
            self.left = ID3Node(data=data_left)
            self.right = ID3Node(data=data_right)

    def is_leaf(self):
        """
        That function checks whether this node is a leaf.
        :return: True if leaf
        :rtype: bool
        """
        return self.diag is not None

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
        separators_list = [(x + y) / 2 for x, y in zip(values_list, values_list[1:])]
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

            # calculate the left son's IG
            fraction = smaller_positive / size_smaller
            entropy_left = -fraction * log(fraction) - ((1 - fraction) * log(1 - fraction))
            # calculate the right son's IG
            fraction = larger_positive / size_larger
            entropy_right = -fraction * log(fraction) - ((1 - fraction) * log(1 - fraction))
            # calculate the root's IG
            fraction = (larger_positive+smaller_positive) / len(values)
            entropy_root = -fraction * log(fraction) - ((1 - fraction) * log(1 - fraction))

            ig = entropy_root - entropy_left * size_smaller / len(values) - entropy_right*size_larger / len(values)
            if ig > best_ig[0]:
                best_ig = ig, separator

        if best_ig[1] is None:
            raise ValueError("separator not found!")
        return best_ig

    def _choose_feature(self):
        """
        Will choose a feature to slice for that node according to all feature's IG.
        :return: (feature name, feature's ig, separator value)
        :rtype: tuple
        """
        features = self.data.keys().tolist()
        features = features[:-1]

        best_ig = None, float("-inf"), None

        for feature in features:
            ig, separator = self._IG_for_feature(feature)
            if ig > best_ig[1]:
                best_ig = feature, ig, separator
        if best_ig[0] is None:
            raise ValueError("feature to separate not found!")
        return best_ig

    def _all_data_has_same_diag(self):
        """
        That function checks whether a node has reached a place where all of his data is either "M" or "B".
        :return: (True, diagnosis) or (False, None)
        :rtype: tuple
        """
        if len(self.data.diagnosis.unique()) == 1:
            result = (True, self.data["diagnosis"].iloc[0])
        else:
            result = (False, None)
        return result


class ID3Classifier(AbstractClassifier):
    def __init__(self):
        self.id3tree = None

    def fit(self, x, y):
        x["diagnosis"] = y
        self.id3tree = ID3Node(data=x)

    def predict(self, x, y):

        def walk_the_tree(node: ID3Node, row):
            """
            This function is a recursive function that walks the tree till it reaches a leaf.
            :param node: An id3 Node
            :type node: ID3Node
            :param row: row number on the dataframe
            :type row: int
            :return: diagnosis
            """
            if node.is_leaf():
                return node.diag
            else:
                feature = node.feature
                value = data[feature].iloc[row]
                if value <= node.slicing_val:
                    return walk_the_tree(node.left, row)
                else:
                    return walk_the_tree(node.right, row)

        if self.id3tree is None:
            raise ReferenceError("There was no fit first!")

        x["diagnosis"] = y
        data = x
        right_predictions = 0
        for row in range(len(data.index)):
            if walk_the_tree(self.id3tree, row) == data["diagnosis"].iloc[row]:
                right_predictions += 1
        return right_predictions / len(data.index)


if __name__ == "__main__":

    # retrieving the data from the csv files
    train_x, train_y = csv2xy("data/train.csv")
    test_x, test_y = csv2xy("data/test.csv")

    # creating a classifier instance
    classifier = ID3Classifier()
    # fitting the classifier
    classifier.fit(train_x, train_y)
    # predicting on the test data set
    accuracy = classifier.predict(test_x, test_y)
    print(accuracy)
