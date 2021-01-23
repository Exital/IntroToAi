from utils import csv2xy, AbstractClassifier, graphPlotAndShow, log, DEFAULT_CLASSIFICATION
import argparse
from sklearn.model_selection import KFold


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

            # calculate the root's IG
            roots_fraction = (larger_positive+smaller_positive) / len(values)
            entropy_root = -roots_fraction * log(roots_fraction) - ((1 - roots_fraction) * log(1 - roots_fraction))
            # calculate the left son's IG
            fraction = smaller_positive / size_smaller if size_smaller != 0 else roots_fraction
            entropy_left = -fraction * log(fraction) - ((1 - fraction) * log(1 - fraction))
            # calculate the right son's IG
            fraction = larger_positive / size_larger if size_larger != 0 else roots_fraction
            entropy_right = -fraction * log(fraction) - ((1 - fraction) * log(1 - fraction))

            ig = entropy_root - entropy_left * size_smaller / len(values) - entropy_right*size_larger / len(values)
            if ig >= best_ig[0]:
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
            if ig >= best_ig[1]:
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
        if len(self.data.index) == 0:
            return True, DEFAULT_CLASSIFICATION
        if len(self.data.diagnosis.unique()) == 1:
            result = (True, self.data["diagnosis"].iloc[0])
        else:
            result = (False, None)
        return result


class ID3Classifier(AbstractClassifier):
    """
    A classifier that utilizes the ID3 tree to fit and predict.
    """
    def __init__(self):
        self.id3tree = None

    def fit(self, x, y):
        """
        fitting the data into a ID3 tree.
        :param x: dataset
        :type x: dataframe
        :param y: labels
        :type y: dataframe
        """
        data = x.copy()
        data["diagnosis"] = y
        self.id3tree = ID3Node(data=data)

    def predict(self, x, y):
        """
        predicts the diagnosis for data set x and computes accuracy and loss for y
        :param x: dataset
        :type x: dataframe
        :param y: diagnosis
        :type y: dataframe
        :return: (accuracy, loss)
        :rtype: tuple
        """
        if self.id3tree is None:
            raise ReferenceError("There was no fit first!")

        data = x.copy()
        data["diagnosis"] = y
        right_predictions = 0
        false_negative = 0
        false_positive = 0
        num_of_samples = len(data.index)
        for row in range(len(data.index)):
            prediction = self.walk_the_tree(self.id3tree, row, data)
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


class ID3PruningNode(ID3Node):
    """
    A Node of a ID3 tree with pruning option.
    """
    def __init__(self, data=None, diag=None, m=8):
        self.feature = None
        self.slicing_val = None
        self.left = None
        self.right = None
        self.data = data
        self.diag = diag
        self.pruning_value = m

        # check if this is a pruned son
        if diag is None:
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
                # check if sons has to be pruned
                prune, diag = self._check_if_son_has_to_be_pruned(data_left)
                if prune:
                    self.left = ID3PruningNode(diag=diag, m=self.pruning_value)
                else:
                    self.left = ID3PruningNode(data=data_left, m=self.pruning_value)
                prune, diag = self._check_if_son_has_to_be_pruned(data_right)
                if prune:
                    self.right = ID3PruningNode(diag=diag, m=self.pruning_value)
                else:
                    self.right = ID3PruningNode(data=data_right, m=self.pruning_value)

    def _check_if_son_has_to_be_pruned(self, data):
        """
        checks if son has to be pruned
        :param data: the data of the son
        :type data: dataframe
        :return: (True, diag) if it has to be pruned otherwise (False, None)
        :rtype: tuple
        """
        if len(data.index) < self.pruning_value:
            diag = self.data['diagnosis'].value_counts().idxmax()
            return True, diag
        else:
            return False, None


class ID3PruneClassifier(ID3Classifier):
    """
    A classifier utilizes the ability of pruning ID3 node.
    """
    def __init__(self, pruning_value=8):
        super().__init__()
        self.pruning_value = pruning_value

    def fit(self, x, y):
        data = x.copy()
        data["diagnosis"] = y
        self.id3tree = ID3PruningNode(data=data, m=self.pruning_value)


def experiment(X=None, y=None, m_values=None, splits=5):
    """
    This function uses sklearn's kFold to cross validate and find the best
    M value for the pruning.
    The only parameter you need is to set verbose to True so you can see output.
    :param X: X dataset
    :type X: dataframe
    :param y: y dataset
    :type y: dataframe
    :param k_values: values to cross validate
    :type k_values: list
    :param verbose: True if you want to see graph and summary
    :type verbose: bool
    :param splits: number of splits for the kfold
    :type splits: int
    """
    # setting up default params
    if X is None or y is None:
        X, y = csv2xy("train.csv")
    if m_values is None:
        m_values = [x for x in range(0, 25)]
    accuracies = []
    print(f"----------------- starting experiment to determine M value -----------------")
    print(f"M values={m_values}")
    for m in m_values:
        classifier = ID3PruneClassifier(pruning_value=m)
        kf = KFold(n_splits=splits, random_state=307965806, shuffle=True)
        kfold_accuracy = []
        for train_index, test_index in kf.split(X):
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]
            classifier.fit(X_train, y_train)
            acc, loss = classifier.predict(X_test, y_test)
            kfold_accuracy.append(acc)
        avg = sum(kfold_accuracy) / len(kfold_accuracy)
        accuracies.append(avg)
        print(f"M={m}, accuracy={avg}")
    graphPlotAndShow(m_values, accuracies, "M value", "Accuracy")
    zipped = list(zip(m_values, accuracies))
    zipped.sort(key=lambda x: x[1], reverse=True)
    best_k = zipped[0]
    print(f"--------------------- Kfold cross validation results --------------\n"
          f"Best M={best_k[0]} with accuracy={best_k[1]}")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('-v', '-verbose', dest="verbose", action='store_true', help="Show more information")
    parser.add_argument('-find_m_value', dest="find_m_value", action='store_true', help="Running a test to find best M")
    args = parser.parse_args()

    # retrieving the data from the csv files
    train_x, train_y = csv2xy("train.csv")
    test_x, test_y = csv2xy("test.csv")
    # creating a classifier instance
    classifier = ID3Classifier()
    # fitting the classifier
    classifier.fit(train_x, train_y)
    # predicting on the test data set
    accuracy, loss = classifier.predict(test_x, test_y)
    if args.verbose:
        print(f"The accuracy of ID3Classifier={accuracy}")
        print(f"loss without cost optimizing={loss}")
    else:
        print(accuracy)

    # Todo - in order to run the experiment of finding the best M value run ID3.py with -find_m_value flag.
    if args.find_m_value:
        experiment()
