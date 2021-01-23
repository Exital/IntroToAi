from ID3 import ID3Node, ID3Classifier
import argparse
from utils import csv2xy, graphPlotAndShow
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold


class ID3CostSensitiveClassifier(ID3Classifier):
    """
    This classifier builds a regular ID3Tree and than pruning the nodes if it will improve loss costs.
    """
    def __init__(self, cost_fn=10, cost_fp=1):
        super().__init__()
        self.validation = None
        self.cost_FN = cost_fn
        self.cost_FP = cost_fp

    def fit(self, x, y, test_size=0.58):
        """
        Builds an ID3Tree and than prune it to improve costs
        :param x: dataset
        :type x: dataframe
        :param y: dataset
        :type y: dataframe
        :param test_size: the fraction for splitting
        :type test_size: float
        """
        X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=test_size, random_state=426)
        train_data = X_train.copy()
        train_data["diagnosis"] = y_train
        validation_data = X_test.copy()
        validation_data["diagnosis"] = y_test
        self.validation = validation_data
        id3tree = ID3Node(data=train_data)

        pruned_tree = self.prune(id3tree, self.validation)
        self.id3tree = pruned_tree

    def prune(self, node: ID3Node, validation):
        """
        This is the function that is pruning the tree for better costs loss
        :param node: The tree to prune
        :type node: ID3Node
        :param validation: validation data to test on the node
        :type validation: dataframe
        :return: A pruned tree with better loss results
        :rtype: ID3Node
        """
        # if no validation data or node is leaf we return the same node
        if len(validation.index) == 0 or node.is_leaf():
            return node

        # slicing the dataframe
        validation_left = validation[validation[node.feature] <= node.slicing_val]
        validation_right = validation[validation[node.feature] > node.slicing_val]

        # recursively pruning the node's sons
        node.left = self.prune(node.left, validation_left)
        node.right = self.prune(node.right, validation_right)

        # checking if its better to prune that node or not with cost in mind
        err_prune, err_no_prune = 0, 0
        prune_diagnostic = self.decide_leaf_diagnosis_by_costs(validation)
        for row in range(len(validation.index)):
            prediction = self.walk_the_tree(node, row, validation)
            real_truth = validation["diagnosis"].iloc[row]
            err_prune += self.evaluate(real_truth, prune_diagnostic)
            err_no_prune += self.evaluate(real_truth, prediction)

        # it will be better to prune
        if err_prune < err_no_prune:
            node.data = None
            node.feature = None
            node.left = None
            node.right = None
            node.slicing_val = None
            node.diag = prune_diagnostic
        return node

    def evaluate(self, real_truth, predicted_truth):
        """
        evaluates the loss for the prediction of that node.
        :param real_truth: the diagnosis of the node
        :type real_truth: str
        :param predicted_truth: the ID3Tree prediction of that node
        :type predicted_truth: str
        :return: the cost of that prediction
        :rtype: int
        """
        if real_truth != predicted_truth:
            return self.cost_FN if real_truth == "M" else self.cost_FP
        else:
            return 0

    def decide_leaf_diagnosis_by_costs(self, validation):
        """
        This function decides whats best diagnosis to give to a pruned node by the cost of it.
        :param validation: the validation data of the node being checked
        :type validation: dataframe
        :return: the node diagnosis
        :rtype: str
        """
        data = validation
        count = data["diagnosis"].value_counts()
        m_count = count["M"] if "M" in count.index else 0
        b_count = count["B"] if "B" in count.index else 0

        return "M" if b_count * self.cost_FP < m_count * self.cost_FN else "B"


def experiment(X=None, y=None, test_size=None, splits=5):
    """
    This function uses sklearn's kFold to cross validate and find the best
    size of split for costs sensitive
    The only parameter you need is to set verbose to True so you can see output.
    :param X: X dataset
    :type X: dataframe
    :param y: y dataset
    :type y: dataframe
    :param test_size: values to cross validate
    :type test_size: list
    :param verbose: True if you want to see graph and summary
    :type verbose: bool
    :param splits: number of splits for kfold
    :type splits: int
    """
    # setting up default params
    if X is None or y is None:
        X, y = csv2xy("train.csv")
    if test_size is None:
        test_size = [x/100 for x in range(1, 99)]
    losses = []
    print(f"------------------- starting validation size test --------------------")
    print(f"test size={test_size}")
    for size in test_size:
        kfold_loss = []
        classifier = ID3CostSensitiveClassifier()
        kf = KFold(n_splits=splits, random_state=307965806, shuffle=True)
        for train_index, test_index in kf.split(X):
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]
            classifier.fit(X_train, y_train, size)
            acc, loss = classifier.predict(X_test, y_test)
            kfold_loss.append(loss)
        avg = sum(kfold_loss) / len(kfold_loss)
        losses.append(avg)
        print(f"size={size}, loss={avg}")
    zipped = list(zip(test_size, losses))
    zipped.sort(key=lambda x: x[1])
    best_size = zipped[0]
    print(f"----------------- Kfold cross validation results ------------------\n"
          f"Best size is={best_size[0]} with loss={best_size[1]}")
    graphPlotAndShow(test_size, losses, "test size", "loss")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('-v', '-verbose', dest="verbose", action='store_true', help="Show more information")
    parser.add_argument('-find_validation_size', dest="find_validation_size",
                        action='store_true', help="Running a test to find best validation size")

    args = parser.parse_args()

    # retrieving the data from the csv files
    train_x, train_y = csv2xy("train.csv")
    test_x, test_y = csv2xy("test.csv")
    # creating a classifier instance
    classifier = ID3CostSensitiveClassifier()
    # fitting the classifier
    classifier.fit(train_x, train_y)
    # predicting on the test data set
    accuracy, loss = classifier.predict(test_x, test_y)
    if args.verbose:
        print(f"accuracy with cost optimizing={accuracy}")
        print(f"loss with cost optimizing={loss}")
    else:
        print(loss)

    # Todo - in order to find the validation size run CostSensitiveID3.py with -find_validation_size flag.
    if args.find_validation_size:
        experiment()
