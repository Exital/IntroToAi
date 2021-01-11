from ID3 import ID3Node, ID3Classifier
import argparse
from utils import csv2xy, log, DEFAULT_CLASSIFICATION, graphPlotAndShow
from sklearn.model_selection import train_test_split


class ID3CostSensitiveClassifier(ID3Classifier):
    def __init__(self, cost_fn=10, cost_fp=1):
        super().__init__()
        self.validation = None
        self.cost_FN = cost_fn
        self.cost_FP = cost_fp

    def fit(self, x, y):
        X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.60, random_state=420)
        train_data = X_train.copy()
        train_data["diagnosis"] = y_train
        validation_data = X_test.copy()
        validation_data["diagnosis"] = y_test
        self.validation = validation_data
        id3tree = ID3Node(data=train_data)

        pruned_tree = self.prune(id3tree, self.validation)
        self.id3tree = pruned_tree

    def prune(self, node: ID3Node, validation):
        if len(validation.index) == 0:
            return node
        if node.is_leaf():
            return node

        # slicing the dataframe
        validation_left = validation[validation[node.feature] <= node.slicing_val]
        validation_right = validation[validation[node.feature] > node.slicing_val]

        node.left = self.prune(node.left, validation_left)
        node.right = self.prune(node.right, validation_right)
        err_prune, err_no_prune = 0, 0
        prune_diagnostic = self.decide_leaf_diagnosis_by_costs(node, validation)
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
        if real_truth != predicted_truth:
            return self.cost_FN if real_truth == "M" else self.cost_FP
        else:
            return 0

    def decide_leaf_diagnosis_by_costs(self, node: ID3Node, validation):
        data = validation
        count = data["diagnosis"].value_counts()
        if "M" in count.index:
            m_count = count["M"]
        else:
            m_count = 0
        if "B" in count.index:
            b_count = count["B"]
        else:
            b_count = 0
        return "M" if b_count * self.cost_FP < m_count * self.cost_FN else "B"


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('-v', '-verbose', dest="verbose", action='store_true', help="Show more information")
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
    print(accuracy)
    if args.verbose:
        print(f"loss with cost optimizing={loss}")
