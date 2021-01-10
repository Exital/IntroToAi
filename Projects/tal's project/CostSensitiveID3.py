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
        X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)
        train_data = X_train.copy()
        train_data["diagnosis"] = y_train
        validation_data = X_test.copy()
        validation_data["diagnosis"] = y_test
        self.validation = validation_data
        id3tree = ID3Node(data=train_data)

    # TODO continue to write the prune function
    # def prune(self, node: ID3Node):
    #     if node.is_leaf():
    #         return node
    #     node.left = self.prune(node.left)
    #     node.right = self.prune(node.right)
    #
    #     err_prune, err_no_prune = 0, 0
    #     for row in range(len(self.validation.index)):
    #         prediction = self.walk_the_tree(node, row, self.validation)
    #         if

    def evaluate(self, real_truth, predicted_truth):
        if real_truth != predicted_truth:
            return self.cost_FN if real_truth == "M" else self.cost_FP
        else:
            return 0

    # TODO continue decide leaf diagnosis
    # def decide_leaf_diagnosis_by_costs(self, node: ID3Node):
    #     data = node.data
    #     diagnosis = data["diagnosis"]


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
