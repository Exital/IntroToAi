from abc import ABC
import pandas as pd
import matplotlib.pyplot as plt
from numpy import log2
DEFAULT_CLASSIFICATION = "M"


def log(x):
    return x if x == 0 else log2(x)


class AbstractClassifier(ABC):
    def fit(self, x, y):
        pass

    def predict(self, x, y):
        pass

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


def csv2xy(file):
    df = pd.read_csv(file)
    x = df.iloc[:, 1:]
    y = df.iloc[:, 0:1]
    return x, y


def graphPlotAndShow(x_values, y_values, x_label="", y_label=""):
    """
    This function will plot a graph
    :param x_values: list of x values
    :type x_values: list
    :param y_values: list of y values
    :type y_values: list
    :param x_label: label for x axis
    :type x_label: str
    :param y_label: label for y axis
    :type y_label: str
    """
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.plot(x_values, y_values)
    plt.show()

