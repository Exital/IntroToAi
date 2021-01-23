from abc import ABC
import pandas as pd
import matplotlib.pyplot as plt
from numpy import log2

DEFAULT_CLASSIFICATION = "M"
# weights for the features were explored with the function "compute_feature_importance" from ID3.py
WEIGHTS = [('radius_mean', 0.0), ('texture_mean', 0.0), ('perimeter_mean', 0.0), ('area_mean', 0.0),
           ('smoothness_mean', 0.12534059945504047), ('compactness_mean', 0.0), ('concavity_mean', 0.0),
           ('concave points_mean', 0.0), ('symmetry_mean', 0.12534059945504047), ('fractal_dimension_mean', 0.0),
           ('radius_se', 0.0), ('texture_se', 0.0), ('perimeter_se', 0.12534059945504172),
           ('area_se', 0.3760217983651227), ('smoothness_se', 0.0), ('compactness_se', 0.12534059945504172),
           ('concavity_se', 0.0), ('concave points_se', 0.12534059945504047), ('symmetry_se', 0.12534059945504047),
           ('fractal_dimension_se', 0.12534059945504047), ('radius_worst', 0.0), ('texture_worst', 0.2506811989100822),
           ('perimeter_worst', 0.5013623978201631), ('area_worst', 0.0), ('smoothness_worst', 0.0),
           ('compactness_worst', 0.25068119891008095), ('concavity_worst', 0.3760217983651227),
           ('concave points_worst', 1.0), ('symmetry_worst', 0.25068119891008095),
           ('fractal_dimension_worst', 0.3760217983651227)]


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

