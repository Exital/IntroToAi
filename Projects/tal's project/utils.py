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
