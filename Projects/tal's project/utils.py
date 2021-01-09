from abc import ABC
import pandas as pd
import matplotlib.pyplot as plt


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
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.plot(x_values, y_values)
    plt.show()
