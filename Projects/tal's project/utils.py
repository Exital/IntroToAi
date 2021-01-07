from abc import ABC
import pandas as pd


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
