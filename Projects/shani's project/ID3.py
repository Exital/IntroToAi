from numpy import log2
import pandas as pd
DEFAULT = "M"


def log(x):
    if x == 0:
        return 0
    else:
        return log2(x)


def get_data_from_csv(file):
    """
    loads the csv data into x,y
    :param file: file name
    :type file: string
    :return: (x, y)
    :rtype: tuple
    """
    pass


class TreeNode:
    """
    This nodes will make an ID3 tree.
    """
    def __init__(self, data: pd.DataFrame):
        """
        TODO init function
        """
        self.data = data
        pass

    def calculate_entropy(self, fraction):
        """

        :param fraction: The fraction of the entropy
        :type fraction: float
        :return: The entropy
        :rtype: float
        """
        pass

    def is_leaf(self):
        """
        checks if that node is leaf
        :return: True if leaf
        :rtype: bool
        """
        pass

    def IG_for_feature(self, feature_name):
        """
        checks what is the best threshold for feature and it's IG
        :param feature_name: The feature name to check
        :type feature_name: string
        :return: (IG for the feature, threshold)
        :rtype: tuple
        """
        pass

    def choose_feature(self):
        """
        checks the IG of all features and chooses the best one.
        :return: (feature name, threshold)
        :rtype: tuple
        """
        pass

    def all_same_data(self):
        """
        checks if the nodes data
        :return:
        :rtype:
        """
        pass


def build_id3_tree(data: pd.DataFrame) -> TreeNode:
    """
    This function will build a ID3 Tree
    :param data: the whole data set
    :type data: pd.Dataframe
    :return: An ID3 tree
    :rtype: TreeNode
    """
    # TODO write your own build tree recursion


class ID3Classifier:
    """
    Classifier for id3 predicting
    """
    def fit(self, x, y):
        """
        fits the classifier and creates a decision tree
        :param x: data without diagnosis
        :type x: Dataframe
        :param y: diagnosis
        :type y: Dataframe
        """
        pass

    def predict(self, x, y):
        """
        predicts new samples with the decision tree made by fit.
        :param x: data without diagnosis
        :type x:Dataframe
        :param y: diagnosis
        :type y: Dataframe
        :return: accuracy [0,1]
        :rtype: float
        """
        pass
