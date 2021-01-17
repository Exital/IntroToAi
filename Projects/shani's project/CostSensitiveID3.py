from ID3 import TreeNode, ID3Classifier
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
import pandas as pd

def get_data_from_csv(file):
    """
    loads the csv data into x,y
    :param file: file name
    :type file: string
    :return: (x, y)
    :rtype: tuple
    """
    temp = pd.read_csv(file)
    y = temp.iloc[:, 0:1]
    x = temp.iloc[:, 1:]
    return x, y



def check_leaf_and_groupValidation(node, group_validation):
    """

    :param group_validation:
    :return:
    """
    if node.is_leaf() or len(group_validation.index) == 0:
        return True
    else:
        False


class costSensitiveID3Classifier(ID3Classifier):
    """

    """
    def __init__(self, FN_val=10, FP_val=1):
        super().__init__()
        self.FN_val = FN_val
        self.FP_val = FP_val
        self.group_validation = None

    def fit(self, x, y, test_size=0.6):
        """

        :param x: dataset
        :type x: dataframe
        :param y: dataset
        :type y: dataframe
        :param test_size:
        :return:
        """
        train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=test_size, random_state=426)
        data_train = train_x.copy()
        data_train["diagnosis"] = train_y
        check_data = test_x.copy()
        check_data["diagnosis"] = test_y
        self.group_validation = check_data
        id3tree = TreeNode(data=data_train)
        # to complete

    def build_prune_tree_validation(self, node: TreeNode, group_validation):
        """
        :param node:
        :param check_valid:
        :return:
        """
        # check if we reached to leaf or if there is no data group validation
        check = check_leaf_and_groupValidation(node, group_validation)
        if check:
            return node
        else:
            # dataframe slicing and creating tree with using recursion
            right_group_validation = group_validation[group_validation[node.feature] > node.threshold]
            left_group_validation = group_validation[group_validation[node.feature] <= node.threshold]

            node.right = self.build_prune_tree_validation(node.right, right_group_validation)
            node.left = self.build_prune_tree_validation(node.left, left_group_validation)

            return node

    def check_if_better_prune(self, node: TreeNode, group_validation):
        """
        function to check if is better to prune this node
        :param node:
        :return:
        """
        node_to_check = self.build_prune_tree_validation(node, group_validation)
        err_prune = 0
        err_no_prune = 0
        # check what is the best detection to give to a pruned node by it's cost
        detection_pruning = 0
        group_valid_data = group_validation
        number_of_values = group_valid_data["diagnosis"].value_counts()
        if "M" in number_of_values.index:
            number_M = number_of_values["M"]
        else:
            number_M = 0
        if "B" in number_of_values.index:
            number_B = number_of_values["B"]
        else:
            number_B = 0

        if number_B * self.FP_val < number_M * self.FN_val:
            detection_pruning = "M"
        else:
            detection_pruning = "B"

        size_group_valid = len(group_validation.index)
        for row in range(size_group_valid):
            res_predict = self.tour_tree(node_to_check, row) # ask Tal
            # evaluates the loss for the prediction result of this node
            detection_node_val = group_validation["diagnosis"].iloc[row]
            # Comparing the error on the validation group with and without pruning
            if detection_pruning != detection_node_val:
                # res includes the cost of this prediction
                if detection_node_val == "M":
                    res = self.FN_val
                else:
                    res = self.FP_val
            else:
                res = 0
            err_prune += res

            if res_predict != detection_node_val:
                # res includes the cost of this prediction
                if detection_node_val == "M":
                    res = self.FN_val
                else:
                    res = self.FP_val
            else:
                res = 0
            err_no_prune += res

        # check iIf the error is small we will perform pruning
        if err_prune <  err_no_prune:
            node_to_check.data = None
            node_to_check.right = None
            node_to_check.left = None
            node_to_check.feature = None
            node_to_check.diag = detection_pruning
            node_to_check.threshold = None

        return node_to_check



if __name__ == "__main__":
    # get tha data from csv files
    train_x, train_y = get_data_from_csv("train.csv")
    test_x, test_y = get_data_from_csv("test.csv")
    # create a ID3Classifier
    classifier = costSensitiveID3Classifier()
    # fit the classifier
    classifier.fit(train_x, train_y)
    # predict in test data set
    res_loss, res_accuracy = classifier.predict(test_x, test_y)
    print(res_accuracy)
    print(res_loss)


