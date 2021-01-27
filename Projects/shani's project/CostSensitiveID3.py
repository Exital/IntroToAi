from ID3 import TreeNode, ID3Classifier, get_data_from_csv, build_id3_tree
from sklearn.model_selection import train_test_split


def tour_tree(node: TreeNode, row, data):
    """
    function that tours in the tree until we get to a leaf
    :param node: id3 tree node
    :type node: TreeNode
    :param row: number of a row in the dataframe
    :type row: int
    :return: detection
    """
    res = None
    while not node.is_leaf():
        feature_node = node.feature
        value = data[feature_node].iloc[row]
        if value <= node.threshold:
            node = node.left
        else:
            node = node.right
    res = node.diag
    return res


def check_leaf_and_groupValidation(node, group_validation):
    """
    Auxiliary function: function to check if the current node is leaf or the size of group validation is zero
    :param node: current node to check
    :param group_validation: group validation
    :return: True if it is a leaf or the size of group validation is zero
    """
    if node.is_leaf() or len(group_validation.index) == 0:
        return True
    else:
        return False


class costSensitiveID3Classifier(ID3Classifier):
    """
    Classifier for costSensitive
    """
    def __init__(self, FN_val=10, FP_val=1):
        super().__init__()
        self.FN_val = FN_val
        self.FP_val = FP_val
        self.group_validation = None

    def fit(self, x, y, test_size=0.56):
        """
        fits the classifier and creates a decision tree
        in the end, the function build prune tree to improve costs
        :param x: dataset
        :type x: dataframe
        :param y: dataset
        :type y: dataframe
        :param test_size: test size I checked and made my experiments to ge the best value
        :return: no return value
        """
        train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=test_size)
        data_train = train_x.copy()
        data_train["diagnosis"] = train_y
        check_data = test_x.copy()
        check_data["diagnosis"] = test_y
        self.group_validation = check_data
        id3tree = build_id3_tree(data=data_train)

        tree_to_prune = self.build_prune_tree(id3tree, self.group_validation)
        self.ID3TreeNode = tree_to_prune

    def build_prune_tree(self, tree_node: TreeNode, group_validation):
        """
        function to build prune tree
        :param tree_node: The tree to prune
        :type tree_node: TreeNode
        :param group_validation: group validation
        :return: pruned tree
        """
        # check if we reached to leaf or if there is no data group validation
        check = check_leaf_and_groupValidation(tree_node, group_validation)
        if check:
            return tree_node
        else:
            # dataframe slicing and creating tree
            right_group_validation = group_validation[group_validation[tree_node.feature] > tree_node.threshold]
            left_group_validation = group_validation[group_validation[tree_node.feature] <= tree_node.threshold]
            tree_node.right = self.build_prune_tree(tree_node.right, right_group_validation)
            tree_node.left = self.build_prune_tree(tree_node.left, left_group_validation)

            # check if is better to prune this node
            err_prune = 0
            err_no_prune = 0
            # check what is the best detection to give to a pruned node by it's cost
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
                # This part evaluates the loss for the prediction result of this node,
                # Comparing the error on the validation group with and without pruning
                # when res includes the cost of this prediction
                res_predict = tour_tree(tree_node, row, group_validation)
                detection_node_val = group_validation["diagnosis"].iloc[row]
                if detection_pruning != detection_node_val:
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
            # check if the error is small we will perform pruning
            if err_prune < err_no_prune:
                tree_node = update_node(tree_node, detection_pruning)
            return tree_node


def update_node(tree_node, detection_pruning):
    """
    function to update node in case the error is small, means when we decide to prune
    :param node: The node to update
    :type tree_node: TreeNode
    :param detection_pruning: value of diagnosis pruning
    :return: the tree_node after update his fields
    """
    tree_node.data = None
    tree_node.right = None
    tree_node.left = None
    tree_node.feature = None
    tree_node.diag = detection_pruning
    tree_node.threshold = None

    return tree_node


if __name__ == "__main__":
    # in the main part, first we get tha data from csv files, after that we create a costSensitiveID3Classifier,
    #  then fit the classifier. in the end we predict in test data set and printing the loss of better algorithm
    train_x, train_y = get_data_from_csv("train.csv")
    test_x, test_y = get_data_from_csv("test.csv")
    classifier = costSensitiveID3Classifier()
    classifier.fit(train_x, train_y)
    res_loss, res_accuracy = classifier.predict(test_x, test_y)
    print(res_loss)


