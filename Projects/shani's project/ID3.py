from numpy import log2
import pandas as pd
DEFAULT_DIAGNOSIS = "M"


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
    temp = pd.read_csv(file)
    y = temp.iloc[:, 0:1]
    x = temp.iloc[:, 1:]
    return x, y

class TreeNode:
    """
    This nodes will make an ID3 tree.
    """
    def __init__(self, data: pd.DataFrame):
        """
        function to generate the whole id3tree
        :param data: pandas data frame with the diagnosis column
        :type data: panda's Dataframe
        """
        self.data = data
        self.right = None
        self.left = None
        self.feature = None
        self.diag = None
        self.threshold = None #slicing value

    def calculate_entropy(self, fraction):
        """
        :param fraction: The fraction of the entropy
        :type fraction: float
        :return: The entropy
        :rtype: float
        """
        entropy_val = -fraction * log(fraction) - ((1-fraction) * log(1-fraction))
        return entropy_val

    def is_leaf(self):
        """
        checks if that node is leaf
        :return: True if leaf
        :rtype: bool
        """
        if self.diag is not None:
            return True
        else:
            return False

    def IG_for_feature(self, feature_name):
        """
        checks what is the best threshold for feature and it's IG
        :param feature_name: The feature name to check
        :type feature_name: string
        :return: (IG for the feature, threshold)
        :rtype: tuple
        """
        detection = self.data["diagnosis"]   #diagnosis
        val_feature = self.data[feature_name]  #values
        val_feature_list = val_feature.tolist()
        # create the threshold's list
        threshold_list = self.cal_threshold_list(val_feature_list)

        ig_best_val = (float("-inf"), None)

        for divider in threshold_list:
            size_biger, biger_posetive, size_smaller, smaller_positive = self.update_values(val_feature, detection, divider)

             # calculate root's IG
            root_ig = smaller_positive + biger_posetive
            size_val_feature = len(val_feature)
            fraction = root_ig/size_val_feature
            root_entropy = self.calculate_entropy(fraction)

            # calculate left son's IG
            if size_smaller != 0:
                fraction = smaller_positive / size_smaller
            else:
                fraction = (biger_posetive+smaller_positive)/ size_val_feature
            left_entropy = self.calculate_entropy(fraction)

            # calculate right son's IG
            if size_biger != 0:
                 fraction = biger_posetive / size_biger
            else:
                fraction = (biger_posetive+smaller_positive)/size_val_feature
            right_entropy = self.calculate_entropy(fraction)

            ig_val = root_entropy - left_entropy * size_smaller / size_val_feature - (right_entropy*size_biger) / size_val_feature
            if ig_val >= ig_best_val[0]: ig_best_val = ig_val, divider

        if ig_best_val[1] is None:
            raise  ValueError("divider does not found")
        return  ig_best_val



    def update_values(self ,val_feature, detection, divider):
        """
        update values of size_biger, biger_posetive, size_smaller, smaller_positive
        :param val_feature: The values of features
        :param detection: The data from diagnosis's column
        :param divider: divider from threshold_list
        :return: size_biger, biger_posetive, size_smaller, smaller_positive
        :rtype: tuple
        """
        size_biger, biger_posetive, size_smaller, smaller_positive = 0, 0, 0, 0
        for i in range(len(val_feature)):
             if val_feature[i] <= divider:
                size_smaller += 1
                if detection[i] == "M":
                    smaller_positive += 1
             else:
                size_biger += 1
                if detection[i] == "M":
                    smaller_positive += 1

        return size_biger, biger_posetive, size_smaller, smaller_positive

    def cal_threshold_list(self, val_feature_list):
        """
        function to sum values of two sequential values
        :param val_feature_list: list of feature's values
        :return: list of the new values
        :rtype: list
        """
        soreted_values = sorted(val_feature_list, key=lambda x: x)
        list_res =[]
        for i in range(len(soreted_values)): #,maybe -1
            sum = soreted_values[i] + soreted_values[i+1]
            res = sum/2
            list_res.append(res)

        return list_res




    def choose_feature(self):
        """
        checks the IG of all features and chooses the best one.
        this one that will be chosen will be the best feature to slice(according to his value)
        :return: (feature name, threshold)
        :rtype: tuple
        """
        features = self.data.keys().tolist()
        features = features[:-1]

        best_ig = []

        for feature in features:
            ig, threshold = self.IG_for_feature(feature)
            value = ig, feature, threshold
            best_ig.append(value)
        best_ig.sort(key=lambda x: x[0], reverse=True)
        ig, feature, threshold = best_ig[0]
        return feature, threshold


    def all_same_data(self):
        """
        checks if the nodes data in some place is either "B" OR "M"
        :return: (None, False) OR (detection, True)
        :rtype: tuple
        """
        if len(self.data.index) == 0:
            return True, DEFAULT_DIAGNOSIS
        values = self.data["diagnosis"].tolist()
        last_val = None
        for val1,val2 in zip(values,values[1:]):
            if val1 != val2:
                return None, False
            else:
                last_val = val2
        return last_val, True


def build_id3_tree(data: pd.DataFrame) -> TreeNode:
    """
    This function will build a ID3 Tree
    :param data: the whole data set
    :type data: pd.Dataframe
    :return: An ID3 tree
    :rtype: TreeNode
    """
    node = TreeNode(data)
    diag, check_leaf = node.all_same_data()
    if check_leaf:
        node.diag = diag
    else:
        # if check_leaf = false, means that this is not a leaf. therefore we update feature for slicing and slicing val
        node.feature, _,node.threshold = node.choose_feature()
        #in this part we slice the dataframe
        right_data = node.data[node.data[node.feature] > node.threshold]
        left_data = node.data[node.data[node.feature] <= node.threshold]
        # Recursive part to create to build the ID3 Tree
        node.left = build_id3_tree(data=left_data)
        node.right = build_id3_tree(data=right_data)


class ID3Classifier:
    """
    Classifier for id3 predicting
    """
    def __init__(self):
        self.ID3TreeNode = None

    def fit(self, x, y):
        """
        fits the classifier and creates a decision tree
        :param x: data without diagnosis
        :type x: Dataframe
        :param y: diagnosis
        :type y: Dataframe
        """
        data = x.copy()
        data["diagnosis"] = y
        self.ID3TreeNode = TreeNode(data=data)

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
        def tour_tree(node: TreeNode, row):
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

        if self.ID3TreeNode is None:
            raise  ReferenceError("fit is missing")

        data = x.copy()
        data["diagnosis"] = y
        correct_predict = 0
        for row in range(len(data.index)):
            if tour_tree(self.ID3TreeNode, row) == data["diagnosis"].iloc[row]:
                    correct_predict += 1
        return correct_predict / len(data.index)
