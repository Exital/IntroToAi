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
        TODO
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

        threshold_list = self.threshold_list(val_feature_list)
        ig_best_val = (float("-inf"), None)

        for divide in threshold_list:
            # size_biger, biger_posetive, size_smaller, smaller_positive = 0, 0 ,0, 0

        feature_name =

    # def update_values(self, val_feature, detection):
    #     for i in range(len(val_feature)):
    #         if val_feature[i] <= detection[i]
    #             si






    def threshold_list(self, val_feature_list):
        soreted_valfe = sorted(val_feature_list, key=lambda x: x)
        list_res =[]
        for i in range(len(soreted_valfe)): #,maybe -1
            sum = soreted_valfe[i] + soreted_valfe[i+1]
            res = sum/2
            list_res.append(res)

        return list_res




    def choose_feature(self):
        """
        checks the IG of all features and chooses the best one.
        :return: (feature name, threshold)
        :rtype: tuple
        """
        features = self.data.keys().tolist()
        features = features[:-1]

        best_ig = []

        for feature in features:
            ig, threshold = self._IG_for_feature(feature)
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
        def walk_the_tree(node: TreeNode, row):
            # if node.is_leaf():
            #     return node.diag
            # else:
            #     feature = node.feature
            #     value = data[feature].iloc[row]
            #     if value <= node.slicing_val:
            #         return walk_the_tree(node.left, row)
            #     else:
            #         return walk_the_tree(node.right, row)
            res = None
            while not node.is_leaf():
                feature_node = node.feature
                value = data[feature].iloc[row]
                if value <= node.threshold:
                    node = node.left
                else:
                    node = node.right
            res = node.diag
            return res


        if cur_node.is_leaf():
             return cur_node.diag
        else:
            feature_node = cur_node.feature
             who is data
