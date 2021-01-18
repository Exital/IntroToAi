# TODO import what you need

def slice_data(data, fraction):
    """
    slices the data and return a fraction of it
    :param data: a dataframe
    :type data: dataframe
    :param fraction: number between [0,1]
    :type fraction: float
    :return: a fraction of the data
    :rtype: dataframe
    """
    sliced_data = None
    # -------- Your code -------
    # TODO write a function that take the dataframe and randomly return a fraction of it.
    # TODO lets say the data has 100 sample and fraction is 0.5 then you return 50 samples randomly.
    # --------------------------
    return sliced_data


def get_centroid(data):
    """
    That function gets a dataframe and returns the vector of mean.
    :param data: a dataframe
    :type data: dataframe
    :return: a mean vector
    :rtype: series
    """
    centroid = None
    # -------- Your code -------
    # TODO create a function that takes the mean out of the whole data to make a mean vector.
    # --------------------------
    return centroid

def distance_between_vectors(v1, v2):
    """
    this function get 2 vectors and return euclidean distance between them
    :param v1: series vector
    :type v1: series
    :param v2: series vector
    :type v2: series
    :return: a distance between the 2 vectors
    :rtype: float
    """
    distance = None
    # -------- Your code -------
    # TODO calculate euclidean distance of the 2 vectors given.
    # --------------------------
    return distance


class KNNForestClassifier:
    """
    This is the classifier for the forest
    """
    def __init__(self, N=20, k=13):
        self.forest = []
        self.centroids = []
        self.N = N
        self.k = k

    def fit(self, x, y):
        pass

    def predict(self, x, y):
        pass
