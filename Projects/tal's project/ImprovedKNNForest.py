from utils import csv2xy
import argparse
import matplotlib.pyplot as plt
from KNNForest import KNNForestClassifier


class ImprovedKNNForestClassifier:
    """
    This is a classifier that uses the regular KNNForestClassifier but normalizes the data before it uses it.
    """
    def __init__(self, N=20, k=7):
        self.knn_classifier = KNNForestClassifier(N=N, k=k)
        self.scaling_consts = []

    def scalar(self, series, mean, deviation):
        """
        This functions scales a series of that to normalized form as we learnt in class.
        :param series: a pd series
        :type series: series
        :param mean: the mean of the series
        :type mean: float
        :param deviation: the std of the series
        :type deviation: float
        :return: a normalized series
        :rtype: a pd series
        """
        for index, value in series.items():
            series.loc[index] = (value - mean) / deviation
        return series

    def fit_scaling(self, x):
        """
        The scaling that has to be done with fitting + saves the values for the prediction fitting.
        :param x: the data
        :type x: dataframe
        :return: a normalized dataframe
        :rtype: dataframe
        """
        # clearing the scaling consts if there is another fit for newer data
        self.scaling_consts = []
        features = x.keys().tolist()
        x_scaled = x.copy()
        for feature in features:
            mean_val, std_val = x_scaled[feature].mean(), x_scaled[feature].std()
            scaling_const = feature, mean_val, std_val
            # saving the values in order to normalize predict data
            self.scaling_consts.append(scaling_const)
            x_scaled[feature] = self.scalar(x_scaled[feature], mean_val, std_val)
        return x_scaled

    def predict_scaling(self, x):
        """
        The scaling that has to be done for prediction using the values that we normalized the fit data.
        Exactly as we have learnt in class.
        :param x: the dataframe that needs to be normalized
        :type x: dataframe
        :return: a normalized dataframe
        :rtype: dataframe
        """
        x_scaled = x.copy()
        for feature, mean_val, std_val in self.scaling_consts:
            x_scaled[feature] = self.scalar(x_scaled[feature], mean_val, std_val)
        return x_scaled

    def fit(self, x, y):
        """
        regular fit just using the normalized data.
        :param x: The data to fit
        :type x: dataframe
        :param y: the labels of that data
        :type y: dataframe
        """
        scaled_x = self.fit_scaling(x)
        self.knn_classifier.fit(scaled_x, y)

    def predict(self, x, y):
        """
        The predict function which will scale the data and then use the regular predictions.
        :param x: the data to be predicted on
        :type x: dataframe
        :param y: the labels of that data
        :type y: dataframe
        :return: (accuracy, loss)
        :rtype: tuple
        """
        scaled_x = self.predict_scaling(x)
        return self.knn_classifier.predict(scaled_x, y)


def experiment(train_x, train_y, test_x, test_y, iterations=5, N=20, k=7, verbose=False):
    accuracy = []
    improved_accuracy = []
    classifier = KNNForestClassifier(N=N, k=k)
    improved_classifier = ImprovedKNNForestClassifier(N=N, k=k)

    for i in range(iterations):
        classifier.fit(train_x, train_y)
        acc, loss = classifier.predict(test_x, test_y)
        accuracy.append(acc)
        improved_classifier.fit(train_x, train_y)
        acc, loss = improved_classifier.predict(test_x, test_y)
        improved_accuracy.append(acc)
    if args.verbose:
        iterations = [i for i in range(iterations)]
        plt.xlabel("Number of iteration")
        plt.ylabel("Accuracy of that iteration")
        plt.plot(iterations, accuracy, label="KNNForest")
        plt.plot(iterations, improved_accuracy, label="ImprovedKNNForest")
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
        plt.show()

        print(f"The average accuracy of KNNForest is {sum(accuracy)/len(accuracy)}")
        print(f"The average accuracy of ImprovedKNNForest is {sum(improved_accuracy)/len(improved_accuracy)}")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('-v', '-verbose', dest="verbose", action='store_true', help="Show more information")
    args = parser.parse_args()

    # retrieving the data from the csv files
    train_x, train_y = csv2xy("train.csv")
    test_x, test_y = csv2xy("test.csv")
    experiment(train_x, train_y, test_x, test_y, verbose=args.verbose, N=20, k=7, iterations=20)
