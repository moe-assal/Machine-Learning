"""
5/13/2020
TODO:
    show how the KMeans converge in animation
"""

import numpy as np
from math import sqrt
from random import sample
import matplotlib.pyplot as plt
data = np.genfromtxt('ex1data1.txt', delimiter=',')


class KMeans:
    def __init__(self):
        self.data = None
        self.cluster_points = None

    def fit(self, x: np.ndarray, cluster_num,
            trails=50, loss_break=0.05, max_iterations=50, train_types=None):
        """
        :param x: the actual data. It will be preprocessed by min-max
        :param cluster_num: the number of clusters
        :param trails: the number of times you want the algorithm to run before choosing the best solution
        :param loss_break: the minimum change in loss per iteration. If not satisfied, the algorithm will converge
        :param max_iterations: Don't ask
        :param train_types: only use this if you want KMode or KMedian.
            demo:
                it's a per feature array -> ['mean', 'median', 'mode']
        :return: the losses of different trails, ndarray
        """
        self.data = x
        if train_types is None:
            train_types = ["mean"] * self.data.shape[1]
        self.pre_process()
        losses = np.zeros(trails)
        trails_clusters = []
        for _ in range(trails):
            trails_clusters.append(self.fit_once(loss_break, max_iterations, cluster_num, train_types=train_types))
            losses[_] = self.loss()
        self.cluster_points = trails_clusters[losses.argmax()]
        return losses

    def fit_once(self, loss_break, max_iterations, cluster_num, x=None, cluster_points=None, train_types=None):
        """
        cluster points are used as initial points instead of choosing cluster points randomly from the data
        """
        if x is not None:
            self.data = x
        if train_types is None:
            train_types = ["median"] * self.data.shape[1]
        if cluster_points is None:
            self.cluster_points = np.array(sample(list(self.data), k=cluster_num))
        else:
            self.cluster_points = cluster_points
        past_loss = 0
        for _ in range(max_iterations):
            predictions = self.predict(self.data)
            for index, cluster in enumerate(self.cluster_points):
                samples = self.data[predictions == index]

                for j, feature in enumerate(samples.T):
                    result = 0
                    if train_types[j] == "mean":
                        result = feature.sum() / feature.shape[0]
                    elif train_types[j] == "median":
                        try:
                            result = feature[int(feature.shape[0] / 2)]
                        except IndexError:
                            print(feature)
                            print(feature.shape)
                            print(feature.shape[0])
                            raise
                    elif train_types[j] == "mode":
                        elements, counts = np.unique(feature, return_counts=True)
                        result = elements[counts.argmax()]
                    self.cluster_points[index][j] = result

            new_loss = self.loss()
            if _ == 0:
                continue
            elif abs(past_loss - new_loss) < loss_break:
                return self.cluster_points
            else:
                past_loss = new_loss
        return self.cluster_points

    def loss(self):
        """

        this calculates the mean of the euclidean distance between the cluster point of the class of all the vectors in
        self.data and the vectors
        note that this function doesn't affect the learning process
        :return: loss for the self.data with respect to the clusters self.cluster_points
        """

        classes = self.predict(self.data)
        loss = 0
        for index, cluster in enumerate(self.cluster_points):
            for vector in self.data[classes == index]:
                loss += self.euclidean_distance(cluster, vector)
        return loss / self.data.shape[0]

    def plot(self, index1, index2, predictions=False, colors=None, ax=None, show=True):
        """
        :param index1: locate the column from self.data to use as x-axis
        :param index2: locate the column from self.data to use as y-axis
        :param predictions: color the data according to their clusters
        :param colors: the colors list to use to color the clusters data points
        :param ax: matplotlib axis, if there is one
        :param show: whether or not to call plt.show()
        :return:
        """
        if ax is None:
            fig, ax = plt.subplots(nrows=1, ncols=1)
        if colors is None:
            colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']

        if predictions:
            classes = self.predict(self.data)
            classes_num = self.cluster_points.shape[0]

            for c in range(classes_num):
                indices = classes == c
                x_axis = self.data.T[index1][indices]
                y_axis = self.data.T[index2][indices]
                ax.scatter(x_axis, y_axis, c=colors[c])
        else:
            ax.scatter(self.data.T[index1], self.data.T[index2])

        if self.cluster_points is not None:
            ax.scatter(self.cluster_points.T[index1], self.cluster_points.T[index2], marker='x', c='r', s=64)
        if show:
            plt.show()

    def pre_process(self):
        """

        """
        def max_min(_x_: np.ndarray):
            minimum = _x_.min()
            maximum = _x_.max()
            sd = maximum - minimum
            if sd == 0:
                return np.array([0.5 for _ in range(_x_.shape[0])])
            return np.array(list(map(lambda n: (n - minimum) / sd, x)))
        for i, x in enumerate(self.data.T):
            self.data.T[i] = max_min(x)

    def predict(self, samples: np.ndarray):
        """
            complex memory inefficient way to predict class number for every vector in the samples
        """
        distances = np.array(list(map(lambda vector: [self.euclidean_distance(cluster_point, vector)
                                                      for cluster_point in self.cluster_points], samples)))
        return distances.argmin(axis=1)

    @staticmethod
    def euclidean_distance(vector1: np.ndarray, vector2: np.ndarray):
        distance = 0
        for component in range(vector1.shape[0]):
            distance += (vector1[component] - vector2[component]) ** 2
        return sqrt(distance)


k = KMeans()
lo = k.fit(data, 3, loss_break=0.0001, max_iterations=150, train_types=['median', 'mean'])
f, (ax1, ax2) = plt.subplots(nrows=1, ncols=2)
k.plot(0, 1, predictions=True, ax=ax1, show=False)
ax2.plot(np.array(list(range(lo.__len__()))), lo)
plt.show()
