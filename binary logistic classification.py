"""
4/7/2020
"""
import numpy as np
import matplotlib.pyplot as plt
data = np.genfromtxt('nba_logreg.csv', delimiter=',')
with open('nba_logreg.csv', 'r') as file:
    features = file.readline().split(',')
    features = ['CONST'] + features[1:20]
theta = np.zeros(shape=(features.__len__(), ))
size = 1340
X = np.concatenate((np.ones(shape=(size, 1)), data[1:, 1:20]), axis=1)
Y = np.array(data[1:, 20][None].T, dtype=int)   # the '1:' is to exclude the feature name
alpha = 1   # learning rate
# fix empty '3P%' fields
X.T[9] = np.divide(X.T[7], X.T[8], out=np.zeros_like(X.T[7]), where=X.T[8] != 0) * 100


# pre-process data
def max_min(x: np.ndarray):
    minimum = x.min()
    maximum = x.max()
    sd = maximum - minimum
    if sd == 0:
        return x
    return np.array(list(map(lambda n: (n-minimum)/sd, x)))


for i, x_column in enumerate(X.T):
    X.T[i] = max_min(x_column)


def plot_two(index_1: int, index_2: int, x: np.ndarray, y: np.ndarray, feature=None, decision_boundary=False, ax=None):
    """
    needs coordinates testing
    :param index_1: index of x_axis located in x data set of shape(size, n)
    :param index_2: index of y_axis located in x data ser of shape(size, n)
    :param x: No need
    :param y: binary ndarray of shape(size, 1)
    :param feature: hhhh
    :return:
    """
    if ax is None:
        fig, ax = plt.subplots()
    x_axis = x[:, index_1]
    y_axis = x[:, index_2]
    y = y.T[0]
    if feature is not None:
        ax.set_xlabel(features[index_1])
        ax.set_ylabel(features[index_2])
    ax.scatter(x_axis[y.astype(bool)], y_axis[y.astype(bool)], c='r', label='one')
    ax.scatter(x_axis[np.invert(y.astype(bool))], y_axis[np.invert(y.astype(bool))], c='b', label='zero')
    ax.legend()
    ax.set_xlim(xmin=-0.2, xmax=1.2)
    ax.set_ylim(ymin=-0.2, ymax=1.2)
    if decision_boundary is True:
        x_prime = np.zeros(shape=(X.shape[1], ))

        for index, column in enumerate(X.T):
            x_prime[index] = column.sum() / column.shape
        x_prime[index_1] = 0
        x_prime[index_2] = 0
        const = 0.5 - np.dot(x_prime, theta)
        x_axis = np.linspace(-0.2, 1.2, 100)
        y_axis = (const - x_axis * theta[index_1]) / theta[index_2]
        ax.plot(x_axis, y_axis)


def plot_two_predictions(index_1: int, index_2: int, x: np.ndarray, feature=None, decision_boundary=False, ax=None):
    """
    needs coordinates testing
    :param index_1: index of x_axis located in x data set of shape(size, n)
    :param index_2: index of y_axis located in x data ser of shape(size, n)
    :param x: No need
    :param y: binary ndarray of shape(size, 1)
    :param feature: hhhh
    :return:
    """
    if ax is None:
        fig, ax = plt.subplots()
    x_axis = x[:, index_1]
    y_axis = x[:, index_2]
    sigmoid_v = np.vectorize(sigmoid)
    # predictions are the probabilities that it is in class 1
    y = sigmoid_v(np.dot(X, theta))
    y = y >= 0.5
    if feature is not None:
        ax.set_xlabel(features[index_1])
        ax.set_ylabel(features[index_2])
    ax.scatter(x_axis[y.astype(bool)], y_axis[y.astype(bool)], c='r', label='one')
    ax.scatter(x_axis[np.invert(y.astype(bool))], y_axis[np.invert(y.astype(bool))], c='b', label='zero')
    ax.legend()
    ax.set_xlim(xmin=-0.2, xmax=1.2)
    ax.set_ylim(ymin=-0.2, ymax=1.2)
    if decision_boundary is True:
        x_prime = np.zeros(shape=(X.shape[1], ))

        for index, column in enumerate(X.T):
            x_prime[index] = column.sum() / column.shape
        x_prime[index_1] = 0
        x_prime[index_2] = 0
        const = 0.5 - np.dot(x_prime, theta)
        x_axis = np.linspace(-0.2, 1.2, 100)
        y_axis = (const - x_axis * theta[index_1]) / theta[index_2]
        ax.plot(x_axis, y_axis)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


log = np.vectorize(np.log)


def gradient_descent():
    global theta
    # define vectorized sigmoid
    sigmoid_v = np.vectorize(sigmoid)
    # predictions are the probabilities that it is in class 1
    predictions = sigmoid_v(np.dot(X, theta))
    theta = theta - (alpha / size) * np.dot(X.T, (predictions - Y.T[0]))
    return theta


def loss():
    y = Y.T[0]
    sigmoid_v = np.vectorize(sigmoid)
    # predictions are the probabilities that it is in class 1
    predictions = sigmoid_v(np.dot(X, theta))
    j = - y * log(predictions) - (1 - y) * log(1 - predictions)
    j = j.sum() / size
    return j


def train(iterations):
    J = []
    global alpha
    for _ in range(iterations):
        if _ == 300:
            alpha = 3
        gradient_descent()
        J.append(loss())
    return J


if __name__ == '__main__':

    steps = 800
    # add more features
    features.append('GP * 3PA')
    o = np.array([X.T[1] * X.T[8]]).T
    X = np.column_stack((X, o))
    features.append('3PA + MIN')
    o = np.array([X.T[8] * X.T[2]]).T
    X = np.column_stack((X, o))
    features.append('MIN^2 * FIM')
    o = np.array([X.T[2] ** 2 * X.T[10]]).T
    X = np.column_stack((X, o))
    features.append('STL * FGA')
    o = np.array([X.T[17] * X.T[5]]).T
    X = np.column_stack((X, o))
    theta = np.zeros(shape=(features.__len__(), ))

    j = train(steps)
    plt.plot(np.array(range(steps)), j)
    plt.show()

    # show feature strength
    for i, g in enumerate(features):
        print(g + '\t', theta[i])
    fig, (ax1, ax2) = plt.subplots(ncols=2, nrows=1)
    plot_two(2, -2, X, Y, features, True, ax1)
    plot_two_predictions(2, -2, X, features, True, ax2)
    plt.show()

    # for i in range(features.__len__()):
    #     for j in range(features.__len__()):
    #         if j != i and i != 0 and j != 0:
    #             print(i, '\t', j)
    #             plot_two(i, j, X, Y, features)

