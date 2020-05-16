import numpy as np
from matplotlib import pyplot as plt
data = np.genfromtxt('ex1data1.txt', delimiter=',')
m = data.__len__()
X = np.concatenate([np.ones(shape=(m, 1)), data[:, 0][None].T], axis=1)
Y = data[:, 1][None].T
theta = np.zeros(shape=(2, 1))
alpha = 0.0001


def compute_cost(x, y, weights):
    predictions = np.dot(x, weights)
    sigma = predictions - y
    sigma = sigma ** 2
    sigma = np.sum(sigma)
    return sigma/(2 * x.shape[0])


def compute_regularized_cost(x, y, weights, regular):
    predictions = np.dot(x, weights)
    sigma = predictions - y
    sigma = sigma ** 2
    sigma = np.sum(sigma)
    cost = sigma + np.sum(regular * (weights ** 2))
    return cost / (2 * x.shape[0])


def plot(x: np.ndarray, y, weights=None, ax=None):
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
    ax.scatter(x.T[1], y)
    if weights is not None:
        x_axis = np.linspace(x.min(), x.max())[None].T
        X_axis = np.concatenate([np.ones(shape=(x_axis.shape[0], 1)), x_axis], axis=1)
        y_axis = np.dot(X_axis, weights).T[0]
        ax.plot(x_axis.T[0], y_axis)
    return ax


def gradient_descent(x, y, weights, alpha, iter):
    loss = []
    sample_size = x.shape[0]
    for i in range(iter):
        gradients = alpha / sample_size * np.dot(x.T, (np.dot(x, weights) - y))
        weights -= gradients
        loss.append(compute_cost(x, y, weights))
    return weights, loss


def regularized_gradient_descent(x, y, weights, alpha, iter, regular):
    loss = []
    loss_reg = []
    sample_size = x.shape[0]
    for i in range(iter):
        gradients = 1 / sample_size * np.dot(x.T, (np.dot(x, weights) - y)) + regular / sample_size * weights
        weights -= alpha * gradients
        loss_reg.append(compute_regularized_cost(x, y, weights, regular))
        loss.append(compute_cost(x, y, weights))
    return weights, loss


if __name__ == '__main__':
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(nrows=2, ncols=2)
    theta, loss = regularized_gradient_descent(X, Y, theta, alpha, 15000, 0.1)
    plot(X, Y, theta, ax1)
    ax3.plot(loss)

    theta = np.zeros(shape=(2, 1))
    theta, loss = gradient_descent(X, Y, theta, alpha, 15000)
    plot(X, Y, theta, ax2)
    ax4.plot(loss)
    plt.show()
