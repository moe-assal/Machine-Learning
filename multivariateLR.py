import numpy as np
from matplotlib import pyplot as plt
data = np.genfromtxt('ex1data2.txt', delimiter=',')
m = data.__len__()
X = np.concatenate([np.ones(shape=(m, 1)), data[:, 0:2]], axis=1)
features = X.shape[1] - 1   # exclude the constant
Y = data[:, features][None].T
theta = np.zeros(shape=(features + 1, 1))
alpha = 0.001


def max_min(x: np.ndarray):
    minimum = x.min()
    maximum = x.max()
    sd = maximum - minimum
    if sd == 0:
        return x
    return np.array(list(map(lambda n: (n-minimum)/sd, x)))


X = np.concatenate([X[:, 0][None].T, max_min(X[:, 1])[None].T, max_min(X[:, 2])[None].T], axis=1)


def compute_cost(x, y, weights):
    predictions = np.dot(x, weights)
    sigma = predictions - y
    sigma = sigma ** 2
    sigma = np.sum(sigma)
    return sigma/(2 * x.shape[0])


def gradient_descent(x, y, weights, alpha, iter):
    loss = []
    sample_size = x.shape[0]
    for i in range(iter):
        gradients = alpha / sample_size * np.dot(x.T, (np.dot(x, weights) - y))
        weights -= gradients
        loss.append(compute_cost(x, y, weights))
    return weights, loss


theta, j = gradient_descent(X, Y, theta, alpha, 150000)
# plt.plot(np.array(list(range(0, 150000))), j)
# plt.show()
theta1 = np.array([[669293.21223208],
                  [-504777.90398791],
                  [34952.07644931]])

pred = np.dot(X, theta)
z = np.concatenate([pred, Y], axis=1)
plt.scatter(pred, Y)
pred1 = np.dot(X, theta1)
z1 = np.concatenate([pred1, Y], axis=1)
plt.scatter(pred1, Y)

x = [0, 700000]
y = [0, 700000]
plt.plot(x, y, 'ro-')
plt.plot([pred.max(), Y.max()], [0, 0])
plt.show()
"""
[[ 665566.1795553 ]
 [-485458.7126757 ]
 [  14037.28273892]]
"""
