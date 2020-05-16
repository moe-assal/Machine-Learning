import numpy as np
from multivariateLR import compute_cost, max_min
data = np.genfromtxt('ex1data2.txt', delimiter=',')
m = data.__len__()
X = np.concatenate([np.ones(shape=(m, 1)), data[:, 0:2]], axis=1)
features = X.shape[1] - 1   # exclude the constant
Y = data[:, features][None].T
X = np.concatenate([X[:, 0][None].T, max_min(X[:, 1])[None].T, max_min(X[:, 2])[None].T], axis=1)
theta = np.dot(np.linalg.inv(np.dot(X.T, X)), np.dot(X.T, Y))
print(compute_cost(X, Y, theta))
print(theta)
"""
[[ 669293.21223208]
 [-504777.90398791]
 [  34952.07644931]]
"""
