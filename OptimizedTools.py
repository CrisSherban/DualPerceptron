from numba import jit, prange
import numpy as np


@jit(nopython=True, parallel=True)
def fit(train_x, train_y, epochs, gram_mat):
    # calculates R, which represents the distance of the furthest element
    norms = np.zeros(len(train_y), dtype=np.float32)
    for i in range(0, len(train_y)):
        norms[i] = np.linalg.norm(train_x[i])
    R = max(norms)

    alpha = np.zeros(len(train_y), dtype=np.float32)

    b = 0
    mistakes = 0

    print("Furthest element distance: ", R)
    print("Fitting data...")
    for k in prange(epochs):
        # print("alpha: ", alpha, " epoch: ", k, "\n")
        for i in range(len(train_y)):
            summation = 0
            for j in range(len(train_y)):
                summation += alpha[j] * train_y[j] * gram_mat[i][j] + b

            if train_y[i] * summation <= 0:
                alpha[i] = alpha[i] + 1
                b += train_y[i] * (R ** 2)
                mistakes += 1

    print("Finished fitting, mistakes made: ", mistakes)
    return alpha, b


@jit(nopython=True, parallel=True)
def predict_set(train_x, train_y, test_x, test_y, alpha, b, kernel, dim, sigma):
    predicted = np.zeros(len(test_y), dtype=np.float32)
    actual = test_y
    print("Testing the test_set...")
    for i in prange(len(test_y)):
        summation = 0
        for j in range(len(test_y)):
            if kernel == 1:
                summation += (alpha[j] * train_y[j] *
                              np.dot(train_x[j], test_x[i]) + b)
            elif kernel == 2:
                summation += (alpha[j] * train_y[j] *
                              (np.dot(train_x[j], test_x[i]) + 1) ** dim + b)
            else:
                summation += (alpha[j] * train_y[j] *
                              np.exp(-(np.linalg.norm(train_x[j] - test_x[i]) ** 2)
                                     / (2.0 * (sigma ** 2))) + b)
        if summation > 0:
            activation = 1
        else:
            activation = -1
        predicted[i] = activation

    return actual, predicted
