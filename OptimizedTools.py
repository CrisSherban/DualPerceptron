from numba import jit, prange
import numpy as np


@jit(nopython=True, parallel=True)
def fit(train_x, train_y, gram_mat, epochs):
    # calculates firstly R, which represents the distance of the furthest element
    norms = np.zeros(len(train_y), dtype=np.float32)
    for i in prange(0, len(train_y)):
        norms[i] = np.linalg.norm(train_x[i])
    R = max(norms)
    print("Furthest element distance: ", R)

    alpha = np.zeros(len(train_y), dtype=np.float32)
    b = 0

    print("Fitting data...")
    for k in range(epochs):
        # print("alpha: ", alpha, " epoch: ", k, "\n")

        mistakes = 0
        for i in range(len(train_y)):
            summation = 0
            for j in range(len(train_y)):
                summation += alpha[j] * train_y[j] * gram_mat[i][j] + b

            if train_y[i] * summation <= 0:
                alpha[i] = alpha[i] + 1
                b += train_y[i] * (R ** 2)
                mistakes += 1

    print("Finished fitting!")
    return alpha, b


@jit(nopython=True, parallel=False)
def predict_set(train_x, train_y, test_x, test_y, alpha, b, kernel, dim, sigma):
    predicted = np.zeros(len(test_y), dtype=np.float32)
    print("Testing the test_set...")
    for i in range(len(test_y)):
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

    return predicted


@jit(nopython=True, parallel=False)
def calculate_gram_mat(train_set, kernel, dim, sigma):
    print("Calculating Gram Matrix...")
    length = len(train_set[:, 0])
    gram_mat = np.zeros(shape=(length, length), dtype=np.float32)

    for i in range(length):
        for j in range(length):
            if kernel == 1:
                gram_mat[i][j] = np.float32(
                    np.dot(train_set[i], train_set[j]))
            elif kernel == 2:
                gram_mat[i][j] = np.float32(
                    (np.dot(train_set[i], train_set[j]) + 1) ** dim)
            else:
                gram_mat[i][j] = np.float32(
                    np.exp(-(np.linalg.norm(train_set[i]
                                            - train_set[j]) ** 2) / (2.0 * (sigma ** 2))))

    print("Gram Matrix calculated")
    return gram_mat
