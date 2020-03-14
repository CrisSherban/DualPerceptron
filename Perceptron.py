import numpy as np


class Perceptron:

    def __init__(self, kernel="linear_ker", threshold=250, dim=None, sigma=None):
        self.threshold = threshold
        self.sigma = sigma
        self.dim = dim
        self.kernel = kernel
        self.X = None
        self.y = None
        self.alpha = None
        self.R = None
        self.b = None

    # Predicts a TestSet and returns the accuracy
    def predict_set(self, X, y):
        actual = y
        predicted = []
        for i in range(0, len(y)):
            predicted.append(self.predict(X[i]))
        return self.accuracy(actual, predicted)

    def predict(self, xi):
        summation = 0

        for j in range(0, len(self.X[0])):
            if self.kernel == "rbf_ker":
                summation += (self.alpha[j] * self.y[j] * self.rbf_ker(self.X[j], xi, self.sigma)) + self.b
            if self.kernel == "polynomial_ker":
                summation += (self.alpha[j] * self.y[j] * self.polynomial_ker(self.X[j], xi, self.dim)) + self.b
            else:
                summation += (self.alpha[j] * self.y[j] * self.linear_ker(self.X[j], xi)) + self.b

        if summation > 0:
            activation = 1
        else:
            activation = -1
        return activation

    def fit(self, X, y):
        self.X = X
        self.y = y
        self.alpha = np.zeros(len(X[0]))
        self.b = 0

        norms = []

        for i in range(0, len(X[0])):
            norm_of_i = np.linalg.norm(X[i])
            norms.append(norm_of_i)

        self.R = max(norms)

        for k in range(0, self.threshold):
            for i in range(len(X[0])):
                if y[i] * self.predict(X[i]) <= 0:
                    # print(self.alpha)
                    self.alpha[i] = self.alpha[i] + 1
                    self.b += y[i] * (self.R ** 2)

    # Calculates accuracy percentage
    def accuracy(self, actual, predicted):
        correct = 0
        for i in range(len(actual)):
            if actual[i] == predicted[i]:
                correct += 1
        return correct / float(len(actual)) * 100.0

    def linear_ker(self, x, z):
        return np.dot(x, z)

    def polynomial_ker(self, x, z, dim):
        return (np.dot(x, z)) ** dim

    def rbf_ker(self, x, z, sigma):
        return np.exp(-(np.linalg.norm(x - z) ** 2) / (2.0 * (sigma ** 2)))
