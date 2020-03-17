import numpy as np
from numpy import savetxt, loadtxt


# Some inspiration is taken from:
# https://github.com/yihui-he/kernel-perceptron/blob/master/kerpercep.py

class Perceptron:

    def __init__(self, gram_mat, epochs=20):
        self.epochs = epochs
        self.train = None
        self.gram_mat = gram_mat
        self.alpha = None
        self.R = None
        self.b = None

    # Predicts a test_set and returns the accuracy
    def predict_set(self, test_set):
        actual = test_set['y']
        predicted = []
        for i in range(0, len(test_set['y'])):
            predicted.append(self.predict(test_set['x'][i]))
        return self.accuracy(actual, predicted)

    def predict(self, an_element):
        summation = 0

        for j in range(0, self.train['n_rows']):
            summation += (self.alpha[j] * self.train['y'][j] *
                          np.dot(self.train['x'][j], an_element) + self.b)

        if summation > 0:
            activation = 1
        else:
            activation = -1
        return activation

    def fit(self, train_set):
        self.train = train_set
        self.alpha = np.zeros(self.train['n_rows'])
        self.b = 0

        # calculates R #
        norms = []

        for i in range(0, self.train['n_rows']):
            norm_of_i = np.linalg.norm(self.train['x'][i])
            norms.append(norm_of_i)

        self.R = max(norms)
        ###############

        for k in range(0, self.epochs):
            print("alpha: ", self.alpha, " epoch: " + str(k))
            for i in range(self.train['n_rows']):
                if self.train['y'][i] * sum(self.alpha * self.train['y'] * self.gram_mat[i] + self.b) <= 0:
                    self.alpha[i] = self.alpha[i] + 1
                    self.b += self.train['y'][i] * (self.R ** 2)

    # Calculates accuracy percentage
    @staticmethod
    def accuracy(actual, predicted):
        correct = 0
        for i in range(len(actual)):
            if actual[i] == predicted[i]:
                correct += 1
        return correct / float(len(actual)) * 100.0
