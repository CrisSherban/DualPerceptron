import numpy as np
import OptimizedTools as op
from numba import jit, prange


# Some inspiration is taken from:
# https://github.com/yihui-he/kernel-perceptron/blob/master/kerpercep.py


class Perceptron:
    dim = None
    sigma = None
    kernel = None

    def __init__(self, dataset_path, train_x, train_y, _kernel_=1, epochs=500, _dim_=2, _sigma_=5.12):
        self.epochs = epochs
        self.train_x = train_x
        self.train_y = train_y
        self.gram_mat = None
        self.R = None
        self.b = None
        self.alpha = None
        self.dataset_path = dataset_path
        Perceptron.dim = _dim_
        Perceptron.sigma = _sigma_
        Perceptron.kernel = _kernel_

    # Predicts a test_set and returns the accuracy
    # The computation is delegated to OptimizedTools
    def predict_set(self, test_x, test_y):
        actual, predicted = op.predict_set(self.train_x,
                                           self.train_y,
                                           test_x,
                                           test_y,
                                           alpha=self.alpha,
                                           b=self.b,
                                           kernel=Perceptron.kernel,
                                           dim=Perceptron.dim,
                                           sigma=Perceptron.sigma)
        return self.accuracy(actual, predicted)

    # Classifies the given element
    def predict_element(self, an_element):

        sv = np.loadtxt(str(self.dataset_path) + "/alpha_and_b_"
                        + str(Perceptron.kernel) + ".csv", delimiter=',')

        types = np.loadtxt(str(self.dataset_path) + "/types.csv", delimiter=',', dtype=str)
        types = {types[0, 0]: types[0, 1], types[1, 0]: types[1, 1]}

        alpha = sv[:-1]
        b = sv[-1]

        summation = 0
        for j in range(len(self.train_y)):
            if Perceptron.kernel == 1:
                summation += (alpha[j] * self.train_y[j] *
                              np.dot(self.train_x[j], an_element) + b)
            elif Perceptron.kernel == 2:
                summation += (alpha[j] * self.train_y[j] *
                              (np.dot(self.train_x[j], an_element) + 1) ** Perceptron.dim + b)
            else:
                summation += (alpha[j] * self.train_y[j] *
                              np.exp(-(np.linalg.norm(self.train_x[j] - an_element) ** 2)
                                     / (2.0 * (Perceptron.sigma ** 2))) + b)

        if summation > 0:
            activation = 1
        else:
            activation = -1
        print("The given element is: ", types[str(activation)], " AKA: ", activation)
        return activation

    # The computation is delegated to OptimizedTools
    def fit(self):
        gram_mat_filename = str(self.dataset_path) + "/gram_mat_" + str(Perceptron.kernel) + ".csv"
        try:
            print("Trying to load the Gram Matrix...")
            self.gram_mat = np.loadtxt(gram_mat_filename,
                                       dtype=np.float32,
                                       delimiter=',')
            print("Gram Matrix loaded successfully")
        except:
            print("Gram Matrix doesn't exist...")
            self.gram_mat = Perceptron.calculate_and_save_gram_mat(np.around(np.float32(self.train_x), decimals=2),
                                                                   len(self.train_y),
                                                                   gram_mat_filename)
            print("Gram Matrix created successfully")

        alpha, b = op.fit(self.train_x, self.train_y, self.epochs, np.float32(self.gram_mat))

        self.alpha = alpha
        self.b = b
        np.savetxt(str(self.dataset_path) + "/alpha_and_b_" + str(Perceptron.kernel)
                   + ".csv", X=np.append(alpha, b), delimiter=',')

    # Calculates accuracy percentage
    @staticmethod
    def accuracy(actual, predicted):
        correct = 0
        for i in range(len(actual)):
            if actual[i] == predicted[i]:
                correct += 1
        return correct / float(len(actual)) * 100.0

    @staticmethod
    @jit(nopython=True, parallel=True)
    def calculate_gram_mat(train_set, length, gram_mat, kernel, dim, sigma):
        print("Calculating Gram Matrix...")
        for i in prange(length):
            for j in range(length):
                if kernel == 1:
                    gram_mat[i][j] = np.float32(
                        np.dot(train_set[i].ravel(), train_set[j].ravel()))
                elif kernel == 2:
                    gram_mat[i][j] = np.float32(
                        (np.dot(train_set[i].ravel(), train_set[j].ravel()) + 1) ** dim)
                else:
                    gram_mat[i][j] = np.float32(
                        np.exp(-(np.linalg.norm(train_set[i].ravel()
                                                - train_set[j].ravel()) ** 2) / (2.0 * (sigma ** 2))))

        print("Gram Matrix calculated")
        return gram_mat

    # wrapper method to allow np.savetxt which is not supported yet in Numba
    @staticmethod
    def calculate_and_save_gram_mat(train_set, length, csv_file_name):
        gram_mat_init = np.empty([length, length], dtype=np.float32)

        gram_mat = Perceptron.calculate_gram_mat(train_set, length, gram_mat_init,
                                                 Perceptron.kernel,
                                                 Perceptron.dim,
                                                 Perceptron.sigma)

        # gram_mat = Perceptron.flatten_gram_mat(gram_mat, length)

        np.savetxt(fname=csv_file_name, X=gram_mat, delimiter=',')
        return gram_mat

    @staticmethod
    @jit(nopython=True, parallel=True)
    def flatten_gram_mat(gram_mat, length):
        print("Flattening Gram Matrix...")
        max_col_val = np.zeros(length, dtype=np.float32)

        # finding max element of each column
        for j in prange(length):
            max_col_val[j] = 0
            for i in range(length):
                if i != j and max_col_val[j] < gram_mat[i][j]:
                    max_col_val[j] = gram_mat[i][j]

        # flattens the values
        for j in prange(length):
            for i in range(length):
                if i != j:
                    gram_mat[i][j] = gram_mat[i][j] / max_col_val[j]
                else:
                    gram_mat[i][j] = 1

        print("Finished flattening Gram Matrix")
        return gram_mat


###########################################################
'''
# KERNELS #
def linear_ker(x, z):
    return np.dot(x, z)


def polynomial_ker(x, z, dim=2):
    return (np.dot(x, z) + 1) ** dim


def rbf_ker(x, z, sigma=5.12):
    return np.exp(-(np.linalg.norm(x - z) ** 2) / (2.0 * (sigma ** 2)))
'''
