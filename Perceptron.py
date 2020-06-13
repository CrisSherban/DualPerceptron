import numpy as np
import OptimizedTools as op


# Some inspiration is taken from:
# https://github.com/yihui-he/kernel-perceptron/blob/master/kerpercep.py

class Perceptron:
    dim = None
    sigma = None
    kernel = None

    def __init__(self, dataset_path, train_x, train_y,
                 _kernel_=1, epochs=10, _dim_=2, _sigma_=5):
        """
            Loads or creates all the necessary files like the Gram Matrix and sets
            all necessary variables in the class. It also looks for a file containing
            pre-trained data like "alpha"s and "b"s.
        :param dataset_path: String
                        path were the already prepared dataset is situated
        :param train_x: array_like
                        a train set in the form of an ndarray containing all test elements
        :param train_y: array_like
                        an ndarray containing binary classes, it can take values 1 or -1
                        it has the same number of rows as train_x
        :param _kernel_: int
                        1: Linear, 2: Polynomial, 3: Gaussian
        :param epochs: int
                        maximum number of epochs
        :param _dim_: int
        :param _sigma_: float
        """

        self.epochs = epochs
        self.train_x = np.ascontiguousarray(train_x, dtype=np.float32)
        self.train_y = np.ascontiguousarray(train_y, dtype=np.float32)
        self.gram_mat = None
        self.R = None
        self.dataset_path = dataset_path
        Perceptron.dim = _dim_
        Perceptron.sigma = _sigma_
        Perceptron.kernel = _kernel_

        try:
            sv = np.loadtxt(str(self.dataset_path) + "/alpha_and_b_"
                            + str(Perceptron.kernel) + ".csv", delimiter=',')

            self.alpha = sv[:-1]
            self.b = sv[-1]
        except:
            print("Alphas an b yet to compute")

        gram_mat_filename = str(self.dataset_path) + "/gram_mat_" + str(Perceptron.kernel) + "_" + \
                            str(Perceptron.dim) + "_" + str(Perceptron.sigma) + ".npy"

        try:
            print("Trying to load the Gram Matrix...")
            self.gram_mat = np.load(gram_mat_filename)
            print("Gram Matrix loaded successfully")
        except:
            print("Gram Matrix doesn't exist...")
            self.gram_mat = np.ascontiguousarray(Perceptron.calculate_and_save_gram_mat(self.train_x,
                                                                                        gram_mat_filename),
                                                 dtype=np.float32)

            print("Gram Matrix created successfully")

    def predict_set(self, test_x, test_y):
        """
            Predicts a test_set and returns the accuracy
            The computation is delegated to OptimizedTools
        :param test_x: array_like
                        a test set in the form of an ndarray containing all test elements

        :param test_y: array_like
                        an ndarray containing binary classes, it can take values 1 or -1
        :return: ndarray
                        an ndarray containing predicted values that can be 1 or -1
        """

        return op.predict_set(self.train_x,
                              self.train_y,
                              np.ascontiguousarray(test_x, dtype=np.float32),
                              np.ascontiguousarray(test_y, dtype=np.float32),
                              alpha=self.alpha,
                              b=self.b,
                              kernel=Perceptron.kernel,
                              dim=Perceptron.dim,
                              sigma=Perceptron.sigma)

    def predict_element(self, an_element):
        """
            Classifies a given element
        :param an_element: ndarray
        :return: int
                1 or -1
        """

        # the classes of the dataset are converted to numerical values
        # thus here we are retrieving corresponding names
        types = np.loadtxt(str(self.dataset_path) + "/types.csv", delimiter=',', dtype=str)
        types = {types[0, 0]: types[0, 1], types[1, 0]: types[1, 1]}

        summation = 0
        for j in range(len(self.train_y)):
            if Perceptron.kernel == 1:
                summation += (self.alpha[j] * self.train_y[j] *
                              np.dot(self.train_x[j], an_element) + self.b)
            elif Perceptron.kernel == 2:
                summation += (self.alpha[j] * self.train_y[j] *
                              (np.dot(self.train_x[j], an_element) + 1) ** Perceptron.dim + self.b)
            else:
                summation += (self.alpha[j] * self.train_y[j] *
                              np.exp(-(np.linalg.norm(self.train_x[j] - an_element) ** 2)
                                     / (2.0 * (Perceptron.sigma ** 2))) + self.b)

        if summation > 0:
            activation = 1
        else:
            activation = -1
        print("The given element is: ", types[str(activation)], " AKA: ", activation)
        return activation

    def fit(self):
        """
            Fits according to the dual perceptron example in :
            "Nello Cristianini, John Shawe-Taylor - An Introduction to
                Support Vector Machines and Other Kernel-based Learning Methods, 1999"

            The computation is delegated to OptimizedTools
            Finally the corresponding parameters that are crucial in the dual form
            of the hyperplane are saved in a .csv file
        :return: None
        """

        self.alpha, self.b = op.fit(self.train_x,
                                    self.train_y,
                                    np.ascontiguousarray(self.gram_mat, dtype=np.float32),
                                    self.epochs)

        np.savetxt(str(self.dataset_path) + "/alpha_and_b_" + str(Perceptron.kernel)
                   + ".csv", X=np.append(self.alpha, self.b), delimiter=',')
        return self.alpha, self.b

    def set_epochs(self, epochs):
        self.epochs = epochs

    def set_kernel(self, kernel):
        self.kernel = kernel

    @staticmethod
    def accuracy(actual_labels, predicted_labels):
        """
            Calculates accuracy percentage
        """

        correct = 0
        for i in range(len(actual_labels)):
            if actual_labels[i] == predicted_labels[i]:
                correct += 1
        return correct / float(len(actual_labels)) * 100.0

    @staticmethod
    def calculate_and_save_gram_mat(train_set, file_name):
        """
            This is a wrapper method to allow np.save which is not supported in Numba yet
        :param train_set: array_like
                            a train set in the form of an ndarray containing all test elements
        :param file_name: String
                            desired file name of the gram mat
        :return: ndarray
        """

        gram_mat = op.calculate_gram_mat(np.ascontiguousarray(train_set, dtype=np.float32),
                                         Perceptron.kernel,
                                         Perceptron.dim,
                                         Perceptron.sigma)

        np.save(file=file_name, arr=gram_mat)
        return gram_mat
