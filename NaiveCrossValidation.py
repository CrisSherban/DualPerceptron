import Perceptron
import DatasetTools as dt
import Tests as tests
import numpy as np

"""
bank_marketing: best_dim = 3, 89.0%; best_sigma = 0.001, 88.9%
gender_voice: best_dim = 1, 96.1%; best_sigma = 0.1, 96.4%
mushroom: best_dim = 5, 96.4%; best_sigma = 0.1, 99.4%
"""

# saving the parameters
# TODO: Implement a better way to pass the parameters automatically
# first row is for dim, the second for sigma
# the cols represent the datasets in the following order:
# bank_marketing, gender_voice, mushroom
best_parameters = np.array([
    [0, 0, 0],  # linear kernel does not need parameters
    [3, 1, 5],
    [0.001, 0.1, 0.1]])


def cross_validate_parameters(dataset_name, train_set, test_set):
    dims = [i for i in range(1, 7)]
    sigmas = [0.001, 0.01, 0.1, 1, 2, 5]

    accuracies = []
    best_accu_and_dim = 0, 0
    max = 0

    for dim in dims:
        clf_poly = Perceptron.Perceptron(dataset_name,
                                         train_set[:, :-1],
                                         train_set[:, -1],
                                         _kernel_=2, epochs=10, _dim_=dim)
        accuracy, accuracy_index = tests.best_epoch(clf_poly, test_set, 10)

        # clf_poly.fit()
        # predicted = clf_poly.predict_set(test_set[:, :-1], test_set[:, -1])
        # accuracy = clf_poly.accuracy(test_set[:, -1], predicted)

        accuracies.append((accuracy, dim))

    for accu, dim in accuracies:
        if accu > max:
            max = accu
            best_accu_and_dim = accu, dim

    accuracies = []
    best_accu_and_sigma = 0, 0
    max = 0

    for sigma in sigmas:
        clf_gaussian = Perceptron.Perceptron(dataset_name,
                                             train_set[:, :-1],
                                             train_set[:, -1],
                                             _kernel_=3, epochs=10, _sigma_=sigma)
        accuracy, accuracy_index = tests.best_epoch(clf_gaussian, test_set, 10)

        # clf_gaussian.fit()
        # predicted = clf_gaussian.predict_set(test_set[:, :-1], test_set[:, -1])
        # accuracy = clf_gaussian.accuracy(test_set[:, -1], predicted)

        accuracies.append((accuracy, sigma))

    for accu, sigma in accuracies:
        if accu > max:
            max = accu
            best_accu_and_sigma = accu, sigma

    print("Best accuracy with associated dim: ", best_accu_and_dim, "\n",
          "Best accuracy with associated sigma :", best_accu_and_sigma)

    return best_accu_and_dim, best_accu_and_sigma


def main():
    dataset_name = "mushroom"
    train_set, test_set = dt.load_dataset(dataset_name, 70, standardize=True)

    cross_validate_parameters(dataset_name, train_set, test_set)


if __name__ == "__main__":
    main()
