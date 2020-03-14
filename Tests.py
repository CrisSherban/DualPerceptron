import pandas as pd
import numpy as np
import Perceptron as neurone


# Credits attributed to: Jason Brownlee for the following 2 functions:
# https://machinelearningmastery.com/implement-perceptron-algorithm-scratch-python/
# Convert string column to integer
def str_column_to_int(dataset, column):
    class_values = [row[column] for row in dataset]
    unique = set(class_values)
    lookup = dict()
    for i, value in enumerate(unique):
        lookup[value] = i
    for row in dataset:
        row[column] = lookup[row[column]]
    return lookup


# Convert string column to float
def str_column_to_float(dataset, column):
    for row in dataset:
        row[column] = float(row[column].strip())


#####################################################################

def setup_dataset(dataset):
    for i in range(0, 14):
        str_column_to_int(dataset, i)

    for i in range(1, len(dataset[:, 14])):
        if dataset[i, 14] == "<=50K":
            dataset[i, 14] = -1
        else:
            dataset[i, 14] = 1

    # 'split' entries for training, the rest are used for testing
    split = 42001

    train = (dataset[1:split, :14], dataset[1:split, 14])
    test = (dataset[split:, :14], dataset[split:, 14])

    return train, test


##################################################################

df = pd.read_csv('census_income_dataset.csv', sep=',', header=None)
dataset = df.to_numpy()

((X_train, y_train), (X_test, y_test)) = setup_dataset(dataset)

clf_linear = neurone.Perceptron(kernel="linear_ker")
clf_poly = neurone.Perceptron(kernel="polynomial_ker", dim=2)
clf_rbf = neurone.Perceptron(kernel="rbf_ker", sigma=5.12)

clf_linear.fit(X_train, y_train)
clf_poly.fit(X_train, y_train)
clf_rbf.fit(X_train, y_train)

accuracy_linear = clf_linear.predict_set(X_test, y_test)
accuracy_poly = clf_poly.predict_set(X_test, y_test)
accuracy_rbf = clf_rbf.predict_set(X_test, y_test)

print("Kernel Lineare: " + str(accuracy_linear))
print("Kernel Poly: " + str(accuracy_poly))
print("Kernel RBF: " + str(accuracy_rbf))
