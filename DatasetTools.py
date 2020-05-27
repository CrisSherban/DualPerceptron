import numpy as np
import pandas as pd


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


# Convert string column that has float values to column to float
def str_column_to_float(dataset, column):
    for row in dataset:
        row[column] = float(row[column].strip())


#####################################################################
def normalize_dataset(dataset):
    for j in range(len(dataset[0])):
        max_of_col = dataset[:, j].max()
        for i in range(len(dataset[:, 0])):
            dataset[i, j] = dataset[i, j] / max_of_col
    return dataset


def standardize_dataset(dataset):
    for j in range(len(dataset[0])):
        mean = dataset[:, j].mean()
        std = dataset[:, j].std()
        for i in range(len(dataset[:, 0])):
            dataset[i, j] = (dataset[i, j] - mean) / std
    return dataset


def setup_dataset(data, split_train_percentage, normalize, standardize):
    data = data[1:, :]  # deletes features' name
    np.random.shuffle(data)  # makes sure data is balanced
    for i in range(0, len(data[1])):
        try:
            str_column_to_float(data, i)
        except:
            lookup_table = str_column_to_int(data, i)
            # the following print is needed in order to understand which value
            # is actually converted in 1 and 0 binary classes
            if i is len(data[0]) - 1:
                print(lookup_table)

    # data has to be a numpy 2D array that has classes in the last column
    rows = len(data[:, 0])

    # if necessary we can either normalize or standardize the dataset
    if normalize:
        data[:, :-1] = normalize_dataset(data[:, :-1])
    if standardize:
        data[:, :-1] = standardize_dataset(data[:, :-1])

    for i in range(0, rows):
        if data[i][-1] == 0:
            data[i][-1] = -1

    # 'split' entries for training, the rest are used for testing
    split = int((split_train_percentage * rows) / 100)

    print("Using " + str(split) + " elements for training and "
          + str(rows - split) + " for testing")

    train = data[0:split][:]
    test = data[split:][:]

    return train, test


def load_dataset(dataset_name, split_train_percentage, normalize=False, standardize=False):
    try:
        print("Loading DATASET...")
        train_set = np.loadtxt(dataset_name + "/train.csv", delimiter=',')
        test_set = np.loadtxt(dataset_name + "/test.csv", delimiter=',')
    except:
        df = pd.read_csv('ml_datasets/' + dataset_name + '_dataset.csv', sep=',', header=None)
        train_set, test_set = setup_dataset(df.to_numpy(), split_train_percentage, normalize, standardize)

        np.savetxt(dataset_name + "/train.csv", train_set, delimiter=',')
        np.savetxt(dataset_name + "/test.csv", test_set, delimiter=',')
    print("DATASET loaded")
    return train_set, test_set
