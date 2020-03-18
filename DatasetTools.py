import numpy as np


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

def setup_dataset(data, split_train_percentage):
    data = data[1:, :]  # deletes labels' name
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
