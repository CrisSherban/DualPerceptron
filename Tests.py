import pandas as pd
import numpy as np
import Perceptron as neurone
import GramMatrix as gram_mat


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
    dataset = dict()
    dataset['n_rows'] = len(data[:, 0])
    dataset['n_cols'] = len(data[0, :-1])
    dataset['x'] = data[:, :(len(data[0]) - 1)]
    dataset['y'] = data[:, len(data[0]) - 1]

    for i in range(0, dataset['n_rows']):
        if dataset['y'][i] == 0:
            dataset['y'][i] = -1

    # 'split' entries for training, the rest are used for testing
    split = int((split_train_percentage * dataset['n_rows']) / 100)

    print("Using " + str(split) + " elements for training and "
          + str(dataset['n_rows'] - split) + " for testing")

    train = {'x': dataset['x'][0:split, :14], 'y': dataset['y'][0:split],
             'n_rows': split, ' n_cols': dataset['n_cols']}

    test = {'x': dataset['x'][split:, :14], 'y': dataset['y'][split:],
            'n_rows': dataset['n_rows'] - split, 'n_cols': dataset['n_cols']}

    return train, test


#####################___main___##############################
#############################################################

# creates Gram matrix if is not given
print("Loading DATASET and Gram Matrix...")
df = pd.read_csv('gender_voice_dataset.csv', sep=',', header=None)
train_set, test_set = setup_dataset(df.to_numpy(), 70)

gram_mat_filename = "gram_mat_gender_voice_dataset_linear.csv"
try:
    gram_matrix = np.loadtxt(gram_mat_filename,
                             dtype=np.float32,
                             delimiter=',')
except:
    print("Creating Gram Matrix")

    gram_matrix = gram_mat.calculate_and_save_gram_mat(np.float32(np.float16(train_set['x'])),
                                                       train_set['n_rows'],
                                                       gram_mat_filename)
print("Gram Matrix Loaded")
#####################################################################


clf_linear = neurone.Perceptron(gram_matrix)

print("Fitting data")
clf_linear.fit(train_set)
print("Finished fitting")

print("Testing the data")
accuracy_linear = clf_linear.predict_set(test_set)
print("Linear Kernel Accuracy: " + str(accuracy_linear))
