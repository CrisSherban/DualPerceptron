import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import DatasetTools as dt
import OptimizedTools as ot

"""
This .py script plots a gram matrix of some given dataset.
The dataset is pre-ordered based on the labels so the gram 
matrix can be used to visualize the classes of the dataset
"""

df = pd.read_csv('ml_datasets/gender_voice_dataset.csv', sep=',', header=None)
dataset = df.to_numpy()
dataset = dataset[1:, :]  # deletes features' name

for i in range(0, len(dataset[1])):
    try:
        dt.str_column_to_float(dataset, i)
    except:
        lookup_table = dt.str_column_to_int(dataset, i)
        # the following print is needed in order to understand which value
        # is actually converted in 1 and 0 binary classes
        if i is len(dataset[0]) - 1:
            print(lookup_table)

# data has to be a numpy 2D array that has classes in the last column
rows = len(dataset[:, 0])

# if necessary we can either normalize or standardize the dataset
dataset[:, :-1] = dt.normalize_dataset(dataset[:, :-1])
# dataset[:, :-1] = dt.standardize_dataset(dataset[:, :-1])

for i in range(0, rows):
    if dataset[i, -1] == 0:
        dataset[i, -1] = -1
print(dataset)

ordered_dataset = dataset
index = 0

for i in range(len(dataset[:, 0])):
    if dataset[i, -1] == 1:
        ordered_dataset[index] = dataset[i]
        index += 1
print("Elements relative to label 1: ", index)
first_class_elements = index

for i in range(len(dataset[:, 0])):
    if dataset[i, -1] != 1:
        ordered_dataset[index] = dataset[i]
        index += 1
print("Elements relative to label -1: ", index - first_class_elements)

gram_mat = ot.calculate_gram_mat(np.ascontiguousarray(ordered_dataset, dtype=np.float32), kernel=3, sigma=1, dim=2)

plt.imshow(gram_mat, cmap='viridis')
plt.savefig("Pictures/gender_voice_gram_mat_3_ordered.png")
plt.show()
