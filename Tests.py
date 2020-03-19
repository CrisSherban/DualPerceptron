import numpy as np
import pandas as pd
import DatasetTools as dt
import Perceptron as model

#####################___main script___#########################

# splits and balances the dataset
dataset_name = "gender_voice"

'''
print("Loading DATASET...")
df = pd.read_csv('ml_datasets/' + dataset_name + '_dataset.csv', sep=',', header=None)
train_set, test_set = dt.setup_dataset(df.to_numpy(), 70)

np.savetxt(dataset_name + "/train.csv", train_set, delimiter=',')
np.savetxt(dataset_name + "/test.csv", test_set, delimiter=',')
'''

train_set = np.loadtxt(dataset_name + "/train.csv", delimiter=',')
test_set = np.loadtxt(dataset_name + "/test.csv", delimiter=',')

kernel = 2  # 1: Linear, 2: Poly, 3:RBF

# last column represents the classes the others are features
clf = model.Perceptron(dataset_name, np.float32(train_set[:, :-1]), np.float32(train_set[:, -1]),
                       _kernel_=kernel, epochs=1000, _sigma_=5.12, _dim_=4)

clf.fit()

accuracy = clf.predict_set(np.float32(train_set[:, :-1]), np.float32(train_set[:, -1]))
print("Kernel Accuracy: " + str(accuracy))

# clf.predict_element(test_set[13, :-1])
# print("The above answer should be: ", test_set[13, -1])
