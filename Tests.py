import numpy as np
import pandas as pd
import DatasetTools as dt
import Perceptron as model

#####################___main___#########################

# splits and balances the dataset
'''
print("Loading DATASET...")
df = pd.read_csv('ml_datasets/gender_voice_dataset.csv', sep=',', header=None)
train_set, test_set = dt.setup_dataset(df.to_numpy(), 70)
np.savetxt("gender_voice/gender_voice_train.csv", train_set, delimiter=',')
np.savetxt("gender_voice/gender_voice_test.csv", test_set, delimiter=',')
'''

train_set = np.loadtxt("gender_voice/gender_voice_train.csv", delimiter=',')
test_set = np.loadtxt("gender_voice/gender_voice_test.csv", delimiter=',')

clf = model.Perceptron("gram_mat_gender_voice_dataset_rbf.csv", _kernel_=3, epochs=1000, _sigma_=5.12)

clf.fit(np.float32(train_set[:, :-1]), np.float32(train_set[:, -1]))

accuracy = clf.predict_set(np.float32(train_set[:, :-1]), np.float32(train_set[:, -1]))
print("Kernel Accuracy: " + str(accuracy))
