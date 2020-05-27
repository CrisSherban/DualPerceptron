import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Perceptron as SkPerceptron

import DatasetTools as dt
import Perceptron
import Plotter


###################### testing functions ##########################

def best_epoch(clf, test_set, max_epochs):
    alphas = []
    bs = []
    accuracies = []

    for i in range(max_epochs):
        clf.set_epochs(i + 1)
        alpha, b = clf.fit()
        alphas.append(alpha)
        bs.append(b)
        predicted = clf.predict_set(test_set[:, :-1], test_set[:, -1])
        accuracies.append(clf.accuracy(test_set[:, -1], predicted))
    return find_max_accuracy(accuracies)


def find_max_accuracy(accuracies):
    best_accuracy = 0
    best_accuracy_index = 0
    for j in range(len(accuracies)):
        if accuracies[j] > best_accuracy:
            best_accuracy = accuracies[j]
            best_accuracy_index = j + 1

    return best_accuracy, best_accuracy_index


def test_for_each_kernel(dataset_name, train_set, test_set, max_epochs, test_on_train):
    best_accuracies_per_kernel = {}
    for i in range(1, 4):
        clf = Perceptron.Perceptron(dataset_name,
                                    train_set[:, :-1],
                                    train_set[:, -1],
                                    _kernel_=i, epochs=10,
                                    _sigma_=5, _dim_=5)
        best_epoch_accuracy, best_epoch_accuracy_index = best_epoch(clf, test_set, max_epochs)

        if test_on_train:
            clf_train = Perceptron.Perceptron(dataset_name,
                                              train_set[:, :-1],
                                              train_set[:, -1],
                                              _kernel_=i, epochs=best_epoch_accuracy_index,
                                              _sigma_=5, _dim_=5)
            clf_train.fit()
            predicted = clf_train.predict_set(train_set[:, :-1], train_set[:, -1])
            best_epoch_accuracy = clf_train.accuracy(train_set[:, -1], predicted)

        best_accuracies_per_kernel[i] = best_epoch_accuracy

    if not test_on_train:
        # adding the results from sklearn perceptron
        clf = SkPerceptron(tol=1e-3, random_state=0)
        clf.fit(train_set[:, :-1], train_set[:, -1])
        best_accuracies_per_kernel[4] = clf.score(train_set[:, :-1], train_set[:, -1]) * 100

    return best_accuracies_per_kernel


def test_for_each_dataset(max_epochs, dataset_names, test_on_train):
    '''
    best_accuracies_per_kernel_per_dataset is meant to be a list of
    dictionaries that contain for keys the kernel type
    and for values the relative accuracy
    '''
    best_accuracies_per_kernel_per_dataset = []
    for dataset in dataset_names:
        train_set, test_set = dt.load_dataset(dataset, 70)
        best_accuracies_per_kernel_per_dataset.append(
            test_for_each_kernel(dataset, train_set, test_set, max_epochs, test_on_train))

    return best_accuracies_per_kernel_per_dataset


def analyze_accuracies(datasets_names, test_on_train=False):
    best_accuracies_per_kernel_per_dataset = test_for_each_dataset(10, datasets_names, test_on_train)

    labels = datasets_names
    linear_ker = [round(best_accuracies_per_kernel_per_dataset[i][1], 1) for i in range(len(datasets_names))]
    poly_ker = [round(best_accuracies_per_kernel_per_dataset[i][2], 1) for i in range(len(datasets_names))]
    gaussian_ker = [round(best_accuracies_per_kernel_per_dataset[i][3], 1) for i in range(len(datasets_names))]
    if not test_on_train:
        sklearn_perceptron = [round(best_accuracies_per_kernel_per_dataset[i][4], 1) for i in
                              range(len(datasets_names))]

    x = np.arange(len(labels))  # the label locations
    width = 0.2  # the width of the bars

    fig, ax = plt.subplots()
    rects1 = ax.bar(x - 0.3, linear_ker, width, label='linear_ker')
    rects2 = ax.bar(x - 0.1, poly_ker, width, label='poly_ker')
    rects3 = ax.bar(x + 0.1, gaussian_ker, width, label='gaussian_ker')
    if not test_on_train:
        rects4 = ax.bar(x + 0.3, sklearn_perceptron, width, label='sklearn_perceptron')

    ax.set_ylabel('Accuracy')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), shadow=True, ncol=2)

    def autolabel(rects):
        """Attach a text label above each bar in *rects*, displaying its height."""
        for rect in rects:
            height = rect.get_height()
            ax.annotate('{}'.format(height),
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 2),  # vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom')

    autolabel(rects1)
    autolabel(rects2)
    autolabel(rects3)
    if not test_on_train:
        autolabel(rects4)

    fig.tight_layout()

    if not test_on_train:
        ax.set_title('Comparison between kernels and sklearn perceptron')
        plt.savefig("Accuracies.png")

    else:
        ax.set_title('Prediction on the train set')
        plt.savefig("test_on_train.png")

    plt.show()


#####################___main script___##############################
# kernels: 1: Linear, 2: Poly, 3: Gaussian

# datasets_names = ["bank_marketing", "gender_voice", "mushroom"]
# analyze_accuracies(datasets_names, test_on_train=True)

# splits and balances the dataset
dataset_name = "gender_voice"
train_set, test_set = dt.load_dataset(dataset_name, 70, standardize=True)

# full_dataset is used to plot graphs
full_dataset = np.vstack((train_set, test_set))
X = full_dataset[:, :-1]
Y = full_dataset[:, -1]

clf = Perceptron.Perceptron(dataset_name,
                            train_set[:, :-1],
                            train_set[:, -1],
                            _kernel_=1, epochs=10,
                            _sigma_=5, _dim_=5)

clf.fit()
predicted = clf.predict_set(train_set[:, :-1], train_set[:, -1])
accuracy = clf.accuracy(train_set[:, -1], predicted)
# print("Kernel Accuracy: " + str(accuracy))
print("Kernel Best Accuracy with 10 maximum epochs: ", best_epoch(clf, test_set, 10))

'''
# plotter functions to visualize the dataset and the hyperplane
Plotter.plot_3d(clf, X, Y)
Plotter.plot_2d(clf, X, Y)
'''

'''
# plots dataset with seaborn
sns.set(style="ticks", color_codes=True)
tmp_dataset = np.loadtxt("ml_datasets/mushroom_dataset.csv", delimiter=',', dtype=str)
tmp_dataset = tmp_dataset[0, :]
#full_dataset = np.vstack((tmp_dataset, full_dataset))


df = pd.DataFrame(data=full_dataset, columns=tmp_dataset)
df.dropna(inplace=True)
sns.pairplot(df, vars=["cap-shape", "cap-surface", "cap-color"], hue='mushroom', corner=True)
plt.savefig("Pictures/" + str(dataset_name) + " scatterplot.png")
plt.show()
'''

'''
clf = SkPerceptron(tol=1e-3, random_state=0)
clf.fit(train_set[:, :-1], train_set[:, -1])
print("Sk Perceptron: ", clf.score(test_set[:, :-1], test_set[:, -1]) * 100)
'''
