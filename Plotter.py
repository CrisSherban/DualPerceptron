import numpy as np
from matplotlib import pyplot as plt

"""
plotter functions are meant to use the whole dataset
and the hyperplane is recalculated based on a fitted perceptron
"""


def plot_3d(clf, X, Y):
    feature1 = 0
    feature2 = 1
    feature3 = 2

    def calculate_weights():
        w = np.zeros(len(clf.train_x[0]))

        for i in range(len(clf.alpha)):
            w += clf.alpha[i] * clf.train_y[i] * clf.train_x[i]

        return w

    w = calculate_weights()
    z = lambda x, y: ((w[feature1] * x) -
                      (w[feature2] * y) - clf.b) / (w[feature3])

    for angle in range(70, 210, 2):
        tmp = np.linspace(-5, 5, 30)
        x, y = np.meshgrid(tmp, tmp)

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_surface(x, y, z(x, y), alpha=0.3)
        ax.plot3D(X[Y == -1, feature1], X[Y == -1, feature2], X[Y == -1, feature3], 'ob')
        ax.plot3D(X[Y == 1, feature1], X[Y == 1, feature2], X[Y == 1, feature3], 'sr')

        ax.view_init(30, angle)

        filename = "3D_plots/plot_step" + str(angle) + ".png"
        plt.savefig(filename, dpi=96)


def plot_2d(clf, X, Y):
    feature1 = 0
    feature2 = 1

    def calculate_weights():
        w = np.zeros(len(clf.train_x[0]))

        for i in range(len(clf.alpha)):
            w += clf.alpha[i] * clf.train_y[i] * clf.train_x[i]

        return w

    x = np.linspace(-5, 5, 30)

    w = calculate_weights()
    y = lambda x: -(w[feature1] * x + clf.b) / (w[feature2])

    plt.title("2D Hyperplane Decision Boundary")
    plt.xlabel("Feature " + str(feature1))
    plt.ylabel("Feature " + str(feature2))

    plt.grid()
    plt.plot(x, y(x))
    plt.plot(X[Y == -1, feature1], X[Y == -1, feature2], 'ro')
    plt.plot(X[Y == 1, feature1], X[Y == 1, feature2], 'bo')
    plt.savefig("Pictures/2D Hyperplane Decision Boundary gender_voice.png")
    plt.show()
