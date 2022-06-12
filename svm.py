# SVM = support vector machine, used to correlate linearly

import sklearn
import sklearn.datasets
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split as tts
import argparse
from sklearn import svm
import numpy as np
import pandas as pd

def arg():
    parser = argparse.ArgumentParser(description='Classification of Irises')
    parser.add_argument('--viz', '-v', action='store_true', help='visualize data')
    args = parser.parse_args()

    return args

def input_fn_train(X_train, y_train):
    num_train = 120
    num_epoch = 500
    dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
    dataset = dataset.shuffle(num_train).repeat(num_epoch).batch(num_train)
    return dataset

def main(args):
    # Import Dataset
    dataset = sklearn.datasets.load_iris()
    X = dataset.data[:, :]
    y = dataset.target

    if args.viz:
        print("The data has shape: ", X.shape)
        print(y.shape)
        input(">>")

        print("We can try visualizing the data")
        x_plot = X[:, :2]
        y_plot = y
        x_min, x_max = x_plot[:, 0].min() - 0.5, x_plot[:, 0].max() + .5
        y_min, y_max = x_plot[:, 1].min() - .5, X[:, 1].max() + .5

        # To getter a better understanding of interaction of the dimensions
        # plot the first three PCA dimensions
        fig = plt.figure(1, figsize=(8, 6))
        ax = Axes3D(fig, elev=-150, azim=110)
        X_reduced = PCA(n_components=3).fit_transform(X)
        ax.scatter(X_reduced[:, 0], X_reduced[:, 1], X_reduced[:, 2], c=y,
                cmap=plt.cm.Set1, edgecolor='k', s=40)
        ax.set_title("First three PCA directions")
        ax.set_xlabel("1st eigenvector")
        ax.w_xaxis.set_ticklabels([])
        ax.set_ylabel("2nd eigenvector")
        ax.w_yaxis.set_ticklabels([])
        ax.set_zlabel("3rd eigenvector")
        ax.w_zaxis.set_ticklabels([])

        input("h")

        plt.show()
        input(">>")
    
    X_train, X_test, y_train, y_test = tts(X, y, test_size=0.2)

    clf = svm.LinearSVC(max_iter=120000, loss='hinge')
    clf.fit(X_train, y_train)

    preds = clf.predict(X_test)
    correct = (preds == y_test).sum()
    print("Regular:", float(correct) / len(preds))

    # SVC = support vector classification
    clf = svm.SVC(max_iter=120000)
    clf.fit(X_train, y_train)
    preds = clf.predict(X_test)
    correct = (preds == y_test).sum()
    print("Non-linear:", float(correct) / len(preds))



if __name__ == '__main__':
    args = arg()
    main(args)
