import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scripts.data_processor import split_input_data


def plot_features(tx, jet=None):
    if jet:
        datasets, _, _ = split_input_data(tx)
        tx = datasets[jet]

    for i in range(tx.shape[1]):
        print(i)
        values = tx[:, i]

        indices_999 = values == -999
        plt.scatter(np.where(indices_999)[0], values[indices_999], color="red", s=4)
        plt.scatter(np.where(~indices_999)[0], values[~indices_999], color="blue", s=4)

        plt.xlabel('samples')
        plt.ylabel('values')
        plt.show()


def plot_points(tx, y):
    pos_values = np.stack([tx[i] for i in range(len(y)) if y[i][0] == 1], axis=0)
    neg_values = np.stack([tx[i] for i in range(len(y)) if y[i][0] == 0], axis=0)
    plt.scatter(np.arange(len(pos_values[:, i])), pos_values[:, i], c='red')
    plt.scatter(np.arange(len(neg_values[:, i])), neg_values[:, i], c='blue', alpha=0.4)
    plt.xlabel('categories')
    plt.ylabel('values')
    plt.show()


def plot_distrib(tx, y):
    pos_values = np.stack([tx[i] for i in range(len(y)) if y[i][0] == 1], axis=0)
    neg_values = np.stack([tx[i] for i in range(len(y)) if y[i][0] == 0], axis=0)
    sns.distplot(pos_values, color="dodgerblue", label="y=1")
    sns.distplot(neg_values, color="orange", label="y=-1")
    plt.show()
