import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from scripts.data_processor import split_input_data


def plot_features(dataset, jet=None):
    """
    Plots the data points of each column feature of the dataset or of a specific jet subdataset.
    The blue shows values different that -999 (meaningless values) and red the -999 values.

    Args:
        dataset: A 2-D Numpy array representing the dataset.
        jet: The number of jet for which to plot the feature columns.
    """
    if jet:
        datasets, _, _ = split_input_data(dataset)
        dataset = datasets[jet]

    for i in range(dataset.shape[1]):
        print('Showing the plot for column feature', i)
        values = dataset[:, i]

        indices_999 = values == -999
        plt.scatter(np.where(indices_999)[0], values[indices_999], color="red", s=4)
        plt.scatter(np.where(~indices_999)[0], values[~indices_999], color="blue", s=4)

        plt.xlabel('samples')
        plt.ylabel('values')
        plt.show()


def plot_points_labels(dataset, labels):
    """
    Plots the data points of each column feature of the dataset and uses the classification to color each datapoint.

    Args:
        dataset: A 2-D Numpy array representing the dataset.
        labels: The labels of the dataset given as 0 or 1.
    """
    for i in range(dataset.shape[1]):
        pos_values = np.stack([dataset[i] for i in range(len(labels)) if labels[i][0] == 1], axis=0)
        neg_values = np.stack([dataset[i] for i in range(len(labels)) if labels[i][0] == 0], axis=0)
        plt.scatter(np.arange(len(pos_values[:, i])), pos_values[:, i], c='red')
        plt.scatter(np.arange(len(neg_values[:, i])), neg_values[:, i], c='blue', alpha=0.4)
        plt.show()


def plot_distribution(dataset, labels):
    """
    Plots the distribution of each column feature of the dataset based on the label value.

    Args:
        dataset: A 2-D Numpy array representing the dataset.
        labels: The labels of the dataset given as 0 or 1.
    """
    for i in range(dataset.shape[1]):
        pos_values = np.stack([dataset[i] for i in range(len(labels)) if labels[i][0] == 1], axis=0)
        neg_values = np.stack([dataset[i] for i in range(len(labels)) if labels[i][0] == 0], axis=0)
        sns.distplot(pos_values, color="dodgerblue", label="y=1")
        sns.distplot(neg_values, color="orange", label="y=-1")
        plt.show()


def plot_losses(losses):
    """
    Plots the given losses as loss per iteration.

    Args:
        losses: A list containing losses values.
    """
    plt.plot(np.arange(len(losses)), losses, label='loss')

    plt.xlabel("Iterations")
    plt.title('Training loss')
    plt.legend(loc='upper right')

    plt.show()


def plot_test_data(test_losses, test_accuracies):
    """
    Plots the loss and accuracy in the same plot.
    
    Args:
        test_losses: A list containing losses values.
        test_accuracies: A list of shape length as losses containg the accuracy values.
    """
    plt.plot(np.arange(len(test_losses)) * 10, test_losses, '-b', label='test loss')
    plt.plot(np.arange(len(test_accuracies)) * 10, test_accuracies, '-r', label='test acc')

    plt.xlabel("Iterations")
    plt.title('Validation loss & acc')
    plt.legend(loc='upper right')

    plt.show()
