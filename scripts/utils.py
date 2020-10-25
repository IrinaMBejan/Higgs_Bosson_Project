import numpy as np


def save_weights_model(model, filename):
    """
    Save weights to file.

    Args:
        model: a instance of a Model class that has been trained
        filename: string value representing the name of the file
    """
    with open(filename, 'wb') as f:
        np.save(filename, model.best_weights)


def load_weights_model(filename):
    """
    Load weights from a file.

    Args:
        filename: string value presenting the name of target file
    """
    with open(filename, 'rb') as file:
        return np.load(file, allow_pickle=True)
