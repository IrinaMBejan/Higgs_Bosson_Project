import numpy as np

from data_processor import split_data, preprocess_inputs, cap_outliers_fn, split_input_data
from plots import plot_test_data
from model import Model, least_squares_fn, logistic_regression_fn
from proj1_helpers import create_csv_submission, load_csv_data, predict_labels
from utils import load_weights_model, save_weights_model

# Constants are set to reflect our submission on AICrowd.

# Change this to use a different model. If set to False, the least_squares is run
USE_LOGISTIC = True
# If set to pretrained, it will load weights for models. Only available for logistic regression where training takes
# a long time.
USE_PRETRAINED = True

# Global values to store training data the mean, standard deviation, logarithmic mean and logarithmic standard deviation
# for each model corresponding to each jet.
means = [None] * 4
stds = [None] * 4
log_means = [None] * 4
log_stds = [None] * 4

num_jets = 4
models = [Model(), Model(), Model(), Model()]

model_weights_filenames = [
    '../pretrained_data/model_0_logistic_reg.npy',
    '../pretrained_data/model_1_logistic_reg.npy',
    '../pretrained_data/model_2_logistic_reg.npy',
    '../pretrained_data/model_3_logistic_reg.npy'
]


def train_model(dataset, output, model, fn, **kwargs):
    """Trains the model.
    Args:
        dataset: The training dataset
        output: The labels of the training dataset
        model: Existing instance of Model() to train
        fn: The method to be used. Current implementation exist for logistic regression and least squares.
    """
    print('Preprocessing inputs...')
    # Preprocess the data and store the mean and std for dataaset
    tx_preprocessed, y_preprocessed, mean, std, log_mean, log_std = preprocess_inputs(dataset, output, use_log=True,
                                                                                      poly_rank=7)
    # Split data into training and validation
    tx_train, tx_test, y_train, y_test = split_data(tx_preprocessed, y_preprocessed, 0.7)

    print('Training...')
    # Train the model and save the weights and loss history
    losses, best_weights, test_losses, test_accs = model.train(y_train, tx_train, y_test, tx_test, fn, **kwargs)
    model.w = best_weights

    # Plot learning artifacts for testing data
    plot_test_data(test_losses, test_accs)
    return model, mean, std, log_mean, log_std


def run_on_test_data(test_data):
    """Computes prediction for given data and using global models trained already."""

    test_data = cap_outliers_fn(test_data)
    datasets, outputs, rows = split_input_data(test_data)
    predictions = np.zeros((test_data.shape[0], 1))

    for jet in range(num_jets):
        preprocessed_data, _, _, _, _, _ = preprocess_inputs(datasets[jet], outputs[jet], poly_rank=7, use_log=True,
                                                             mean=means[jet],
                                                             std=stds[jet], log_std=log_stds[jet],
                                                             log_mean=log_means[jet])
        jet_predictions = models[jet].predict(preprocessed_data)
        predictions[rows[jet]] = jet_predictions
    return predictions


def get_train_data_accuracy(tx_train, y_train):
    """Computes the accuracy of the models combined on the given dataset."""
    predictions = run_on_test_data(tx_train)
    number_correct = np.sum(predictions == y_train)
    accuracy = number_correct / len(predictions)
    return accuracy


def create_submission(name, tx_test):
    """Creates the submission file using the given test data and filename."""
    predictions = run_on_test_data(tx_test)
    predictions[predictions == 0] = -1
    create_csv_submission(list(range(350000, 350000 + len(predictions))), predictions, name)


def load_data(change_labels=True):
    """
    Loads the training and testing data from disk.
    Args:
        change_labels: Convert the labels from -1/1 to 0/1 for logistic regression.
    """
    train_path = "../data/train.csv"
    test_path = "../data/test.csv"

    print('Reading from file {}'.format(train_path))
    y, tx, ids = load_csv_data(train_path, sub_sample=False)
    y = np.expand_dims(y, axis=1)

    if change_labels:
        y = np.where(y == -1, 0, y)

    print('Reading from file {}'.format(test_path))
    _, tx_submission, _ = load_csv_data(test_path, sub_sample=False)

    return tx, y, tx_submission


def run_least_squares(generate_submission_file=True):
    """
    Runs the least square method and trains all models.
    Args:
        generate_submission_file: If True, an output csv file is generated on test data after training ends.
    """
    tx, y, tx_submission = load_data(change_labels=False)
    tx_c = cap_outliers_fn(tx)
    datasets, outputs, _ = split_input_data(tx_c, y)

    for jet in range(num_jets):
        print('Training model for jet', jet)
        models[jet] = Model()

        models[jet], mean, std, log_mean, log_std = train_model(datasets[jet], outputs[jet], models[jet],
                                                                least_squares_fn, batch_size=None, max_iters=10000,
                                                                gamma=0.05, reg_lambda=1e-6, regularization='l2')
        save_weights_model(models[jet], '../output_files/model_{}_least_squares.npy'.format(jet))

        means.append(mean)
        stds.append(std)
        log_means.append(log_mean)
        log_stds.append(log_std)

    print('Accuracy on entire training dataset is', get_train_data_accuracy(tx, y))

    if generate_submission_file:
        create_submission('../output_files/output.csv', tx_submission)


def run_logistic_regression(pretrained=True, generate_submission_file=False) -> None:
    """
    Runs the logistic regression method and trains all models.
    Args:
        pretrained: If True, it will load existing weights used for the model submission on AICrowd.
        generate_submission_file: If True, an output csv file is generated on test data after training ends.
    """
    tx, y, tx_submission = load_data()
    tx_c = cap_outliers_fn(tx)
    datasets, outputs, _ = split_input_data(tx_c, y)

    for jet in range(num_jets):
        print('Training model for jet', jet)
        if pretrained:
            models[jet].w = load_weights_model(model_weights_filenames[jet])

            models[jet], mean, std, log_mean, log_std = train_model(datasets[jet], outputs[jet], models[jet],
                                                                    logistic_regression_fn, max_iters=300,
                                                                    batch_size=8192, gamma_decay=None, gamma=0.1,
                                                                    reg_lambda=1e-6, regularization='l2')
        else:
            models[jet] = Model()
            gammas = [0.3, 0.2, 0.2, 0.2]
            batch_sizes = [8192, 1024, 512, 128]
            max_iters = [8000, 15000, 14000, 30000]

            models[jet], mean, std, log_mean, log_std = train_model(datasets[jet], outputs[jet], models[jet],
                                                                    logistic_regression_fn,
                                                                    batch_size=batch_sizes[jet],
                                                                    max_iters=max_iters[jet],
                                                                    gamma=gammas[jet],
                                                                    reg_lambda=1e-6, regularization='l2')
            save_weights_model(models[jet], '../output_files/model_{}_logistic_regression.npy'.format(jet))

        means.append(mean)
        stds.append(std)
        log_means.append(log_mean)
        log_stds.append(log_std)
    print('Accuracy on whole training is', get_train_data_accuracy(tx, y))

    if generate_submission_file:
        create_submission('../output_files/output.csv', tx_submission)


def main():
    # Runs our best configuration
    if USE_LOGISTIC:
        run_logistic_regression(pretrained=USE_PRETRAINED, generate_submission_file=True)
    else:  # use least_squares
        run_least_squares()


if __name__ == "__main__":
    main()
