import numpy as np

from scripts.data_processor import split_data, preprocess_inputs, cap_outliers_fn, split_input_data
from scripts.plots import plot_test_data
from scripts.model import Model, least_squares_fn, logistic_regression_fn
from scripts.proj1_helpers import create_csv_submission, load_csv_data, predict_labels
from scripts.utils import load_weights_model, save_weights_model

# Change this to use a different model
# If set to False, the least_squares is run
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
models = [Model()] * 4

model_weights_filenames = [
    '../pretrained_data/model_0_poly_7_cap_extradrop_log.npy',
    '../pretrained_data/model_1_poly_7_cap_extradrop_log.npy',
    '../pretrained_data/model_2_poly_7_cap_extradrop_log.npy',
    '../pretrained_data/model_3_poly_7_cap_extradrop_log.npy'
]


def train_model(dataset, output, model, fn, **kwargs):
    print('Preprocessing inputs...')
    tx_preprocessed, y_preprocced, mean, std, log_mean, log_std = preprocess_inputs(dataset, output, use_log=True,
                                                                                    poly_rank=7)
    tx_train, tx_test, y_train, y_test = split_data(tx_preprocessed, y_preprocced, 0.7)

    print('Training...')
    losses, best_weights, test_losses, test_accs = model.train(y_train, tx_train, y_test, tx_test, fn, **kwargs)
    model.w = best_weights

    plot_test_data(test_losses, test_accs)
    return model, mean, std, log_mean, log_std


def run_on_test_data(test_data):
    test_data = cap_outliers_fn(test_data)
    datasets, outputs, rows = split_input_data(test_data)
    predictions = np.zeros((test_data.shape[0], 1))
    for jet in range(num_jets):
        preprocessed_data, _, _, _, _, _ = preprocess_inputs(datasets[jet], outputs[jet], poly_rank=7, use_log=True,
                                                             mean=means[jet],
                                                             std=stds[jet], log_std=log_stds[jet],
                                                             log_mean=log_means[jet])
        pred = models[jet].predict(preprocessed_data)
        predictions[rows[jet]] = pred
    return predictions


def get_train_data_accuracy(tx_train, y_train):
    predictions = run_on_test_data(tx_train)
    number_correct = np.sum(predictions == y_train)
    accuracy = number_correct / len(predictions)
    return accuracy


def create_submission(name, tx_test):
    predictions = run_on_test_data(tx_test)
    predictions[predictions == 0] = -1
    create_csv_submission(list(range(350000, 350000 + len(predictions))), predictions, name)


def load_data(change_labels=True):
    train_path = "../data/train.csv"
    test_path = "../data/test.csv"

    print('Reading from file {}'.format(train_path))
    y, tx, ids = load_csv_data(train_path, sub_sample=False)
    y = np.expand_dims(y, axis=1)

    if change_labels:  # for logistic regression only
        y = np.where(y == -1, 0, y)

    print('Reading from file {}'.format(test_path))
    _, tx_submission, _ = load_csv_data(test_path, sub_sample=False)

    return tx, y, tx_submission


def run_least_squares(generate_submission_file=True):
    tx, y, tx_submission = load_data(change_labels=False)
    tx_c = cap_outliers_fn(tx)
    datasets, outputs, _ = split_input_data(tx_c, y)

    for jet in range(num_jets):
        print('Training model for jet', jet)
        models[jet] = Model()

        models[jet], mean, std, log_mean, log_std = train_model(datasets[jet], outputs[jet], models[jet],
                                                                least_squares_fn, batch_size=None, max_iters=10000,
                                                                gamma=0.05, reg_lambda=1e-6, regularization='l2')
        save_weights_model(models[jet], 'model_{}_least_squares.npy'.format(jet))

        means.append(mean)
        stds.append(std)
        log_means.append(log_mean)
        log_stds.append(log_std)

    print('Accuracy on whole training is', get_train_data_accuracy(tx, y))


    if generate_submission_file:
        create_submission('output.csv', tx_submission)


def run_logistic_reggression(pretrained=True, generate_submission_file=False):
    tx, y, tx_submission = load_data()
    tx_c = cap_outliers_fn(tx)
    datasets, outputs, _ = split_input_data(tx_c, y)

    for jet in range(num_jets):
        print('Training model for jet', jet)
        if pretrained:
            models[jet].w = load_weights_model(model_weights_filenames[jet])
            models[jet], mean, std, log_mean, log_std = train_model(datasets[jet], outputs[jet], models[jet],
                                                                    logistic_regression_fn, max_iters=250,
                                                                    batch_size=8192, gamma_decay=None, gamma=0.1,
                                                                    reg_lambda=1e-6, regularization='l2')
        else:
            models[jet] = Model()
            models[jet], mean, std, log_mean, log_std = train_model(datasets[jet], outputs[jet], models[jet],
                                                                    logistic_regression_fn, batch_size=8192,
                                                                    max_iters=40000,
                                                                    gamma=0.1, reg_lambda=1e-6, regularization='l2')
            save_weights_model(models[jet], '../output_files/model_{}_logistic_regression.npy'.format(jet))

        means.append(mean)
        stds.append(std)
        log_means.append(log_mean)
        log_stds.append(log_std)
    # print('Accuracy on whole training is', get_train_data_accuracy(tx, y))

    if generate_submission_file:
        create_submission('output.csv', tx_submission)


def main():
    if USE_LOGISTIC:
        run_logistic_reggression(pretrained=USE_PRETRAINED)
    else:  # use least_squares
        run_least_squares()


if __name__ == "__main__":
    main()

# tx, y, tx_submission = load_data(change_labels=False)
# tx_c = cap_outliers_fn(tx)
# datasets, outputs, _ = split_input_data(tx_c, y)
#
# jet = 0
# models[jet], mean, std, log_mean, log_std = train_model(datasets[jet], outputs[jet], models[jet], least_squares_fn, batch_size=None, max_iters=10000, gamma=0.1, reg_lambda=0.0001, regularization='l1')
#
# jet = 1
# models[jet], mean, std, log_mean, log_std = train_model(datasets[jet], outputs[jet], models[jet], least_squares_fn, batch_size=None, max_iters=10000, gamma=0.015, reg_lambda=0.0001, regularization='l1')
#
# jet = 2
# models[jet], mean, std, log_mean, log_std = train_model(datasets[jet], outputs[jet], models[jet], least_squares_fn, batch_size=None, max_iters=10000, gamma=0.015, reg_lambda=0.0001, regularization='l1')
#
# jet = 3
# models[jet], mean, std, log_mean, log_std = train_model(datasets[jet], outputs[jet], models[jet], least_squares_fn, batch_size=None, max_iters=40000, gamma=0.015, reg_lambda=0.0001, regularization='l1')
