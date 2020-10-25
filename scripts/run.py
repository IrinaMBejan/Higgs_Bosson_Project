import numpy as np
from scripts.data_processor import split_data, preprocess_inputs, cap_outliers_fn, split_input_data
from scripts.plots import plot_test_data
from scripts.model import Model, least_squares_fn
from scripts.proj1_helpers import create_csv_submission, load_csv_data

means = []
stds = []
log_means = []
log_stds = []
num_jets = 4
models = [Model()] * 4

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


def run_on_test_data(test_data, means, stds, log_stds, log_means):
  test_data = cap_outliers_fn(test_data)
  datasets, outputs, rows = split_input_data(test_data)
  predictions = np.zeros((test_data.shape[0],1))
  for jet in range(num_jets):
    preprocessed_data, _, _, _, _, _ = preprocess_inputs(datasets[jet], outputs[jet], poly_rank=7, mean=means[jet], std=stds[jet], log_std=log_stds[jet], log_mean=log_means[jet])
    pred = models[jet].predict(preprocessed_data)
    predictions[rows[jet]] = pred
  return predictions

def get_train_data_accuracy(tx_train, y_train):
  predictions = run_on_test_data(tx_train, means, stds, log_stds, log_means)
  number_correct = np.sum(predictions == y_train)
  accuracy = number_correct/len(predictions)
  return accuracy

def create_submission(name, tx_test):
  predictions = run_on_test_data(tx_test, means, stds, log_stds, log_means)
  predictions[predictions==0] = -1
  create_csv_submission(list(range(350000, 350000 + len(predictions))), predictions, name)
  #files working for colab only?
  #files.download(name)


def load_data(change_labels=True):
    train_path = '/data/train.csv'
    test_path = '/data/test.csv'

    print('Reading from file {}'.format(train_path))
    y, tx, ids = load_csv_data(train_path, sub_sample=False)
    y = np.expand_dims(y, axis=1)

    if change_labels:  # for logistic regression only
        y = np.where(y == -1, 0, y)

    print('Reading from file {}'.format(test_path))
    _, tx_submission, _ = load_csv_data(test_path, sub_sample=False)

    return tx, y, tx_submission

tx, y, tx_submission = load_data(change_labels=False)
tx_c = cap_outliers_fn(tx)
datasets, outputs, _ = split_input_data(tx_c, y)

jet = 0
models[jet], mean, std, log_mean, log_std = train_model(datasets[jet], outputs[jet], models[jet], least_squares_fn, batch_size=None, max_iters=10000, gamma=0.015, reg_lambda=0.0001, regularization='l1')

jet = 1
models[jet], mean, std, log_mean, log_std = train_model(datasets[jet], outputs[jet], models[jet], least_squares_fn, batch_size=None, max_iters=10000, gamma=0.015, reg_lambda=0.0001, regularization='l1')

jet = 2
models[jet], mean, std, log_mean, log_std = train_model(datasets[jet], outputs[jet], models[jet], least_squares_fn, batch_size=None, max_iters=10000, gamma=0.015, reg_lambda=0.0001, regularization='l1')

jet = 3
models[jet], mean, std, log_mean, log_std = train_model(datasets[jet], outputs[jet], models[jet], least_squares_fn, batch_size=None, max_iters=40000, gamma=0.015, reg_lambda=0.0001, regularization='l1')