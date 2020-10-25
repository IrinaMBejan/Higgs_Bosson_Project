import numpy as np
from scripts.proj1_helpers import create_csv_submission
from scripts.proj1_helpers import predict_labels


def compute_mse(y, tx, w):
    "Compute mse"
    e = y - np.dot(tx, w)
    mse = np.dot(e, e) / (2 * len(e))
    return mse


def calculate_mse(e):
    """Calculate the mse for vector e."""
    return 1 / 2 * np.mean(e ** 2)


def ls_compute_gradient(y, tx, w):
    """Compute gradient"""
    e = y - np.dot(tx, w)
    gradient = - np.dot(tx.T, e) / len(y)
    return gradient, e


def sigmoid(t):
    """Apply the sigmoid function on t."""
    return 1.0 / (1.0 + np.exp(-t))


def logistic_regression_fn(y, tx, w, lambda_=None):
    """Compute the loss: negative log likelihood."""
    epsilon = 1e-10
    pred = sigmoid(np.dot(tx, w))
    pred = np.where(pred == 0.0, epsilon, pred)
    pred = np.where(pred == 1.0, 1 - epsilon, pred)
    loss = (1 / tx.shape[0]) * (0.66 * y.T.dot(np.log(pred)) + 0.33 * (1 - y).T.dot(np.log(1 - pred)))
    gradient = np.dot(tx.T, (pred - y)) / tx.shape[0]
    return np.squeeze(-loss), gradient


def least_squares_fn(y, tx, w, lambda_=None):
    """""Compute the gradient using least squares"""
    gradient, error = ls_compute_gradient(y, tx, w)
    loss = calculate_mse(error)

    return loss, gradient


def he_weights_initialization(shape):
    """Initialize weights"""
    w = np.random.randn(shape[0], shape[1]) * np.sqrt(2 / shape[1])
    return w


def compute_class_weights(y):
    """Computing the class weights"""
    #i think we don't use this?
    return len(y) / (2 * np.bincount(np.squeeze(y).astype(int)))


class Model(object):

    def __init__(self):
        pass

    def predict(self, inputs):
        if self.use_logistic:
            return self.predict_labels_logistic(self.w, inputs)
        return predict_labels(self.w, inputs)

    def predict_labels_logistic(self, weights, data):
        y_pred = sigmoid(np.dot(data, weights))
        y_pred[np.where(y_pred < 0.5)] = 0
        y_pred[np.where(y_pred >= 0.5)] = 1
        return y_pred

    def create_submission(self, inputs, name):
        pred = self.predict(inputs)

        if self.use_logistic:
            pred[np.where(pred == 0)] = -1

        create_csv_submission(list(range(350000, 350000 + len(pred))), pred, name)
        files.download(name)
        return name

    def initialize_weights(self, shape):
        # Our function is not convex
        return he_weights_initialization(shape)

    def compute_accuracy(self, inputs, true_values):
        predicted_values = self.predict(inputs)
        number_correct = np.sum(predicted_values == true_values)
        accuracy = number_correct / len(predicted_values)

        return accuracy

    def stopping_condition(self, losses, accs):
        # if validation loss started to decrease during last iters
        should_stop = True
        for i in np.arange(2, 4):
            if accs[-1] > accs[-i]:
                should_stop = False
        return should_stop

    def apply_regularization(self, w, loss, gradient, regularization, lambda_, m):
        if regularization == 'l2':
            loss += lambda_ / (2 * m) * np.squeeze(w.T.dot(w))
            gradient += lambda_ / m * w
        elif regularization == 'l1':
            loss += lambda_ / (2 * m) * np.sum(np.abs(w))
            gradient += lambda_ / m * np.sum((w >= 0) * 1 + (w < 0) * -1)
        return loss, gradient

    def _learn_using_GD(self, y, tx, w, fn, gamma, lambda_, regularization):
        loss, grad = fn(y, tx, w, lambda_)
        loss, grad = self.apply_regularization(w, loss, grad, regularization, lambda_, tx.shape[0])
        w = w - gamma * grad
        return loss, w

    def _learn_using_SGD(self, y, tx, w, batch_size, fn, gamma, lambda_, regularization):
        """Stochastic gradient descent."""
        for y_batch, tx_batch in self._batch_iter(y, tx, batch_size=batch_size, num_batches=1):
            loss, grad = fn(y_batch, tx_batch, w, lambda_)
            loss, grad = self.apply_regularization(w, loss, grad, regularization, lambda_, tx.shape[0])
            w = w - gamma * grad
        return loss, w

    def _batch_iter(self, y, tx, batch_size, num_batches=1, shuffle=True):
        """
        Generate a minibatch iterator for a dataset.
        Takes as input two iterables (here the output desired values 'y' and the input data 'tx')
        Outputs an iterator which gives mini-batches of `batch_size` matching elements from `y` and `tx`.
        Data can be randomly shuffled to avoid ordering in the original data messing with the randomness of the minibatches.
        Example of use :
        for minibatch_y, minibatch_tx in batch_iter(y, tx, 32):
            <DO-SOMETHING>
        """
        data_size = len(y)

        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_y = y[shuffle_indices]
            shuffled_tx = tx[shuffle_indices]
        else:
            shuffled_y = y
            shuffled_tx = tx
        for batch_num in range(num_batches):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            if start_index != end_index:
                yield shuffled_y[start_index:end_index], shuffled_tx[start_index:end_index]

    def train(self, y, tx, y_test, tx_test, fn, max_iters=8000, gamma=1, batch_size=None, all_output=False,
              reg_lambda=None, regularization=None, gamma_decay=None, reinitialize_weights=False):
        """
        If batch size is set, the SGD is used instead of GD.
        """
        self.tx = tx
        self.gamma = gamma
        self.use_logistic = fn == logistic_regression_fn

        if not hasattr(self, 'w') or reinitialize_weights:
            self.w = self.initialize_weights((tx.shape[1], 1))
            self.best_weights = self.w

        self.best_acc = 0
        self.losses = []
        self.test_accs = []
        self.test_losses = []
        for iter in range(max_iters):
            if batch_size:
                # Using stochastic gradient descent
                loss, self.w = self._learn_using_SGD(y, tx, self.w, batch_size, fn, gamma=self.gamma,
                                                     lambda_=reg_lambda, regularization=regularization)
            else:
                loss, self.w = self._learn_using_GD(y, tx, self.w, fn, gamma=self.gamma, lambda_=reg_lambda,
                                                    regularization=regularization)

            # Save previous losses to check convergence
            self.losses.append(loss)

            if all_output or (iter % 50 == 0):
                acc = self.compute_accuracy(tx, y)
                test_loss, _ = fn(y_test, tx_test, self.w, lambda_=reg_lambda)
                test_acc = self.compute_accuracy(tx_test, y_test)

                self.test_losses.append(test_loss)
                self.test_accs.append(test_acc)
                print(
                    "Iter={}, train_loss={:.5f}, train_acc={:.4f}, test_loss={:.5f}, test_acc={:.4f}, lr={:.4f}".format(
                        iter, loss, acc, test_loss, test_acc, self.gamma))

                # To reload best model
                if acc > self.best_acc:
                    self.best_weights = self.w
                    self.best_acc = acc

                # if len(self.test_accs) > 3 and self.stopping_condition(self.test_losses, self.test_accs) and not batch_size:
                #   break

            if iter > 0 and iter % 100 == 0 and gamma_decay:
                self.gamma *= gamma_decay
                self.gamma = max(self.gamma, 0.1)

        return self.losses, self.best_weights, self.test_losses, self.test_accs