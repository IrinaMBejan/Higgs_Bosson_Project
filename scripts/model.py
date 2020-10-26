import numpy as np
from proj1_helpers import create_csv_submission
from proj1_helpers import predict_labels
from utils import batch_iter
from implementations import compute_mse, calculate_mse, compute_gradient, sigmoid


def logistic_regression_fn(y, tx, w, lambda_=None):
    """Compute the loss: negative log likelihood and penalize the loss based on class weights."""

    # Class weights computed according to the percentage of inputs of each classification in the dataset
    class_weights = {0: 0.666, 1: 0.333}

    pred = sigmoid(np.dot(tx, w))

    # Make sure that we log will not get 0 values, therefore resulting in -inf.
    epsilon = 1e-30
    pred = np.where(pred == 0.0, epsilon, pred)
    pred = np.where(pred == 1.0, 1 - epsilon, pred)

    # Compute the loss and gradient scaled by the size of the dataset
    loss = (1 / tx.shape[0]) * (class_weights[0] * y.T.dot(np.log(pred)) +
                                class_weights[1] * (1 - y).T.dot(np.log(1 - pred)))
    gradient = np.dot(tx.T, (pred - y)) / tx.shape[0]
    return np.squeeze(-loss), gradient


def least_squares_fn(y, tx, w, lambda_=None):
    """Compute the gradient using least squares"""
    gradient, error = compute_gradient(y, tx, w)
    loss = calculate_mse(error)

    return loss, gradient


def he_weights_initialization(shape):
    """Initialize weights using He initialization."""
    w = np.random.randn(shape[0], shape[1]) * np.sqrt(2 / shape[1])
    return w


class Model(object):
    """Class to represent a model and encapsulate training details."""

    def __init__(self):
        pass

    def predict(self, inputs):
        """Computes prediction on given inputs based on model's weights and method used."""
        if self.use_logistic:
            return self.predict_labels_logistic(self.w, inputs)
        return predict_labels(self.w, inputs)

    def predict_labels_logistic(self, weights, data):
        """Predicts labels given the input features and weights when using logistic regression"""
        y_pred = sigmoid(np.dot(data, weights))
        y_pred[np.where(y_pred < 0.5)] = 0
        y_pred[np.where(y_pred >= 0.5)] = 1
        return y_pred

    def create_submission(self, inputs, name):
        """Create the submission csv file.
        Args:
            inputs: Test data to run the predictions on
            name: The name of the submission file
        """
        pred = self.predict(inputs)

        if self.use_logistic:
            pred[np.where(pred == 0)] = -1

        create_csv_submission(list(range(350000, 350000 + len(pred))), pred, name)
        return name

    def initialize_weights(self, shape):
        """Initialize the weights of the model using the given shape."""
        # Our function is not convex, so initialization with zero is not helpful
        return he_weights_initialization(shape)

    def compute_accuracy(self, inputs, true_values):
        """"Computation of accuracy on given inputs and their true value."""
        predicted_values = self.predict(inputs)
        number_correct = np.sum(predicted_values == true_values)
        accuracy = number_correct / len(predicted_values)

        return accuracy

    def stopping_condition(self, losses):
        """Checks the stop condition for training to prevent overfitting."""

        # if last validation loss is higher than during last three iterations, stop training
        should_stop = True
        for iteration in np.arange(2, 5):
            if losses[-1] < losses[-iteration]:
                should_stop = False
        return should_stop

    def apply_regularization(self, w, loss, gradient, regularization, lambda_, m):
        """
        Applies the specified regularization method to the loss and gradients.
        Args:
            w: weights of the model
            loss: the current loss
            gradient: the current gradient
            regularization: can be either 'l1' or 'l2'
            lambda_: the regularization factor
            m: size of the dataset for scaling.
        """
        if regularization == 'l2':
            loss += lambda_ / (2 * m) * np.squeeze(w.T.dot(w))
            gradient += lambda_ / m * w
        elif regularization == 'l1':
            loss += lambda_ / (2 * m) * np.sum(np.abs(w))
            gradient += lambda_ / m * np.sum((w >= 0) * 1 + (w < 0) * -1)
        return loss, gradient

    def _learn_using_GD(self, y, tx, w, fn, gamma, lambda_, regularization):
        """
        Optimizes the model using Gradient Descent.
        Args:
            y: Labels of the dataset
            tx: Input features
            w: Current weights
            fn: Method for computing loss and gradients
            gamma: The learning rate, to be between 0 and 1.
            lambda_: The regularization factor
            regularization: The specified regularization. It can be None.
        """
        loss, grad = fn(y, tx, w, lambda_)
        loss, grad = self.apply_regularization(w, loss, grad, regularization, lambda_, tx.shape[0])
        w = w - gamma * grad
        return loss, w

    def _learn_using_SGD(self, y, tx, w, batch_size, fn, gamma, lambda_, regularization):
        """
        Optimizes the model using Stochastic Gradient Descent.
        Args:
            y: Labels of the dataset
            tx: Input features
            w: Current weights
            batch_size: The size of the batch used
            fn: Method for computing loss and gradients
            gamma: The learning rate, to be between 0 and 1.
            lambda_: The regularization factor
            regularization: The specified regularization. It can be None.
        """
        for y_batch, tx_batch in batch_iter(y, tx, batch_size=batch_size, num_batches=1):
            loss, grad = fn(y_batch, tx_batch, w, lambda_)
            loss, grad = self.apply_regularization(w, loss, grad, regularization, lambda_, tx.shape[0])
            w = w - gamma * grad
        return loss, w

    def train(self, y, tx, y_test, tx_test, fn, max_iters=8000, gamma=1, batch_size=None, all_output=False,
              reg_lambda=None, regularization=None, gamma_decay=None, reinitialize_weights=False):
        """
        Trains the model.
        Args:
            y: Labels given for the training dataset
            tx: Training dataset
            y_test: Labels given for the validation dataset
            tx_test: Validation dataset
            fn: Method for computing loss and gradients. It must respect the signature of existing methods.
            max_iters: The maximum number of iterations for the model
            gamma: The learning rate, to be between 0 and 1.
            batch_size: The size of the batch used. If specified, the SGD is used instead of GD.
            all_output: Output the loss after each iteration
            reg_lambda: The regularization factor
            regularization: The specified regularization. It can be None.
            gamma_decay: The factor to decay the value of the learning rate.
            reinitialize_weights: If true, reinitialize the weights before training.
        """

        self.tx = tx
        self.gamma = gamma
        self.use_logistic = fn == logistic_regression_fn

        if not hasattr(self, 'w') or reinitialize_weights:
            self.w = self.initialize_weights((tx.shape[1], 1))
            self.best_weights = self.w

        self.best_acc = 0

        # Store all losses and test data to check convergence
        self.losses = []
        self.test_accs = []
        self.test_losses = []

        for iter in range(max_iters):
            if batch_size:
                loss, self.w = self._learn_using_SGD(y, tx, self.w, batch_size, fn, gamma=self.gamma,
                                                     lambda_=reg_lambda, regularization=regularization)
            else:
                loss, self.w = self._learn_using_GD(y, tx, self.w, fn, gamma=self.gamma, lambda_=reg_lambda,
                                                    regularization=regularization)

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

                # To reload best model when training ends
                if acc > self.best_acc:
                    self.best_weights = self.w
                    self.best_acc = acc

                # The stopping condition is applied only in the case of gradient descent.
                if len(self.test_accs) > 100 and self.stopping_condition(self.test_accs) and not batch_size:
                   break

            if iter > 0 and iter % 100 == 0 and gamma_decay:
                self.gamma *= gamma_decay
                # The learning rate is capped to 0.1 to not slow down the training too hard
                self.gamma = max(self.gamma, 0.1)

        return self.losses, self.best_weights, self.test_losses, self.test_accs
