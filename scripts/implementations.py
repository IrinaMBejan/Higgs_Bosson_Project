import numpy as np


def compute_mse(y, tx, w):
    """Calculate the loss using mse."""
    e = y - np.dot(tx, w)
    mse = np.dot(e.T, e) / (2 * len(e))
    return mse


def calculate_mse(e):
    """Calculate the mse for vector e."""
    return 1 / 2 * np.mean(e ** 2)


def compute_gradient(y, tx, w):
    """Computes the gradient."""
    e = y - np.dot(tx, w)
    gradient = - np.dot(tx.T, e) / len(y)
    return gradient, e


def sigmoid(t):
    """Apply the sigmoid function on t."""
    return 1.0 / (1.0 + np.exp(-t))


def logistic_loss_gradient(y, tx, w):
    """Computes the cost by negative log likelihood and the gradient of the loss."""
    pred = sigmoid(np.dot(tx, w))
    loss = y.T.dot(np.log(pred)) + (1 - y).T.dot(np.log(1 - pred))
    gradient = np.dot(tx.T, (pred - y))

    return np.squeeze(-loss), gradient


def penalized_logistic_loss_gradient(y, tx, w, lambda_):
    """Computes the loss and the gradient of the loss."""
    loss, gradient = logistic_loss_gradient(y, tx, w)
    loss += lambda_ * np.squeeze(w.T.dot(w))
    gradient += 2 * lambda_ * w

    return loss, gradient


def batch_iter(y, tx, batch_size, num_batches=1, shuffle=True):
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


def least_squares_GD(y, tx, initial_w, max_iters, gamma):
    """
    Linear regression using gradient descent.

    :param initial_w: initial weight vector
    :param max_iters: number of iterations to run
    :param gamma: step-size
    :return: the last weight vector, the value of the corresponding cost function
    """
    # define parameters to store w and loss
    ws = [initial_w]
    losses = []
    w = initial_w
    for n_iter in range(max_iters):
        # compute gradient and loss
        gradient, error = compute_gradient(y, tx, w)
        loss = calculate_mse(error)
        # update w by gradient
        w = w - gamma * gradient
        # store w and loss
        ws.append(w)
        losses.append(loss)
        # print("Gradient Descent({bi}/{ti}): loss={l}".format(bi=n_iter, ti=max_iters - 1, l=loss))

    return ws[-1], losses[-1]


def least_squares_SGD(y, tx, initial_w, max_iters, gamma):
    """
    Linear regression using stochastic gradient descent.

    :param initial_w: initial weight vector
    :param max_iters: number of iterations to run
    :param gamma: step-size
    :return: the last weight vector, the value of the corresponding cost function
    """
    ws = [initial_w]
    losses = []
    w = initial_w
    batch_size = 1
    num_batches = 32
    for n_iter in range(max_iters):
        # compute stochastic gradient and loss
        for batch_y, batch_x in batch_iter(y, tx, batch_size, num_batches):
            gradient, error = compute_gradient(batch_y, batch_x, w)
            loss = calculate_mse(error)
            # update w by gradient
            w = w - gamma * gradient
            # store w and loss
            ws.append(w)
            losses.append(loss)
            # print("Stochastic gradient Descent({bi}/{ti}): loss={l}".format(
            #     bi=n_iter, ti=max_iters - 1, l=loss))

        return ws[-1], losses[-1]


def least_squares(y, tx):
    """
    Least squares regression using normal equations.

    :return: the last weight vector, the value of the corresponding cost function
    """
    a = np.dot(tx.T, tx)
    b = np.dot(tx.T, y)
    weights = np.linalg.solve(a, b)
    mse = compute_mse(y, tx, weights)

    return weights, mse


def ridge_regression(y, tx, lambda_):
    """
    Ridge regression using normal equations.

    :return: the last weight vector, the value of the corresponding cost function
    """
    lambdaPrim = lambda_ * np.identity(tx.shape[1]) * 2 * tx.shape[0]
    a = np.dot(tx.T, tx) + lambdaPrim
    b = np.dot(tx.T, y)
    weights = np.linalg.solve(a, b)
    loss = compute_mse(y, tx, weights)

    return weights, loss


def logistic_regression(y, tx, initial_w, max_iters, gamma):
    """
    Logistic regression using gradient descent or SGD.

    :param initial_w: initial weight vector
    :param max_iters: number of iterations to run
    :param gamma: step-size
    :return: the last weight vector, the value of the corresponding cost function
    """
    # init parameters
    threshold = 1e-8
    ws = [initial_w]
    losses = []
    w = initial_w

    # start the logistic regression
    for iter in range(max_iters):
        loss, gradient = logistic_loss_gradient(y, tx, w)
        w = w - gamma * gradient
        # if iter % 10 == 0:
        #     print("Current iteration={}, loss={}".format(iter, loss))
        losses.append(loss)
        ws.append(w)
        # added any.
        if len(losses) > 1 and np.abs((losses[-1] - losses[-2]).any()) < threshold:
            break

    return ws[-1], losses[-1]


def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma):
    """
    Regularized logistic regression using gradient descent or SGD.

    :param initial_w: initial weight vector
    :param max_iters: number of iterations to run
    :param gamma: step-size
    :return: the last weight vector, the value of the corresponding cost function
    """
    ws = [initial_w]
    losses = []
    w = initial_w

    for iter in range(max_iters):
        loss, gradient = penalized_logistic_loss_gradient(y, tx, w, lambda_)
        w = w - gamma * gradient
        # if iter % 10 == 0:
        #     print("Current iteration={i}, loss={l}".format(i=iter, l=loss))
        losses.append(loss)
        ws.append(w)

    return ws[-1], losses[-1]
