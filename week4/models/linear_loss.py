import numpy as np


def linear_loss_naive(W, X, y, reg):
    """
    Linear loss function, naive implementation (with loops)

    Inputs have dimension D, there are N examples.

    Inputs:
    - W: A numpy array of shape (D, 1) containing weights.
    - X: A numpy array of shape (N, D) containing data.
    - y: A numpy array of shape (N,) containing training labels; y[i] = c means
        that X[i] has label c, where c is a real number.
    - reg: (float) regularization strength

    Returns a tuple of:
    - loss as single float
    - gradient with respect to weights W; an array of same shape as W
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the linear loss and its gradient using explicit loops.     #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    N, D = np.shape(X)
    dW = np.reshape(dW, (D, 1))

    # compute loss
    for i in range(N):
        sum = 0
        for j in range(D):
            sum += X[i][j] * W[j]
        loss += (y[i] - sum) ** 2
    loss = 1 / (2 * N) * loss

    # compute L1 regularization
    regularization = 0
    for i in range(D):
        regularization += W[i] ** 2

    # compute loss function with regularization
    loss = loss + reg / (2 * N) * regularization

    # compute gradient
    for i in range(D):
        for j in range(N):
            dW[i] += X[j][i] * (np.sum(X[j] * np.transpose(W)) - y[j])
        dW[i] = dW[i] / N + reg / N * W[i]
    #############################################################################
    #                          END OF YOUR CODE                                 #
    #############################################################################

    return loss, dW


def linear_loss_vectorized(W, X, y, reg):
    """
    Linear loss function, vectorized version.

    Inputs and outputs are the same as linear_loss_naive.
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the linear loss and its gradient using no explicit loops.  #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    N, D = np.shape(X)
    y = np.reshape(y, (N, 1))
    W = np.reshape(W, (D, 1))
    loss = 1 / (2 * N) * (np.sum((np.matmul(X, W) - y) ** 2)) + reg / (2 * N) * np.sum(W ** 2)
    dW = 1 / N * np.matmul(np.transpose(X), np.matmul(X, W) - y) + (reg / N) * W
    #############################################################################
    #                          END OF YOUR CODE                                 #
    #############################################################################

    return loss, dW
