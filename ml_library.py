#!/usr/bin/env python3

"""
Simple machine learning library implementing methods for logistic regression
and MLP.
"""

import numpy as np
import matplotlib.pyplot as plt
import timeit
import cv2
from extract_features import *


def sigmoid(z):
    """Compute the sigmoid function (assumed as the probability of the galaxy being spiral)."""
    return 1 / (1 + np.exp(-z))


def logreg_inference(x, w, b):
    """Infer the probability that the galaxies are spiral."""
    logit = np.dot(x, w) + b
    p = sigmoid(logit)
    return p


def cross_entropy(P, Y):
    """Compute binary cross-entropy."""
    return np.mean(-Y * np.log(P) - (1 - Y) * np.log(1 - P))


def logreg_train(X, Y, lr, n_step):
    """
    Train logistic regression using gradient descent with momentum.

    Parameters:
    X : array-like, shape (m, n)
        Feature matrix.
    Y : array-like, shape (m,)
        Labels.
    lr : float
        Learning rate.
    n_step : int
        Number of gradient descent steps.
    """
    m, n = np.shape(X)
    w = np.random.rand(n)  # Random initialization of weights
    b = 0

    v_w = np.zeros_like(w)  # Velocity for gradient descent
    v_b = 0

    start = timeit.default_timer()  # Timer to estimate training time
    print("Please wait a few seconds...")

    for step in range(n_step):
        P = logreg_inference(X, w, b)

        if step == 100:
            end = timeit.default_timer()
            print("Estimated training time = ", round((end - start) / 100 * n_step / 60), " min")

        grad_w = np.dot(np.transpose(X), (P - Y)) / m  # Gradient
        grad_b = np.mean(P - Y)

        v_w = 0.99 * v_w + grad_w
        v_b = 0.99 * v_b

        w -= lr * v_w  # Gradient descent step
        b -= lr * v_b

    return w, b


def logreg_train_batch(X, Y, X_validation, Y_validation, epochs, batch_size, lr, lambd):
    """
    Train logistic regression using stochastic gradient descent with momentum and L2 regularization.

    Parameters:
    X : array-like, shape (m, n)
        Feature matrix.
    Y : array-like, shape (m,)
        Labels.
    X_validation : array-like, shape (m_v, n)
        Validation feature matrix.
    Y_validation : array-like, shape (m_v,)
        Validation labels.
    epochs : int
        Number of epochs.
    batch_size : int
        Size of each batch.
    lr : float
        Learning rate.
    lambd : float
        L2 regularization parameter.
    """
    m, n = np.shape(X)
    w = np.random.rand(n)  # Random initialization of weights
    b = 0

    v_w = np.zeros_like(w)  # Velocity for gradient descent
    v_b = 0

    start = timeit.default_timer()  # Timer to estimate training time
    print("Please wait a few seconds...")

    for epoch in range(epochs):
        perm = np.random.permutation(m)  # Shuffle dataset
        X = X[perm, :]
        Y = Y[perm]

        if epoch == 1:
            end = timeit.default_timer()
            print("Estimated training time = ", round((end - start) * (epochs - 1) / 60), " min")

        for i in range(0, m, batch_size):
            X_batch = X[i: i + batch_size, :]  # Batch features
            Y_batch = Y[i: i + batch_size]  # Batch labels

            P = logreg_inference(X_batch, w, b)

            grad_w = np.dot(np.transpose(X_batch), (P - Y_batch)) / batch_size  # Gradient
            grad_b = np.mean(P - Y_batch)

            v_w = 0.99 * v_w + grad_w + 2 * lambd * w  # L2 regularization
            v_b = 0.99 * v_b + grad_b

            w -= lr * v_w  # Gradient descent step
            b -= lr * v_b

    return w, b


def ReLU(x):
    """ReLU activation function."""
    return np.maximum(x, 0)


def der_ReLU(x):
    """Derivative of ReLU."""
    return np.heaviside(x, 0)


def softmax(z):
    """Compute softmax probabilities."""
    return np.exp(z) / np.sum(np.exp(z), axis=1, keepdims=True)


def hot_state(Y, nc):
    """Convert labels to one-hot encoding."""
    ns = np.shape(Y)[0]  # Number of samples

    y = np.zeros((ns, nc), dtype=float)
    for i in range(ns):
        y[i, Y[i]] = 1

    return y


def forward(X, W, b, layers):
    """Forward propagation of activation values."""
    ns = np.shape(X)[0]  # Number of samples
    nl = np.shape(layers)[0]  # Number of layers

    x = [np.zeros((ns, n), dtype=float) for n in layers]  # Initialize activation values
    x[0] = X
    z = [np.zeros((ns, n), dtype=float) for n in layers]

    for l in range(nl - 1):
        z[l + 1] = np.dot(x[l], W[l]) + b[l]  # Compute activations
        x[l + 1] = ReLU(z[l + 1])

    return x, z


def gradient(W, b, x, y, z, nl, layers):
    """Compute the gradient of weights and biases for each layer."""

    grad_W = [np.zeros((m, n)) for m, n in zip(layers[:-1], layers[1:])]   # Initialize gradients
    grad_b = [np.zeros(n, dtype=float) for n in layers[1:]]

    p = softmax(z[nl - 1])  # Estimate class probabilities

    delta = p - y

    for l in range(nl - 1, 0, -1):  # Compute gradient for each layer
        ns = np.shape(x[l - 1])[0]  # Number of samples in batch
        aux = delta * der_ReLU(z[l])

        for s in range(ns):  # Iterate over samples
            grad_W[l - 1] += np.outer(x[l - 1][s], aux[s])  # Gradient for weights
            grad_b[l - 1] += aux[s]

        grad_W[l - 1] /= ns
        grad_b[l - 1] /= ns

        delta = np.dot(aux, W[l - 1].T)  # Update delta for previous layer

    return grad_W, grad_b


def mlp_train(X, Y, X_validation, Y_validation, layers, epochs, batch_size, lr, lambd):
    """
    Train multi-layer perceptron using stochastic gradient descent with momentum and L2 regularization.

    Parameters:
    X : array-like, shape (m, n)
        Feature matrix.
    Y : array-like, shape (m,)
        Labels.
    X_validation : array-like, shape (m_v, n)
        Validation feature matrix.
    Y_validation : array-like, shape (m_v,)
        Validation labels.
    layers : list of int
        List specifying the number of neurons in each layer.
    epochs : int
        Number of epochs.
    batch_size : int
        Size of each batch.
    lr : float
        Learning rate.
    lambd : float
        L2 regularization parameter.
    """

    nl = np.shape(layers)[0]  # Number of layers
    ns = np.shape(X)[0]  # Number of samples
    nc = np.max(Y) + 1  # Number of classes

    layers[0] = np.shape(X)[1]  # Input neurons = number of features
    layers[-1] = nc  # Output neurons = number of classes

    W = [np.random.normal(size=(m, n)) * np.sqrt(2 / m) for m, n in zip(layers[:-1], layers[1:])]  # Initialize weights
    b = [np.zeros(n, dtype=float) for n in layers[1:]]

    v_W = [np.zeros((m, n)) for m, n in zip(layers[:-1], layers[1:])]   # Velocity for gradient descent
    v_b = [np.zeros(n, dtype=float) for n in layers[1:]]

    start = timeit.default_timer()  # Timer to estimate training time
    print("Please wait a few seconds...")

    for epoch in range(epochs):
        perm = np.random.permutation(ns)  # Shuffle dataset
        X = X[perm, :]
        Y = Y[perm]

        if epoch == 1:
            end = timeit.default_timer()
            print("Estimated training time = ", round((end - start) * (epochs - 1) / 60), " min")

        for i in range(0, ns, batch_size):
            X_batch = X[i: i + batch_size, :]  # Batch features
            Y_batch = Y[i: i + batch_size]  # Batch labels

            y = hot_state(Y_batch, nc)  # One-hot encoding of labels

            x, z = forward(X_batch, W, b, layers)  # Forward propagation

            grad_W, grad_b = gradient(W, b, x, y, z, nl, layers)  # Compute gradient

            for l in range(len(v_b)):
                v_W[l] = 0.99 * v_W[l] + grad_W[l] + lambd * W[l]  # L2 regularization
                v_b[l] = 0.99 * v_b[l] + grad_b[l]

                W[l] -= lr * v_W[l]  # Gradient descent step
                b[l] -= lr * v_b[l]

    return W, b


def mlp_inference(X, W, b, layers):
    """
    Classify samples based on estimated probabilities using a multi-layer perceptron.

    Parameters:
    X : array-like, shape (m, n)
        Feature matrix.
    W : list of arrays
        Weights of each layer.
    b : list of arrays
        Biases of each layer.
    layers : list of int
        List specifying the number of neurons in each layer.
    """
    ns = np.shape(X)[0]  # Number of samples

    x, z = forward(X, W, b, layers)  # Forward propagation

    p = softmax(z[-1])  # Softmax probabilities

    guess = np.zeros(ns, dtype=int)  # Classification results

    for s in range(ns):
        guess[s] = np.argmax(p[s])

    return guess, p
