#!/usr/bin/env python3

'''
Signal detection and removal of celestial bodies using the k-means clustering method.
'''

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import scipy.optimize as opt
from extract_features import *

def k_means(img, k):
    """
    k-means algorithm to find clusters in the image.

    Parameters:
        img (numpy.ndarray): The input image.
        k (int): The number of clusters.

    Returns:
        tuple: Points considered, cluster centers, and vector with point classifications.
    """
    ind = np.where(img > 0.05)  # Consider parts of the signal > 5% of the central signal
    X = np.array([i for i in zip(ind[0], ind[1])])
    n = np.shape(X)[0]  # Number of points considered
    cluster = np.zeros(n, dtype=int)  # Vector that collects cluster memberships for each point in X

    c = (np.random.rand(k, 2) - 0.5) * 10 + np.full((k, 2), 29.5)  # Initialize k centers near the center of the image
    d = np.zeros((k, n), dtype=float)  # Initialize distances from centers

    tol = 10**(-3)
    aux = True

    while aux:
        delta = 0  # When delta < tol, the loop stops

        for i in range(k):  # Calculate distances from centers
            d[i] = np.linalg.norm(X - c[i], axis=1)

        for i in range(n):  # Classify points
            cluster[i] = np.argmin(d[:, i])

        for i in range(k):  # Update cluster centers
            ind = np.where(cluster == i)
            if np.shape(ind)[1] > 0:  # Avoid calculating mean of an empty array
                delta += np.linalg.norm(c[i] - np.mean(X[ind], axis=0))
                c[i] = np.mean(X[ind], axis=0)

        if delta < tol:
            aux = False

    return X, c, cluster

def linear_regression(x, g):
    """
    Returns the mean slope and offset given x-axis and g values.

    Parameters:
        x (numpy.ndarray): x-axis values.
        g (numpy.ndarray): Corresponding g values.

    Returns:
        tuple: Mean slope and offset.
    """
    ind_max = np.argmax(g)
    G = g[ind_max]
    m = np.zeros(np.shape(x)[0] - 1, dtype=float)  # Collect slope values, all negative (2D function will be a cone)

    aux = 0
    for j in range(np.shape(x)[0]):
        if j == ind_max:
            continue
        delta_x = ind_max - j
        delta_y = g[ind_max] - g[j]
        m[aux] = -np.abs(delta_y / delta_x)
        aux += 1

    return np.mean(m), G

def cone(x, y, mu, m, G):
    """
    Defines a 2D cone function, assuming extraneous signals have this shape.

    Parameters:
        x (float): x-coordinate.
        y (float): y-coordinate.
        mu (tuple): Center of the cone.
        m (float): Slope.
        G (float): Offset.

    Returns:
        float: Cone value at (x, y).
    """
    mux = mu[0]
    muy = mu[1]

    return ReLU(m * np.sqrt((x - mux)**2 + (y - muy)**2) + G)

def ReLU(x):
    """
    ReLU activation function.

    Parameters:
        x (float): Input value.

    Returns:
        float: Output value.
    """
    return np.maximum(x, 0)

def find_c(c):
    """
    Finds the central signal (29.5, 29.5), returns the cluster index l (l-th cluster -> galaxy).

    Parameters:
        c (numpy.ndarray): Cluster centers.

    Returns:
        int: Index of the central cluster.
    """
    mean = np.mean(c, axis=1)  # Mean of galactic center coordinates should be 29.5
    mean = np.abs(mean - 29.5)
    l = np.argmin(mean)

    return l

def clean_k(img, X, c, cluster, k, l):
    """
    Removes signal belonging to any cluster other than l.

    Parameters:
        img (numpy.ndarray): Input image.
        X (numpy.ndarray): Points considered.
        c (numpy.ndarray): Cluster centers.
        cluster (numpy.ndarray): Vector with point classifications.
        k (int): Number of clusters.
        l (int): Central cluster index.

    Returns:
        numpy.ndarray: Cleaned image.
    """
    for i in range(k):
        if i == l:  # Skip l-th cluster
            continue

        mean_y = round(c[i, 0])  # Mean on y-axis, assuming spherical symmetry -> study 1D signal shape with fixed y = mean_y

        ind = np.where(cluster == i)
        ind_x = np.where(X[ind][:, 0] == mean_y)[0]
        x = np.array(X[ind][ind_x, 1])  # Considered x points
        g = np.array(img[mean_y, x])[:, 0]  # Values that img assumes in (mean_y, x), assumed to follow a linear decrease

        m, G = linear_regression(x, g)  # Mean slope

        signal = np.zeros((60, 60))
        for I in range(60):
            for J in range(60):
                signal[I, J] = cone(I, J, c[i], m, G)

    return np.abs(img[:, :, 0] - signal)

if __name__ == "__main__":
    # Load image
    img = np.load('Set/Train/Ellittiche/el_1139.npy')

    k = 2  # Number of clusters

    # Apply k-means clustering
    X, c, cluster = k_means(img, k)
    l = find_c(c)  # Find central cluster
    img_cleaned = clean_k(img, X, c, cluster, k, l)  # Clean image

    # Plot original image
    plt.imshow(img)
    plt.colorbar()
    plt.show()
    plt.close()

    # Plot cleaned image
    plt.imshow(img_cleaned)
    plt.colorbar()
    plt.show()
    plt.close()
