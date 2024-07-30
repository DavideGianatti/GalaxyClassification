#!/usr/bin/env python3

'''
Methods to construct and extract Low-Level Features from images of spiral and elliptical galaxies
'''

import numpy as np
import matplotlib.pyplot as plt
import os
import timeit

def brightness(X):
    """
    Returns the intensity of the radiation by summing the value of each pixel.
    """
    return np.sum(X)

def circle_contour(X):
    """
    Evaluates the distance d of each point on the galaxy's contour from the center
    and returns the mean and the relative standard deviation of d.
    Elliptical galaxies are well approximated by a circular profile, which leads to a lower dispersion of d.
    """
    aux = (X > 0.05) * (X < 0.075)  # the galaxy's contour is defined as the set of pixels with values between 5% and 7.5% of the central signal
    ind = np.array(np.where(aux))
    N = np.shape(ind)[1]
    d = np.zeros(N, dtype=float)  # vector that collects the distances from the center

    for i in range(N):
        d[i] = np.sqrt((ind[0][i] - 29.5) ** 2 + (ind[1][i] - 29.5) ** 2)

    if len(d) != 0:  # to avoid strange behavior when the vector d is empty
        std = np.std(d)
        mean = np.mean(d)
    else:
        std = mean = 10**(-3)  # not equal to 0 to avoid divergences

    return mean, std / mean

def upndown(X):
    """
    Returns the number of times a point, following two trajectories around the galaxy,
    records a change in the slope in the profile of the norm of the gradient field.
    This exposes the presence of depressions, as valleys in spirals are local minima in brightness,
    well evidenced by variations in the gradient.
    Empirically, the following trajectories are good: (35,15) --> (35,45) and (15,25) --> (45,25).
    """
    norm = mean(norm_grad(X))  # it is essential to regularize the field to avoid derivative instabilities
    ind = np.where(norm < 0.01)  # zero all pixels below a certain threshold to avoid noise (ellipticals halve the count, spirals decrease by a few units)
    norm[ind] = 0

    conteggio = 0
    for i in range(30):  # the slope is described by a forward derivative
        if (norm[35][15 + i] - norm[35][15 + i - 1]) * (norm[35][15 + i + 1] - norm[35][15 + i]) < 0:
            # if derivatives are opposite = True (along the first path)
            conteggio += 1

        if (norm[15 + i][25] - norm[15 + i - 1][25]) * (norm[15 + i + 1][25] - norm[15 + i][25]) < 0:
            # if derivatives are opposite = True (along the second path)
            conteggio += 1

    return conteggio

def cumulative(X):
    """
    Returns the number of radius increments, on which the 2D cumulative is calculated, required to reach a plateau.
    Ellipticals should reach it sooner as they are more compact, also solving the problem of other celestial bodies in the image.
    """
    N = int(np.shape(X)[0] / 2)
    dcum = np.zeros(N, dtype=float)  # vector containing the increments of the cumulative normalized to the number of pixels added at each step (essentially the mean intensity of new pixels added to the cumulative count at each step), the idea is to find a plateau after a certain number of steps (this occurs when dcum[i] < tol)
    tol = 0.05

    for i in range(0, N):
        dcum[i] = (np.sum(cut(X, i + 1)) - np.sum(cut(X, i))) / ((i + 1) * 4)
        # see the cut method, the cumulative is the sum of pixels inside a circle of radius r centered at the galactic center, the circle is "approximated" by a square of side 2r

    r = np.where(dcum < tol)[0]  # first index such that increment is below tol

    if len(r) > 0:  # to avoid the case where r is an empty string
        r = r[0]
    else:
        r = N

    return r

def fourier(X):
    """
    Returns the weights of the first three components of the 1D Fourier analysis on a slice of the galaxy.
    The raw image is not considered; instead, the norm of the gradient is used because depressions are better visualized.
    """
    Y = mean(norm_grad(X))

    N = 3  # number of Fourier components to store
    a = np.zeros(N, dtype=float)  # vector to save the Fourier weights
    t = 20  # slice size
    T = t * 2  # period = double the interval considered, since we consider an even extension of the function (slice of t pixels)
    galaxy_slice = Y[32][30:30 + t]  # a slice starting from the center of the galaxy extending up to t pixels, the periodic extension is made so that the function is even (origin at the galaxy center and reflected about it, considering only cosine terms)

    galaxy_slice = galaxy_slice - np.min(galaxy_slice)  # offset is removed
    galaxy_slice = galaxy_slice / np.sum(galaxy_slice)  # normalized vector (dx = 1), making the analysis independent of the galaxy's intensity

    for i in range(N):
        n = i + 1  # skipping the term n = 0

        integral = 0
        for j in range(t):
            integral = integral + galaxy_slice[j] * np.cos(2 * np.pi / T * n * j)

        integral = 4 / T * integral

        a[i] = integral

    return a

def mirror(X):
    """
    Returns the sum of differences between the pixels of the image and the pixels of the image mirrored with respect to the bisector of xy.
    Elliptical galaxies appear to be more symmetric upon reflection.
    """
    mir = np.absolute(X - np.transpose(X)) / np.sum(X)  # divided by np.sum(X) to make the feature independent of brightness

    return np.sum(mir)

def features(X):
    """
    Returns the feature vector associated with the image X.
    """
    features = np.zeros(6)
    X = clean(X)  # removes potential secondary celestial bodies and attenuates noise

    features[0] = brightness(X)
    features[1] = circle_contour(X)[0]
    features[2] = circle_contour(X)[1]
    features[3] = upndown(X)
    features[4] = cumulative(X)  # note that this is not exactly analogous to circle_contour[0], they will surely be correlated, but for very flattened galaxies, they take very different values
    features[5] = mirror(X)

    return features

def extract_features(path):
    """
    Returns the feature matrix X and the class vector Y.
    The path to the Train or Test directory should be specified.
    """
    listdata_s = os.listdir(path + "/Spirali")
    listdata_el = os.listdir(path + "/Ellittiche")

    N_s = len(listdata_s)  # number of spiral galaxies
    N_el = len(listdata_el)  # number of elliptical galaxies
    N_f = len(features(np.load(path + "/Spirali/" + listdata_s[0])))  # number of features

    X = np.zeros((N_s + N_el, N_f), dtype=float)  # row i-th --> i-th galaxy, column j-th --> j-th feature
    Y = np.append(np.ones(N_s, dtype=int), np.zeros(N_el, dtype=int))  # 1 --> spiral, 0 --> elliptical

    print("Please wait a few seconds...")
    start = timeit.default_timer()  # timer to estimate the actual feature extraction time

    i = 0
    for data_s in listdata_s:
        X[i] = features(np.load(path + "/Spirali/" + data_s))
        i += 1

        if i == 10:
            end = timeit.default_timer()
            print("Estimated time for feature extraction = ", round((end - start) / 10 * (N_el + N_s) / 60), " min")

    for data_el in listdata_el:
        X[i] = features(np.load(path + "/Ellittiche/" + data_el))
        i += 1

    return X, Y

def extract_features_pixel(path):
    """
    Returns the feature matrix X (where the features are considered as the pixels of the images) and the class vector Y.
    The path to the Train or Test directory should be specified.
    This method is used in mlp.py.
    """
    listdata_s = os.listdir(path + "/Spirali")
    listdata_el = os.listdir(path + "/Ellittiche")

    N_s = len(listdata_s)  # number of spiral galaxies
    N_el = len(listdata_el)  # number of elliptical galaxies
    N_f = 60 * 60  # number of features = number of pixels

    X = np.zeros((N_s + N_el, N_f), dtype=float)  # row i-th --> i-th galaxy, column j-th --> j-th feature
    Y = np.append(np.ones(N_s, dtype=int), np.zeros(N_el, dtype=int))  # 1 --> spiral, 0 --> elliptical

    print("Please wait a few seconds...")
    start = timeit.default_timer()  # timer to estimate the actual feature extraction time

    i = 0
    for data_s in listdata_s:
        X[i] = np.reshape(np.load(path + "/Spirali/" + data_s), -1)  # 60x60 matrix flattened into a 3600-dimension vector, to return to the original matrix simply use np.reshape(X, (60, -1))
        i += 1
        print(i / (N_el + N_s))
        if i == 10:
            end = timeit.default_timer()
            print("Estimated time for feature extraction = ", round((end - start) / 10 * (N_el + N_s) / 60), " min")

    for data_el in listdata_el:
        X[i] = np.reshape(np.load(path + "/Ellittiche/" + data_el), -1)
        print(i / (N_el + N_s))
        i += 1

    return X, Y

def gradient(X):
    """
    Returns the gradient vector field.
    grad[i][j][0] is the gradient component along x at point (i,j).
    """
    n_x = np.shape(X)[1]  # number of points along x-axis (= number of columns)
    n_y = np.shape(X)[0]  # number of points along y-axis (= number of rows)

    grad = np.zeros((n_x, n_y, 2), dtype=float)

    for i in range(1, n_y-1):  # note that the gradient on the edge is set to 0, be cautious as this might negatively affect the behavior of the features
        for j in range(1, n_x-1):

            grad[i][j][0] = (X[i][j+1] - X[i][j-1]) / 2  # gradient component along x
            grad[i][j][1] = (X[i-1][j] - X[i+1][j]) / 2  # increments set to 1, also note how the increments are oriented (when viewing the 2D matrix, it is immediately noticeable that the y-axis is given by the index 60 - row index)

    return grad

def rotore(X):
    """
    Returns the curl vector field.
    """
    N = np.shape(X)[0]
    rotore = np.zeros((N, N))

    for i in range(1, N-1):  # delicate edges
        for j in range(1, N-1):

            fy_x = (X[i-1][j][0] - X[i+1][j][0]) / 2  # derivative with respect to y of the x component of the field X at point (i,j)
            fx_y = (X[i][j+1][1] - X[i][j-1][1]) / 2  # vice versa

            rotore[i][j] = fy_x - fx_y

    return rotore

def contour(X):
    """
    Defines the contours of the galaxy given a certain luminosity threshold.
    Returns a matrix with values equal to 1 near the contours.
    """
    aux = (X > 0.05) * (X < 0.075)  # contours defined at about 5% of the central signal
    ind = np.where(aux)
    cont = np.zeros((60, 60))
    cont[ind[0:2][:]] = 1

    return cont

def resize_mean(X):
    """
    Changes the number of pixels in the image from 60x60 to 20x20 by averaging 3x3 blocks.
    """
    X_new = np.zeros((20, 20), dtype=float)

    for i in range(0, 20):
        for j in range(0, 20):
            cx = i * 3 + 1
            cy = j * 3 + 1

            for k in range(3):
                for l in range(3):
                    X_new[i][j] += X[cx-1+k][cy-1+l]

            X_new[i][j] = X_new[i][j] / 9

    return X_new

def resize_max(X):
    """
    Changes the number of pixels in the image from 60x60 to 20x20 by taking the maximum in 3x3 blocks.
    """
    X_new = np.zeros((20, 20))

    for i in range(0, 20):
        for j in range(0, 20):
            cx = i * 3 + 1
            cy = j * 3 + 1

            max_val = 0
            for k in range(3):
                for l in range(3):
                    if X[cx-1+k][cy-1+l] > max_val:
                        max_val = X[cx-1+k][cy-1+l]
            X_new[i][j] = max_val

    return X_new

def mean(X):
    """
    Averages the content of individual pixels considering adjacent ones.
    """
    X_new = np.zeros((60, 60), dtype=float)

    for i in range(1, 59):  # exclude the edges, which are delicate
        for j in range(1, 59):

            for k in range(3):  # consider a 3x3 square centered on the pixel under consideration
                for l in range(3):
                    X_new[i][j] += X[i-1+k][j-1+l]

            X_new[i][j] = X_new[i][j] / 9

    return X_new

def eclipse(X):
    """
    Dims the brightest parts.
    """
    ind = np.where(X > 0.3)
    X[ind] = 0.3

    return X

def visualize_grad(grad):
    """
    Returns grid x, y with the corresponding values of the gradient field vectors.
    """
    N = np.shape(grad)[0]
    x, y = np.meshgrid(range(N), range(N))
    u = np.zeros((N, N))
    v = np.zeros((N, N))

    for i in range(N):
        for j in range(N):
            u[N-1-i][j] = grad[i][j][0]
            v[N-1-i][j] = grad[i][j][1]

    return x, y, u, v

def norm_grad(X):
    """
    Returns the magnitude of the gradient field of X.
    """
    grad = gradient(X)
    N = np.shape(grad)[0]
    abs_val = np.zeros((N, N), dtype=float)

    for i in range(N):
        for j in range(N):
            abs_val[i][j] = np.linalg.norm(grad[i][j])

    return abs_val

def cut(X, n):
    """
    Returns the central 2n x 2n part of X, n < dimX / 2.
    """
    N = int(np.shape(X)[0] / 2)  # half of one dimension of X, used to find the center of the matrix
    X_new = np.zeros((2*n, 2*n), dtype=float)

    for i in range(2*n):
        for j in range(2*n):
            X_new[i][j] = X[N-n+i][N-n+j]  # X[N-n][N-n] is the top right corner of the central part of the matrix, X_new is built under this

    return X_new

def clean(X):
    """
    Returns X keeping only the most central part of the signal, any secondary celestial bodies are removed,
    and the mean() method is applied to reduce noise.
    """
    max_val = np.max(cut(X, 2))  # maximum of the central signal (used to correctly renormalize the signal, if there was a very bright secondary signal useful information would be lost, it also gives meaning to the tolerances of other methods --> now they will be a percentage of the central signal)

    if max_val != 0:  # to avoid divergences
        X = X / max_val

    ind = np.where(X > 1)  # problematic pixels with intensity greater than 1, due to division by max, are zeroed out (summary procedure, all pixels associated with foreign signals should be nullified)
    X[ind] = 0

    N = int(np.shape(X)[0] / 2)
    n = cumulative(X)  # number of steps to reach a cumulative plateau --> central signal is included within a square centered at the galactic center with side 2n

    if n == 0:  # to avoid issues related to very faint galaxies
        n = N

    X_new = np.zeros((2*N, 2*N), dtype=float)  # matrix of the same dimensions as X with zero signal

    for i in range(2*n):
        for j in range(2*n):
            X_new[N-n+i][N-n+j] = X[N-n+i][N-n+j]  # rewrite the central part of the new matrix = X

    X_new = mean(X_new)

    return X_new
