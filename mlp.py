#!/usr/bin/env python3

'''
Image classification using a Multilayer Perceptron (MLP).
'''

from ml_library import *
from extract_features import *

if __name__ == "__main__":

    already_processed = False       # mark True if you want to skip feature extraction
    low_level = False               # mark True if you want to consider low level features,
                                    # otherwise individual pixels will be considered
    aux = ""
    if low_level:
        aux = "_ll"

    if not already_processed:
        # Clean images and extract features from training, test, and validation sets
        if low_level:
            X, Y = extract_features('Set/Train')
            X_test, Y_test = extract_features('Set/Test')
            X_validation, Y_validation = extract_features('Set/Validation')

        else:
            X, Y = extract_features_pixels('Set/Train')
            X_test, Y_test = extract_features_pixels('Set/Test')
            X_validation, Y_validation = extract_features_pixels('Set/Validation')

        # Save extracted features and labels for future use
        os.makedirs('Save', exist_ok=True)
        np.save('Save/X_mlp' + aux, X)
        np.save('Save/Y_mlp' + aux, Y)
        np.save('Save/X_mlp_test' + aux, X_test)
        np.save('Save/Y_mlp_test' + aux, Y_test)
        np.save('Save/X_mlp_validation' + aux, X_validation)
        np.save('Save/Y_mlp_validation' + aux, Y_validation)

    # Load saved features and labels
    X = np.load('Save/X_mlp'  + aux + '.npy')
    Y = np.load('Save/Y_mlp'  + aux + '.npy')
    X_test = np.load('Save/X_mlp_test'  + aux + '.npy')
    Y_test = np.load('Save/Y_mlp_test'  + aux + '.npy')
    X_validation = np.load('Save/X_mlp_validation'  + aux + '.npy')
    Y_validation = np.load('Save/Y_mlp_validation'  + aux + '.npy')

    # Define the architecture of the MLP (number of neurons in each layer)
    layers = [3600, 10, 2]

    # Train the MLP using the training and validation sets
    W, b = mlp_train(X, Y, X_validation, Y_validation, layers, 1000, 100, 10**(-3), 10**(-3))

    # Save trained weights and biases for future use
    np.save('Save/W_mlp.npy', W)
    np.save('Save/b_mlp.npy', b)

    # Perform inference on the training set and calculate accuracy
    guess, P = mlp_inference(X, W, b, layers)
    accuracy = (Y == guess).mean()
    print("Accuracy on the training set = ", accuracy * 100)

    # Perform inference on the test set and calculate accuracy
    guess_test, P_test = mlp_inference(X_test, W, b, layers)
    accuracy_test = (Y_test == guess_test).mean()
    print("Accuracy on the test set = ", accuracy_test * 100)
