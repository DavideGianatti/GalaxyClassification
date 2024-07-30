July 2022

## Galaxy Classification

This project provides scripts for training and testing machine learning models to classify galaxies based on their morphology, distinguishing between elliptical and spiral galaxies. The project includes methods for extracting low-level features, implementing a Multilayer Perceptron (MLP) for image classification, and using k-means clustering for signal detection.
The project leverages a simple custom machine learning library implemented in Python.

### Project Structure
The project is organized as follows:
- **ml_library.py**: implementation of a simple ML library (logistic regression, MLP, backward propagation algorithm);
- **extract_features.py**: contains functions to pre-process images and extract low-level features;
- **mlp.py**: trains and saves different MLP architectures;
- **k_cluster.py**: signal detection and removal of celestial bodies using the k-means clustering method.

### Data Set
The training, validation and test sets have been obtained from SDSS survey and they can be downloaded from https://drive.google.com/file/d/1-KJpx1bx-b9sjJ7c4jfbrnSvEdqibJQg/view?usp=drive_link

