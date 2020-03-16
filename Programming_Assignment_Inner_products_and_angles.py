# PACKAGE: DO NOT EDIT
import PyQt5
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import scipy

# GRADED FUNCTION: DO NOT EDIT THIS LINE

# ===YOU SHOULD EDIT THIS FUNCTION===
def distance(x, y):
    """Compute distance between two vectors x, y using the dot product"""
    x = np.array(x, dtype=np.float).ravel() # ravel() "flattens" the ndarray
    y = np.array(y, dtype=np.float).ravel()
    distance=np.sqrt(np.transpose(x-y)@(x-y))
    return distance


# ===YOU SHOULD EDIT THIS FUNCTION===
def angle(x, y):
    """Compute the angle between two vectors x, y using the dot product"""
    angle = np.arccos(np.dot(x,y)/np.sqrt(np.transpose(x)@x*np.transpose(y)@y))
    return angle

# ===YOU SHOULD EDIT THIS FUNCTION===
def pairwise_distance_matrix(X, Y):
    """Compute the pairwise distance between rows of X and rows of Y

    Arguments
    ----------
    X: ndarray of size (N, D)
    Y: ndarray of size (M, D)
    
    Returns
    --------
    D: matrix of shape (N, M), each entry D[i,j] is the distance between
    X[i,:] and Y[j,:] using the dot product.
    """
    N, D = X.shape
    M, _ = Y.shape
    distance_matrix = np.zeros((N, M), dtype=np.float)
    for i in range(N):
        x=X[i,:]
        for j in range(M):
            y=Y[j,:]
            d=distance(x,y)
            distance_matrix[i,j]=d
    return distance_matrix

# GRADED FUNCTION: DO NOT EDIT THIS LINE

# ===YOU SHOULD EDIT THIS FUNCTION===
def most_similar_image():
    """Find the index of the digit, among all MNIST digits (excluding the first),
       that is the closest to the first image in the dataset, your answer should be a single integer

       Note:
       Use another cell to write some code snippets to find out this index.
       Then fill it in here. The autograder does not have access to the MNIST dataset,
       so any mentions of MNIST inside this will fail.
    """
    most_similar_index = 61
    return most_similar_index