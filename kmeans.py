"""
    This is a file you will have to fill in.
    It contains helper functions required by K-means method via iterative improvement
"""
import numpy as np
from random import sample

def init_centroids(k, inputs):
    """
    Selects k random rows from inputs and returns them as the chosen centroids
    :param k: number of cluster centroids
    :param inputs: a 2D Numpy array, each row of which is one input
    :return: a Numpy array of k cluster centroids, one per row
    """
    # TODO
    pass


def assign_step(inputs, centroids):
    """
    Determines a centroid index for every row of the inputs using Euclidean Distance
    :param inputs: inputs of data, a 2D Numpy array
    :param centroids: a Numpy array of k current centroids
    :return: a Numpy array of centroid indices, one for each row of the inputs
    """
    # TODO
    pass


def update_step(inputs, indices, k):
    """
    Computes the centroid for each cluster
    :param inputs: inputs of data, a 2D Numpy array
    :param indices: a Numpy array of centroid indices, one for each row of the inputs
    :param k: number of cluster centroids, an int
    :return: a Numpy array of k cluster centroids, one per row
    """
    # TODO
    pass


def kmeans(inputs, k, max_iter, tol):
    """
    Runs the K-means algorithm on n rows of inputs using k clusters via iterative improvement
    :param inputs: inputs of data, a 2D Numpy array
    :param k: number of cluster centroids, an int
    :param max_iter: the maximum number of times the algorithm can iterate trying to optimize the centroid values, an int
    :param tol: relative tolerance with regards to inertia to declare convergence, a float number
    :return: a Numpy array of k cluster centroids, one per row
    """
    # TODO
    pass