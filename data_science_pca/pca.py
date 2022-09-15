"""Module for calculation of PCA.
"""

import numpy as np


def check_data_validity(data) -> None:
    """Runs checks for whether given data is valid.

    Args:
        data (np.array): Target data. Must be 2D.
    """
    # Check data type
    if type(data) != type(np.array([])):
        raise TypeError("Input must be a np.array.")
    # Check dimensions
    try:
        data.shape[1]
    except IndexError:
        print("Input must be 2-dimensional.")
        raise


def standardize_2d(data):
    """Calculates normalization by taking zero mean and dividing by stdev.

    Args:
        data (np.array): Target data. Must be 2D.

    Returns:
        np.array: Tranfsformed data.
    """
    check_data_validity(data=data)
    transformed_data = np.zeros(shape=data.shape)
    for col in range(0, data.shape[1]):
        old_col = data[:, col]
        new_col = [(value - old_col.mean())/old_col.std() for value in old_col]
        transformed_data[:, col] = new_col
    return transformed_data


def whitener(data):
    """Calculates whitening.

    Taken for colab week 2.

    Args:
        Inputs 2D data set.

    Returns:
        Output whiten & inv. whiten functions.
    """
    sigma = np.cov(data, rowvar=False)
    mu = np.mean(data, axis=0)
    values, vectors = np.linalg.eig(sigma)
    l = np.diag(values ** -0.5)


    def func(datapoints):
        return np.array( [ l.dot(vectors.T.dot(d - mu)) for d in datapoints ])


    def inv_func(datapoints):
        return np.array( [ np.linalg.inv(vectors.T).dot(np.linalg.inv(l).dot(d)) + mu for d in datapoints ] )
    return func, inv_func


def pca(X, n_components):
    """Calculates PCA.

    Taken for colab week 2.

    Args:
        Input 2D np.array of data sets.

    Returns:
        Output np.array of PCA data sets.
    """
    pca = PCA(n_components=n_components)
    pca.fit(X)
    X_pca = pca.transform(X)
    return X_pca


def pca_transform(data):
    """

    Taken for colab week 2.

    Args:
        Input 2D np.array of data sets.

    Returns:
        Output np.array of PCA data sets.
    """
    mu = np.mean(data, axis=0)
    sigma = np.cov(data, rowvar=False)
    values, vectors = np.linalg.eig(sigma)
    components = sorted( zip(values, vectors.T), key = lambda vv: vv[0], reverse=True )


    def func(datapoints):
        result = []
        for d in datapoints:
            t = [ vector.dot(d - mu) for (value, vector) in components ]
            result.append(t)
        return np.array(result)
    return func
