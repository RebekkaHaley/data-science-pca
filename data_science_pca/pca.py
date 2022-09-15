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
        raise TypeError(f"Found {type(data)} type. Input must be a np.array.")
    # Check dimensions
    if data.ndim != 2:
        raise ValueError(f"Found {data.ndim} dimensions. Input must be 2-dimensional.")


class TwoDimensionStandardizer():
    """Calculates standardization.

    Args:
        data (np.array): Target data. Must be 2D.
    """
    def __init__(self):
        self.data = None


    def fit_transform(self, data):
        """todo

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


class Whitener():
    """Calculates whitening.

    Args:
        data (np.array): Target data. Must be 2D.
    """
    def __init__(self):
        self.data = None


    def _func_(self, datapoints):
        """todo
        """
        output = np.zeros(datapoints.shape)
        for i, point in enumerate(datapoints):
            output[i, :] = self.l.dot(self.vectors.T.dot(point - self.mu))
        return output


    def _inv_func_(self, datapoints):
        """todo
        """
        output = np.zeros(datapoints.shape)
        for i, point in enumerate(datapoints):
            output[i, :] = np.linalg.inv(self.vectors.T).dot(np.linalg.inv(self.l).dot(point)) + self.mu
        return output


    def fit_transform(self, data):
        """todo

        Returns:
            func (todo): Whiten function.
            inv_func (todo): Inverse whiten function.
        """
        check_data_validity(data=data)
        self.sigma = np.cov(data, rowvar=False)
        self.mu = np.mean(data, axis=0)
        self.values, self.vectors = np.linalg.eig(self.sigma)
        self.l = np.diag(self.values ** -0.5)
        func = self._func_(datapoints=data)
        inv_func = self._inv_func_(datapoints=data)
        return func, inv_func


class PrincipalComponentAnalysis():
    """Calculates PCA.

    Args:
        Input 2D np.array of data sets.
    """
    def __init__(self, n_components=None):
        self.n_components = n_components


    def fit(self, data):
        """todo

        Returns:
            Output np.array of PCA data sets.
        """
        check_data_validity(data=data)
        self.data_mean = np.mean(data, axis=0)
        self.covariance = np.cov(data, rowvar=False)
        self.eigenvalues, self.eigenvectors = np.linalg.eig(self.covariance)
        self.components = sorted(zip(self.eigenvalues, self.eigenvectors.T), key=lambda vv: vv[0], reverse=True)
        if self.n_components is None:
            self.n_components == data.shape[1]
        self.feature_vector = self.components[:self.n_components]


    def transform(self, data):
        """todo

        Returns:
            Output np.array of PCA data sets.
        """
        check_data_validity(data=data)
        output_matrix = []
        for point in data:
            output_vector = [vector.dot(point - self.data_mean) for (value, vector) in self.feature_vector]
            output_matrix.append(output_vector)
        return np.array(output_matrix)
