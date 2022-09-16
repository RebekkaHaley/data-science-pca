"""Module for calculation of PCA.
"""

import numpy as np


def check_data_validity(data) -> None:
    """Runs checks for whether given data is valid.

    Args:
        data (numpy.ndarray): Target data. Must be 2D.
    """
    # Check data type
    if not isinstance(data, (np.ndarray, np.generic)):
        raise TypeError(f"Found {type(data)} type. Input must be a numpy.ndarray.")
    # Check dimensions
    if data.ndim != 2:
        raise ValueError(f"Found {data.ndim} dimensions. Input must be 2-dimensional.")


class TwoDimensionStandardizer():
    """Calculates standardization via normalization of data.
    """
    def __init__(self):
        self.data = None


    def fit_transform(self, data):
        """Calculates normalization transform for given data.

        Args:
            data (numpy.ndarray): Target data. Must be 2D.

        Returns:
            numpy.ndarray: Tranfsformed data.
        """
        check_data_validity(data=data)
        transformed_data = np.zeros(shape=data.shape)
        for col in range(0, data.shape[1]):
            old_col = data[:, col]
            new_col = [(value - old_col.mean())/old_col.std() for value in old_col]
            transformed_data[:, col] = new_col
        return transformed_data


class PrincipalComponentAnalysis():
    """Calculates PCA.

    NB: The input data matrix should be observations-by-components.

    Args:
        n_components (int): Num of componets to retain. Max value is num of columns in input data.
        whiten (bool): Whitens output if True.
    """
    def __init__(self, n_components=None, whiten=False):
        self.n_components = n_components
        self.whiten = whiten
        self.data_mean = None
        self.covariance = None
        self.eigenvalues = None
        self.eigenvectors = None
        self.components = None
        self.decomp = None


    def fit(self, data):
        """todo

        Args:
            data (numpy.ndarray): Target data. Must be 2D.

        Returns:
            numpy.ndarray: Output numpy.ndarray of PCA data sets.
        """
        check_data_validity(data=data)
        self.data_mean = np.mean(data, axis=0)
        self.covariance = np.cov(data, rowvar=False)
        self.eigenvalues, self.eigenvectors = np.linalg.eig(self.covariance)
        zipped = zip(self.eigenvalues, self.eigenvectors.T)
        self.components = sorted(zipped, key=lambda vv: vv[0], reverse=True)
        if self.n_components is None:
            self.n_components = data.shape[1]
        self.components = self.components[:self.n_components]


    def whitener(self, data):
        """Whitens data by calculating eigenvalue decomposition of the covariance matrix.

        Args:
            data (numpy.ndarray): Target data. Must be 2D.

        Returns:
            numpy.ndarray: todo
        """
        check_data_validity(data=data)
        self.decomp = np.diag([eigenvalue ** -0.5 for (eigenvalue, _) in self.components])
        return data.dot(self.decomp)


    def transform(self, data):
        """todo

        Args:
            data (numpy.ndarray): Target data. Must be 2D.

        Returns:
            numpy.ndarray: Output numpy.ndarray of PCA data sets.
        """
        check_data_validity(data=data)
        output = np.zeros(data.shape)
        for i, point in enumerate(data):
            row = [eigen_vec.dot(point - self.data_mean) for (_, eigen_vec) in self.components]
            output[i, :] = row
        output = output[:, :self.n_components]
        if self.whiten:
            return self.whitener(data=output)
        return output


    def reverse_transform(self, data):
        """todo. Reverse PCA transform.

        Args:
            data (numpy.ndarray): Target data. Must be 2D.

        Returns:
            numpy.ndarray: todo.
        """
        check_data_validity(data=data)
        #     = self.decomp.dot(self.eigenvectors.T.dot(point - self.data_mean))  # WIP: whiten func
        #     = np.linalg.inv(self.eigenvectors.T).dot(np.linalg.inv(self.decomp).dot(point)) + self.data_mean  # WIP: inv whiten func
        eigenvectors = np.array([ei_vec for (_, ei_vec) in self.components])
        return np.dot(data, eigenvectors) + self.data_mean
