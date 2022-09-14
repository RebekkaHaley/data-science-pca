"""Tests for PCA module.
"""

import numpy as np
import pytest

from data_science_pca.pca import normalize_2d


def test_normalize_2d_invalid_input_type():
    with pytest.raises(TypeError):
        bad_input = 'apple'
        normalize_2d(data=bad_input)


def test_normalize_2d_invalid_input_shape():
    with pytest.raises(IndexError):
        bad_input = np.array([1, 2, 3])
        normalize_2d(data=bad_input)


def test_normalize_2d_valid_input():
    good_input = np.array([[3, 6], [5, 10]])
    good_output = np.array([[-1, -1], [1, 1]])
    np.testing.assert_array_equal(normalize_2d(data=good_input), good_output)
