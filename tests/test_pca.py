"""Tests for PCA module.
"""

import numpy as np
import pytest
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from data_science_pca.pca import check_data_validity, TwoDimensionStandardizer, PrincipalComponentAnalysis

RAW_DATA = np.array([
    [2.5, 0.5, 2.2, 1.9, 3.1, 2.3, 2, 1, 1.5, 1.1],
    [2.4, 0.7, 2.9, 2.2, 3.0, 2.7, 1.6, 1.1, 1.6, 0.9]]).T  # shape: (10, 2)


def test_check_data_validity_invalid_input_type():
    invalid_input = 'apple'
    with pytest.raises(TypeError):
        check_data_validity(data=invalid_input)


def test_check_data_validity_invalid_input_shape():
    invalid_input = np.array([1, 2, 3])
    with pytest.raises(ValueError):
        check_data_validity(data=invalid_input)


def test_check_data_validity_valid_input():
    valid_input = np.array([[3, 6], [5, 10]])
    check_data_validity(data=valid_input)


def test_two_dimension_standardizer_init():
    test_scaler = TwoDimensionStandardizer()
    assert test_scaler.__dict__ == {}


def test_two_dimension_standardizer_compare_sklearn():
    test_scaler = TwoDimensionStandardizer()
    test_output = test_scaler.fit_transform(data=RAW_DATA)
    skl_scaler = StandardScaler()
    skl_output = skl_scaler.fit_transform(RAW_DATA)    
    np.allclose(test_output, skl_output, rtol=1e-09, atol=1e-09)


def test_principal_component_analysis_init():
    n_comp = RAW_DATA.shape[1]
    whiten_bool = False
    test_pca = PrincipalComponentAnalysis(n_components=n_comp, whiten=whiten_bool)
    assert test_pca.__dict__ == {
        'n_components': n_comp,
        'whiten': whiten_bool,
        'data_mean': None,
        'covariance': None,
        'eigenvalues': None,
        'eigenvectors': None,
        'components': None,
        'decomp': None,
        'feature_vector': None
        }


def test_principal_component_analysis_compare_sklearn():
    n_comp = RAW_DATA.shape[1]
    whiten_bool = False
    test_pca = PrincipalComponentAnalysis(n_components=n_comp, whiten=whiten_bool)
    test_pca.fit(RAW_DATA)
    test_output = test_pca.transform(RAW_DATA)
    skl_pca = PCA(n_components=n_comp, whiten=whiten_bool)
    skl_pca.fit(RAW_DATA)
    skl_output = skl_pca.transform(RAW_DATA)
    np.allclose(test_output, skl_output, rtol=1e-09, atol=1e-09)


def test_principal_component_analysis_whitening_compare_sklearn():
    n_comp = RAW_DATA.shape[1]
    whiten_bool = True
    test_pca = PrincipalComponentAnalysis(n_components=n_comp, whiten=whiten_bool)
    test_pca.fit(RAW_DATA)
    test_output = test_pca.transform(RAW_DATA)
    skl_pca = PCA(n_components=n_comp, whiten=whiten_bool)
    skl_pca.fit(RAW_DATA)
    skl_output = skl_pca.transform(RAW_DATA)
    np.allclose(test_output, skl_output, rtol=1e-09, atol=1e-09)


def test_principal_component_analysis_reverse_transform_full_recovery():
    n_comp = RAW_DATA.shape[1]
    whiten_bool = False
    test_pca = PrincipalComponentAnalysis(n_components=n_comp, whiten=whiten_bool)
    test_pca.fit(RAW_DATA)
    test_output = test_pca.transform(RAW_DATA)
    test_reverse = test_pca.reverse_transform(data=test_output)
    np.allclose(test_reverse, RAW_DATA, rtol=1e-09, atol=1e-09)
