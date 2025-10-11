import numpy as np
import pytest

from ariastrotools.operations import ari_operations
from ariastrotools.operations import combine_data
from ariastrotools.operations import combine_data_full
from ariastrotools.operations import weighted_mean_and_variance


@pytest.mark.parametrize(
    "arr1, arr2, var_arr1, var_arr2, operation, expected, expected_var",
    [
        (np.array([1, 2]), np.array([3, 4]), None, None,
         '+', np.array([4, 6]), None),
        (np.array([5, 7]), np.array([2, 3]), None, None,
         '-', np.array([3, 4]), None),
        (np.array([2, 3]), np.array([4, 5]), None, None,
         '*', np.array([8, 15]), None),
        (np.array([8, 9]), np.array([2, 3]), None, None,
         '/', np.array([4.0, 3.0]), None),
        (1, 1, None, None, '+', 2, None),
        (1, 1, 1, 1, '+', 2, 2)
    ]
)
def test_add(arr1, arr2,
             var_arr1, var_arr2,
             operation, expected, expected_var):
    result, var_result = ari_operations(arr1, arr2,
                                        var_arr1, var_arr2,
                                        operation)

    # Check main result
    assert np.allclose(result, expected)

    # If expected_var is given, check it
    if expected_var is not None:
        assert np.allclose(var_result, expected_var)
    else:
        assert var_result is None


@pytest.mark.parametrize(
    "dataarr, var, method, expected_data, expected_var",
    [
        # Mean without var
        (np.array([[1, 2], [3, 4]]), None, 'mean', np.array([2, 3]), None),

        # Median without var
        (np.array([[1, 2], [3, 4]]), None, 'median', np.array([2, 3]), None),

        # Mean with var
        (np.array([[1, 2], [3, 4]]), np.array([[0.1, 0.2], [0.1, 0.4]]),
         'mean', np.array([2, 3]), np.array([0.05, 0.15])),

        # Median with var
        (np.array([[1, 2], [3, 4]]), np.array([[0.1, 0.2], [0.1, 0.4]]),
         'median', np.array([2, 3]), np.array([0.05, 0.15])),
    ]
)
def test_combine_data(dataarr, var, method, expected_data, expected_var):
    comb_data, comb_var = combine_data(dataarr, var=var, method=method)
    assert np.allclose(comb_data, expected_data)
    if expected_var is not None:
        assert np.allclose(comb_var, expected_var)


def test_combine_data_full_basic():
    key0 = [None, None]
    scifluxes = np.arange(3*3*3).reshape(3, 3, 3)
    scivars = np.arange(3*3*3).reshape(3, 3, 3)
    wls = np.array([[500, 600],
                    [700, 800]])
    # mean_scifluxes = np.mean(scifluxes, axis=0)
    mean_flux, var_flux = combine_data(scifluxes, scivars, method='mean')
    datadict = {
        "KEY0": key0,
        "SCIFLUX": scifluxes,
        "VAR": scivars,
        "WAVELENGTH": wls
    }
    print(datadict["WAVELENGTH"])
    combined = combine_data_full(datadict, dataext=[1], varext=[2],
                                 method='mean')
    # Check that non-flux/var keys are replaced by first element
    print(datadict["WAVELENGTH"])
    assert np.array_equal(combined["SCIFLUX"], mean_flux)
    assert np.array_equal(combined["VAR"], var_flux)
    assert np.array_equal(combined["WAVELENGTH"], wls[0])
    assert combined["KEY0"] == key0[0]
    # Check combined flux and variance (example expectations)
    # Depend on fixed combine_data behavior
    # For example:
    # expected_flux = np.array([2.0, 3.0])  # mean of [1,3] and [2,4]
    # expected_var = np.array([0.01, 0.025])  # example propagated variance


def test_weighted_mean_and_variance():
    values = np.array([10.0, 20.0, 30.0])
    variances = np.array([1.0, 4.0, 9.0])

    mean, var = weighted_mean_and_variance(values, variances)

    expected_mean = 13.469387755102039
    expected_var = 0.7346938775510203

    assert np.allclose(mean, expected_mean, rtol=1e-6)
    assert np.allclose(var, expected_var, rtol=1e-6)


def test_weighted_mean_in_combine_data():
    values = np.array([10.0, 20.0, 30.0])
    variances = np.array([1.0, 4.0, 9.0])
    mean, var = combine_data(values, variances, method='weightedavg')

    expected_mean = 13.469387755102039
    expected_var = 0.7346938775510203

    assert np.allclose(mean, expected_mean, rtol=1e-6)
    assert np.allclose(var, expected_var, rtol=1e-6)

# End
