# The functions to perform mathematical operations
import numpy as np
from astropy.stats import biweight_location


'''
Mathematical operations
'''


def ari_operations(arr1, arr2, var_arr1=None, var_arr2=None, operation='+'):
    """
    Perform element-wise arithmetic operations on two input arrays with
    optional variance propagation.

    Parameters
    ----------
    arr1 : numpy.ndarray
        First input array.
    arr2 : numpy.ndarray
        Second input array, must be broadcastable to the shape of `arr1`.
    var_arr1 : numpy.ndarray or None, optional
        Variance (uncertainty) array corresponding to `arr1`. Default is None.
    var_arr2 : numpy.ndarray or None, optional
        Variance (uncertainty) array corresponding to `arr2`. Default is None.
    operation : str, optional
        Arithmetic operation to apply (default is 'sum').
        Supported values:
        - '+'  : element-wise addition
        - '-' : element-wise subtraction (`arr1 - arr2`)
        - '*' : element-wise multiplication
        - '/'  : element-wise division (`arr1 / arr2`)

    Returns
    -------
    numpy.ndarray or tuple of numpy.ndarray
        If `var_arr1` and `var_arr2` are not provided, returns the result of
        element-wise operation on `arr1` and `arr2`.
        If both variances are provided, returns the propagated variance array
        computed according to the operation:
        - For '+' and '-': variances are added.
        - For '*' and '/': variance propagated as
    product.

    Raises
    ------
    ZeroDivisionError
        When division by zero occurs in 'div' operation.
    ValueError
        If `operation` is not one of the supported strings.

    Examples
    --------
    >>> import numpy as np
    >>> a = np.array([1.0, 2.0, 3.0])
    >>> b = np.array([4.0, 5.0, 6.0])
    >>> ari_operations(a, b, operation='+')
    array([5., 7., 9.])
    >>> var_a = np.array([0.1, 0.1, 0.1])
    >>> var_b = np.array([0.2, 0.2, 0.2])
    >>> ari_operations(a, b, var_a, var_b, operation='+')
    array([0.3, 0.3, 0.3])
    """
    if operation == '+':
        answer = arr1 + arr2
    elif operation == '-':
        answer = arr1 - arr2
    elif operation == '*':
        answer = arr1 * arr2
    elif operation == '/':
        answer = arr1 / arr2
    else:
        raise ValueError(
            f"Unsupported operation '{operation}'. Supported: ",
            "'+', '-', '*', '/'.")

    if (var_arr1 is not None) & (var_arr2 is not None):
        if (operation == '+') or (operation == '-'):
            var_tot = var_arr1 + var_arr2
        elif (operation == '*') or (operation == '/'):
            var_tot = answer**2 * ((var_arr1/arr1**2) + (var_arr2/arr2**2))
        return answer, var_tot

    return answer, None


'''
Combine
'''


def combine_data(dataarr, var=None, method='mean',
                 mask=None):
    """
    Combine multiple arrays along the first axis using a specified method.

    Parameters
    ----------
    dataarr : array_like
        Input data array of shape (N, ...), where `N` is the number of
        individual datasets to combine. The combination is performed
        along axis=0.
    var : array_like, optional
        Variance array of the same shape as `dataarr`. If provided,
        error propagation is performed assuming independent errors,
        yielding the variance of the combined data. Default is None.
    method : {'mean', 'median', 'biweight', 'weightedavg'}, optional
        Method used for combining the data:

        - 'mean' : arithmetic mean ignoring NaNs.
        - 'median' : median ignoring NaNs.
        - 'biweight' : robust biweight location (from `astropy.stats`).

        Default is 'mean'.

    Returns
    -------
    comb_data : ndarray
        Combined data array, same shape as a single input array
        (i.e., shape of `dataarr[0]`).
    comb_var : ndarray, optional
        Combined variance array of the same shape as `comb_data`.
        Returned only if `var` is provided.

    Notes
    -----
    - NaN values in `dataarr` are ignored during combination.
    - Variance is propagated as if the combination method were the mean,
      even if `median` or `biweight` are chosen. This provides an
      approximate uncertainty estimate.
    - The biweight method is less sensitive to outliers than the mean
      or median.
    """
    if method == 'weightedavg':
        comb_data, comb_var = weighted_mean_and_variance(dataarr, var)
        return comb_data, comb_var
    dataarr = np.array(dataarr)
    N = dataarr.shape[0]
    if mask is not None:
        mask_full = np.broadcast_to(mask, dataarr.shape)
        dataarr_ma = np.ma.array(dataarr, mask=mask_full)
        if method == 'mean':
            comb_data = np.ma.nanmean(dataarr_ma, axis=0).filled(np.nan)
        elif method == 'median':
            comb_data = np.ma.nanmedian(dataarr_ma, axis=0).filled(np.nan)
        elif method == 'biweight':
            comb_data = biweight_location(dataarr_ma, axis=0).filled(np.nan)

    else:
        if method == 'mean':
            comb_data = np.nanmean(dataarr, axis=0)
        elif method == 'median':
            comb_data = np.nanmedian(dataarr, axis=0)
        elif method == 'biweight':
            comb_data = biweight_location(dataarr, axis=0)
    # Propagating error.
    # Treating the error propagation
    # as mean for median also.
    if var is not None:
        if mask is None:
            comb_var = np.sum(var, axis=0) / N**2
            return comb_data, comb_var
        else:
            var_ma = np.ma.array(var, mask_full)
            comb_var = np.ma.sum(var_ma, axis=0) / N**2
            comb_var = comb_var.filled(np.nan)
            return comb_data, comb_var
    return comb_data, None


def weighted_mean_and_variance(values, variances):
    r"""
    Compute the weighted mean and variance of the mean,
    given measurements and their variances.

    Parameters
    ----------
    values : array-like
        Measured values (x_i)
    variances : array-like
        Variances of the measurements.

    Returns
    ----------
    mean : float
        Weighted mean.
    variance_of_mean : float
        Variance of the weighted mean

    Raises
    ------
    ValueError
        If `variances` is None.
    TypeError
        If `values` or `variances` are not array-like.

    Notes
    -----
    The weighted mean is computed as:

    .. math::

        \bar{x} = \frac{\sum_i w_i x_i}{\sum_i w_i}, \quad
        w_i = \frac{1}{\sigma_i^2}

    The variance of the weighted mean is:

    .. math::

        \sigma_{\bar{x}}^2 = \frac{1}{\sum_i w_i}
    """
    if variances is None:
        raise TypeError("variances must be an array-like object")

    weights = 1.0 / variances
    mean = np.sum(weights * values, axis=0) / np.sum(weights, axis=0)
    variance_of_mean = 1.0 / np.sum(weights, axis=0)

    return mean, variance_of_mean


def combine_data_full(datadict, dataext=[1, 2, 3],
                      varext=[4, 5, 6],
                      method='mean'):
    """
    Combine flux and variance data from multiple FITS files into a single
    dictionary.

    This function takes a dictionary of arrays (typically produced from
    reading multiple FITS files), selects specific keys for flux and
    variance, and combines them using a specified method (e.g. mean,
    median, or biweight). Non-flux/variance entries are copied from the
    first element of the corresponding arrays.

    Parameters
    ----------
    datadict : dict
        Dictionary containing data arrays. Each key corresponds to a FITS
        extension or metadata. Flux and variance arrays are stacked along
        the first axis (i.e., shape ``(n_files, n_points, ...)``).
    dataext : list of int, optional
        Indices of ``datadict.keys()`` that correspond to flux data.
        Default is ``[1, 2, 3]``.
    varext : list of int, optional
        Indices of ``datadict.keys()`` that correspond to variance data.
        Default is ``[4, 5, 6]``.
    method : {'mean', 'median', 'biweight'}, optional
        Method used to combine the fluxes and variances.

        - ``'mean'`` : compute the mean across input files
        - ``'median'`` : compute the median across input files
        - ``'biweight'`` : compute the biweight across input files

    Returns
    -------
    comb_dicts : dict
        New dictionary with combined flux and variance arrays.

        - Keys in ``flux_keys`` and ``var_keys`` contain the combined
          arrays.
        - Other keys are reduced to the first element of their array.

    Notes
    -----
    - This function assumes dictionary key order matches ``dataext`` and
      ``varext``. Since Python 3.7, dictionary order is guaranteed to be
      insertion order.
    - Non-flux/variance arrays (e.g., wavelength grids, headers) are taken
      from the first FITS file. If you want to preserve the full stack,
      you should modify the loop that reduces them to index ``[0]``.
    - To prevent modifying the original input, the function creates a
      shallow copy of ``datadict``. Arrays themselves are *not*
      deep-copied.

    Examples
    --------
    >>> import numpy as np
    >>> datadict = {
    ...     "KEY0": ["header1", "header2"],
    ...     "SCIFLUX": np.array([[1, 2, 3], [4, 5, 6]]),
    ...     "VAR": np.array([[1, 1, 1], [2, 2, 2]]),
    ...     "WAVELENGTH": np.array([[500, 600], [700, 800]])
    ... }
    >>> result = combine_data_full(datadict, dataext=[1], varext=[2],
    ...                            method='mean')
    >>> result["SCIFLUX"]
    array([2.5, 3.5, 4.5])
    >>> result["VAR"]
    array([1.5, 1.5, 1.5])
    >>> result["KEY0"]
    'header1'
    """

    dictkeys = list(datadict.keys())
    print(dictkeys)
    comb_dicts = datadict.copy()
    # print("comb data full", np.array(datadict["SCIFLUX"]).shape)
    flux_keys = [dictkeys[int(i)] for i in dataext]
    var_keys = [dictkeys[int(i)] for i in varext]

    # Avoiding the extensions that are not flux or variance.
    # Taking only the first element of that. i.e.,
    # The data from first fits file will
    # be copied to the final output.
    # For spectrum, wavelengths are interpolated to
    # same array. So, that also
    # copied in the same way.
    for cro, keys in enumerate(dictkeys):
        print(keys, flux_keys, var_keys)
        if keys not in flux_keys + var_keys:
            print('keys', keys)
            comb_dicts[keys] = comb_dicts[keys][0]
    # Doing for flux and variance.
    for index, extk in enumerate(flux_keys):
        fluxes = comb_dicts[flux_keys[index]]
        variances = comb_dicts[var_keys[index]]
        comb_flux, comb_var = combine_data(fluxes, variances,
                                           method=method)

        comb_dicts[flux_keys[index]] = comb_flux
        comb_dicts[var_keys[index]] = comb_var
    # print(datadict[flux_keys[0]].shape)
    # print(comb_dicts)
    return comb_dicts

# End
