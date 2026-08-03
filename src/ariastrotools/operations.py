# The functions to perform mathematical operations
import numpy as np
from astropy.stats import biweight_location
from astropy.table import Table
from astropy.io.fits.fitsrec import FITS_rec


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

    # Handle binary tables
    if len(dataarr) > 0 and isinstance(dataarr[0], (FITS_rec, Table)):
        return combine_bintable(dataarr, method=method)

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


def combine_bintable(dataarr,
                     value_cols=None,
                     uncertainity_cols=None,
                     combine_cols=None,
                     method='mean'):

    if value_cols is None:
        value_cols = []

    if uncertainity_cols is None:
        uncertainity_cols = []

    if combine_cols is None:
        combine_cols = []

    if len(value_cols) != len(uncertainity_cols):
        raise ValueError(
            "'value_cols' and 'uncertainity_cols' must have the same length"
            )

    tables = [Table(t) if not isinstance(t, Table) else t
              for t in dataarr]

    ref = tables[0]
    comb = ref.copy(copy_data=True)

    handled = set()

    # Combine quantities with propagated uncertainities
    for value_col, err_col in zip(value_cols, uncertainity_cols):

        values = np.stack([t[value_col] for t in tables])
        var = np.stack([t[err_col]**2 for t in tables])

        comb[value_col], comb_var = combine_data(
            values,
            var=var,
            method=method
            )
        comb[err_col] = np.sqrt(comb_var)
        handled.update([value_col, err_col])

    # Combine quantities without uncertainities
    for col in combine_cols:

        values = np.stack([t[col] for t in tables])

        comb[col], _ = combine_data(
            values,
            method=method
            )
        handled.add(col)

    # Verify all remaining columns are identical
    for col in ref.colnames:

        if col in handled:
            continue

        for table in tables[1:]:
            if not np.array_equal(ref[col], table[col]):
                raise ValueError(
                    f"Column '{col}' differs between tables."
                    )

    return comb, None


def combine_data_full(datadict, dataext=[1, 2, 3],
                      varext=[4, 5, 6],
                      extras=[],
                      table_info=None,
                      method='mean'):
    """
    Combine data from multiple FITS files into a single dictionary.

    This function combines data arrays, variance arrays, additional
    numeric arrays, and binary tables stored in a dictionary (typically
    produced by reading multiple FITS files). Flux and variance arrays
    are combined using ``combine_data``, while binary tables are combined
    using ``combine_bintable``.

    Parameters
    ----------
    datadict : dict
        Dictionary containing data from multiple FITS files. Each key
        corresponds to a FITS extension or metadata item. Arrays to be
        combined are expected to be stacked along the first axis
        (i.e., shape ``(n_files, ...)``).

    dataext : list of int, optional
        Indices of ``datadict.keys()`` corresponding to data arrays
        (e.g., flux) that should be combined. Default is
        ``[1, 2, 3]``.

    varext : list of int, optional
        Indices of ``datadict.keys()`` corresponding to variance arrays.
        Each entry must correspond to the matching entry in
        ``dataext``. Default is ``[4, 5, 6]``.

    extras : list of int, optional
        Indices of additional numeric arrays that should be combined
        using ``combine_data`` without associated variance arrays.
        Default is ``[]``.

    table_info : dict, optional
        Dictionary describing binary table extensions to combine. Keys
        are indices into ``datadict.keys()`` and values are dictionaries
        passed to ``combine_bintable``.

        Each value may contain the following entries:

        - ``value_cols`` : list of columns whose values are combined
          with propagated uncertainties.
        - ``uncertainty_cols`` : list of uncertainty columns
          corresponding to ``value_cols``.
        - ``combine_cols`` : list of numeric columns that are combined
          without uncertainty propagation.

        Columns not listed above are assumed to be identical in all
        input tables and are copied from the first table after verifying
        they are unchanged.

        Default is ``None``.

    method : {'mean', 'median', 'biweight'}, optional
        Method used to combine the data.

        - ``'mean'`` : arithmetic mean.
        - ``'median'`` : median.
        - ``'biweight'`` : biweight location.

    Returns
    -------
    comb_dicts : dict
        Dictionary containing the combined data.

        - Data arrays in ``dataext`` are combined.
        - Variance arrays in ``varext`` are propagated.
        - Arrays in ``extras`` are combined.
        - Binary tables in ``table_info`` are combined using
          ``combine_bintable``.
        - All remaining entries are copied from the first input file.

    Notes
    -----
    - The order of ``dataext`` and ``varext`` must correspond.
    - Dictionary insertion order is assumed to match the FITS extension
      order.
    - Entries not listed in ``dataext``, ``varext``, ``extras``, or
      ``table_info`` are copied from the first input file.
    - ``table_info`` provides a generic mechanism for combining
      arbitrary FITS binary tables without requiring instrument-specific
      code.

    Examples
    --------
    >>> table_info = {
    ...     7: {
    ...         "value_cols": ["VALUE"],
    ...         "uncertainty_cols": ["UNCERTAINTY"],
    ...         "combine_cols": []
    ...     }
    ... }
    >>> combined = combine_data_full(
    ...     datadict,
    ...     dataext=[1],
    ...     varext=[2],
    ...     table_info=table_info,
    ...     method="mean",
    ... )
    """
    dictkeys = list(datadict.keys())
    print(dictkeys)
    comb_dicts = datadict.copy()

    flux_keys = [dictkeys[int(i)] for i in dataext]
    var_keys = [dictkeys[int(i)] for i in varext]
    if len(extras) > 0:
        extra_keys = [dictkeys[int(i)] for i in extras]
    else:
        extra_keys = []

    table_keys = [dictkeys[i] for i in table_info] if table_info else []

    # Avoiding the extensions that are not flux or variance.
    # Taking only the first element of that. i.e.,
    # The data from first fits file will
    # be copied to the final output.
    # For spectrum, wavelengths are interpolated to
    # same array. So, that also
    # copied in the same way.
    for cro, keys in enumerate(dictkeys):

        if keys not in flux_keys + var_keys + extra_keys + table_keys:

            comb_dicts[keys] = comb_dicts[keys][0]
    # Doing for flux and variance.
    for index, extk in enumerate(flux_keys):
        fluxes = comb_dicts[flux_keys[index]]
        variances = comb_dicts[var_keys[index]]
        comb_flux, comb_var = combine_data(fluxes, variances,
                                           method=method)

        comb_dicts[flux_keys[index]] = comb_flux
        comb_dicts[var_keys[index]] = comb_var

    for index, ext in enumerate(extra_keys):
        data = comb_dicts[ext]
        comb_data, _ = combine_data(data,
                                    method=method)
        comb_dicts[ext] = comb_data
    if table_info is not None:
        for extnum, info in table_info.items():
            key = dictkeys[extnum]
            comb_table, _ = combine_bintable(
                comb_dicts[key],
                value_cols=info.get("value_cols", []),
                uncertainity_cols=info.get("uncertainty_cols", []),
                combine_cols=info.get("combine_cols", []),
                method=method,
            )

            comb_dicts[key] = comb_table

    return comb_dicts

# End
