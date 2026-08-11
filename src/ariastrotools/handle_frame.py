#!/usr/bin/env python3

"""
This module contains functions for processing astronomical FITS frames.
Includes functions for arithmetic operations, combining data,
removing cosmic rays, and smoothing gradients.

Functions:
- operate_process
- combine_process
- divide_smoothgradient
- remove_cosmic_rays
"""

import numpy as np
import astroscrappy
from scipy.ndimage import filters
from skimage.restoration import inpaint

from pathlib import Path
from astropy.io import fits

from .operations import ari_operations
from .operations import combine_data
from .spectral_utils import combine_spectra
from .utils import call_mask

from .logger import logger


def scale_datacube(datacube,
                   varcube=None,
                   scale="p50",
                   scale_mask=None):
    """
    Scale the frames in a data cube using percentile-based scaling.

    Each frame is assigned a scaling factor based on the specified
    percentile. The scaling factors are normalized by their median value
    before scaling the data cube. If a variance cube is provided, the
    variance is scaled consistently with the data.

    Parameters
    ----------
    datacube : numpy.ndarray
        Input data cube with shape (n_frames, ny, nx).

    varcube : numpy.ndarray or None, optional
        Variance cube corresponding to `datacube`, with the same shape.
        If provided, the variance is scaled by the square of the scaling
        factor. Default is None.

    scale : str, optional
        Scaling scheme to use. Currently, only percentile-based scaling
        in the form ``'pXX'`` is supported, where ``XX`` specifies the
        percentile used to determine the scaling factor. For example,
        ``'p50'`` uses the 50th percentile (median) of each frame.
        The scaling factors are normalized by their median value.
        Default is ``'p50'``.

    scale_mask : numpy.ndarray or None, optional
        Boolean mask identifying pixels to exclude when calculating the
        percentile scaling factors. The same mask is applied to every
        frame. If None, all valid (non-NaN) pixels are used.
        Default is None.

    Returns
    -------
    scaled_datacube : numpy.ndarray
        Scaled data cube with the same shape as `datacube`.

    scaled_varcube : numpy.ndarray or None
        Scaled variance cube with the same shape as `varcube`, or None
        if `varcube` was not provided.

    scale_array : numpy.ndarray
        One-dimensional array containing the normalized scaling factor
        for each frame. The median of the scaling factors is one.

    Raises
    ------
    ValueError
        If `scale` does not follow the supported ``'pXX'`` format.

    Notes
    -----
    The scaling factor for each frame is calculated from the specified
    percentile and then normalized by the median of all frame scaling
    factors:

    ``scale_factor = percentile(frame) / median(percentile_values)``

    The data and variance are then scaled as:

    ``scaled_data = data / scale_factor``

    ``scaled_variance = variance / scale_factor**2``

    For example, with ``scale='p50'``, each frame is normalized using
    its median value relative to the median of the frame medians.
    """

    logger.info(f"Using {scale} scheme for scaleing.")

    scale_mask = call_mask(scale_mask)

    if scale[0] == 'p':  # Using percentie scaling
        percentile = float(scale[1:])

        if scale_mask is None:
            scale_array = np.nanpercentile(datacube, percentile, axis=(1, 2))
        else:
            scale_array = np.array(
                [
                    np.nanpercentile(
                        d[~scale_mask], percentile
                        ) for d in datacube
                    ]
                )
    else:
        logger.error(
            f"Scale method {scale} which is not pXX not yet implemented"
            )

        raise ValueError(
            f"Scale method {scale} which is not pXX not yet implemented"
            )

    scale_array = scale_array / np.nanmedian(scale_array)

    scaled_datacube = datacube / scale_array[:, np.newaxis, np.newaxis]

    logger.info(f"The datacube is being scaled with {scale_array}")
    if varcube is None:
        return scaled_datacube, None, scale_array

    scaled_varcube = varcube / (
        scale_array[:, np.newaxis, np.newaxis]
        ) ** 2

    logger.info("Scaleing variance")

    return scaled_datacube, scaled_varcube, scale_array


def masking_frame(frame, mask, variance=None, method='interpolate'):
    """
    Apply a bad-pixel mask to a data frame.

    Pixels where the mask is not equal to 1 are either replaced with
    NaN values or interpolated from neighboring pixels, depending on
    the selected method.

    Parameters
    ----------
    frame : numpy.ndarray
        Input 2D data array.

    mask : numpy.ndarray, str, pathlib.Path, or list
        Mask information. Supported inputs are:

        - numpy.ndarray :
          Boolean/integer mask array with the same shape as ``frame``.
          Pixels with value 1 are considered valid.

        - str or pathlib.Path :
          Path to a ``.npy`` mask file that will be loaded using
          ``numpy.load``.

        - list :
          If a list is provided, only the first element is used.
          This is useful when arguments are parsed using
          ``argparse`` with ``nargs='+'``.

    variance : numpy.ndarray, optional
        Variance array corresponding to ``frame``. If provided,
        variances of masked pixels are multiplied by 1000 after
        masking/interpolation to reflect their reduced reliability.

    method : {'nan', 'interpolate'}, optional
        Method used to handle masked pixels.

        - ``'nan'`` :
          Replace masked pixels with ``NaN`` values.

        - ``'interpolate'`` :
          Fill masked pixels using biharmonic inpainting from
          neighboring valid pixels.

        Default is ``'interpolate'``.

    Returns
    -------
    numpy.ndarray or tuple
        If ``variance`` is not provided, returns the processed frame.

        If ``variance`` is provided, returns ``(frame, variance)``,
        where the variance of masked pixels has been increased.

    Notes
    -----
    For ``method='nan'``, the input ``frame`` is modified in place.

    For ``method='interpolate'``, a new array is returned by the
    interpolation routine.

    Interpolated pixels should be treated with caution, particularly
    for large masked regions. When a variance map is supplied, the
    variance of masked pixels is increased to reduce their influence
    in subsequent weighted analyses.
    """

    if isinstance(mask, list):
        mask = mask[0]
    mask = call_mask(mask)
    mask_bool = mask == 1

    if method == 'nan':
        frame[~mask_bool] = np.nan
        logger.info("Replacing bad pixels with NaN")

    elif method == 'interpolate':
        frame = inpaint.inpaint_biharmonic(
            frame,
            ~mask_bool)
        logger.info("Interpolating bad pixels.")
    if variance is not None:
        variance[~mask_bool] = 1000 * variance[~mask_bool]
        logger.info("Multiplying bad pixel variance with 1000")
        return frame, variance
    return frame


def operate_process(ip1, ip2,
                    opfilename,
                    operation='+',
                    fluxext=[0],
                    varext=None):
    """
    Perform arithmetic operations on FITS file extensions and write results.

    This function takes one FITS file (``ip1``) and either another FITS file
    or a constant value (``ip2``), performs the specified operation on the
    selected extensions, and writes the result to a new FITS file.

    Parameters
    ----------
    ip1 : str
        Path to the first FITS file.
    ip2 : str or float
        Path to the second FITS file, or a constant value to apply the
        operation.
        - If a filename, the same extensions as in ``fluxext`` will be read.
        - If a float, the value is broadcasted to the data in ``ip1``.
    opfilename : str
        Output FITS filename where the result will be written.
    operation : {'+', '-', '*', '/', ...}, optional
        Arithmetic operation to perform. Default is ``'+'``.
        The valid set depends on what ``ari_operations`` supports.
    fluxext : list of int, optional
        List of extension numbers containing flux data in the input files.
        Each extension in this list will be processed. Default is ``[0]``
        (primary HDU).
    varext : list of int or None, optional
        List of extension numbers containing variance data corresponding to
        each entry in ``fluxext``. If ``None`` (default), variance propagation
        is skipped.

    Notes
    -----
    - For each extension in ``fluxext``:

      1. Data are read from ``ip1`` and ``ip2``.
      2. The operation is applied using ``ari_operations``.
      3. Results are stored in the output HDUList.
      4. If ``varext`` is provided, the corresponding variance extensions are
         also operated on and appended to the output.

    - If an extension index is ``0``, the result is stored in the
      ``PrimaryHDU``. Otherwise, results are stored as ``ImageHDU``
      extensions.

    - A ``HISTORY`` entry is added to the output headers to track
      the operation.

    Examples
    --------
    Add fluxes in the primary HDU of two FITS files::

        operate_process("file1.fits", "file2.fits",
                        "sum.fits", operation='+', fluxext=[0])

    Subtract a constant value from a flux extension::

        operate_process("file1.fits", 10.0,
                        "output.fits", operation='-', fluxext=[1])

    Perform multiplication with variance propagation::

        operate_process("file1.fits", "file2.fits",
                        "multiplied.fits", operation='*',
                        fluxext=[1, 2], varext=[3, 4])
    """

    # primary_hdu = fits.PrimaryHDU()
    hdul1 = fits.open(ip1)
    hdul = fits.HDUList([hdu.copy() for hdu in hdul1])
    for index, ext in enumerate(fluxext):
        ext = int(ext)
        header = hdul[ext].header
        data1 = hdul1[ext].data
        header.add_history('{} {} {}'.format(Path(ip1).name,
                                             operation,
                                             Path(ip2).name))
        if varext is None:
            var1 = None
        else:
            var1 = hdul1[int(varext[index])].data
        hdul1.close()

        if isinstance(ip2, float):
            data2 = ip2
            var2 = 0
        else:
            if ip2[-5:] == ".fits":
                hdul2 = fits.open(ip2)
                data2 = hdul2[ext].data
                if varext is None:
                    var2 = None
                else:
                    var2 = hdul2[int(varext[index])].data
            elif ip2[-4:] == ".npy":
                data2 = np.load(ip2)
                var2 = 0
        result, var = ari_operations(data1, data2,
                                     var1, var2,
                                     operation=operation)
        hdul[ext].data = result
        hdul[ext].header = header
        if varext is not None:
            var_ext = int(varext[index])
            hdul[var_ext].data = var

    hdul1.close()
    hdul.writeto(opfilename, overwrite=True)


def combine_process(files,
                    opfilename,
                    path='.',
                    method='mean',
                    scale=None,
                    fluxext=[0],
                    varext=None,
                    mask=None,
                    mask_method='interpolate',
                    instrument=None
                    ):
    """
    Combine spectral or image data from multiple FITS files into a single
    output FITS file.

    This function supports two modes of operation:

    1. If an instrument is specified, it calls an instrument-specific routine
       (`combine_spectra`).
    2. Otherwise, it manually reads data arrays and (optionally) variance
       arrays from the input files, combines them using the given method,
       and writes the results into a new FITS file.

    Parameters
    ----------
    files : list of str or str
        Input FITS files. May be either:

        - A list of FITS file paths.
        - A glob pattern used to match FITS files within ``path``.
        - A string specifying a pattern/regular expression to match files in
          `path`.

    opfilename : str
        Output FITS filename to write the combined data.

    path : str, optional
        Path to search for FITS files if `files` is provided as a string
        pattern.
        Default is `'.'`.

    method : str, optional
        Combination method for data arrays (e.g., 'mean', 'median').
        Passed to `combine_data`. Default is `'mean'`.

    scale : str or None, optional
        Percentile-based scaling scheme used to normalize the input frames
        before combination. The value should follow the ``'pXX'`` format,
        where ``XX`` specifies the percentile used to determine the scaling
        factor. For example, ``'p50'`` uses the 50th percentile (median) of
        each frame for scaling. The corresponding variance arrays are
        scaled consistently. If ``None``, no scaling is applied.
        Default is ``None``.

    fluxext : list of int, optional
        List of FITS extensions containing flux (or image) data.
        Default is `[0]`.

    varext : list of int or None, optional
        List of FITS extensions containing variance data corresponding
        to `fluxext`. If `None`, variance is not processed. Default is `None`.

    mask : array_like or str or None, optional
        Bad-pixel mask to apply to the input data. If provided, bad pixels
        are either interpolated or replaced with NaN according to
        `mask_method`. Default is ``None``.

    mask_method : {'interpolate', 'nan'}, optional
        Method used to handle bad pixels when `mask` is provided and variance
        data are available. ``'interpolate'`` replaces bad pixels by
        interpolating from surrounding valid pixels, while ``'nan'`` replaces
        bad pixels with NaN. Default is ``'interpolate'``.

    instrument : str or None, optional
        Instrument name. If provided, the function calls
        `combine_spectra` instead of the default combination logic.
        Default is `None`.

    Returns
    -------
    None
        The combined FITS data is written directly to `opfilename`.

    Raises
    ------
    TypeError
        If ``files`` is neither a list of filenames nor a glob pattern.

    FileNotFoundError
        If ``files`` is given as a glob pattern and no matching files are
        found in ``path``.

    Notes
    -----
    - If `instrument` is not `None`, this function delegates to
      `combine_spectra` and returns immediately.
    - Input data are combined using ``combine_data``
    - Variance extensions are processed only if ``varext`` is provided.
    - If ``mask`` is supplied, bad pixels are masked or interpolated
      using ``masking_frame`` before the output is written.
    - The primary HDU is replaced when ``fluxext`` contains extension 0.

    Examples
    --------
    Combine the primary extension of a list of FITS files using the mean:

    >>> combine_process(files=["file1.fits", "file2.fits"],
    ...                 opfilename="combined.fits",
    ...                 fluxext=[0],
    ...                 method="mean")

    Combine flux and variance from extensions 1 and 2:

    >>> combine_process(files=["obs1.fits", "obs2.fits"],
    ...                 opfilename="combined.fits",
    ...                 fluxext=[1],
    ...                 varext=[2],
    ...                 method="median")
    """
    if instrument is not None:
        combine_spectra(files, opfilename=opfilename,
                        instrumentname=instrument,
                        method=method,
                        fluxext=fluxext,
                        varext=varext)
        return

    primary_hdu = fits.PrimaryHDU()
    hdul = fits.HDUList([primary_hdu])
    if isinstance(files, list):
        files_list = files
    elif isinstance(files, str):
        files_path = Path(path)
        files_list = list(files_path.glob(files))
        if not files_list:
            raise FileNotFoundError(
                f"No files found matching {files} in {path}"
            )
    else:
        raise TypeError(
            "'files must be either a list of filenames or a glob pattern."
        )

    for index, ext in enumerate(fluxext):
        ext = int(ext)
        header = fits.getheader(files_list[0], ext=ext)
        data_array = []
        var_array = []
        for fname in files_list:
            data = fits.getdata(fname, ext=ext)
            data_array.append(data)
            if varext is not None:
                var = fits.getdata(fname, ext=int(varext[index]))
                var_array.append(var)
        if len(files_list) == 1:
            result = data_array[0]
            if varext is not None:
                variance = var_array[0]
        else:
            if scale is not None:
                data_array, var_array, scale_array = scale_datacube(
                    datacube=data_array,
                    varcube=var_array,
                    scale=scale,
                    scale_mask=mask
                )
                header.add_history(
                    f"Arrays are scaled with {list(scale_array)}"
                )

            result, variance = combine_data(dataarr=data_array,
                                            var=var_array,
                                            method=method)
        to_history = [Path(i).name for i in files_list]
        header.add_history(method + str(to_history))
        if mask is not None:
            if varext is None:
                result = masking_frame(result, mask)
            else:
                result, variance = masking_frame(result,
                                                 mask,
                                                 variance,
                                                 method=mask_method)
            header.add_history(f"Mask used: {mask}")
            header.add_history(
                f"Bad pixels handled using method: {mask_method}"
            )
        if int(ext) == 0:
            hdul[0] = fits.PrimaryHDU(result, header=header)
        else:
            imagehdu = fits.ImageHDU(result, header=header,
                                     name="FLUX")
            hdul.append(imagehdu)
        if varext is not None:
            hdul.append(
                fits.ImageHDU(variance,
                              header=fits.getheader(
                                  files_list[0], ext=int(varext[index])
                                  ),
                              name="VARIANCE"
                              )
                )
        hdul.writeto(opfilename, overwrite=True)


def divide_smoothgradient(filename,
                          opfilename,
                          path='.',
                          medsmoothsize=(25, 51),
                          fluxext=[0],
                          varext=None):
    """
    Apply a median filter to an astronomical FITS image and normalize it
    by dividing the original image by the smoothed background gradient.

    This is typically used to remove large-scale background gradients
    while preserving smaller-scale features in the image.

    Parameters
    ----------
    filename : str
        Input FITS file containing the data to be processed.
    opfilename : str
        Output FITS file where the processed result will be saved.
    path : str, optional
        Path to the input file. Default is the current directory ('.').
    medsmoothsize : tuple of int, optional
        Size of the median filter window. Larger sizes smooth more strongly.
        Default is (25, 51).
    fluxext : list of int, optional
        List of extensions in the FITS file that contain the flux/image data
        to be normalized. Default is [0] (primary extension).
    varext : list of int, optional
        List of extensions corresponding to variance maps for each flux
        extension. If provided, the variance maps will also be normalized by
        the squared smoothed gradient. Default is None.

    Notes
    -----
    - The function clips the input image values to avoid division by zero:
      `inputimgdata = np.clip(inputimgdata, 1, np.max(inputimgdata+1))`.
    - Median filtering may be memory intensive. If a `MemoryError` occurs,
      try using a smaller `medsmoothsize`.
    - For each extension processed:
        * The flux is divided by the median-smoothed version of itself.
        * If variance data are provided, they are divided by the square
          of the median-smoothed image.
    - The output FITS file contains the normalized data (and variance maps,
      if applicable) with updated headers recording the operation history.

    Output
    ------
    FITS file
        A FITS file (`opfilename`) containing the normalized image(s) and
        optional variance extensions.

    Example
    -------
    >>> divide_smoothgradient("input.fits", "output.fits",
    ...                       medsmoothsize=(25, 51),
    ...                       fluxext=[0, 1],
    ...                       varext=[2, 3])
    """
    primary_hdu = fits.PrimaryHDU()
    hdul = fits.HDUList([primary_hdu])
    for index, ext in enumerate(fluxext):
        inputimgdata = fits.getdata(filename, ext=int(ext))
        inputimgdata = np.clip(inputimgdata, 1, np.max(inputimgdata+1))
        logger.info("Applying median filter with size %s", medsmoothsize)
        logger.info('It takes sometime (> 100 sec) to finish. Wait ...')
        try:
            smoothGrad = filters.median_filter(inputimgdata,
                                               size=medsmoothsize)

        except MemoryError:
            logger.error(
                "Skipping extension %d because median filtering "
                "ran out of memory.",
                ext,
                )
            continue
        else:
            header = fits.getheader(filename, ext=0)
            NormContdata = inputimgdata / smoothGrad
            if varext is not None:
                var = fits.getdata(filename, ext=int(varext[index]))
                NormCont_var = var / smoothGrad ** 2
            header.add_history('Divided median filter size: {}'.format(
                medsmoothsize))
            if int(ext) == 0:
                hdul[0] = fits.PrimaryHDU(NormContdata, header=header)
            else:
                imagehdu = fits.ImageHDU(NormContdata, header=header,
                                         name="FLUX")
                hdul.append(imagehdu)
            if varext is not None:
                hdul.append(
                    fits.ImageHDU(NormCont_var,
                                  header=fits.getheader(
                                      filename, ext=int(varext[index])
                                      ),
                                  name="VARIANCE"
                                  )
                    )
            hdul.writeto(opfilename, overwrite=True)


def remove_cosmic_rays(input_fname,
                       opfilename,
                       fluxext=[0],
                       varext=None):
    """
    Remove cosmic rays from FITS image extensions using ``astroscrappy``.

    This function reads one or more image extensions from a FITS file,
    detects and removes cosmic rays using the ``astroscrappy.detect_cosmics``
    algorithm, and writes the cleaned images (along with cosmic-ray masks
    and optional variance extensions) into a new output FITS file.

    Parameters
    ----------
    input_fname : str
        Path to the input FITS file containing the image data.

    opfilename : str
        Path to the output FITS file where the cosmic-ray-cleaned data
        will be written.

    fluxext : list of int, optional
        List of extension indices in the input FITS file that contain
        image data to be cleaned. Default is ``[0]`` (the primary HDU).

    varext : list of int or None, optional
        List of extension indices corresponding to variance data for
        each flux extension. If provided, the same indices are used to
        fetch the variance arrays and pass them to
        ``astroscrappy.detect_cosmics`` for improved detection. If
        ``None`` (default), cosmic-ray detection is run without variance
        information.

    Notes
    -----
    - The function uses the ``astroscrappy`` implementation of the LA
      Cosmic algorithm to detect and remove cosmic rays.
    - For each processed image extension, the following are written to
      the output file:

        * The cleaned image data
        * (Optionally) the corresponding variance extension, if
          ``varext`` is given
        * A binary mask extension named ``CRMASK`` with 1 where cosmic
          rays were detected

    - A ``HISTORY`` keyword is added to the header indicating that
      cosmic rays were removed with ``astroscrappy``.

    Output Structure
    ----------------
    The output FITS file will contain, in order:

        1. Cleaned image(s) in the same order as ``fluxext``
        2. Optional variance image(s), if ``varext`` is provided
        3. Corresponding cosmic-ray mask(s) named ``CRMASK``

    Example
    -------
    >>> remove_cosmic_rays(
    ...     "raw_image.fits",
    ...     "cleaned_image.fits",
    ...     fluxext=[1, 2],
    ...     varext=[3, 4]
    ... )

    This reads extensions 1 and 2 as flux images, uses extensions 3 and
    4 as variance maps, removes cosmic rays, and writes a cleaned file
    containing the corrected images, variance maps, and cosmic-ray masks.
    """
    primary_hdu = fits.PrimaryHDU()
    hdul = fits.HDUList([primary_hdu])
    for index, ext in enumerate(fluxext):
        inputimgdata = fits.getdata(input_fname, ext=int(ext))
        if varext is None:
            crmask, cleararr = astroscrappy.detect_cosmics(inputimgdata)
        else:
            inputvardata = fits.getdata(input_fname, ext=int(varext[index]))
            crmask, cleararr = astroscrappy.detect_cosmics(inputimgdata,
                                                           inputvardata)
        header = fits.getheader(input_fname, ext=0)
        header.add_history("Cosmic Rays removed with astroscrappy")
        if int(ext) == 0:
            hdul[0] = fits.PrimaryHDU(cleararr, header=header)
        else:
            imagehdu = fits.ImageHDU(cleararr, header=header)
            hdul.append(imagehdu)
        if varext is not None:
            hdul.append(
                fits.ImageHDU(inputvardata,
                              header=fits.getheader(
                                  input_fname, ext=int(varext[index])
                                  ),
                              name="VARIANCE"
                              )
                )
        hdul.append(
            fits.ImageHDU(crmask.astype(int), name="CRMASK")
        )
        hdul.writeto(opfilename, overwrite=True)


def shifting_frame(input_fname,
                   opfilename,
                   shifttoapply=np.array([0., 0.]),
                   fluxext=[0],
                   varext=None):
    """
    Shift the image extensions of a FITS file by an integer pixel offset.

    The specified flux extensions are shifted using ``numpy.roll``, which
    performs a circular shift (pixels shifted off one edge reappear on the
    opposite edge). If corresponding variance extensions are provided, they
    are shifted by the same amount.

    Parameters
    ----------
    input_fname : str or pathlib.Path
        Path to the input FITS file.

    opfilename : str or pathlib.Path
        Path to the output FITS file.

    shifttoapply : array-like of int, optional
        Pixel shift to apply in the form ``(row_shift, column_shift)``.
        Positive values shift the image towards increasing row or column
        indices. The default is ``(0, 0)``.

    fluxext : list of int, optional
        List of FITS extensions containing flux images to be shifted.
        The default is ``[0]``.

    varext : list of int, optional
        List of FITS extensions containing variance images corresponding
        to ``fluxext``. If provided, each variance extension is shifted by
        the same amount as its corresponding flux extension. The default
        is ``None``.

    Notes
    -----
    - Shifts are performed using ``numpy.roll`` and therefore are circular.
    - Only integer pixel shifts are supported.
    - The length of ``varext`` must match the length of ``fluxext`` when
      provided.

    Returns
    -------
    None
        The shifted FITS file is written to ``opfilename``.
    """
    primary_hdu = fits.PrimaryHDU()
    hdul = fits.HDUList([primary_hdu])
    header = fits.getheader(input_fname, ext=0)
    header.add_history("Shifted by {}".format(shifttoapply))
    for index, ext in enumerate(fluxext):
        inputimgdata = fits.getdata(input_fname, ext=int(ext))
        shifted = np.roll(inputimgdata,
                          shift=tuple(shifttoapply),
                          axis=(0, 1))
        if varext is not None:
            var = fits.getdata(input_fname, ext=int(varext[index]))
            shifted_var = np.roll(var,
                                  shift=tuple(shifttoapply),
                                  axis=(0, 1)
                                  )
        if int(ext) == 0:
            hdul[0] = fits.PrimaryHDU(shifted, header=header)
        else:
            imagehdu = fits.ImageHDU(shifted, header=header,
                                     name="FLUX")
            hdul.append(imagehdu)
        if varext is not None:
            hdul.append(
                fits.ImageHDU(shifted_var,
                              header=fits.getheader(
                                  input_fname, ext=int(varext[index])
                                  ),
                              name="VARIANCE"
                              )
                )
        hdul.writeto(opfilename, overwrite=True)

# End
