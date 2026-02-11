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

from pathlib import Path
from astropy.io import fits
from scipy.ndimage import shift

from .operations import ari_operations
from .operations import combine_data
from .spectral_utils import combine_spectra


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

    primary_hdu = fits.PrimaryHDU()
    hdul = fits.HDUList([primary_hdu])
    for index, ext in enumerate(fluxext):
        ext = int(ext)
        header = fits.getheader(ip1, ext=ext)
        hdul1 = fits.open(ip1)
        data1 = hdul1[ext].data
        header['HISTORY'] = '{} {} {}'.format(Path(ip1).name,
                                              operation,
                                              Path(ip2).name)
        if varext is None:
            var1 = None
        else:
            var1 = hdul1[int(varext[index])].data
        hdul1.close()

        if isinstance(ip2, float):
            data2 = ip2
            var2 = 0
        else:
            hdul2 = fits.open(ip2)
            data2 = hdul2[ext].data
            if varext is None:
                var2 = None
            else:
                var2 = hdul2[int(varext[index])].data
            hdul2.close()
        result, var = ari_operations(data1, data2,
                                     var1, var2,
                                     operation=operation)
        if int(ext) == 0:
            hdul[0] = fits.PrimaryHDU(result, header=header)
        else:
            imagehdu = fits.ImageHDU(result, header=header,
                                     name="FLUX")
            hdul.append(imagehdu)
        if varext is not None:
            hdul.append(
                fits.ImageHDU(var,
                              header=fits.getheader(
                                  ip1, ext=int(varext[index])
                              ),
                              name="VARIANCE"
                              )
            )
    hdul.writeto(opfilename, overwrite=True)


def combine_process(files,
                    opfilename,
                    path='.',
                    method='mean',
                    fluxext=[0],
                    varext=None,
                    mask=None,
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
        Input FITS files. Can be:

        - A list of FITS file paths.
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

    fluxext : list of int, optional
        List of FITS extensions containing flux (or image) data.
        Default is `[0]`.

    varext : list of int or None, optional
        List of FITS extensions containing variance data corresponding
        to `fluxext`. If `None`, variance is not processed. Default is `None`.

    instrument : str or None, optional
        Instrument name. If provided, the function calls
        `combine_spectra` instead of the default combination logic.
        Default is `None`.

    Returns
    -------
    None
        The combined FITS data is written directly to `opfilename`.

    Notes
    -----
    - If `instrument` is not `None`, this function delegates to
      `combine_spectra` and returns immediately.
    - The combination of data is handled by `combine_data`, which is expected
      to return `(result, variance)`.
    - The variance extension in the output file is only written if
      `varext` is provided.
    - The primary HDU (extension 0) is replaced if `fluxext` includes 0.

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
        files_list = files_path.glob(files)
        if not files_list:
            raise FileNotFoundError(
                f"No files found matching {files} in {path}"
            )
    else:
        print("Enter either files list or the regular expression")

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
            result, variance = combine_data(dataarr=data_array,
                                            var=var_array,
                                            method=method)
        to_history = [Path(i).name for i in files_list]
        header["HISTORY"] = method + str(to_history)
        if int(ext) == 0:
            hdul[0] = fits.PrimaryHDU(result, header=header)
        else:
            imagehdu = fits.ImageHDU(result, header=header,
                                     name="FLUX")
            hdul.append(imagehdu)
        if varext is not None:
            hdul.append(
                fits.ImageHDU(var,
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
        List of extensions corresponding to variance maps for each flux extension.
        If provided, the variance maps will also be normalized by the squared
        smoothed gradient. Default is None.

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
        print("Smoothing the frame")
        print('It takes sometime (> 100 sec) to finish. Wait ...')
        try:
            smoothGrad = filters.median_filter(inputimgdata,
                                               size=medsmoothsize)

        except MemoryError:
            print("*** MEMORY ERROR : Skipping median filter Division ***")
            print("Try giving a smaller smooth size for medial filtter insted")
        else:
            header = fits.getheader(filename, ext=0)
            NormContdata = inputimgdata / smoothGrad
            if varext is not None:
                var = fits.getdata(filename, ext=int(varext[index]))
                NormCont_var = var / smoothGrad ** 2
            header['HISTORY'] = 'Divided median filter size: {}'.format(
                medsmoothsize)
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
        header['HISTORY'] = "Cosmic Rays removed with astroscrappy"
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
    primary_hdu = fits.PrimaryHDU()
    hdul = fits.HDUList([primary_hdu])
    header = fits.getheader(input_fname, ext=0)
    header['HISOTRY'] = "Shifted by {}".format(shifttoapply)
    for index, ext in enumerate(fluxext):
        inputimgdata = fits.getdata(input_fname, ext=int(ext))
        shifted = shift(inputimgdata,
                        shifttoapply,
                        order=3)
        if varext is not None:
            var = fits.getdata(input_fname, ext=int(varext[index]))
            shifted_var = shift(var,
                                shifttoapply,
                                order=3)
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
