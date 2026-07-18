from astropy.io import fits
from pathlib import Path

from .logger import logger


def shrink_fits(filename, extensions, replace=False,
                strict=True):
    """
    Shrink a FITS file by retaining data only in selected extensions.

    Extensions not listed in ``extensions`` are replaced with empty HDUs,
    preserving their headers and extension names.

    Parameters
    ----------
    filename : str or pathlib.Path
        Input FITS file.

    extensions : list of str or int
        Extension names and/or extension numbers to retain.
        The primary HDU (extension 0) is always preserved.

    replace : bool, optional
        If True, overwrite the original file. Otherwise create a new file
        with suffix ``.shrink.fits``. Default is False.

    strict : bool, optional
        If True (default), raise an exception if any requested extension
        does not exist. Otherwise, issue a warning and continue.

    Returns
    -------
    str
        Name of the output FITS file.
    """
    filename = Path(filename)
    logger.info("shrinking file {}".format(filename))
    if replace:
        outfile = filename
    else:
        outfile = filename.with_suffix("").with_suffix(".shrink.fits")

    removed = []
    with fits.open(filename) as hdul:
         # ----------------------------------------------------------
        # Validate requested extensions
        # ----------------------------------------------------------
        available_names = {hdu.name for hdu in hdul[1:]}
        available_numbers = set(range(1, len(hdul)))

        missing = []

        for ext in extensions:
            if isinstance(ext, str):
                if ext not in available_names:
                    missing.append(ext)
            elif isinstance(ext, int):
                if ext not in available_numbers:
                    missing.append(ext)

        if missing:
            msg = (
                "The following requested extensions do not exist: "
                + ", ".join(map(str, missing))
            )
            if strict:
                raise ValueError(msg)
            logger.warning(msg)
        primary = hdul[0].copy()
        primary.header.add_history("File shrunk using shrink_fits().")
        new_hdus = [primary]  # Always keep the primary HDU

        for i, hdu in enumerate(hdul[1:], start=1):
            keep = (i in extensions) or (hdu.name in extensions)

            if keep:
                new_hdus.append(hdu.copy())
            else:
                removed.append(hdu.name)
                header = hdu.header.copy()
                if isinstance(hdu, fits.ImageHDU):
                    new_hdu = fits.ImageHDU(
                        data=None,
                        header=header,
                        name=hdu.name)
                elif isinstance(hdu, fits.BinTableHDU):
                    new_hdu = fits.BinTableHDU(
                        data=None,
                        header=header,
                        name=hdu.name
                    )
                else:
                    # Fallback for any other extension type
                    new_hdu = fits.ImageHDU(
                        data=None,
                        header=header,
                        name=hdu.name
                    )
                new_hdus.append(new_hdu)
        if removed:
            primary.header.add_history(
                "Removed data from extensions: "
                + ", ".join(removed)
            )
    fits.HDUList(new_hdus).writeto(outfile, overwrite=True)

    return str(outfile)


def extract_data_header(hdu, ext=0):
    """
    Function to open the fits file and
    extract the data and header.
    """
    opened = False
    if isinstance(hdu, str):
        hdu = fits.open(hdu)
        opened = False
    data = hdu[ext].data
    header = hdu[ext].header
    extname = hdu[ext].header.get("EXTNAME")
    if opened:
        hdu.close()
    return data, header, extname


def extract_allexts(fname):
    """
    Extract all extensions from a FITS file.

    Parameters
    ----------
    fname : str
        Path to the FITS file.

    Returns
    -------
    datadict : dict
        Dictionary mapping extension keywords to numpy arrays
        containing the data.
    headerdict : dict
        Dictionary mapping extension keywords to FITS headers.
        """

    hdu = fits.open(fname)
    datadict = {}
    headerdict = {}
    for ext in range(len(hdu)):
        data, header, extname = extract_data_header(hdu, ext=ext)
        datadict[extname] = data
        headerdict[extname] = header
    return datadict, headerdict


def create_fits(datadict, header_dict, filename="Avg_neid_data.fits"):
    """
    Create a multi-extension FITS file from a dictionary of data arrays and
    headers.

    Parameters
    ----------
    datadict : dict
        Dictionary mapping extension names (str) to their corresponding data.
        - The first entry in `datadict` is treated as the *primary HDU*.
        - Other entries are written as either `ImageHDU` (for numeric arrays)
          or `BinTableHDU` (for tabular/structured arrays, e.g. 'ACTIVITY').

    header_dict : dict
        Dictionary mapping extension names (str) to FITS header information.
        Each value must be compatible with `astropy.io.fits.Header`.

    filename : str, optional
        Name of the FITS file to create. Default is `"Avg_neid_data.fits"`.

    Notes
    -----
    - The function automatically selects `BinTableHDU` for extensions
      listed in `tablehdu` (currently only `'ACTIVITY'`).
    - All other extensions are written as `ImageHDU`.
    - Existing files with the same name are overwritten.

    Examples
    --------
    >>> datadict = {
    ...     "PRIMARY": np.zeros((100, 100)),        # primary HDU data
    ...     "SCIENCE": np.random.random((50, 50)),  # image extension
    ...     "ACTIVITY": structured_array            # table extension
    ... }
    >>> header_dict = {
    ...     "PRIMARY": {"OBSERVER": "Varghese"},
    ...     "SCIENCE": {"EXTNAME": "SCIENCE"},
    ...     "ACTIVITY": {"COMMENT": "Activity indices"}
    ... }
    >>> create_fits(datadict, header_dict, filename="output.fits")

    This will produce a FITS file with:
    - A primary HDU containing the first dataset.
    - An image extension for "SCIENCE".
    - A binary table extension for "ACTIVITY".
    """
    header_names = list(datadict.keys())
    hdus = []

    # --- Primary HDU ---
    primary_data = datadict[header_names[0]]

    primary_header = fits.Header(header_dict[header_names[0]])

    primary_hdu = fits.PrimaryHDU(data=primary_data, header=primary_header)
    hdus.append(primary_hdu)

    # --- Extensions ---
    tablehdu = ['ACTIVITY']
    for exts in header_names[1:]:
        data = datadict[exts]
        ext_header = fits.Header(header_dict[exts])
        if exts in tablehdu:
            hdu = fits.BinTableHDU(data=data, header=ext_header, name=exts)
        else:
            hdu = fits.ImageHDU(data=data, header=ext_header, name=exts)
        hdus.append(hdu)

    # --- Write FITS ---
    hdul = fits.HDUList(hdus)
    hdul.writeto(filename, overwrite=True)

# End
