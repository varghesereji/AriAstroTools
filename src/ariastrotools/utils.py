from astropy.io import fits


def extract_data_header(hdu, ext=0):
    """
    Function to open the fits file and
    extract the data and header.
    """
    data = hdu[ext].data
    header = hdu[ext].header
    extname = hdu[ext].header.get("EXTNAME")

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
