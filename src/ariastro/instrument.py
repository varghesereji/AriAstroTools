
import numpy as np

from .utils import extract_allexts


class Handle_NEID:
    """
    Class to handle NEID spectrograph FITS data.

    This class provides methods for reading NEID FITS files, applying
    barycentric corrections to the wavelength solution, and performing
    basic blaze correction on the extracted orders.

    Attributes
    ----------
    name : str
        Instrument identifier, set to "NEID".

    Methods
    -------
    __init__():
        Initializes the NEID handler.

    getfull_data(fname):
        Reads all extensions from a FITS file and returns data and headers.

    barycorr(wl_array, header):
        Applies barycentric correction to the wavelength array using
        header keywords `SSBZxxx`.

    process_data(fname):
        Reads the FITS file, applies barycentric and blaze corrections,
        and returns corrected data and headers.
    """

    name = "NEID"

    def __init__(self):
        """Initialize the NEID handler instance."""
        pass

    def fits_extensions(self):
        fluxext = [1, 2, 3]
        varext = [4, 5, 6]
        wlext = [7, 8, 9]
        return fluxext, varext, wlext

    def getfull_data(self, fname):
        """
        Extract all extensions from a NEID FITS file.

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
        datadict, headerdict = extract_allexts(fname)
        return datadict, headerdict

    def barycorr(self, wl_array, header):
        """
        Apply barycentric correction to the wavelength array.

        For each spectral order, the barycentric correction factor is
        retrieved from the header keywords ``SSBZxxx`` (where xxx is the
        order number, zero-padded to 3 digits). The factor is then applied
        multiplicatively to the wavelength array.

        Parameters
        ----------
        wl_array : ndarray
            2D array of wavelength values (orders x pixels).
        header : astropy.io.fits.Header
            FITS header containing the barycentric correction keywords.

        Returns
        -------
        corr_wl_array : ndarray
            Wavelength array corrected for barycentric motion.
        header : astropy.io.fits.Header
            Updated header with ``SSBZxxx`` values reset to zero.
        """

        n_orders = wl_array.shape[0]
        zfacts = []
        for index in range(n_orders):
            order = 173 - index
            strnum = str(order)
            while len(strnum) < 3:
                strnum = '0' + strnum
            zfact = header['SSBZ'+strnum]
            header['SSBZ'+strnum] = 0
            zfacts.append(float(zfact))
        zfacts = np.array(zfacts)
        corr_wl_array = (wl_array.T * (1+zfacts)).T

        return corr_wl_array, header

    def process_data(self, fname, contnorm=False):
        """
        Process a NEID FITS file: barycentric correction, blaze correction,
        and variance correction.

        This method:
        - Reads in the flux, variance, wavelength, blaze, and header data.
        - Applies barycentric correction to the wavelength arrays.
        - Replaces blaze arrays with ones (effectively removing blaze shape).
        - Corrects flux and variance for blaze.
        - Continuum normalization. (optional)

        Parameters
        ----------
        fname : str
            Path to the NEID FITS file.

        Returns
        -------
        datadict : dict
            Dictionary with corrected data arrays (flux, variance, blaze).
        headerdict : dict
            Dictionary with updated FITS headers.
        contnorm  :  bool
            Do continuum division with the function continuum_normalize.
        """
        datadict, headerdict = self.getfull_data(fname)
        # print(datadict)

        sci_ext = [1, 2, 3]
        var_ext = [4, 5, 6]
        wl_ext = [7, 8, 9]
        blaze_ext = [15, 16, 17]
        header_kws = list(datadict.keys())
        # print(header_kws)
        for n, ext in enumerate(sci_ext):

            flux_kw = header_kws[sci_ext[n]]
            var_kw = header_kws[var_ext[n]]
            wl_kw = header_kws[wl_ext[n]]
            blaze_kw = header_kws[blaze_ext[n]]
            # print(flux_kw, var_kw, wl_kw, blaze_kw)

            flux = datadict[flux_kw].astype(np.float64)
            var = datadict[var_kw].astype(np.float64)
            wl = datadict[wl_kw].astype(np.float64)
            blaze = datadict[blaze_kw].astype(np.float64)
            header_ext = headerdict[header_kws[0]]
            # print(header_ext)
            corr_wl, corr_header = self.barycorr(wl, header_ext)
            headerdict[header_kws[0]] = corr_header
            newblaze = np.ones(blaze.shape)
            corr_flux = flux / blaze
            corr_var = var / blaze ** 2
            datadict[flux_kw] = corr_flux
            datadict[var_kw] = corr_var
            datadict[wl_kw] = corr_wl
            datadict[blaze_kw] = newblaze
        if contnorm:
            from .spectral_utils import continuum_normalize
            datadict = continuum_normalize(datadict, sci_ext, var_ext, wl_ext)
        return datadict, headerdict

    def req_qtys(self):
        """
        Dictonary format:
        {Name of extension: List of keys from that extension}
        """
        qty = {'CCFS': ['CCFRVMOD', 'BISMOD', 'FWHMMOD']}
        return qty


instrument_dict = {
    "NEID": Handle_NEID
    }

# End
