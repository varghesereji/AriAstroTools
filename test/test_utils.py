import numpy as np
import pytest
from astropy.io import fits

from ariastrotools.utils import shrink_fits


def test_shrink_fits(tmp_path):
    """Test shrinking a FITS file."""

    infile = tmp_path / "test.fits"

    # Create a test FITS file
    hdus = [
        fits.PrimaryHDU(data=np.arange(10)),
        fits.ImageHDU(np.ones((5, 5)), name="FLUX"),
        fits.ImageHDU(np.zeros((5, 5)), name="VAR"),
        fits.BinTableHDU.from_columns(
            [
                fits.Column(
                    name="TIME",
                    format="D",
                    array=np.arange(5, dtype=float),
                ),
                fits.Column(
                    name="VALUE",
                    format="E",
                    array=np.arange(5, dtype=np.float32),
                ),
            ],
            name="ACTIVITY",
        ),
    ]

    fits.HDUList(hdus).writeto(infile)

    outfile = shrink_fits(infile, ["FLUX"])

    with fits.open(outfile) as hdul:

        # Primary HDU unchanged
        np.testing.assert_array_equal(
            hdul[0].data,
            np.arange(10),
        )

        # Extension names preserved
        assert [h.name for h in hdul] == [
            "PRIMARY",
            "FLUX",
            "VAR",
            "ACTIVITY",
        ]

        # FLUX kept
        np.testing.assert_array_equal(
            hdul["FLUX"].data,
            np.ones((5, 5)),
        )

        # VAR emptied
        assert hdul["VAR"].data is None

        # ACTIVITY emptied
        assert len(hdul["ACTIVITY"].data) == 0

        # HISTORY added
        history = hdul[0].header.get("HISTORY", [])

        if isinstance(history, str):
            history = [history]

        print("\nHISTORY entries:")
        for h in history:
            print(repr(h))

        assert any(
            "Removed data from extensions:" in h
            for h in history
        )

        assert any(
            "VAR" in h and "ACTIVITY" in h
            for h in history
        )


def test_shrink_fits_invalid_extension(tmp_path):
    """Requesting a non-existent extension should raise an error."""

    infile = tmp_path / "test.fits"

    fits.HDUList([
        fits.PrimaryHDU(),
        fits.ImageHDU(np.ones((2, 2)), name="FLUX"),
    ]).writeto(infile)

    with pytest.raises(ValueError, match="do not exist"):
        shrink_fits(infile, ["NOT_REAL"])
