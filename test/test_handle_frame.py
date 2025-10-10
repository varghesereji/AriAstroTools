import numpy as np
import pytest
from astropy.io import fits

from ariastro.handle_frame import remove_cosmic_rays


@pytest.fixture
def sample_fits(tmp_path):
    """Create a synthetic FITS file with cosmic-ray–like artifacts."""
    # Create fake data with cosmic rays
    img = np.random.normal(1000, 5, (100, 100))
    img[30, 40] = 10000  # a bright cosmic ray
    img[70, 80] = 12000  # another cosmic ray

    var = np.full((100, 100), 25.0)  # simple variance map

    # Create FITS HDUs
    hdu_flux = fits.PrimaryHDU(img)
    hdu_var = fits.ImageHDU(var, name="VARIANCE")

    fname = tmp_path / "test_input.fits"
    hdu_flux.writeto(fname, overwrite=True)

    # Append using an ImageHDU explicitly
    with fits.open(fname, mode="update") as hdul:
        hdul.append(hdu_var)
        hdul.flush()

    return fname


def test_remove_cosmic_rays_basic(sample_fits, tmp_path):
    """Test that remove_cosmic_rays runs and produces valid output."""
    outname = tmp_path / "cleaned_output.fits"

    remove_cosmic_rays(
        input_fname=sample_fits,
        opfilename=outname,
        fluxext=[0],
        varext=[1]
    )

    assert outname.exists(), "Output FITS file was not created"

    with fits.open(outname) as hdul:
        assert len(hdul) == 3, "Unexpected number of extensions"

        clean_flux = hdul[0].data
        var_data = hdul[1].data
        crmask = hdul[2].data

        assert clean_flux.shape == (100, 100)
        assert crmask.shape == (100, 100)

        assert set(np.unique(crmask)).issubset({0, 1}), "CRMASK not binary"

        orig = fits.getdata(sample_fits, ext=0)
        diff_pixels = np.sum(orig != clean_flux)
        assert diff_pixels > 0, "No pixels were \
        cleaned — cosmic rays not removed"

        np.testing.assert_allclose(var_data, 25.0)


def test_remove_cosmic_rays_without_variance(sample_fits, tmp_path):
    """Test function works even when no variance extension is provided."""
    outname = tmp_path / "cleaned_no_var.fits"

    remove_cosmic_rays(
        input_fname=sample_fits,
        opfilename=outname,
        fluxext=[0],
        varext=None
    )

    with fits.open(outname) as hdul:
        # Expect: cleaned flux + CRMASK only
        assert len(hdul) == 2
        assert "CRMASK" in hdul[1].name
