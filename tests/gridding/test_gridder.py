import pytest
import os
import numpy.testing as npt

from aneris.gridding import Gridder
from aneris.gridding.gridder import convert_to_target_unit
from aneris.unit_registry import ur
from aneris.gridding.masks import MaskLoader


@pytest.fixture
def country_emissions(test_data_dir):
    return os.path.join(test_data_dir, "gridding", "country_timeseries.csv")


def test_gridder_setup(grid_dir):
    gridder = Gridder(grid_dir=grid_dir)

    assert isinstance(gridder.mask_loader, MaskLoader)
    assert gridder.mask_loader.grid_dir == grid_dir


def test_gridder_grid(grid_dir, country_emissions):
    gridder = Gridder(grid_dir=grid_dir)

    res = gridder.grid(country_emissions)
    pass


def test_unit_conversion():
    result = convert_to_target_unit("Mt CH4/yr", "kg")

    assert result.magnitude == 1e9
    assert result.u == ur.Unit("kg CH4/yr")

    value = ur.Quantity(12, "Mt CH4/yr")
    scale_factor = convert_to_target_unit(value.units, target_unit="kg")
    assert ur.Quantity(value.m * scale_factor.m, scale_factor.u) == ur(
        "12 * 1e9 kg CH4/yr"
    )


def test_unit_conversion_complex():
    result = convert_to_target_unit("Mt CH4 yr^-1 km^-2", "kg s^-1 m^-2")

    npt.assert_almost_equal(result.magnitude, 1e9 / 1000**2 / (365.25 * 24 * 60 * 60))
    assert result.u == ur.Unit("kg CH4 / s / m^2")
