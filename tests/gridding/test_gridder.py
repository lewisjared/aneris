import os
from unittest import mock

import numpy as np
import numpy.testing as npt
import pytest
import scmdata
import xarray as xr

from aneris.gridding import Gridder
from aneris.gridding.gridder import convert_to_target_unit
from aneris.gridding.masks import MaskLoader
from aneris.unit_registry import ur


@pytest.fixture
def country_emissions(test_data_dir):
    return os.path.join(test_data_dir, "gridding", "country_timeseries.csv")


@pytest.fixture
def country_emissions_clean(test_data_dir, country_emissions):
    emms = scmdata.ScmRun(country_emissions)
    return emms.filter(region="World|R5*", keep=False)


def test_gridder_setup(grid_dir):
    gridder = Gridder(grid_dir=grid_dir)

    assert isinstance(gridder.mask_loader, MaskLoader)
    assert gridder.mask_loader.grid_dir == grid_dir


@mock.patch.object(Gridder, "grid_sector")
def test_gridder_grid(mock_grid_sector, grid_dir, country_emissions):
    mock_grid_sector.return_value = xr.DataArray(np.zeros())

    gridder = Gridder(grid_dir=grid_dir)

    res = gridder.grid(country_emissions)

    pass


def test_gridder_grid_sector(grid_dir, country_emissions_clean):
    gridder = Gridder(grid_dir=grid_dir)
    emissions = country_emissions_clean.filter(variable="Emissions|CH4|Total")

    res = gridder.grid_sector(
        model="test",
        scenario="ssp126",
        variable="Emissions|CH4|Total",
        emissions=emissions,
    )
    assert res["scenario"] == "ssp126"
    assert res["model"] == "test"
    assert res["species"] == "CH4"
    assert res["sector"] == "Total"

    assert ur(str(res["unit"].values)) == ur("kg CH4 / s/ m^2")


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
