import pytest
import os

from aneris.gridding import Gridder
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

    gridder.grid(country_emissions)
