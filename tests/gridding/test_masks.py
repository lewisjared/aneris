import os

import xarray as xr

from aneris.gridding.masks import MaskLoader


def test_mask_iso_list(grid_dir):
    loader = MaskLoader(grid_dir)
    iso_list = loader.iso_list()

    assert len(iso_list) > 0


def test_mask_get(grid_dir):
    loader = MaskLoader(grid_dir)
    res = loader.get_iso("aus")

    assert isinstance(res, xr.DataArray)
    assert res.ndim == 2
