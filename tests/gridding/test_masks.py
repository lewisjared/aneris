import os

import pytest
import xarray as xr

from aneris.gridding.masks import MaskLoader


def test_mask_iso_list(emissions_downscaling_archive):
    loader = MaskLoader(os.path.join(emissions_downscaling_archive, "gridding"))
    iso_list = loader.iso_list()

    assert len(iso_list) > 0


def test_mask_get(emissions_downscaling_archive):
    loader = MaskLoader(os.path.join(emissions_downscaling_archive, "gridding"))
    res = loader.get_iso("aus")

    assert isinstance(res, xr.DataArray)
    assert res.ndim == 2
