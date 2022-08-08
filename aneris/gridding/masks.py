import os.path

import pandas as pd
import pyreadr
import xarray as xr
import numpy as np
from glob import glob
from typing import List

GRID_RESOLUTION = 0.5
LAT_CENTERS = np.arange(90 - GRID_RESOLUTION / 2, -90, -GRID_RESOLUTION)
LON_CENTERS = np.arange(
    -180 + GRID_RESOLUTION / 2, 180 + GRID_RESOLUTION / 2, GRID_RESOLUTION
)
DEFAULT_ISO_LIST = []


def read_mask_as_da(grid_dir, iso_code, grid_mappings):
    iso_code = iso_code.lower()

    fname = f"{grid_dir}/mask/{iso_code}_mask.Rd"
    mask = pyreadr.read_r(fname)[f"{iso_code}_mask"]

    mapping = grid_mappings.loc[iso_code]
    lats = LAT_CENTERS[int(mapping.start_row) - 1 : int(mapping.end_row)]
    lons = LON_CENTERS[int(mapping.start_col) - 1 : int(mapping.end_col)]

    return xr.DataArray(mask, coords=(lats, lons), dims=("lat", "lon"))


class MaskLoader:
    """
    Loads and processes country masks

    Currently the country masks come from the emissions_downscaling data archive, but
    these could one day be swapped out for other grids if needed by subclassing.
    """

    def __init__(self, grid_dir):
        self.grid_dir = grid_dir
        self.grid_mappings = self._read_grid_mappings()

    def _read_grid_mappings(self):
        return pd.read_csv(
            # TODO: link to config
            os.path.join(
                self.grid_dir, "gridding-mappings", "country_location_index_05.csv"
            )
        ).set_index("iso")

    def get_iso(self, iso_code: str) -> xr.DataArray:
        return read_mask_as_da(
            self.grid_dir, iso_code, grid_mappings=self.grid_mappings
        )

    def iso_list(self) -> List[str]:
        """
        Get the list of available ISOs

        Returns
        -------
        list of str
        """

        fnames = glob(os.path.join(self.grid_dir, "mask", "*.Rd"))

        return [f.split("_")[0] for f in fnames]
