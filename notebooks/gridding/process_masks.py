# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -pycharm
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.14.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Process Masks
#
# This notebook does some preprocessing on the masks provided as part of the input data provided by the Feng et al 2020 paper.
#
# This requires that the output from https://zenodo.org/record/2538194 has been downloaded an extracted to `data/raw/emissions_downlscaling_archive`.

# %%
import os

import pyreadr
import numpy as np
import pandas as pd
import xarray as xr

import cartopy.crs as ccrs
import matplotlib.pyplot as plt

# %%
PROXY_DATA = os.path.join(
    "..", "..", "data", "raw", "emissions_downscaling_archive", "gridding"
)

# %%
grid_mappings = pd.read_csv(
    os.path.join(PROXY_DATA, "gridding-mappings", "country_location_index_05.csv")
).set_index("iso")
grid_mappings

# %%
grid_resolution = 0.5
lat_centers = np.arange(90 - grid_resolution / 2, -90, -grid_resolution)
lon_centers = np.arange(
    -180 + grid_resolution / 2, 180 + grid_resolution / 2, grid_resolution
)


# %%
def read_mask_as_da(iso_code, grid_mappings):
    iso_code = iso_code.lower()

    fname = f"{PROXY_DATA}/mask/{iso_code}_mask.Rd"
    mask = pyreadr.read_r(fname)[f"{iso_code}_mask"]

    mapping = grid_mappings.loc[iso_code]
    lats = lat_centers[int(mapping.start_row) - 1 : int(mapping.end_row)]
    lons = lon_centers[int(mapping.start_col) - 1 : int(mapping.end_col)]

    return xr.DataArray(mask, coords=(lats, lons), dims=("lat", "lon"))


# %%


# %%
plt.figure(figsize=(12, 8))
ax = plt.axes(projection=ccrs.PlateCarree())
read_mask_as_da("usa", grid_mappings).plot.contour()
ax.coastlines()

# %%
read_mask_as_da("fin", grid_mappings)

# %%
