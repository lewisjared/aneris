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
# # Prepare the input data
#
# This notebook does some preprocessing on the masks provided as part of the input data provided by the Feng
# et al 2020 paper to make them easier to use.
#
# This requires that the output from https://zenodo.org/record/2538194 has been downloaded an extracted to
# `data/raw/emissions_downscaling_archive`.
#
# This also requires an additional `pyreadr` dependency.

# %%
# !pip install pyreadr

# %%
import os
from typing import Union
from glob import glob
import shutil

import xarray as xr
import pandas as pd
import numpy as np
import pyreadr

import matplotlib.pyplot as plt

# %%
DATA_DIR = os.path.join("..", "..", "data")
ZENODO_DATA_ARCHIVE = os.path.join(DATA_DIR, "raw", "emissions_downscaling_archive")
RAW_GRIDDING_DIR = os.path.join(ZENODO_DATA_ARCHIVE, "gridding")
GRID_RESOLUTION = 0.5
LAT_CENTERS = np.arange(90 - GRID_RESOLUTION / 2, -90, -GRID_RESOLUTION)
LON_CENTERS = np.arange(
    -180 + GRID_RESOLUTION / 2, 180 + GRID_RESOLUTION / 2, GRID_RESOLUTION
)

# %%
assert os.path.isdir(ZENODO_DATA_ARCHIVE)

# %%
# Load grid mapping files
grid_mappings = pd.read_csv(
    os.path.join(RAW_GRIDDING_DIR, "gridding-mappings", "country_location_index_05.csv")
).set_index("iso")


# %%
def read_mask_as_da(grid_dir, iso_code, grid_mappings):
    iso_code = iso_code.lower()

    fname = f"{grid_dir}/mask/{iso_code}_mask.Rd"
    mask = pyreadr.read_r(fname)[f"{iso_code}_mask"]

    mapping = grid_mappings.loc[iso_code]
    lats = LAT_CENTERS[int(mapping.start_row) - 1 : int(mapping.end_row)]
    lons = LON_CENTERS[int(mapping.start_col) - 1 : int(mapping.end_col)]

    da = xr.DataArray(mask, coords=(lats, lons), dims=("lat", "lon"))

    da.attrs["region"] = iso_code
    da.attrs["source"] = fname
    da.attrs["history"] = f"read_mask_as_da {fname}"
    return da


def read_proxy_file(proxy_fname: str) -> Union[xr.DataArray, None]:
    """
    Read a proxy file from disk

    We are using the existing proxy data from the Feng et al zenodo archive for now.
    These data are stored as Rd files (a proprietary format from R), but can be later
    expanded to also use proxies calculated as part of aneris.

    Parameters
    ----------
    proxy_fname : str
        Path to the proxy data

    Raises
    ------
    FileNotFoundError
        Requested proxy file cannot be found
    Returns
    -------
    xr.DataArray
        Proxy data augmented with latitude and longitude coordinates

    """
    fname = os.path.join(proxy_fname)
    if not os.path.exists(proxy_fname):
        return None

    data = pyreadr.read_r(fname)
    assert len(data) == 1
    data = data[list(data.keys())[0]]

    if data.ndim == 2:
        coords, dims = (LAT_CENTERS, LON_CENTERS), ("lat", "lon")
    elif data.ndim == 3:
        # AIR data also contain a y dimension
        levels = range(data.shape[2])
        coords, dims = (LAT_CENTERS, LON_CENTERS, levels), (
            "lat",
            "lon",
            "level",
        )
    else:
        raise ValueError(f"Unexpected dimensionality for proxy : {data.shape}")

    return xr.DataArray(data, coords=coords, dims=dims)


# %%
plt.figure(figsize=(12, 8))
read_mask_as_da(RAW_GRIDDING_DIR, "usa", grid_mappings).plot.contour()

# %%
read_mask_as_da(RAW_GRIDDING_DIR, "usa", grid_mappings)

# %%
output_grid_dir = os.path.join(DATA_DIR, "processed", "gridding")

# %% [markdown]
# # Masks

# %%
fnames = glob(os.path.join(RAW_GRIDDING_DIR, "mask", "*.Rd"))
country_codes = [os.path.basename(f).split("_")[0].upper() for f in fnames]
country_codes.remove("GLOBAL")
len(country_codes)

# %%
mask_dir = os.path.join(output_grid_dir, "masks")

if os.path.exists(mask_dir):
    shutil.rmtree(mask_dir)

os.makedirs(mask_dir)

for code in country_codes:
    mask = read_mask_as_da(RAW_GRIDDING_DIR, code, grid_mappings)
    mask.to_netcdf(os.path.join(mask_dir, f"mask_{code.upper()}.nc"))

# %% [markdown]
# # Proxies
#
# The proxies are also stored as Rdata
#
# There are 3 folders of interest:
# * proxy-CEDS9
# * proxy-CEDS16
# * proxy-backups

# %%
proxy_dirs = ["proxy-CEDS9", "proxy-CEDS16", "proxy-backups"]


# %%
for proxy_dir in proxy_dirs:
    output_proxy_dir = os.path.join(output_grid_dir, "proxies", proxy_dir)
    if os.path.exists(output_proxy_dir):
        shutil.rmtree(output_proxy_dir)

    os.makedirs(output_proxy_dir)

    fnames = glob(os.path.join(RAW_GRIDDING_DIR, proxy_dir, "*.Rd"))
    for fname in fnames:
        proxy = read_proxy_file(fname)
        fname_out, _ = os.path.splitext(os.path.basename(fname))

        variable, sector, year = fname_out.split("_")
        proxy.attrs["source"] = fname
        proxy.attrs["sector"] = sector
        proxy.attrs["year"] = year

        proxy.to_dataset(name=variable).to_netcdf(
            os.path.join(output_proxy_dir, f"{fname_out}.nc")
        )

# %%

# Seasonality

output_seasonality_dir = os.path.join(output_grid_dir, "seasonality", proxy_dir)
if os.path.exists(output_seasonality_dir):
    shutil.rmtree(output_seasonality_dir)

os.makedirs(output_seasonality_dir)
fnames = glob(os.path.join(RAW_GRIDDING_DIR, "seasonality-CEDS9", "*.Rd"))

for fname in fnames:
    try:
        proxy = read_proxy_file(fname)
    except pyreadr.LibrdataError:
        print(f"failed to read {fname}")
        continue

    toks = os.path.basename(fname).split("_")
    proxy.attrs["source"] = fname
    proxy.attrs["sector"] = toks[0]

    if len(toks) == 3:
        variable = toks[1]
    else:
        variable = "ALL"
    proxy.attrs["sector"] = variable
    fname_out = f"{toks[0]}_{variable}_seasonality.nc"

    proxy.to_dataset(name=variable).to_netcdf(
        os.path.join(output_seasonality_dir, fname_out)
    )
