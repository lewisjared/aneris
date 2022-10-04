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
from joblib import delayed, Parallel

import xarray as xr
import pandas as pd
import numpy as np
import pyreadr

# %%
DATA_DIR = os.path.join("..", "..", "data")
ZENODO_DATA_ARCHIVE = os.path.join(DATA_DIR, "raw", "emissions_downscaling_archive")
RAW_GRIDDING_DIR = os.path.join(ZENODO_DATA_ARCHIVE, "gridding")
GRID_RESOLUTION = 0.5
LAT_CENTERS = np.arange(90 - GRID_RESOLUTION / 2, -90, -GRID_RESOLUTION)
LON_CENTERS = np.arange(
    -180 + GRID_RESOLUTION / 2, 180 + GRID_RESOLUTION / 2, GRID_RESOLUTION
)
LEVELS = [
    0.305,
    0.915,
    1.525,
    2.135,
    2.745,
    3.355,
    3.965,
    4.575,
    5.185,
    5.795,
    6.405,
    7.015,
    7.625,
    8.235,
    8.845,
    9.455,
    10.065,
    10.675,
    11.285,
    11.895,
    12.505,
    13.115,
    13.725,
    14.335,
    14.945,
]

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

    if iso_code in grid_mappings.index:
        mapping = grid_mappings.loc[iso_code]
        lats = LAT_CENTERS[int(mapping.start_row) - 1 : int(mapping.end_row)]
        lons = LON_CENTERS[int(mapping.start_col) - 1 : int(mapping.end_col)]
    else:
        lats = LAT_CENTERS
        lons = LON_CENTERS

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
    elif data.ndim == 3 and data.shape[2] != 12:
        # AIR data also contain a y dimension
        coords, dims = (LAT_CENTERS, LON_CENTERS, LEVELS), (
            "lat",
            "lon",
            "level",
        )
    elif data.ndim == 3 and data.shape[2] == 12:
        # AIR data also contain a y dimension
        coords, dims = (LAT_CENTERS, LON_CENTERS, range(1, 12 + 1)), (
            "lat",
            "lon",
            "month",
        )
    else:
        raise ValueError(f"Unexpected dimensionality for proxy : {data.shape}")

    return xr.DataArray(data, coords=coords, dims=dims)

# %%
output_grid_dir = os.path.join(DATA_DIR, "processed", "gridding")

# %% [markdown]
# # Masks

# %%
print("Masks")

fnames = glob(os.path.join(RAW_GRIDDING_DIR, "mask", "*.Rd"))
country_codes = [os.path.basename(f).split("_")[0].upper() for f in fnames]
len(country_codes)

# %%
mask_dir = os.path.join(output_grid_dir, "masks")

if os.path.exists(mask_dir):
    shutil.rmtree(mask_dir)

os.makedirs(mask_dir)


def read_mask(code):
    mask = read_mask_as_da(RAW_GRIDDING_DIR, code, grid_mappings)
    mask.to_netcdf(os.path.join(mask_dir, f"mask_{code.upper()}.nc"))


# Parallel(n_jobs=16)(delayed(read_mask)(code) for code in country_codes)

# %% [markdown]
# # Proxies
#
# The proxies are also stored as Rdata
#
# There are 3 folders of interest:
# * proxy-CEDS9
# * proxy-CEDS16
# * proxy-backup

# %%
proxy_dirs = ["proxy-CEDS9", "proxy-CEDS16", "proxy-backup"]


# %%
def write_proxy_file(output_proxy_dir, fname):
    proxy = read_proxy_file(fname)
    fname_out, _ = os.path.splitext(os.path.basename(fname))

    toks = fname_out.split("_")
    if len(toks) == 3:
        variable, sector, year = toks
    else:
        variable, year = toks
        sector = "Total"

    proxy.attrs["source"] = fname
    proxy.attrs["sector"] = sector
    proxy.attrs["year"] = year

    proxy.to_dataset(name=variable).to_netcdf(
        os.path.join(output_proxy_dir, f"{fname_out}.nc")
    )


for proxy_dir in proxy_dirs:
    print("Proxies " + proxy_dir)
    output_proxy_dir = os.path.join(output_grid_dir, "proxies", proxy_dir)
    if os.path.exists(output_proxy_dir):
        shutil.rmtree(output_proxy_dir)

    os.makedirs(output_proxy_dir)

    fnames = glob(os.path.join(RAW_GRIDDING_DIR, proxy_dir, "*.Rd"))

    Parallel(n_jobs=8)(delayed(write_proxy_file)(output_proxy_dir, fname) for fname in fnames)

# %%

# Seasonality

print("Seasonality")

output_seasonality_dir = os.path.join(output_grid_dir, "seasonality")
if os.path.exists(output_seasonality_dir):
    shutil.rmtree(output_seasonality_dir)

os.makedirs(output_seasonality_dir)
fnames = glob(os.path.join(RAW_GRIDDING_DIR, "seasonality-CEDS9", "*.Rd"))

def read_seasonality(fname):
    try:
        proxy = read_proxy_file(fname)
    except pyreadr.LibrdataError:
        print(f"failed to read {fname}")
        return

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

read_seasonality(fnames[0])
Parallel(n_jobs=16)(delayed(read_seasonality)(fname) for fname in fnames)
