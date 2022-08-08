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

import pandas as pd

import cartopy.crs as ccrs
import matplotlib.pyplot as plt

from aneris.gridding.masks import read_mask_as_da

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
plt.figure(figsize=(12, 8))
ax = plt.axes(projection=ccrs.PlateCarree())
read_mask_as_da(PROXY_DATA, "usa", grid_mappings).plot.contour()
ax.coastlines()

# %%
read_mask_as_da(PROXY_DATA, "fin", grid_mappings)

# %%
