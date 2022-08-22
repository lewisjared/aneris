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

# %%
import os
from aneris.gridding import Gridder
import scmdata
import logging

# %%
logging.basicConfig(level=logging.INFO)

# %%
test_data_dir = os.path.join("..", "..", "tests", "test_data", "gridding")

# %%

# %%
input_emissions = scmdata.ScmRun(os.path.join(test_data_dir, "country_timeseries.csv"))

# %%

g = Gridder()
