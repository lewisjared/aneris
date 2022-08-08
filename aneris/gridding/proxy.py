import logging
import os

import pandas as pd
import xarray as xr
import pyreadr
from typing import List
from .masks import LAT_CENTERS, LON_CENTERS

logger = logging.getLogger(__name__)


def read_proxy_file(proxy_fname: str) -> xr.DataArray:
    fname = os.path.join(proxy_fname)
    logger.debug(f"Reading {fname}")

    data = pyreadr.read_r(fname)

    return xr.DataArray(data, coords=(LAT_CENTERS, LON_CENTERS), dims=("lat", "lon"))


class ProxyDataset:
    """
    A proxy dataset is used
    """

    def __init__(self, masker, data):
        self.masker = masker
        self.data = data

    def get_weighted(self, iso_code: str) -> xr.DataArray:
        return self.data * self.masker.get(iso_code)

    @classmethod
    def load_from_proxy_file(
        cls, proxy_definition_file, masker, species: str, sector: str, years: List[int]
    ):
        proxy_definitions = pd.read_csv(proxy_definition_file)

        selected_proxies = proxy_definitions[
            (proxy_definitions.em == species) & (proxy_definitions.sector == sector)
        ]
        if not len(selected_proxies):
            raise ValueError(f"Could not find proxy definition for {species}/{sector}")

        proxies = []

        for y in years:
            proxy_fname = selected_proxies[selected_proxies.year == y]
            if len(proxy_fname) != 1:
                raise ValueError(
                    f"Could not select a single proxy for {species}/{sector}/{y}"
                )
            proxy_fname = proxy_fname.squeeze()

            try:
                proxy = read_proxy_file(proxy_fname["proxy_file"])
            except FileNotFoundError:
                logger.error(
                    f"Could not load proxy data for {species}/{sector}/{y}. Falling back to backup"
                )
                proxy = read_proxy_file(proxy_fname["proxybackup_file"])
            proxy["year"] = y

            proxies.append(proxy)

        return cls(masker, xr.concat(proxies, dim="year"))
