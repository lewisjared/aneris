import logging
import os

import pandas as pd
import xarray as xr
import pyreadr
from typing import List
from .masks import LAT_CENTERS, LON_CENTERS

logger = logging.getLogger(__name__)


def read_proxy_file(proxy_fname: str) -> xr.DataArray:
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
        raise FileNotFoundError(proxy_fname)
    logger.debug(f"Reading {fname}")

    data = pyreadr.read_r(fname)

    return xr.DataArray(data, coords=(LAT_CENTERS, LON_CENTERS), dims=("lat", "lon"))


class ProxyDataset:
    """
    A proxy dataset which is used to scale country data into gridded data
    """

    def __init__(self, data: xr.DataArray):
        self.data = data

    def get_weighted(self, mask: xr.DataArray) -> xr.DataArray:
        """
        Get data which has been masked and weighted

        Parameters
        ----------
        mask : xr.DataArray
            A lat, lon array used to select a subset of the data

            In this application the mask is generally used to denote the percentage of
            land area in any given cell that corresponds to a given region/country.

            Data should be in the range [0, 1]
        Returns
        -------
        xr.DataArray
        """
        weighted_proxy = self.data * mask
        norm_weighted_proxy = weighted_proxy / weighted_proxy.sum()
        norm_weighted_proxy = norm_weighted_proxy.fillna(0)

        return norm_weighted_proxy

    @classmethod
    def load_from_proxy_file(
        cls, proxy_definition_file, species: str, sector: str, years: List[int]
    ) -> "ProxyDataset":
        """
        Load a proxy dataset from disk

        This factory method uses a proxy definition file to define the proxy that will be
        used and a backup for if that isn't available.

        Parameters
        ----------
        proxy_definition_file : str
            Path to a CSV file containing the definitions of the proxies being used
        species : str
            Species name
        sector : str
            Sector name
        years : list of int
            List of years of interest.

            A ValueError will be raised if proxy data are not available for any of the
            requested years.
        Returns
        -------
        ProxyDataset
            Proxy dataset ready for use
        """
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

        return cls(xr.concat(proxies, dim="year"))
