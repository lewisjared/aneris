import logging
import os
from typing import List, Optional
from attrs import define

import pandas as pd
import xarray as xr

logger = logging.getLogger(__name__)


def read_proxy_file(fname: str) -> Optional[xr.DataArray]:
    if not os.path.exists(fname):
        return None

    return xr.load_dataarray(fname)


@define
class ProxyInfo:
    sector: str
    sector_type: str
    proxy_file: str


def load_proxy(proxy_dir: str, proxy_info: pd.DataFrame) -> xr.DataArray:
    if len(proxy_info) > 1:
        raise ValueError("Could not select a single proxy")

    fallback_proxy = os.path.join(
        proxy_dir,
        "proxy-backup",
        "population_2015.nc",
    )
    if len(proxy_info) == 0:
        logger.error(f"No selected proxies. Falling back to population_2015")
        proxy = read_proxy_file(fallback_proxy)
        if proxy is None:
            raise ValueError(f"Could not load {fallback_proxy}")
        return proxy

    proxy_info = proxy_info.squeeze()
    proxy = read_proxy_file(
        os.path.join(
            proxy_dir,
            f"proxy-{ proxy_info['sector_type']}",
            proxy_info["proxy_file"] + ".nc",
        )
    )

    if proxy is None:
        logger.error(
            f"Could not load proxy {proxy_info['proxy_file']}. Falling back to backup"
        )

        proxy = read_proxy_file(
            os.path.join(
                proxy_dir,
                "proxy-backup",
                proxy_info["proxybackup_file"] + ".nc",
            )
        )
    if proxy is None:
        logger.error(
            f"Could not load proxy {proxy_info['proxybackup_file']}. Falling back to population_2015"
        )

        proxy = read_proxy_file(fallback_proxy)
    if proxy is None:
        raise ValueError("Could not find any appropriate proxies")
    return proxy


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
        norm_weighted_proxy = weighted_proxy / weighted_proxy.sum(dim=("lat", "lon"))
        norm_weighted_proxy = norm_weighted_proxy.fillna(0)

        return norm_weighted_proxy

    @classmethod
    def load_from_proxy_file(
        cls,
        proxy_definition_file: str,
        proxy_dir: str,
        species: str,
        sector: str,
        sector_type: str,
        years: List[int],
    ) -> "ProxyDataset":
        """
        Load a proxy dataset from disk

        This factory method uses a proxy definition file to define the proxy that will be
        used and a backup for if that isn't available.

        Parameters
        ----------
        proxy_definition_file : str
            Path to a CSV file containing the definitions of the proxies being used
        proxy_dir : str
            Directory containing the proxy data files
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
        proxy_definitions["sector_type"] = sector_type

        selected_proxies = proxy_definitions[
            (proxy_definitions.em == species) & (proxy_definitions.sector == sector)
        ]
        if not len(selected_proxies):
            logger.warning(f"Could not find proxy definition for {species}/{sector}")

        proxies = []

        for y in years:
            proxy = load_proxy(proxy_dir, selected_proxies[selected_proxies.year == y])
            proxy["year"] = y

            proxies.append(proxy)

        return cls(xr.concat(proxies, dim="year"))


class SeasonalityStore:
    def __init__(self, grid_dir: str, mapping: pd.DataFrame):
        self.grid_dir = grid_dir
        self.mapping = mapping

    def get(self, species: str, sector: str, year: int, allow_close=False):
        match = self._find(species, sector, year, allow_close=allow_close)

        return os.path.join(self.grid_dir, "seasonality", match + ".nc")

    def load(self, species: str, sector: str, year: int, allow_close=False):
        # TODO: cache result
        return read_proxy_file(self.get(species, sector, year, allow_close=allow_close))

    def _find(self, species, sector, year, allow_close=False):
        matching = self.mapping[
            (self.mapping.em == species) & (self.mapping.sector == sector)
        ]
        if not len(matching):
            raise ValueError(f"Could not find a match for {species}/{sector}")

        exact = matching[matching.year == year]
        if len(exact) == 1:
            return matching.seasonality_file.squeeze()
        elif allow_close:
            #
            return matching.seasonality_file.iloc[0]
        else:
            raise ValueError(f"Could not find a match for {species}/{sector}/{year}")

    @classmethod
    def load_from_csv(cls, grid_dir, fname):
        mapping = pd.read_csv(fname)

        return cls(grid_dir, mapping)
