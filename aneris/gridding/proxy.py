import logging
import os
from typing import List, Optional, Any
from attrs import define

import pandas as pd
import xarray as xr

from aneris.gridding.sectors import SECTOR_TYPE

logger = logging.getLogger(__name__)


def read_proxy_file(fname: str) -> Optional[xr.DataArray]:
    if not os.path.exists(fname):
        return None

    return xr.load_dataarray(fname)


@define
class ProxyInfo:
    em: str
    sector: str
    year: int
    sector_type: str
    proxy_file: str


@define
class Store:
    """
    Finds an appropriate proxy for a given species, sector and year
    """

    grid_dir: str
    mapping: pd.DataFrame
    allow_close: bool = False
    file_column: str = "seasonality_file"

    @classmethod
    def load_from_csv(cls, grid_dir: str, fname: str, **kwargs: Any):
        mapping = pd.read_csv(fname)

        return cls(grid_dir, mapping, **kwargs)

    def _find(self, species, sector, year) -> str:
        matching = self.mapping[
            (self.mapping.em == species) & (self.mapping.sector == sector)
        ]
        if not len(matching):
            raise ValueError(f"Could not find a match for {species}/{sector}")

        exact = matching[matching.year == year]
        if len(exact) == 1:
            return matching[self.file_column].iloc[0]
        elif self.allow_close:
            closest_idx = (year - matching.year).abs().argmin()
            logger.info(
                f"Using close match for {species}/{sector}/{year} (year={matching.year.iloc[closest_idx]})"
            )
            return matching[self.file_column].iloc[closest_idx]
        else:
            raise ValueError(f"Could not find a match for {species}/{sector}/{year}")


@define
class ProxyDataset:
    """
    A proxy dataset which is used to scale country data into gridded data
    """

    data: xr.DataArray

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


@define
class ProxyStore(Store):
    sector_type: SECTOR_TYPE = "CEDS9"
    file_column: str = "proxy_file"

    def get(self, species: str, sector: str, year: int) -> str:
        try:
            match = self._find(species, sector, year)
        except ValueError:
            match = None

        if match is not None:
            out_fname = os.path.join(
                self.grid_dir,
                "proxies",
                f"proxy-{self.sector_type}",
                match + ".nc",
            )

            if os.path.exists(out_fname):
                return out_fname
            else:
                logger.error(f"Could not load {out_fname}")

        logger.error(f"No selected proxies. Falling back to population_2015")
        return os.path.join(
            self.grid_dir,
            "proxies",
            "proxy-backup",
            "population_2015.nc",
        )

    def load_year(self, species: str, sector: str, year: int) -> xr.DataArray:
        fname = self.get(species, sector, year)
        proxy = read_proxy_file(fname)
        if proxy is None:
            raise ValueError(f"Could not load {fname}. Invalid format?")
        return proxy

    def load(self, species: str, sector: str, years: List[int]) -> ProxyDataset:
        proxies = []

        for year in years:
            proxy = self.load_year(species, sector, year)
            if proxy is None:
                raise ValueError(
                    f"Could not find an appropriate proxy {species}/{sector}/{year}"
                )

            proxy["year"] = year

            proxies.append(proxy)

        return ProxyDataset(xr.concat(proxies, dim="year"))


class SeasonalityStore(Store):
    def get(self, species: str, sector: str, year: int) -> str:
        """
        Get the filename of the proxy to load

        Parameters
        ----------
        species
        sector
        year

        Returns
        -------
        A filename for a selected proxy file
        """
        match = self._find(species, sector, year)

        return os.path.join(self.grid_dir, "seasonality", match + ".nc")

    def load(self, species: str, sector: str, year: int):
        # TODO: cache result
        return read_proxy_file(self.get(species, sector, year))
