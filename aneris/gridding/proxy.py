import logging
import os
from typing import List

import pandas as pd
import xarray as xr

logger = logging.getLogger(__name__)


def load_proxy(proxy_dir: str, proxy_info: pd.DataFrame) -> xr.DataArray:
    if len(proxy_info) > 1:
        raise ValueError("Could not select a single proxy")
    if len(proxy_info) == 0:
        logger.error(f"No selected proxies. Falling back to population_2015")
        return read_proxy_file(
            # TODO: fix the implicit relative path in the location of the backup proxy
            os.path.join(
                proxy_dir,
                "..",
                "proxy-backup",
                "population_2015.Rd",
            )
        )

    proxy_info = proxy_info.squeeze()
    proxy = read_proxy_file(os.path.join(proxy_dir, proxy_info["proxy_file"] + ".Rd"))

    if proxy is None:
        logger.error(
            f"Could not load proxy {proxy_info['proxy_file']}. Falling back to backup"
        )

        proxy = read_proxy_file(
            # TODO: fix the implicit relative path in the location of the backup proxy
            os.path.join(
                proxy_dir,
                "..",
                "proxy-backup",
                proxy_info["proxybackup_file"] + ".Rd",
            )
        )
    if proxy is None:
        logger.error(
            f"Could not load proxy {proxy_info['proxybackup_file']}. Falling back to population_2015"
        )

        proxy = read_proxy_file(
            # TODO: fix the implicit relative path in the location of the backup proxy
            os.path.join(
                proxy_dir,
                "..",
                "proxy-backup",
                "population_2015.Rd",
            )
        )
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

        sector_mapping = pd.read_csv(
            proxy_definition_file.replace("proxy_mapping", "gridding_sector")
        )

        # Hack: replace the short names with the full sector names
        # The CEDS9 file uses short names and the CEDS16 file uses long names
        for mapping in sector_mapping.to_dict("records"):
            proxy_definitions["sector"] = proxy_definitions["sector"].str.replace(
                mapping["sector_short"], mapping["sector_name"]
            )

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
