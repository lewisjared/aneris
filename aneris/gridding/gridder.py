import xarray as xr

from typing import List, Union
from aneris.gridding.proxy import ProxyDataset
from aneris.gridding.masks import MaskLoader
import scmdata
import logging

IAMCDataset = Union["scmdata.ScmRun", "pyam.IamDataFrame"]

logger = logging.getLogger(__name__)


def grid_sector(
    iso_list: List[str],
    masker: MaskLoader,
    emissions: scmdata.ScmRun,
    proxy: ProxyDataset,
):
    global_grid_area = 100
    flux_factor = (
        1000000000 / global_grid_area / (365 * 24 * 60 * 60)
    )  # from Mt to kg m-2 s-1

    iso_sectoral_emissions = [
        grid_iso(masker.get_iso(iso), emissions.filter(region=iso), proxy)
        for iso in iso_list
    ]

    global_emissions = aggregate(iso_sectoral_emissions)

    return add_seasonality(global_emissions)


def grid_iso(mask: xr.DataArray, emissions: scmdata.ScmRun, proxy: ProxyDataset):
    weighted_proxy = proxy.get_weighted(mask)

    return emissions.values * weighted_proxy


class Gridder:
    """
    Grids a set of input emissions
    """

    def __init__(self, grid_dir):
        self.mask_loader = MaskLoader(grid_dir)

    def grid(self, emissions: IAMCDataset) -> xr.Dataset:
        """
        Attempt to grid a set of emissions

        Parameters
        ----------
        emissions : scmdata.ScmRun or pyam.IamDataFrame

        Returns
        -------
        xr.Dataset
        """
        emissions = scmdata.ScmRun(emissions.timeseries())

        # Remove unknown regions
        unknown_regions = emissions.filter(region=self.mask_loader.iso_list())
        if len(unknown_regions):
            logger.warning(
                f"Dropping unknown regions: {unknown_regions.get_unique_meta('region')}"
            )
            emissions = emissions.filter(region=self.mask_loader.iso_list(), keep=False)

        for emissions_sector in emissions.groupby(["scenario", "model", "variable"]):
            scenario = emissions_sector.get_unique_meta("scenario", True)
            model = emissions_sector.get_unique_meta("model", True)
            variable = emissions_sector.get_unique_meta("variable", True)
            logger.info(f"Gridding {model} / {scenario} / {variable}")
            species, sector = self._parse_variable_name(variable)

            target_years = emissions_sector["years"]
            regions = emissions_sector.get_unique_meta("region")

            # todo: check region availability

            proxy = ProxyDataset.load_from_proxy_file(
                "", species=species, sector=sector, years=target_years
            )

            res = grid_sector(regions, self.mask_loader, emissions, proxy)

    def _parse_variable_name(self, variable: str) -> (str, str):
        # TODO: implement
        return "CO2", "AIR"
