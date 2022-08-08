import xarray as xr

from typing import List, Union
from aneris.gridding.proxy import ProxyDataset
from aneris.gridding.masks import MaskLoader
import scmdata
import logging

IAMCDataset = Union["scmdata.ScmRun", "pyam.IamDataFrame"]

logger = logging.getLogger(__name__)


def grid_sector(iso_list: List[str], emissions: scmdata.ScmRun, proxy: ProxyDataset):
    global_grid_area = 100
    flux_factor = (
        1000000000 / global_grid_area / (365 * 24 * 60 * 60)
    )  # from Mt to kg m-2 s-1

    years = emissions["year"]

    iso_sectoral_emissions = [
        grid_iso(iso, emissions.filter(region=iso), proxy) for iso in iso_list
    ]

    global_emissions = aggregate(iso_sectoral_emissions)

    return add_seasonality(global_emissions)


def grid_iso(iso: str, emissions: scmdata.ScmRun, proxy: ProxyDataset):
    weighted_proxy = proxy.get(iso)
    norm_weighted_proxy = weighted_proxy / weighted_proxy.sum()
    norm_weighted_proxy = norm_weighted_proxy.fillna(0)

    return emissions.get()


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
            scenario = emissions_sector.get_unique_meta("scenario")
            model = emissions_sector.get_unique_meta("model")
            variable = emissions_sector.get_unique_meta("variable")
            logger.info(f"Gridding {model} / {scenario} / {variable}")
            species, sector = self._parse_variable_name(variable)

            target_years = emissions_sector["years"]
            regions = emissions_sector.get_unique_meta("region")

            # todo: check region availability

            proxy = ProxyDataset.load_from_proxy_file(
                self.mask_loader, species=species, sector=sector, years=target_years
            )

            res = grid_sector(regions, emissions, proxy)

    def _parse_variable_name(self, variable: str) -> (str, str):
        return "CO2", "AIR"
