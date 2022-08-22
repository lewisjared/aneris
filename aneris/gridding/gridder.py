import xarray as xr
import os
from typing import List, Union
from aneris.gridding.proxy import ProxyDataset
from aneris.gridding.masks import MaskLoader
import scmdata
import logging
from aneris.unit_registry import unit_registry

IAMCDataset = Union["scmdata.ScmRun", "pyam.IamDataFrame"]

logger = logging.getLogger(__name__)


def add_seasonality(data: xr.DataArray) -> xr.DataArray:
    # TODO: implement seasonality

    return data


def grid_sector(
    species: str,
    iso_list: List[str],
    masker: MaskLoader,
    emissions: scmdata.ScmRun,
    proxy: ProxyDataset,
):
    global_grid_area = masker.latitude_grid_size()
    emissions_units = unit_registry.parse_units(emissions.get_unique_meta("unit", True))

    iso_sectoral_emissions = [
        grid_iso(iso, masker.get_iso(iso), emissions.filter(region=iso), proxy)
        for iso in iso_list
    ]

    # Aggregate and scale to area
    global_emissions = xr.concat(iso_sectoral_emissions, dim="region").sum(dim="region")

    # Calculate factor to go from Mt X year-1 km-2 to kg m-2 s-1
    flux_factor = (
        (emissions_units / unit_registry("km^2"))
        .to(f"kg {species} km^-2 s^-1")
        .magnitude
    )
    global_emissions = global_emissions / global_grid_area * flux_factor
    global_emissions["unit"] = f"kg {species} km^-2 s^-1"

    return add_seasonality(global_emissions)


def grid_iso(
    iso: str, mask: xr.DataArray, emissions: scmdata.ScmRun, proxy: ProxyDataset
) -> xr.DataArray:
    weighted_proxy = proxy.get_weighted(mask)

    emissions_da = xr.DataArray(
        emissions.values[0], coords=(emissions["year"],), dims=("year",)
    )

    res = emissions_da * weighted_proxy
    res["region"] = iso

    return res


class Gridder:
    """
    Grids a set of input emissions
    """

    def __init__(
        self,
        grid_dir: str,
        proxy_dir: Union[str, None] = None,
        proxy_definition_file: Union[str, None] = None,
    ):
        self.mask_loader = MaskLoader(grid_dir)

        if proxy_definition_file is None:
            proxy_definition_file = os.path.join(
                grid_dir,
                "gridding-mappings",
                "proxy_mapping_CEDS16.csv",
            )
        self.proxy_definition_file = proxy_definition_file

        if proxy_dir is None:
            proxy_dir = os.path.join(grid_dir, "proxy-CEDS16")
        self.proxy_dir = proxy_dir

    def grid_sector(self, model, scenario, variable, emissions) -> xr.DataArray:
        species, sector = self._parse_variable_name(variable)

        target_years = emissions["year"]
        regions = emissions.get_unique_meta("region")

        # todo: check region availability

        proxy_dataset = ProxyDataset.load_from_proxy_file(
            self.proxy_definition_file,
            self.proxy_dir,
            species=species,
            sector=sector,
            years=target_years,
        )

        gridded_sector = grid_sector(
            species, regions, self.mask_loader, emissions, proxy_dataset
        )
        gridded_sector["scenario"] = scenario
        gridded_sector["model"] = model
        gridded_sector["species"] = species
        gridded_sector["sector"] = sector

        return gridded_sector

    def grid(self, emissions: Union[IAMCDataset, "str"], **kwargs) -> xr.Dataset:
        """
        Attempt to grid a set of emissions

        Parameters
        ----------
        emissions : scmdata.ScmRun or pyam.IamDataFrame

        Returns
        -------
        xr.Dataset
        """
        if isinstance(emissions, str):
            emissions = scmdata.ScmRun(emissions, **kwargs)
        else:
            emissions = scmdata.ScmRun(emissions.timeseries())

        # Remove unknown regions
        unknown_regions = emissions.filter(
            region=self.mask_loader.iso_list(), keep=False
        )
        if len(unknown_regions):
            logger.warning(
                f"Dropping unknown regions: {unknown_regions.get_unique_meta('region')}"
            )
            emissions = emissions.filter(region=self.mask_loader.iso_list())

        if len(emissions) == 0:
            raise ValueError("No emissions remain to be gridded")

        result = xr.Dataset()

        for emissions_variable in emissions.groupby(["variable"]):
            grids = []
            variable = emissions_variable.get_unique_meta("variable", True)

            for emissions_sector in emissions_variable.groupby(["scenario", "model"]):
                scenario = emissions_sector.get_unique_meta("scenario", True)
                model = emissions_sector.get_unique_meta("model", True)

                logger.info(f"Gridding {model} / {scenario} / {variable}")
                res = self.grid_sector(
                    model=model,
                    scenario=scenario,
                    variable=variable,
                    emissions=emissions_sector,
                )

                grids.append(res)

            result[variable] = xr.concat(
                grids, dim="run_id", coords=["scenario", "model"]
            )

        return result

    def _parse_variable_name(self, variable: str) -> (str, str):
        toks = variable.split("|")

        return toks[-2], toks[-1]
