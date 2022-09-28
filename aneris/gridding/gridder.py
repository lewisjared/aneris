import logging
import os
from typing import List, Union

import pint
import scmdata
import xarray as xr

from aneris.gridding.masks import MaskLoader
from aneris.gridding.proxy import ProxyDataset
from aneris.unit_registry import ur

IAMCDataset = Union["scmdata.ScmRun", "pyam.IamDataFrame"]

logger = logging.getLogger(__name__)


def convert_to_target_unit(
    initial_unit: Union[str, pint.Unit], target_unit: Union[str, pint.Unit]
) -> pint.Quantity:
    """
    Calculate the scale factor required to convert between units

    This function supports converting a subset of the input units dimensions which
    is helpful in situations where arbitary dimensions can be provided i.e. Mt X/yr
    where X could be a range of species.

    >>> value = ur.Quantity(12, "Mt CH4/yr")
    >>> scale_factor = convert_to_target_unit(value.units, target_unit="kg")
    >>> ur.Quantity(value.m * scale_factor.m, scale_factor.u)

    Parameters
    ----------
    initial_unit
        Units of input
    target_unit
        The expected output

        Any dimensions present in the initial_unit, but not in the target unit will
        be kept the same.

    Returns
    -------
    pint.Quantity
        The magnitude of the quantity represents the required scale factor
        The units of the quantity represent the resulting unit
    """
    start = ur.Quantity(1, initial_unit)

    # Get pint to find the conversion factor for you
    start_mass_in_kg = (start / ur.Quantity(1, target_unit)).to_reduced_units()

    # Put the intended mass units back in
    start_mass_in_kg_correct_units = start_mass_in_kg * ur.Quantity(1, target_unit)
    return start_mass_in_kg_correct_units


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
    emissions_units: pint.Unit = ur(emissions.get_unique_meta("unit", True))

    iso_sectoral_emissions = [
        grid_iso(iso, masker.get_iso(iso), emissions.filter(region=iso), proxy)
        for iso in iso_list
    ]

    # Aggregate and scale to area
    global_emissions = xr.concat(iso_sectoral_emissions, dim="region").sum(dim="region")

    # Calculate factor to go from Mt X year-1 km-2 to kg m-2 s-1
    flux_factor = convert_to_target_unit(
        (emissions_units / ur("km^2")), f"kg m^-2 s^-1"
    )
    global_emissions = global_emissions / global_grid_area * flux_factor.m
    global_emissions["unit"] = str(flux_factor.u)

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
        proxy_definition_file: Union[str, None] = None,
        sectoral_map="CEDS16",
        global_sectors=("Aircraft", "International Shipping"),
    ):
        self.grid_dir = grid_dir
        self.mask_loader = MaskLoader(grid_dir)
        self.global_sectors = global_sectors

        if proxy_definition_file is None:
            proxy_definition_file = os.path.join(
                grid_dir,
                "gridding-mappings",
                f"proxy_mapping_{sectoral_map}.csv",
            )
        self.proxy_definition_file = proxy_definition_file

    def grid_sector(
        self,
        model: str,
        scenario: str,
        variable: str,
        emissions: scmdata.ScmRun,
    ) -> xr.DataArray:
        """
        Grid an individual sector

        Parameters
        ----------
        model : str
        scenario : str
        variable : str
            Used to define the gas and sector.
            Should match the form: "*|{variable}|{sector}"
        emissions : scmdata.ScmRun
            Emissions timeseries for a single model, scenario and sector

            Should contain the timeseries for the regions of interest

        Returns
        -------
        xr.DataArray
        """
        species, sector = self._parse_variable_name(variable)

        target_years = emissions["year"]

        if sector in self.global_sectors:
            regions = ["World"]
        else:
            regions = self.mask_loader.iso_list()

        # Check region availability
        available_regions = emissions.get_unique_meta("region")
        missing_regions = set(regions) - set(available_regions)
        if missing_regions:
            logger.warning(
                f"Missing {missing_regions} regions from gridding {species} / {sector}"
            )
            available_regions = list(set(available_regions) - missing_regions)

        proxy_dataset = ProxyDataset.load_from_proxy_file(
            self.proxy_definition_file,
            os.path.join(self.grid_dir, "proxies"),
            species=species,
            sector=sector,
            years=target_years,
        )

        gridded_sector = grid_sector(
            species, available_regions, self.mask_loader, emissions, proxy_dataset
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
        emissions : scmdata.ScmRun or pyam.IamDataFrame or str

            If a string is provided, the emissions input will be loaded from disk.

        Returns
        -------
        xr.Dataset
        """
        if isinstance(emissions, str):
            emissions = scmdata.ScmRun(emissions, **kwargs)
        else:
            emissions = scmdata.ScmRun(emissions.timeseries())

        # Remove unknown regions
        expected_regions = self.mask_loader.iso_list() + ["World"]
        unknown_regions = emissions.filter(region=expected_regions, keep=False)
        if len(unknown_regions):
            logger.warning(
                f"Dropping unknown regions: {unknown_regions.get_unique_meta('region')}"
            )
            emissions = emissions.filter(region=expected_regions)

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
