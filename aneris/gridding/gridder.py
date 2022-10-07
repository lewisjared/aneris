import logging
import os
from typing import List, Union

import pint
import scmdata
import xarray as xr
from cftime import DatetimeNoLeap
import numpy as np

from aneris.gridding.masks import MaskStore
from aneris.gridding.proxy import ProxyDataset, SeasonalityStore
from aneris.gridding.sectors import SECTOR_TYPE
from aneris.unit_registry import ur

IAMCDataset = Union["scmdata.ScmRun", "pyam.IamDataFrame"]

logger = logging.getLogger(__name__)

# Days per month in a 365-day calendar
DAYS_IN_MONTH = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
MID_DAY_OF_MONTH = [16, 15, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16]


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


def convert_to_monthly(years, date_type=DatetimeNoLeap):
    vals = []

    for y in years:
        vals.extend(
            [date_type(y, month, MID_DAY_OF_MONTH[month - 1]) for month in range(1, 13)]
        )

    return xr.IndexVariable(
        "time",
        vals,
        attrs={
            "long_name": "time",
            "standard_name": "time",
            "axis": "T",
        },
        encoding={"units": f"days since {years[0]}-01-01 00:00:00"},
    )


def add_seasonality(
    seasonality_store: SeasonalityStore, data: xr.DataArray, species, sector
) -> xr.DataArray:

    if not seasonality_store:
        logger.info("Skipping adding seasonality")
        return data

    # Loop over years and apply the seasonality mapping
    num_years = data.shape[0]
    new_shape = (num_years * 12, *data.shape[1:])

    time = convert_to_monthly(data.year)

    monthly_data = xr.DataArray(
        np.zeros(new_shape),
        dims=("time", *data.dims[1:]),
        coords={
            "time": time,
            **{dim_name: data.coords[dim_name] for dim_name in data.dims[1:]},
        },
        attrs=data.attrs,
    )

    for i, year in enumerate(data.year.values):
        seas_data = seasonality_store.load(species, sector, 2015)

        if seas_data is None:
            raise ValueError("Could not find a seasonality map")

        # Adjust seasonality for 365-day calendar
        # I don't understand why this is needed
        seas_adj = 365 / (seas_data * DAYS_IN_MONTH * 12).sum("month")

        # Data is in kg/m^2/s
        data_seasonal = data[i] * seas_data * seas_adj * 12
        monthly_data[i * 12 : (i + 1) * 12] = data_seasonal.transpose(
            "month", *data.dims[1:]
        )

    return monthly_data


def grid_sector(
    species: str,
    sector: str,
    iso_list: List[str],
    mask_store: MaskStore,
    seasonality_store: SeasonalityStore,
    emissions: scmdata.ScmRun,
    proxy: ProxyDataset,
) -> xr.DataArray:
    global_grid_area = mask_store.latitude_grid_size()
    emissions_units: pint.Unit = ur(emissions.get_unique_meta("unit", True))

    iso_sectoral_emissions = [
        grid_iso(iso, mask_store.get_iso(iso), emissions.filter(region=iso), proxy)
        for iso in iso_list
    ]

    # Aggregate and scale to area
    global_emissions = xr.concat(iso_sectoral_emissions, dim="region").sum(dim="region")
    # Reformat data to globe
    global_emissions, _ = xr.align(
        global_emissions, mask_store.get_iso("World"), join="right", fill_value=0
    )

    # Calculate factor to go from Mt X year-1 km-2 to kg m-2 s-1
    flux_factor = convert_to_target_unit(
        (emissions_units / ur("km^2")), f"kg m^-2 s^-1"
    )
    global_emissions = global_emissions / global_grid_area * flux_factor.m
    global_emissions.attrs["units"] = "kg m^-2 s^-1"

    global_emissions.lat.attrs.update(
        {
            "units": "degrees_north",
            "long_name": "latitude",
            "axis": "Y",
            "standard_name": "latitude",
            "topology": "linear",
        }
    )

    global_emissions.lon.attrs.update(
        {
            "units": "degrees_east",
            "long_name": "longitude",
            "axis": "X",
            "modulo": "360",
            "standard_name": "longitude",
            "topology": "circular",
        }
    )

    return add_seasonality(seasonality_store, global_emissions, species, sector)


def grid_iso(
    iso: str, mask: xr.DataArray, emissions: scmdata.ScmRun, proxy: ProxyDataset
) -> xr.DataArray:
    weighted_proxy = proxy.get_weighted(mask)

    emissions_da = xr.DataArray(
        emissions.values[0], coords=(emissions["year"],), dims=("year",)
    )

    res = emissions_da * weighted_proxy
    res.attrs["region"] = iso

    return res


def get_matching_regions(
    available_regions: List[str], allowed_regions: List[str]
) -> List[str]:
    missing_regions = set(allowed_regions) - set(available_regions)
    if missing_regions:
        logger.warning(f"Missing {missing_regions} regions from gridding")
        available_regions = list(set(available_regions) - missing_regions)

    extra_regions = set(available_regions) - set(allowed_regions)
    if extra_regions:
        logger.warning(f"Additional regions {extra_regions} will be ignored")
        available_regions = list(set(available_regions) - extra_regions)

    return available_regions


class GriddedResults:
    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        self._results = []

    def __repr__(self):
        return f"<GriddedResults {self._results}>"

    def _data_filename(
        self,
        model: str,
        scenario: str,
        variable: str,
    ):
        return os.path.join(
            self.output_dir, f'{variable.replace("|", "_")}_{model}_{scenario}.nc'
        )

    def load(
        self,
        model: str,
        scenario: str,
        variable: str,
    ) -> xr.DataArray:
        return xr.load_dataarray(self._data_filename(variable, model, scenario))

    def save(
        self,
        data: xr.DataArray,
        model: str,
        scenario: str,
        variable: str,
    ):
        output_filename = self._data_filename(model, scenario, variable)
        data.name = variable
        data.to_dataset().to_netcdf(
            output_filename, encoding={variable: {"zlib": True, "complevel": 5}}
        )


class Gridder:
    """
    Grids a set of input emissions
    """

    def __init__(
        self,
        grid_dir: str,
        proxy_definition_file: Union[str, None] = None,
        seasonality_mapping_file: Union[str, None] = None,
        sector_type: SECTOR_TYPE = "CEDS9",
        global_sectors=("Aircraft", "International Shipping"),
    ):
        self.grid_dir = grid_dir
        self.mask_store = MaskStore(grid_dir)

        self.global_sectors = global_sectors
        self.sector_type = sector_type

        if proxy_definition_file is None:
            proxy_definition_file = os.path.join(
                grid_dir,
                "gridding-mappings",
                f"proxy_mapping_{sector_type}.csv",
            )
        self.proxy_definition_file = proxy_definition_file

        if seasonality_mapping_file:
            self.seasonality_store = SeasonalityStore.load_from_csv(
                self.grid_dir, seasonality_mapping_file
            )
        else:
            self.seasonality_store = None

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
        species, sector_name = self._parse_variable_name(variable)

        target_years = emissions["year"]

        if sector_name in self.global_sectors:
            regions = ["World"]
        else:
            regions = self.mask_store.iso_list()

        # Check region availability
        available_regions = get_matching_regions(
            emissions.get_unique_meta("region"), regions
        )

        if not available_regions:
            raise ValueError("No regions available for regridding")

        proxy_dataset = ProxyDataset.load_from_proxy_file(
            self.proxy_definition_file,
            os.path.join(self.grid_dir, "proxies"),
            species=species,
            sector=sector_name,
            sector_type=self.sector_type,
            years=target_years,
        )

        gridded_sector = grid_sector(
            species,
            sector_name,
            available_regions,
            self.mask_store,
            self.seasonality_store,
            emissions,
            proxy_dataset,
        )
        gridded_sector.attrs["scenario"] = scenario
        gridded_sector.attrs["model"] = model
        gridded_sector.attrs["species"] = species
        gridded_sector.attrs["sector"] = sector_name

        return gridded_sector

    def grid(
        self, output_dir: str, emissions: Union[IAMCDataset, "str"], **kwargs
    ) -> GriddedResults:
        """
        Attempt to grid a set of emissions

        Parameters
        ----------
        output_dir : str
            Directory to store the gridded outputs

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

        results = GriddedResults(output_dir)

        for emissions_scenario in emissions.groupby(["scenario", "model"]):
            self.grid_scenario(emissions_scenario, results)

        return results

    def grid_scenario(self, emissions: scmdata.ScmRun, results=None, output_dir=None):
        if results is None:
            results = GriddedResults(output_dir or ".")

        scenario = emissions.get_unique_meta("scenario", True)
        model = emissions.get_unique_meta("model", True)

        for emissions_variable in emissions.groupby(["variable"]):
            variable = emissions_variable.get_unique_meta("variable", True)

            logger.info(f"Gridding {model} / {scenario} / {variable}")
            res = self.grid_sector(
                model=model,
                scenario=scenario,
                variable=variable,
                emissions=emissions_variable,
            )

            results.save(res, model, scenario, variable)
        return results

    def _parse_variable_name(self, variable: str) -> (str, str):
        toks = variable.split("|")

        return toks[-2], toks[-1]
