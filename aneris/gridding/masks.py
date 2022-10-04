import os.path
from glob import glob
from typing import List

import numpy as np
import xarray as xr

EARTH_RADIUS = 6371000.0  # m


def get_grid_centers(resolution):
    lat_centers = np.arange(90 - resolution / 2, -90, -resolution)
    lon_centers = np.arange(-180 + resolution / 2, 180 + resolution / 2, resolution)

    return lat_centers, lon_centers


def _guess_bounds(points, bound_position=0.5):
    """
    Guess bounds of grid cells.

    Simplified function from iris.coord.Coord.

    Parameters
    ----------
    points: numpy.array
        Array of grid points of shape (N,).
    bound_position: float, optional
        Bounds offset relative to the grid cell centre.
    Returns
    -------
    Array of shape (N, 2).
    """
    diffs = np.diff(points)
    diffs = np.insert(diffs, 0, diffs[0])
    diffs = np.append(diffs, diffs[-1])

    min_bounds = points - diffs[:-1] * bound_position
    max_bounds = points + diffs[1:] * (1 - bound_position)

    return np.array([min_bounds, max_bounds]).transpose()


def _quadrant_area(radian_lat_bounds, radian_lon_bounds, radius_of_earth):
    """
    Calculate spherical segment areas.
    Taken from SciTools iris library.
    Area weights are calculated for each lat/lon cell as:
        .. math::
            r^2 (lon_1 - lon_0) ( sin(lat_1) - sin(lat_0))
    The resulting array will have a shape of
    *(radian_lat_bounds.shape[0], radian_lon_bounds.shape[0])*
    The calculations are done at 64 bit precision and the returned array
    will be of type numpy.float64.
    Parameters
    ----------
    radian_lat_bounds: numpy.array
        Array of latitude bounds (radians) of shape (M, 2)
    radian_lon_bounds: numpy.array
        Array of longitude bounds (radians) of shape (N, 2)
    radius_of_earth: float
        Radius of the Earth (currently assumed spherical)
    Returns
    -------
    Array of grid cell areas of shape (M, N).
    """
    # ensure pairs of bounds
    if (
        radian_lat_bounds.shape[-1] != 2
        or radian_lon_bounds.shape[-1] != 2
        or radian_lat_bounds.ndim != 2
        or radian_lon_bounds.ndim != 2
    ):
        raise ValueError("Bounds must be [n,2] array")

    # fill in a new array of areas
    radius_sqr = radius_of_earth**2
    radian_lat_64 = radian_lat_bounds.astype(np.float64)
    radian_lon_64 = radian_lon_bounds.astype(np.float64)

    ylen = np.sin(radian_lat_64[:, 1]) - np.sin(radian_lat_64[:, 0])
    xlen = radian_lon_64[:, 1] - radian_lon_64[:, 0]
    areas = radius_sqr * np.outer(ylen, xlen)

    # we use abs because backwards bounds (min > max) give negative areas.
    return np.abs(areas)


def grid_cell_areas(lon1d, lat1d, radius=EARTH_RADIUS):
    """
    Calculate grid cell areas given 1D arrays of longitudes and latitudes
    for a planet with the given radius.

    Parameters
    ----------
    lon1d: numpy.array
        Array of longitude points [degrees] of shape (M,)
    lat1d: numpy.array
        Array of latitude points [degrees] of shape (M,)
    radius: float, optional
        Radius of the planet [metres] (currently assumed spherical)
    Returns
    -------
    Array of grid cell areas [metres**2] of shape (M, N).
    """
    lon_bounds_radian = np.deg2rad(_guess_bounds(lon1d))
    lat_bounds_radian = np.deg2rad(_guess_bounds(lat1d))
    area = _quadrant_area(lat_bounds_radian, lon_bounds_radian, radius)
    return area


def read_mask(grid_dir: str, iso_code: str) -> xr.DataArray:
    fname = f"mask_{iso_code.upper()}.nc"

    return xr.load_dataarray(os.path.join(grid_dir, "masks", fname))


class MaskLoader:
    """
    Loads and processes country masks

    Currently the country masks come from the emissions_downscaling data archive, but
    these could one day be swapped out for other grids if needed by subclassing.
    """

    def __init__(self, grid_dir):
        self.grid_dir = grid_dir

    def get_iso(self, iso_code: str) -> xr.DataArray:
        if iso_code.upper() == "WORLD":
            return read_mask(
            self.grid_dir,
            "GLOBAL",
        )

        return read_mask(
            self.grid_dir,
            iso_code,
        )

    def iso_list(self) -> List[str]:
        """
        Get the list of available ISOs

        Returns
        -------
        list of str
        """

        fnames = glob(os.path.join(self.grid_dir, "masks", "*.nc"))

        return [os.path.basename(f)[:-3].split("_")[1].upper() for f in fnames]

    def latitude_grid_size(self) -> xr.DataArray:
        lon_centers, lat_centers = get_grid_centers(0.5)

        return xr.DataArray(
            grid_cell_areas(lon_centers[:2], lat_centers)[:, 0],
            coords=(lat_centers,),
            dims=("lat",),
            attrs={"units": "km ^ 2"},
        )
