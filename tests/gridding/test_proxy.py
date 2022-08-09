import pytest

from aneris.gridding.proxy import ProxyDataset
import os


def test_proxy_loading_aircraft(emissions_downscaling_archive):
    proxy_dir = os.path.join(emissions_downscaling_archive, "gridding", "proxy-CEDS9")

    ds = ProxyDataset.load_from_proxy_file(
        os.path.join(
            emissions_downscaling_archive,
            "gridding",
            "gridding-mappings",
            "proxy_mapping_CEDS9.csv",
        ),
        proxy_dir,
        "CO2",
        "AIR",
        years=[2015, 2100],
    )

    assert isinstance(ds, ProxyDataset)
    assert ds.data.shape == (2, 360, 720, 25)


def test_proxy_loading(emissions_downscaling_archive):
    proxy_dir = os.path.join(emissions_downscaling_archive, "gridding", "proxy-CEDS9")

    ds = ProxyDataset.load_from_proxy_file(
        os.path.join(
            emissions_downscaling_archive,
            "gridding",
            "gridding-mappings",
            "proxy_mapping_CEDS9.csv",
        ),
        proxy_dir,
        "CO2",
        "RCO",
        years=[2015, 2100],
    )

    assert isinstance(ds, ProxyDataset)
    assert ds.data.shape == (2, 360, 720)
