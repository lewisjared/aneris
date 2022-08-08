import pytest

from aneris.gridding.proxy import ProxyDataset
from aneris.gridding.masks import MaskLoader
import os


@pytest.fixture()
def mask_loader(emissions_downscaling_archive):
    return MaskLoader(os.path.join(emissions_downscaling_archive, "gridding"))


def test_proxy_loading(mask_loader, emissions_downscaling_archive):
    ds = ProxyDataset.load_from_proxy_file(
        os.path.join(
            emissions_downscaling_archive,
            "gridding",
            "gridding-mappings",
            "proxy_mapping_CEDS9.csv",
        ),
        mask_loader,
        "CO2",
        "AIR",
        years=[2015, 2100],
    )
