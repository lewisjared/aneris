from aneris.gridding.proxy import ProxyDataset
import os


def test_proxy_loading_aircraft(grid_dir):
    proxy_dir = os.path.join(grid_dir, "proxy-CEDS9")

    ds = ProxyDataset.load_from_proxy_file(
        os.path.join(
            grid_dir,
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


def test_proxy_loading(grid_dir):
    proxy_dir = os.path.join(grid_dir, "proxy-CEDS9")

    ds = ProxyDataset.load_from_proxy_file(
        os.path.join(
            grid_dir,
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
