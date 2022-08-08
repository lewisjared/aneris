import os.path

import pytest


# TODO: utilize this for regression or take it out completely
# def pytest_addoption(parser):
#     parser.addoption("--runslow", action="store_true",
#                      help="run slow tests")


@pytest.fixture()
def raw_data_dir() -> str:
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "raw"))


@pytest.fixture
def emissions_downscaling_archive(raw_data_dir):
    dir_name = os.path.join(raw_data_dir, "emissions_downscaling_archive")

    if not os.path.exists(dir_name):
        pytest.skip(
            "Archive from emissions_downscaling is not present. "
            "Download from https://zenodo.org/record/2538194"
        )
    return dir_name
