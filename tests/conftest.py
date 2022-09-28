import os.path

import pytest


# TODO: utilize this for regression or take it out completely
# def pytest_addoption(parser):
#     parser.addoption("--runslow", action="store_true",
#                      help="run slow tests")


@pytest.fixture()
def test_data_dir() -> str:
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "test_data"))


@pytest.fixture()
def raw_data_dir() -> str:
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "raw"))


@pytest.fixture()
def processed_data_dir() -> str:
    return os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "data", "processed")
    )


@pytest.fixture
def emissions_downscaling_archive(raw_data_dir) -> str:
    dir_name = os.path.join(raw_data_dir, "emissions_downscaling_archive")

    if not os.path.exists(dir_name):
        pytest.skip(
            "Archive from emissions_downscaling is not present. "
            "Download from https://zenodo.org/record/2538194"
        )
    return dir_name


@pytest.fixture
def grid_dir(processed_data_dir, emissions_downscaling_archive) -> str:
    dir_name = os.path.join(processed_data_dir, "gridding")

    if not os.path.exists(dir_name):
        pytest.skip(
            "Need to preprocess the input data. See notebooks/gridding/000_prepare_intput_data.py"
        )
    return dir_name
