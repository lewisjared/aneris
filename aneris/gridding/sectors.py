from attrs import define
from typing import Literal


SECTOR_TYPE = Literal["ceds9", "ceds16"]


@define
class Sector:
    name: str
    long_name: str
    type: SECTOR_TYPE
