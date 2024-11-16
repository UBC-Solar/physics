import tomli
from pydantic import BaseModel
from typing import List
import os
import pathlib

class BatteryModelConfig(BaseModel):
    R_0_data: List[float]
    R_P: float
    C_P: float
    Q_total: float
    SOC_data: List[float]
    Uoc_data: List[float]
    max_current_capacity: float
    max_energy_capacity: float

def load_battery_config(absolute_path: str) -> BatteryModelConfig:
    # Build the full path to the config file
    full_path = pathlib.Path(absolute_path)
    with open(full_path, 'rb') as f:
        data = tomli.load(f)
    return BatteryModelConfig.model_validate(data)
