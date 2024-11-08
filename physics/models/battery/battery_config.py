import tomli
from pydantic import BaseModel
from typing import List
import os

class BatteryModelConfig(BaseModel):
    R_0_data: List[float]
    R_P: float
    C_P: float
    Q_total: float
    SOC_data: List[float]
    Uoc_data: List[float]
    max_current_capacity: float
    max_energy_capacity: float

def load_battery_config(file_name: str = "battery_config.toml") -> BatteryModelConfig:

     # Get the directory where this file is located
    base_dir = os.path.dirname(__file__)

    # Build the full path to the config file
    full_path = os.path.join(base_dir, file_name)

    with open(full_path, 'rb') as f:
        data = tomli.load(f)
    return BatteryModelConfig.model_validate(data)
