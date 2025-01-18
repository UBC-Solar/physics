from physics.models.battery.kalman_filter import EKF_SOC
from physics.models.battery.battery_config import BatteryModelConfig, load_battery_config
import pathlib

config_path = pathlib.Path(__file__).parent.parent / "battery_config.toml"
config: BatteryModelConfig = load_battery_config(config_path.absolute())

Kalman_Filter = EKF_SOC(config, 1.0, 0.0)

def test_SOC_Value():
    SOC = Kalman_Filter.get_SOC()
    assert type(SOC) == float
    assert SOC <= 1.0
    assert SOC >= 0.10

def test_Uc_Value():
    Uc = Kalman_Filter.get_Uc()
    assert Uc >= 0
    assert type(Uc) == float

def test_update_filter_invalid_arguments():
    # Test invalid current value (out of range)
    try:
        Kalman_Filter.update_filter(3.5, 50.0)
    except ValueError as e:
        assert "Invalid value for current" in str(e)

    # Test invalid current type (not a float)
    try:
        Kalman_Filter.update_filter(3.5, 30)
    except TypeError as e:
        assert "Invalid type for current I" in str(e)

    # Test invalid terminal voltage value (out of range)
    try:
        Kalman_Filter.update_filter(6.0, 10.0)
    except ValueError as e:
        assert "Invalid value for terminal voltage" in str(e)

    # Test invalid terminal voltage type (not a float)
    try:
        Kalman_Filter.update_filter("3.7", 10.0)
    except TypeError as e:
        assert "Invalid type for measured_Ut" in str(e)


