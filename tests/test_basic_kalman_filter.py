from types import SimpleNamespace
from physics.models.battery.kalman_filter import EKF_SOC
from physics.models.battery.battery_config import BatteryModelConfig, load_battery_config

config = SimpleNamespace(
    R_0_data=[0.002564, 0.002541, 0.002541, 0.002558, 0.002549, 0.002574, 0.002596, 0.002626, 0.002676, 0.002789],
    R_P=0.000530,
    C_P=14646,
    Q_total=259200,
    SOC_data=[0.0752, 0.1705, 0.2677, 0.366, 0.4654, 0.5666, 0.6701, 0.7767, 0.8865, 1.0],
    Uoc_data=[3.481, 3.557, 3.597, 3.623, 3.660, 3.750, 3.846, 3.946, 4.056, 4.183],
    max_current_capacity=40,
    max_energy_capacity=500
)
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


