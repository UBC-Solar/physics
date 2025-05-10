import pytest
import numpy as np
from unittest.mock import Mock
from physics.models.battery import FilteredBatteryModel, KalmanFilterConfig

@pytest.fixture
def mock_config():
    # Mock the battery model config
    battery_model_config = Mock()
    battery_model_config.Q_total = 3600.0
    R_0_data = [0.17953765302439662, 0.15580951404728172, 0.14176929930784543, 0.11043950958574644, 0.13930042505446938,
                0.1552885289394773, 0.044070982259896085, 0.2208806896239539, 0.15116267852908616, 0.6553961767519164]
    R_P_data = [0.04153180244191346, 0.10674683402208612, 0.061085424180509884, 0.0781407642082238, 0.05537901113775878,
                0.09732054673529467, 0.07662520885708152, 0.09799857401036915, 0.42622740149661487, 0.2718418915736874]
    C_P_data = [14824.398495212006, 1587.5971318119796, 341.1064063616048, 1243.182413110655, 619.5791066439332,
                2252.7885790042164, 954.5884882581622, 515.7219779825028, 431.10892633451135, 195.14394897766627]
    Uoc_data = [131.88002282453857, 129.4574321366064, 125.5750277614186, 121.99586066440303, 118.69893412178982,
                115.71854177322408, 111.99025635444923, 108.29354777060836, 98.23397960300946, 95.24125831782388]
    Q_total = 151000.0
    Soc_data = [1.0000113624123392, 0.8815263722745977, 0.7671918526292492, 0.6206071038045673, 0.4911613638651783,
                0.3606311083423134, 0.23687514228021178, 0.12073345089992571, 0.01456057818183809,
                0.0070648691224265425]



    battery_model_config.get_Uoc = lambda: lambda soc: 3.5 + 0.5 * soc
    battery_model_config.get_R_0 = lambda: lambda soc: 0.01 + 0.005 * soc
    battery_model_config.get_R_P = lambda: lambda soc: 0.02 + 0.002 * soc
    battery_model_config.get_C_P = lambda: lambda soc: 1000.0 + 100.0 * soc

    # Mock the full filtered battery model config
    config = Mock()
    config.battery_model_config = battery_model_config
    config.state_covariance_matrix = np.eye(2) * 0.01
    config.process_noise_matrix = np.eye(2) * 1e-6
    config.measurement_noise_vector = np.array([[0.001]])

    return config


def test_initialization(mock_config):
    model = FilteredBatteryModel(mock_config, initial_SOC=0.9, initial_Uc=0.1)
    assert np.isclose(model.SOC, 0.9)
    assert np.isclose(model.Uc, 0.1)
    assert model.Ut == 0


def test_predict_then_update_changes_state(mock_config):
    model = FilteredBatteryModel(mock_config)
    SOC_before = model.SOC
    Uc_before = model.Uc

    # Run a prediction and update step
    model.predict_then_update(measured_Ut=3.7, current=2.0, time_step=1.0)

    SOC_after = model.SOC
    Uc_after = model.Uc

    # Ensure state is updated
    assert not np.isclose(SOC_before, SOC_after)
    assert not np.isclose(Uc_before, Uc_after)


def test_soc_clipping(mock_config):
    model = FilteredBatteryModel(mock_config, initial_SOC=1.5)
    # SOC should be clipped to 1.1 in constructor
    assert model.SOC <= 1.1

    # Force EKF state to an invalid SOC, then run update to ensure clipping
    model._ekf.x[0] = 1.5
    model.update_filter(measured_Ut=3.7, current=5.0)
    assert 0.0 <= model.SOC <= 1.1


def test_measurement_function_and_jacobian_shape(mock_config):
    model = FilteredBatteryModel(mock_config)
    x = np.array([0.8, 0.05])
    H = model._measurement_jacobian(x)
    z = model._measurement_function(x)

    assert H.shape == (1, 2)
    assert isinstance(z, float)
