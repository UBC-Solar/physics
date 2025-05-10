import pytest
import numpy as np
from unittest.mock import Mock
from physics.models.battery import FilteredBatteryModel, KalmanFilterConfig


@pytest.fixture
def mock_config():
    # Mock the battery model config
    battery_model_config = Mock()
    battery_model_config.Q_total = 3600.0

    battery_model_config.get_Uoc = lambda soc: 3.5 + 0.5 * soc
    battery_model_config.get_R_0 = lambda soc: 0.01 + 0.005 * soc
    battery_model_config.get_R_P = lambda soc: 0.02 + 0.002 * soc
    battery_model_config.get_C_P = lambda soc: 1000.0 + 100.0 * soc

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


def test_failure(mock_config):
    with pytest.raises(AssertionError):
        model = FilteredBatteryModel(mock_config, initial_SOC=1.5)

    with pytest.raises(AssertionError):
        model = FilteredBatteryModel(mock_config, initial_SOC=-0.2)

    with pytest.raises(AssertionError):
        model = FilteredBatteryModel(mock_config, initial_SOC=1.0, alpha=-0.1)

    with pytest.raises(AssertionError):
        model = FilteredBatteryModel(mock_config, initial_SOC=1.0, alpha=1.1)


def test_measurement_function_and_jacobian_shape(mock_config):
    model = FilteredBatteryModel(mock_config)
    x = np.array([0.8, 0.05])
    H = model._measurement_jacobian(x)
    z = model._measurement_function(x)

    assert H.shape == (1, 2)
    assert isinstance(z, float)
