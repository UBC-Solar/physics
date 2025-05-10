import tomli as tomllib
import pathlib
from scipy import optimize
import numpy as np
from physics.models.battery import SOCDependent
from typing import cast
from numpy.typing import NDArray


class BatteryModelConfig:
    """
    A concrete implementation of the `EquivalentCircuitModelConfig` protocol.

    This implementation fits values of U_oc, R_0, R_P, and C_P at various state-of-charge (SOC) values
    to a seventh degree polynomial to generate a smooth function mapping SOC to each battery parameter.

    For example, R_0 = R_0_data[i] when Soc = Soc_data[i].
    """
    def __init__(self, R_0_data, Soc_data, R_P_data, C_P_data, Uoc_data, Q_total):
        # ----- Initialize Parameters -----
        def quintic_polynomial(x, x0, x1, x2, x3, x4, x5, x6, x7):
            return np.polyval(np.array([x0, x1, x2, x3, x4, x5, x6, x7]), x)

        self._U_oc_coefficients, _ = optimize.curve_fit(quintic_polynomial, Soc_data, Uoc_data)
        self._R_0_coefficients, _ = optimize.curve_fit(quintic_polynomial, Soc_data, R_0_data)
        self._C_P_coefficients, _ = optimize.curve_fit(quintic_polynomial, Soc_data, C_P_data)
        self._R_P_coefficients, _ = optimize.curve_fit(quintic_polynomial, Soc_data, R_P_data)

        # Casts are just for the type-checker to know that np.polyval will work as SOCDependent
        self._U_oc: SOCDependent = cast(SOCDependent, lambda soc: np.polyval(self._U_oc_coefficients, soc))  # V
        self._R_0: SOCDependent = cast(SOCDependent, lambda soc: np.polyval(self._R_0_coefficients, soc))    # Ohms
        self._R_P: SOCDependent = cast(SOCDependent, lambda soc: np.polyval(self._R_P_coefficients, soc))    # Ohms
        self._C_P: SOCDependent = cast(SOCDependent, lambda soc: np.polyval(self._C_P_coefficients, soc))    # Farads

        self._Q_total = Q_total

    @property
    def get_Uoc(self) -> SOCDependent:
        return self._U_oc

    @property
    def get_R_0(self) -> SOCDependent:
        return self._R_0

    @property
    def get_R_P(self) -> SOCDependent:
        return self._R_P

    @property
    def get_C_P(self) -> SOCDependent:
        return self._C_P

    @property
    def Q_total(self) -> float:
        return self._Q_total


class KalmanFilterConfig:
    def __init__(
            self,
            battery_model_config: BatteryModelConfig,
            process_noise_matrix: NDArray,
            state_covariance_matrix: NDArray,
            measurement_noise_vector: NDArray
    ):
        self._battery_model_config = battery_model_config
        self._process_noise_matrix = process_noise_matrix
        self._state_covariance_matrix = state_covariance_matrix
        self._measurement_noise_vector = measurement_noise_vector

    @property
    def battery_model_config(self) -> BatteryModelConfig:
        """
        Configuration of the underlying `EquivalentCircuitModel`.
        """
        return self._battery_model_config

    @property
    def process_noise_matrix(self) -> NDArray[float]:
        """
        A 2x2 matrix containing the process noise covariance matrix where [0, 0] is the SOC evolution
        noise and [1, 1] is the polarization potential evolution noise.
        """
        return self._process_noise_matrix

    @property
    def state_covariance_matrix(self) -> NDArray[float]:
        """
        A 2x2 matrix containing the state covariance matrix where [0, 0] is the SOC covariance
        noise and [1, 1] is the polarization potential covariance.
        """
        return self._state_covariance_matrix

    @property
    def measurement_noise_vector(self) -> NDArray[float]:
        """
        A 1x1 vector containing the noise expected in the terminal voltage measurement.
        """
        return self._measurement_noise_vector


def load_battery_config(absolute_path: str | pathlib.Path) -> BatteryModelConfig:
    # Build the full path to the config file
    full_path = pathlib.Path(absolute_path)
    with open(full_path, 'rb') as f:
        data = tomllib.load(f)
    return BatteryModelConfig(**data)
