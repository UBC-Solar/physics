import numpy as np
from filterpy.kalman import ExtendedKalmanFilter
from typing import Protocol, runtime_checkable, cast, Callable
from physics.models.battery import EquivalentCircuitModelConfig
from numpy.typing import NDArray


@runtime_checkable
class FilteredBatteryModelConfig(Protocol):
    """
    A specification for a configuration object which contains the requisite data to specify
    a `FilteredBatteryModel`.
    """
    @property
    def battery_model_config(self) -> EquivalentCircuitModelConfig:
        """
        Configuration of the underlying `EquivalentCircuitModel`.
        """
        ...

    @property
    def process_noise_matrix(self) -> NDArray[float]:
        """
        A 2x2 matrix containing the process noise covariance matrix where [0, 0] is the SOC evolution
        noise and [1, 1] is the polarization potential evolution noise.
        """
        ...

    @property
    def state_covariance_matrix(self) -> NDArray[float]:
        """
        A 2x2 matrix containing the state covariance matrix where [0, 0] is the SOC covariance
        noise and [1, 1] is the polarization potential covariance.
        """
        ...

    @property
    def measurement_noise_vector(self) -> NDArray[float]:
        """
        A 1x1 vector containing the noise expected in the terminal voltage measurement.
        """
        ...


class FilteredBatteryModel:
    """
    `FilteredBatteryModel` is a first-order Thevenin equivalent model of a lithium-ion battery packed, wrapped
    in a Kalman filter which uses voltage measurements with model predictions.
    """
    def __init__(
            self,
            battery_config: FilteredBatteryModelConfig,
            initial_SOC: float = 1.0,
            initial_Uc: float = 0.0,
            alpha: float = 0.9
    ):
        """
        :param FilteredBatteryModelConfig battery_config: Contains Kalman filter state estimation configuration and
            underlying equivalent circuit model configuration.
        :param float initial_SOC: Initial SOC of the battery, in the range (0, 1].
        :param float initial_Uc: Initial polarization voltage of the battery in Volts.
        """
        # Initial state
        assert 0.0 <= initial_SOC <= 1.1, "`initial_SOC` must be in (0, 1.1]!"

        self._SOC = initial_SOC     # State of Charge
        self._Uc = initial_Uc       # Polarization Voltage

        # Load Config data
        self._Q_total = battery_config.battery_model_config.Q_total

        # These `cast` calls just promise to the type-checker that these will map floats to floats
        self._U_oc = cast(Callable[[float], float], battery_config.battery_model_config.get_Uoc)
        self._R_0 = cast(Callable[[float], float], battery_config.battery_model_config.get_R_0)
        self._R_P = cast(Callable[[float], float], battery_config.battery_model_config.get_R_P)
        self._C_P = cast(Callable[[float], float], battery_config.battery_model_config.get_C_P)

        self._tau: Callable[[float], float] = lambda soc: self._R_P(soc) * self._C_P(soc)  # Characteristic Time in s

        def central_difference_derivative(func, value, h=1e-6):
            """
            Compute the derivative of an arbitrary function `func` at `SOC` using central difference.

            :param func: The function to differentiate.
            :param value: The point at which to compute the derivative.
            :param h: Step size for the finite difference.
            :return: The derivative of the function at `SOC`.
            """
            return (func(value + h) - func(value - h)) / (2 * h)

        self._dUoc_dSOC = lambda soc: central_difference_derivative(self._U_oc, np.minimum(1.0, soc))  # dUOC wrt to SOC
        self._dR0_dSOC = lambda soc: central_difference_derivative(self._R_0, np.minimum(1.0, soc))    # dR0 wrt to SOC

        # Initializing the EKF object
        self._ekf = ExtendedKalmanFilter(dim_x=2, dim_z=1)

        # State Vector
        self._ekf.x = np.array([
            self._SOC,
            self._Uc]
        )

        self._ekf.P = battery_config.state_covariance_matrix
        self._ekf.Q = battery_config.process_noise_matrix
        self._ekf.R = battery_config.measurement_noise_vector

        assert 0 <= alpha <= 1, "`alpha` should be between 0 and 1!"
        self._alpha = alpha

        self._filtered_I = 0
        self._predicted_measurement = 0

    @property
    def SOC(self) -> float:
        """
        Return the current SOC of the battery.

        :return: The current state of charge.
        """
        return self._SOC

    @property
    def Uc(self) -> float:
        """
        Return the polarization voltage of the battery.

        :return: The current polarization voltage.
        """
        return self._Uc

    @property
    def Ut(self) -> float:
        """
        Return the predicted terminal voltage for the last prediction step.

        :return: The predicted terminal voltage.
        """
        return self._predicted_measurement

    def update_filter(self, measured_Ut, current):
        """
        Update the filter based on a new measurement and the predicted state.
        This function should be called after `predict_state` in a typical predict-update workflow.

        :param float measured_Ut: The actual voltage across the terminals of the battery.
        :param float current: The current being sourced by the battery.
        """
        # Simple low-pass filter to current
        self._filtered_I = self._alpha * self._filtered_I + (1 - self._alpha) * current

        self._ekf.update(z=measured_Ut, HJacobian=self._measurement_jacobian, Hx=self._measurement_function)

        self._SOC, self._Uc = self._ekf.x
        self._SOC = np.clip(self._SOC, 0.0, 1.1)

    def predict_state(self, current, time_step):
        """
        Predict the next evolution of the state vector (SOC, Uc).
        This function should be called before updating the filter in a typical predict-update workflow.

        :param float current: The current being sourced by the battery.
            Sign convention is that positive indicates current being drawn.
        :param float time_step: Time elapsed between this prediction and the last updated state of the filter (seconds).
        """
        # Control matrix B (for input current I_k)
        self._ekf.B = np.array([
            -time_step / self._Q_total,
            self._R_P(self._SOC) * (1 - np.exp(-time_step / (self._tau(self._SOC)))),
        ])
        self._ekf.F = self._state_jacobian(time_step)

        self._ekf.predict(u=current)
        self._SOC, self._Uc = self._ekf.x

    def predict_then_update(self, measured_Ut: float, current: float, time_step: float):
        """
        Predict the next evolution of the state vector (SOC, Uc), then update the filter
        based on this prediction and a measurement. Abstracts the full predict-update workflow of the EKF.

        :param float measured_Ut: The actual voltage across the terminals of the battery.
        :param float current: The current being sourced by the battery. Positive indicates current being drawn.
        :param float time_step: Time elapsed between this prediction and the last updated state of the filter (seconds).
        """
        self.predict_state(current, time_step)
        self.update_filter(measured_Ut, current)

    def _state_jacobian(self, time_step):
        """
        Return the state Jacobian matrix for the current time step.

        :param float time_step: Time elapsed between this prediction and the last updated state of the filter (seconds).
        :return: The state Jacobian matrix.
        :rtype: np.ndarray
        """
        return np.array([[1, 0], [0, np.exp(-time_step / self._tau(self._SOC))]])

    def _measurement_jacobian(self, x):
        """
        Return the measurement Jacobian matrix for the current state vector.

        :param list[float, float] x: The state vector [SOC, Uc].
        :return: The measurement Jacobian matrix.
        :rtype: np.ndarray
        """
        SOC = x[0]
        dUoc_dSOC = self._dUoc_dSOC(SOC)
        dR0_dSOC = self._dR0_dSOC(SOC)

        return np.array([[dUoc_dSOC - dR0_dSOC * self._filtered_I, -1]])

    def _measurement_function(self, x) -> float:
        """
        Return the measurement function relating terminal voltage to SOC and polarization voltage.

        :param list[float, float] x: The state vector [SOC, Uc].
        :return: The predicted terminal voltage.
        """
        SOC, Uc = x
        Uoc = self._U_oc(SOC)
        R0 = self._R_0(SOC)
        self._predicted_measurement = Uoc - Uc - R0 * self._filtered_I

        return self._predicted_measurement
