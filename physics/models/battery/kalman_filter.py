import numpy as np
from scipy import optimize
from filterpy.kalman import ExtendedKalmanFilter as EKF
from physics.models.battery.battery_config import BatteryModelConfig


class EKF_SOC:
    def get_SOC(self):
        """
        Return the current state of charge of the battery.

        :return: The current state of charge.
        :rtype: float
        """
        return self.SOC

    def get_Uc(self):
        """
        Return the polarization voltage of the battery.

        :return: The current polarization voltage.
        :rtype: float
        """
        return self.Uc

    def get_predicted_Ut(self):
        """
        Return the predicted terminal voltage for the last prediction step.

        :return: The predicted terminal voltage.
        :rtype: float
        """
        return self.predicted_measurement

    def update_filter(self, measured_Ut, I):
        """
        Update the filter based on a new measurement and the predicted state.
        This function should be called after `predict_state` in a typical predict-update workflow.

        :param float measured_Ut: The actual voltage across the terminals of the battery.
        :param float I: The current being sourced by the battery.
        """
        alpha = 0.9
        self._filtered_I = alpha * self._filtered_I + (1 - alpha) * I

        self.ekf.update(z=measured_Ut, HJacobian=self._measurement_jacobian, Hx=self._measurement_function)

        self.SOC, self.Uc = self.ekf.x
        self.SOC = np.clip(self.SOC, 0.0, 1.1)
        # print(f"Kalman gain: {self.ekf.K}")
        # Uoc = self.U_oc(self.SOC)
        # R0 = self.R_0(self.SOC)
        # self.predicted_measurement = Uoc - self.Uc - R0 * I

    def predict_state(self, I, time_step):
        """
        Predict the next evolution of the state vector (SOC, Uc).
        This function should be called before updating the filter in a typical predict-update workflow.

        :param float I: The current being sourced by the battery. Positive indicates current being drawn.
        :param float time_step: Time elapsed between this prediction and the last updated state of the filter (seconds).
        """
        # check_current(I)
        # Control matrix B (for input current I_k)
        self.ekf.B = np.array(
            [-time_step / self.Q_total, self.R_P(self.SOC) * (1 - np.exp(-time_step / self.tau(self.SOC)))])
        self.ekf.F = self._state_jacobian(time_step)

        #     # 🔧 Incorporate uncertainty in current measurement into Q
        # I_noise_std = 0.5  # adjust based on sensor datasheet or empirical tests
        # dq = (time_step / self.Q_total) * I_noise_std
        # duc = self.R_P(self.SOC) * (1 - np.exp(-time_step / self.tau(self.SOC))) * I_noise_std
        # # if I > 5:
        # #     print(f"dq: {dq * 1e2}, duc: {duc * 1e4}, I: {I}")
        #
        # self.ekf.Q = np.diag([
        #     1e-1 * 0.1,  # add floor to maintain numerical stability
        #     1e3
        # ])

        self.ekf.predict(u=I)
        self.SOC, self.Uc = self.ekf.x

        # print(f'ekf prediction: {self.ekf.x_prior}')

    def predict_then_update(self, measured_Ut, I, time_step):
        """
        Predict the next evolution of the state vector (SOC, Uc), then update the filter
        based on this prediction and a measurement. Abstracts the full predict-update workflow of the EKF.

        :param float measured_Ut: The actual voltage across the terminals of the battery.
        :param float I: The current being sourced by the battery. Positive indicates current being drawn.
        :param float time_step: Time elapsed between this prediction and the last updated state of the filter (seconds).
        """
        # check_current(I)
        # check_Terminal_V(measured_Ut)

        self.predict_state(I, time_step)
        # print(f'predicted: {self.ekf.x_prior}')

        self.update_filter(measured_Ut, I)
        # print(f'SOC: {self.ekf.x[0]}, Uc: {self.ekf.x[1]}')

    def _state_jacobian(self, time_step):
        """
        Return the state Jacobian matrix for the current time step.

        :param float time_step: Time elapsed between this prediction and the last updated state of the filter (seconds).
        :return: The state Jacobian matrix.
        :rtype: np.ndarray
        """
        return np.array([[1, 0], [0, np.exp(-time_step / self.tau(self.SOC))]])

    def _measurement_jacobian(self, x):
        """
        Return the measurement Jacobian matrix for the current state vector.

        :param list[float, float] x: The state vector [SOC, Uc].
        :return: The measurement Jacobian matrix.
        :rtype: np.ndarray
        """
        SOC = x[0]
        dUoc_dSOC = self.Uoc_derivative(SOC)
        dR0_dSOC = self.R_0_derivative(SOC)
        return np.array([[dUoc_dSOC - dR0_dSOC * self._filtered_I, -1]])

    def _measurement_function(self, x):
        """
        Return the measurement function relating terminal voltage to SOC and polarization voltage.

        :param list[float, float] x: The state vector [SOC, Uc].
        :param float I: The current being sourced by the battery.
        :return: The predicted terminal voltage.
        :rtype: float
        """
        SOC, Uc = x
        Uoc = self.U_oc(SOC)
        R0 = self.R_0(SOC)
        self.predicted_measurement = Uoc - Uc - R0 * self._filtered_I
        return self.predicted_measurement

    def __init__(self, battery_config: BatteryModelConfig, initial_SOC=1, initial_Uc=0):
        """
        EKF_SOC represents the Kalman filter used for predicting state of charge.

        :param BatteryModelConfig battery_config: Contains the HPPC parameters of the battery model.
        :param float initial_SOC: Initial state of charge of the battery (ranges from 0 to 1 inclusive, default is 1).
        :param float initial_Uc: Initial polarization voltage of the battery in volts (default is 0).
        """
        # Initial state
        self.SOC = initial_SOC
        self.Uc = initial_Uc  # Polarization Voltage

        # Load Config data
        self.Q_total = battery_config.Q_total
        SOC_data = battery_config.SOC_data
        Uoc_data = battery_config.Uoc_data
        R_0_data = battery_config.R_0_data
        R_P_data = battery_config.R_P_data
        C_P_data = battery_config.C_P_data

        def quintic_polynomial(x, x0, x1, x2, x3, x4, x5, x6, x7):
            return np.polyval([x0, x1, x2, x3, x4, x5, x6, x7], x)

        U_oc_coefficients, _ = optimize.curve_fit(quintic_polynomial, SOC_data, Uoc_data)
        R_0_coefficients, _ = optimize.curve_fit(quintic_polynomial, SOC_data, R_0_data)
        R_P_coefficients, _ = optimize.curve_fit(quintic_polynomial, SOC_data, R_P_data)
        C_P_coefficients, _ = optimize.curve_fit(quintic_polynomial, SOC_data, C_P_data)
        self.U_oc = lambda soc: np.polyval(U_oc_coefficients, soc)  # Open-circuit voltage as a function of SOC
        self.R_0 = lambda soc: np.polyval(R_0_coefficients, soc)  # Resistance as a function of SOC
        self.R_P = lambda soc: np.polyval(R_P_coefficients, soc)  # Resistance as a function of SOC
        self.C_P = lambda soc: np.polyval(C_P_coefficients, soc)  # Resistance as a function of SOC
        self.Uoc_derivative = lambda soc: np.polyval(np.polyder(U_oc_coefficients),
                                                     np.minimum(1.0, soc))  # Derivative of Uoc wrt SOC
        self.R_0_derivative = lambda soc: np.polyval(np.polyder(R_0_coefficients), np.minimum(1.0, soc))

        self.tau = lambda soc: self.R_P(soc) * self.C_P(soc)

        # initializing the ekf object
        self.ekf = EKF(dim_x=2, dim_z=1)
        self.ekf.x = np.array([self.SOC, self.Uc])
        self.ekf.Q = np.diag([
            1e-10 * 0.1,  # add floor to maintain numerical stability
            1e-6 * 0.1
        ])
        self.ekf.P = np.diag(
            [1e-2 * 0.5,
             1e-1]
        )
        self.ekf.R = np.eye(1) * 1e0 * 0.5

        # For logs
        self._filtered_I = 0
        self.predicted_measurement = 0

