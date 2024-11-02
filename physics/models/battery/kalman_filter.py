import numpy as np
from filterpy.kalman import ExtendedKalmanFilter as EKF
from .battery_config import BatteryModelConfig
import numpy as np

class EKF_SOC():
    def __init__(self, battery_model_config: BatteryModelConfig,  initial_SOC = 1, initial_Uc = 0):
        """
        EKF_SOC represents the kalman filter used for predicting state of charge.

        Attributes:
            battery_model_config (BatteryModelConfig): This contains the HPPC parameters of the battery model
            initial_SOC (float/int): Ranges from 0 to 1 inclusive. The initial state of charge of the battery.
            initial_Uc (float/int):  The initial polarization volatge of the battery. (V)
        """
        # Inital state
        self.SOC = initial_SOC
        self.Uc = initial_Uc  # Polarization Volatge

        # Covariance Matrices
        self.Q_covariance = np.eye(2) * 0.0001
        self.R_covariance = np.eye(1) * 0.5     # currently not really trusting the predicted state

        # Load Config
        self.R_P = battery_model_config.R_P
        self.C_P = battery_model_config.C_P
        self.Q_total = battery_model_config.Q_total
        SOC_data = np.array(battery_model_config.SOC_data)
        Uoc_data = np.array(battery_model_config.Uoc_data)
        R_0_data = np.array(battery_model_config.R_0_data)

        # polynomial interpolation
        self.Uoc_coefficients = np.polyfit(SOC_data, Uoc_data, 7)
        self.R_0_coefficients = np.polyfit(SOC_data, R_0_data, 7)
        self.Uoc_derivative_coefficients = np.polyder(self.Uoc_coefficients)

        self.tau = self.R_P / self.C_P

        # initializing the ekf object
        self.ekf = EKF(dim_x=2, dim_z=1)
        self.ekf.x = np.array([self.SOC, self.Uc])
        self.ekf.P = np.diag([1e-6, 1e-6])  # I'm keeping low uncertainty in initial SOC and Uc
        self.ekf.Q = self.Q_covariance
        self.ekf.R = self.R_covariance

        # For logs
        self.predicted_measurment = 0
    
    def get_SOC(self): 
        """ Return the state of charge of the battery """
        return self.SOC
    
    def get_Uc(self):
        """ Return the polarization voltage of the battery """
        return self.Uc
    
    def get_predicted_Ut(self):
        """ Return the predicted terminal voltage for the last prediction step """
        return self.predicted_measurment
        
    def update_filter(self, measured_Ut, I):
        """
        Update the filter based on a new measurment, and the predicted state.
        This function should be called after predict_state in a typical predict update workflow.

        Attributes:
            measured_Ut (float/integer): The actual voltage across the terminals of the battery.
            I (float/integer): The current being sourced by the battery
        """
        check_Terminal_V(measured_Ut)

        h_jacobian = self._measurement_jacobian
        Hx = self._measurement_function

        self.ekf.update(z=measured_Ut, HJacobian=h_jacobian, Hx=Hx, hx_args=I)

        self.SOC, self.Uc = self.ekf.x

    def predict_state(self, I, time_step):
        """
        Predicts the next evolution of the state vector (SOC, Uc).
        This function should be called before updating the filter in a typical predict update workflow.

        Attributes:
            I (float/integer): The current being sourced by the battery. Positive indicated current being drawn.
            time_step (float/integer): Time elapsed between this prediction and the last updated state of filter. (Seconds)
        """        
        check_current(I)
        # Control matrix B (for input current I_k)
        self.ekf.B = np.array([-time_step / self.Q_total, self.R_P * (1 - np.exp(-time_step / self.tau))])
        self.ekf.F = self._state_jacobian(time_step)
        
        self.ekf.predict(u=I)
        print(f'ekf prediction: {self.ekf.x_prior}')

    def predict_then_update(self, measured_Ut, I, time_step):
        """
        Predicts the next evolution of the state vector (SOC, Uc), then updates the filter
        based on this prediction and a measurement.
        This function abstracts the full predict update workflow of the EKF. 

        Attributes:
            measured_Ut (float/integer): The actual voltage across the terminals of the battery.
            I (float/integer): The current being sourced by the battery. Positive indicated current being drawn.
            time_step (float/integer): Time elapsed between this prediction and the last updated state of filter. (Seconds)
        """  
        check_current(I)
        check_Terminal_V(measured_Ut)

        self.predict_state(I, time_step)
        print(f'predicted: {self.ekf.x_prior}')

        self.update_filter(measured_Ut, I)
        print(f'SOC: {self.ekf.x[0]}, Uc: {self.ekf.x[1]}')

    def _state_jacobian(self, time_step):
        """ 
        Returns the state jacobian for the current time 

        Attributes:
            time_step (float/integer): Time elapsed between this prediction and the last updated state of filter. (Seconds)
        """
        return np.array([[1, 0], [0, np.exp(-time_step / self.tau)]])

    def _measurement_jacobian(self, x):
        """ 
        Returns the measurement jacobian for the current time 

        Attributes:
            x [float, float]: The state vector [SOC, Uc], where both values are floats or integers.
        """
        SOC = x[0]
        derivative = np.polyval(self.Uoc_derivative_coefficients, SOC)
        return np.array([[derivative, -1]])

    def _measurement_function(self, x, I):
        """
        The customized measurement equation relating Ut to SOC and Uc

        Attributes:
            x [float, float]: The state vector [SOC, Uc], where both values are floats or integers.
            I (float/integer): The current being sourced by the battery. Positive indicated current being drawn.
        """
        SOC, Uc = x
        R_0 = np.polyval(self.R_0_coefficients, SOC)
        Uoc = np.polyval(self.Uoc_coefficients, SOC)
        self.predicted_measurment = Uoc - Uc - R_0*I
        return self.predicted_measurment

def check_current(I):
    if not isinstance(I, (float, int)):
        raise TypeError(f"Invalid type for current I: {type(I)}. Expected float or int.")
    if not (-45.0 <= I <= 45.0):
        raise ValueError(f"Invalid value for current (I): {I}. Must be between -45.0A and 45.0A.")

def check_Terminal_V(Ut):
    if not isinstance(Ut, (float, int)):
        raise TypeError(f"Invalid type for measured_Ut: {type(Ut)}. Expected float or int.")
    if not (0.0 <= Ut <= 5.0):
        raise ValueError(f"Invalid value for terminal voltage (measured_Ut): {Ut}. Must be between 0.0 and 5.0 volts.")
