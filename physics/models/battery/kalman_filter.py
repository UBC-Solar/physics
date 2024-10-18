import numpy as np
from filterpy.kalman import ExtendedKalmanFilter as EKF

class EKF_SOC():
    def __init__(self, inital_SOC = 1, initial_Uc = 0):
        self.SOC = inital_SOC
        self.Uc = initial_Uc  # Polarization Volatge

        # Covariance Matrices
        self.Q_covariance = np.eye(2) * 0.001
        self.R_covariance = np.eye(1) * 0.001

        # HPPC coefficients go here
        R_0_coefficients = [2.12293662e-08, -5.14521450e-06, 4.68039977e-04, -2.00597542e-02, 2.91277259e+00]
        self.R_0 = lambda soc: np.polyval(R_0_coefficients, soc) / 1000     # Ohms
        self.R_P = 0.530 / 1000     # Ohms
        self.C_P = 14646            # F
        self.tau = self.R_P / self.C_P
        self.Q_total = 259200       # 72Amp hours

        # Creating the SOC derivative curve
        SOC_data = np.array([1, 0.8865, .7767, .6701, .5666, .4654, .3660, .2677, .1705, .0752])
        Uoc_data = np.array([4.183, 4.056, 3.946, 3.846, 3.750, 3.660, 3.623, 3.597, 3.557, 3.481])
        Uoc_coefficients = np.polyfit(SOC_data, Uoc_data, 4)
        self.Uoc_derivative_coefficients = np.polyder(Uoc_coefficients)

        # initializing the ekf object
        self.ekf = EKF(dim_x=2, dim_z=1)
        self.ekf.x = np.array([self.SOC, self.Uc])
        self.ekf.P = self.Q_covariance     # common practice is to initialize P using Q
        self.ekf.Q = self.Q_covariance
        self.ekf.R = self.R_covariance
        print("AT THE TOPPPP")
        print(self.ekf.x)
        print("_________")

    def get_SOC_curve_derivative(self):
        return np.polyval(self.Uoc_derivative_coefficients, self.SOC)
    
    def get_SOC(self): 
        return self.SOC
    
    def get_Uc(self):
        return self.Uc
    
    def _check_current(self, I):
        if not (-45.0 <= I <= 45.0):
            raise ValueError(f"Invalid value for current (I): {I}. Must be between -45.0A and 45.0A.")
        if not type(I) == float:
            raise TypeError(f"Invalid type for current I: {type(I)}. Expected float.")
        
    def _check_Terminal_V(self, Ut):
        if not type(Ut) == float:
            raise TypeError(f"Invalid type for measured_Ut: {type(Ut)}. Expected float.")
        if not (0.0 <= Ut <= 5.0):
            raise ValueError(f"Invalid value for terminal voltage (measured_Ut): {Ut}. Must be between 0.0 and 5.0 volts.")
        
    def update_filter(self, measured_Ut):
        self._check_Terminal_V(measured_Ut)

        h_jacobian = self.measurement_jacobian
        Hx = self.measurement_function

        self.ekf.update(z=measured_Ut, HJacobian=h_jacobian, Hx=Hx)

        self.SOC, self.Uc = self.ekf.x

    def predict_state(self, I, time_step):        
        print("AT THE TOP OF PREDICTTTT")
        print(self.ekf.x)
        print("_________")
        self._check_current(I)
        # Control matrix B (for input current I_k)
        self.ekf.B = np.array([[-time_step / self.Q_total], [self.R_P * (1 - np.exp(-time_step / self.tau))]])
        state_jacobian = self.state_jacobian(time_step)
        self.ekf.F = state_jacobian
        
        self.ekf.predict(u=I)
        print(self.ekf.x)

    def state_jacobian(self, time_step):
        return np.array([[1, 0], [0, np.exp(-time_step / self.tau)]])

    def measurement_jacobian(self, x):
        SOC = x[0]
        derivative = self.get_SOC_curve_derivative()
        return np.array([[derivative, -1]])

    # the customized measurement equation relating Ut to SOC and Uc
    def measurement_function(self, x):
        print(x)
        print("_______________________")
        SOC, Uc = x
        # return self.Uoc_derivative_curve(SOC) * SOC - Uc - I*self.R_0(SOC) + self.R_covariance
        derivative = self.get_SOC_curve_derivative()
        return derivative * SOC - Uc
    


test_EKF = EKF_SOC(1, 0)

print(test_EKF.get_SOC())

test_EKF.predict_state(5.0, 10)

test_EKF.update_filter(3.0)

print(test_EKF.get_SOC())


