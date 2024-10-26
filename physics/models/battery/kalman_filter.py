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
        R_0_data = np.array([2.564, 2.541, 2.541, 2.558, 2.549, 2.574, 2.596, 2.626, 2.676, 2.789]) / 1000  # In ohms (Ω)
        self.R_P = 0.530 / 1000     # Ohms
        self.C_P = 14646            # F
        self.tau = self.R_P / self.C_P
        self.Q_total = 259200       # 72Amp hours

        # Creating the SOC derivative curve
        SOC_data = np.array([0.0752, 0.1705, 0.2677, 0.366, 0.4654, 0.5666, 0.6701, 0.7767, 0.8865, 1.0])
        Uoc_data = np.array([3.481, 3.557, 3.597, 3.623, 3.660, 3.750, 3.846, 3.946, 4.056, 4.183])
        self.Uoc_coefficients = np.polyfit(SOC_data, Uoc_data, 6)
        self.R_0_coefficients = np.polyfit(SOC_data, R_0_data, 6)
        self.Uoc_derivative_coefficients = np.polyder(self.Uoc_coefficients)

        # initializing the ekf object
        self.ekf = EKF(dim_x=2, dim_z=1)
        self.ekf.x = np.array([self.SOC, self.Uc])
        self.ekf.P = self.Q_covariance     # common practice is to initialize P using Q
        self.ekf.Q = self.Q_covariance
        self.ekf.R = self.R_covariance

    def get_SOC_curve_derivative(self, SOC):
        return np.polyval(self.Uoc_derivative_coefficients, SOC)
    
    def get_SOC_value(self, SOC):
        return np.polyval(self.Uoc_coefficients, SOC)
    
    def get_R_0_value(self, SOC):
        return np.polyval(self.R_0_coefficients, SOC)
    
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
        
    def update_filter(self, measured_Ut, I):
        self._check_Terminal_V(measured_Ut)

        h_jacobian = self.measurement_jacobian
        Hx = self.measurement_function

        self.ekf.update(z=measured_Ut, HJacobian=h_jacobian, Hx=Hx, hx_args=I)

        self.SOC, self.Uc = self.ekf.x

    def predict_state(self, I, time_step):        
        self._check_current(I)
        # Control matrix B (for input current I_k)
        self.ekf.B = np.array([-time_step / self.Q_total, self.R_P * (1 - np.exp(-time_step / self.tau))])
        state_jacobian = self.state_jacobian(time_step)
        self.ekf.F = state_jacobian
        
        self.ekf.predict(u=I)
        print(f'ekf prediction: {self.ekf.x_prior}')

    def predict_then_update(self, measured_Ut, I, time_step):
        self._check_current(I)
        self._check_Terminal_V(measured_Ut)

        self.predict_state(I, time_step)
        print(f'predicted: {self.ekf.x_prior}')

        self.update_filter(measured_Ut, I)
        print(f'SOC: {self.ekf.x[0]}, Uc: {self.ekf.x[1]}')

    def state_jacobian(self, time_step):
        return np.array([[1, 0], [0, np.exp(-time_step / self.tau)]])

    def measurement_jacobian(self, x):
        SOC = x[0]
        derivative = self.get_SOC_curve_derivative(SOC)
        return np.array([[derivative, -1]])

    # the customized measurement equation relating Ut to SOC and Uc
    def measurement_function(self, x, I):
        SOC, Uc = x
        print("here in measurement function", SOC, Uc)
        # return self.Uoc_derivative_curve(SOC) * SOC - Uc - I*self.R_0(SOC) + self.R_covariance
        derivative = self.get_SOC_curve_derivative(SOC)
        R_0 = self.get_R_0_value(SOC)

        print("result: ", derivative * SOC - Uc - R_0*I)
        print(f'resistance: {R_0}')
        # return derivative * SOC - Uc - self.R_0*I
        return self.get_SOC_value(SOC) - Uc - R_0*I
    




# iterations = 10
# time_step = 1000
# test_EKF = EKF_SOC(1, 0)
# SOCs = np.zeros(iterations)
# Ucs = np.zeros(iterations)
# def test():
#     Ut = 4.183
#     delta_Ut = 0.15
#     for i in range(10):
#         test_EKF.predict_then_update(Ut, 20.0, time_step)
#         SOCs[i] = test_EKF.get_SOC()
#         Ucs[i] = test_EKF.get_Uc()
#         Ut -= delta_Ut

# import matplotlib.pyplot as plt


# test()

# # Create a figure and axis
# fig, ax1 = plt.subplots()

# # Plot SOC on the first y-axis
# color = 'tab:blue'
# ax1.set_xlabel('Iteration')
# ax1.set_ylabel('SOC (State of Charge)', color=color)
# ax1.plot(np.arange(iterations), SOCs, color=color, marker='o', label='SOC')
# ax1.tick_params(axis='y', labelcolor=color)
# ax1.grid(True)

# # Create a second y-axis for Uc on the same x-axis
# ax2 = ax1.twinx()  # Create a twin Axes sharing the x-axis
# color = 'tab:green'
# ax2.set_ylabel('Uc (Polarization Voltage)', color=color)
# ax2.plot(np.arange(iterations), Ucs, color=color, marker='o', linestyle='--', label='Uc')
# ax2.tick_params(axis='y', labelcolor=color)

# # Add a title
# plt.title('SOC and Uc over Iterations')

# # Show the plot
# plt.tight_layout()  # Adjust layout so labels don't overlap
# plt.show()
