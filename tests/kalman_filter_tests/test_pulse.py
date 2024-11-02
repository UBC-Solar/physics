import numpy as np
import pandas as pd
from physics.models.battery.kalman_filter import EKF_SOC
import matplotlib.pyplot as plt



ekf_full = EKF_SOC(1, 0)
ekf_depleted = EKF_SOC(0.4, 0)
time_steps = np.linspace(0, 3610 * 100, 3610 * 100)

current_array = np.zeros(len(time_steps))
SOC_array_full = np.zeros(len(time_steps))
Ut_array_full = np.zeros(len(time_steps))
polarization_voltage_array_full = np.zeros(len(time_steps))

SOC_array_depeleted = np.zeros(len(time_steps))
Ut_array_depeleted = np.zeros(len(time_steps))
polarization_voltage_array_depeleted = np.zeros(len(time_steps))

for i in range(0, 10 * 100):
    # Calculate time difference between current and previous measurements
    time_difference = 1 / 100 # seconds
    
    I = 40.0 # 40 amps discharge
    Ut = 3.0
    
    ekf_full.predict_then_update(Ut, I, time_difference)
    ekf_depleted.predict_then_update(Ut, I, time_difference)

    SOC_array_full[i] = ekf_full.get_SOC()
    SOC_array_depeleted[i] = ekf_depleted.get_SOC()
    current_array[i] = I

for i in range(10 * 100, 3610 * 100):
    # Calculate time difference between current and previous measurements
    time_difference = 1 / 100 # seconds
    
    Ut = 3.0
    I = 0.0 # 40 amps discharge
    
    ekf_full.predict_then_update(Ut, I, time_difference)
    ekf_depleted.predict_then_update(Ut, I, time_difference)

    SOC_array_full[i] = ekf_full.get_SOC()
    SOC_array_depeleted[i] = ekf_depleted.get_SOC()
    Ut_array_depeleted[i] = ekf_depleted.get_predicted_Ut()
    Ut_array_full[i] = ekf_full.get_predicted_Ut()
    current_array[i] = I


def plot_kalman_results(data_arrays, labels):
    """
    Plots multiple data arrays against a common time axis on separate y-axes.

    Parameters:
    - time_axis: Array of time values
    - data_arrays: List of data arrays to plot (each array should be of equal length)
    - labels: List of labels for each data array
    - colors: List of colors for each data array

    Example:
    plot_kalman_results([SOC_array, voltage_array, current_array], 
                        ["SOC", "Voltage (V)", "Current (A)"], 
                        ["tab:blue", "tab:red", "tab:green"])
    """
    time_axis = time_steps
    # Predefined color list (10 colors)
    colors = [
        "tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple",
        "tab:brown", "tab:pink", "tab:gray", "tab:olive", "tab:cyan"
    ]
    
    # Ensure input arrays are of the same length
    if not all(len(arr) == len(time_axis) for arr in data_arrays):
        raise ValueError("All data arrays must be of the same length as the time axis.")
    
    fig, ax1 = plt.subplots()
    ax1.set_xlabel("Time")
    
    # Plot each array on a new y-axis
    for i, (data, label, color) in enumerate(zip(data_arrays, labels, colors)):
        if i == 0:
            ax = ax1  # First plot on primary y-axis
        else:
            ax = ax1.twinx()  # Subsequent plots on secondary y-axes
            ax.spines['right'].set_position(('outward', 60 * (i - 1)))
        
        ax.set_ylabel(label, color=color)
        ax.plot(time_axis, data, color=color)
        ax.tick_params(axis='y', labelcolor=color)

    fig.tight_layout()  # Adjust layout to prevent overlap
    plt.show()

plot_kalman_results(
    [Ut_array_depeleted, Ut_array_full, current_array], 
    ["SOC depleted", "SOC full", "Current (A)"]
)


