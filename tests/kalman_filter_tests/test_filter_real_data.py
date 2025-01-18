import numpy as np
import pandas as pd
import pytest
from physics.models.battery.kalman_filter import EKF_SOC
from physics.models.battery.battery_config import BatteryModelConfig, load_battery_config
import matplotlib.pyplot as plt
import pathlib



# This test requires a voltage.csv and current.csv in the same directory to run
def csv_to_timeseries_tuples(csv_file):
    path = pathlib.Path(__file__).parent / csv_file
    df = pd.read_csv(path)
    df['Time'] = pd.to_datetime(df['Time'])
    return np.array(list(zip(df['Time'].dt.to_pydatetime(), df['Value'])))


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
    time_axis = [entry[0] for entry in data_arrays[0]]
    # Predefined color list
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



@pytest.mark.skip(reason="this isn't a real test and takes quite long")
def test_kalman_filter():
    
    voltage_data = csv_to_timeseries_tuples('voltage.csv')
    current_data = csv_to_timeseries_tuples('current.csv')



    config_path = pathlib.Path(__file__).parent.parent / "battery_config.toml"
    config: BatteryModelConfig = load_battery_config(config_path.absolute())
    
    ekf = EKF_SOC(config, 1, 0)

    SOC_array = np.zeros(len(voltage_data))
    Ut_array = np.zeros(len(voltage_data))
    current_array = np.zeros(len(voltage_data))
    polarization_voltage_array = np.zeros(len(voltage_data))
    predicted_Ut_array = np.zeros(len(voltage_data))

    for i in range(1, len(voltage_data)):
        # Calculate time difference between current and previous measurements
        time_difference = (voltage_data[i][0] - current_data[i - 1][0]).total_seconds()
        
        Ut = voltage_data[i][1] / 32  # Normalize voltage for cell count
        I = current_data[i][1]
        
        ekf.predict_then_update(Ut, I, time_difference)

        SOC_array[i] = ekf.get_SOC()
        Ut_array[i] = Ut
        current_array[i] = I
        polarization_voltage_array[i] = ekf.get_Uc()
        predicted_Ut_array[i] = ekf.get_predicted_Ut()

    # example usage
    plot_kalman_results(
        [SOC_array, Ut_array, predicted_Ut_array], 
        ["SOC", "Measured Terminal Voltage (V)", "Predicted Terminal Voltage (V)"]
    )

