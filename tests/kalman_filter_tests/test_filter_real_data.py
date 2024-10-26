import numpy as np
import pandas as pd
from physics.models.battery.kalman_filter import EKF_SOC
import matplotlib.pyplot as plt


# This test requires a voltage.csv and current.csv in the same directory to run
def csv_to_timeseries_tuples(csv_file):
    df = pd.read_csv(csv_file)
    df['Time'] = pd.to_datetime(df['Time'])
    return np.array(list(zip(df['Time'].dt.to_pydatetime(), df['Value'])))


voltage_data = csv_to_timeseries_tuples('voltage.csv')
current_data = csv_to_timeseries_tuples('current.csv')

ekf = EKF_SOC(1, 0)

SOC_array = np.zeros(len(voltage_data))
voltage_array = np.zeros(len(voltage_data))
current_array = np.zeros(len(voltage_data))

for i in range(1, len(voltage_data)):
    current_time = voltage_data[i][0]
    prev_time = current_data[i - 1][0]
    time_difference = (current_time - prev_time).total_seconds()
    
    Ut = voltage_data[i][1] / 32 # account for number of cells
    I = current_data[i][1]
    # print(f'Ut: {Ut}')
    # print(f'I: {I}')
    ekf.predict_then_update(Ut, I, time_difference)

    SOC_array[i] = ekf.get_SOC()
    voltage_array[i] = Ut
    # current_array[i] = I




def plot_kalman_results(SOC_array, voltage_array, current_array):
    time_axis = [entry[0] for entry in voltage_data]

    # Create the plot with multiple y-axes
    fig, ax1 = plt.subplots()

    # Plot SOC on the first y-axis (left)
    color = 'tab:blue'
    ax1.set_xlabel('Time')
    ax1.set_ylabel('SOC', color=color)
    ax1.plot(time_axis, SOC_array, color=color)
    ax1.tick_params(axis='y', labelcolor=color)

    # Create a second y-axis for Voltage
    ax2 = ax1.twinx()
    color = 'tab:red'
    ax2.set_ylabel('Voltage (V)', color=color)
    ax2.plot(time_axis, voltage_array, color=color)
    ax2.tick_params(axis='y', labelcolor=color)

    # Create a third y-axis for Current
    ax3 = ax1.twinx()
    ax3.spines['right'].set_position(('outward', 60))
    color = 'tab:green'
    ax3.set_ylabel('Current (A)', color=color)
    ax3.plot(time_axis, current_array, color=color)
    ax3.tick_params(axis='y', labelcolor=color)

    # Finalize the layout
    fig.tight_layout()

    # Show the plot
    plt.show()


# plot_kalman_results(SOC_array, voltage_array, current_array)
