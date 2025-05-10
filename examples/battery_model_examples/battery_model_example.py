import numpy as np
import pandas as pd
import pathlib
import matplotlib.pyplot as plt
from physics.models.battery import EquivalentCircuitBatteryModel, BatteryModelConfig, load_battery_config


# This test requires a voltage.csv and current.csv in the same directory to run
def csv_to_timeseries_tuples(csv_file):
    path = pathlib.Path(__file__).parent / csv_file
    df = pd.read_csv(path)
    df['Time'] = pd.to_datetime(df['Time'])
    return np.array(list(zip(df['Time'].dt.to_pydatetime(), df['Value'])))


def plot_results(soc_array, predicted_ut_array, voltage_data, window=None):
    fig, ax = plt.subplots()

    if window is None:
        window = slice(0, len(predicted_ut_array), 1)

    ax.plot(predicted_ut_array, label=r"Predicted $U_t$", color="tab:red")
    ax.plot(voltage_data, label=r"Measured $U_t$", color="tab:orange")
    ax.set_ylim(75, 140)
    ax.set_xticks([])
    ax.set_ylabel("Voltage")

    ax2 = ax.twinx()
    ax2.plot(soc_array[window], color="tab:blue", label="Filtered SOC")
    ax2.set_ylabel("SOC")

    ax.legend(loc='upper right')
    ax2.legend(loc='lower left')

    plt.title("Simulation of first-order Thevenin equivalent battery model")
    plt.show()


def battery_model():
    voltage_data = csv_to_timeseries_tuples('voltage.csv')
    current_data = csv_to_timeseries_tuples('current.csv')

    # This dataset has 0.1s period between measurements
    time_difference = 0.1

    current_raw = current_data[:, 1]
    current_error = np.polyval([-0.00388, 1547], current_raw * 1000.0)

    current = current_raw - (current_error / 1000)
    voltage = voltage_data[:, 1]

    energy_array = current * voltage * time_difference

    model_config: BatteryModelConfig = load_battery_config(pathlib.Path(__file__).parent / 'battery_config.toml')

    battery_model = EquivalentCircuitBatteryModel(model_config, state_of_charge=1.04)

    soc_array, predicted_ut_array = battery_model.update_array(tick=time_difference, current_array=np.array(-current, dtype=float))
    # soc_array, predicted_ut_array = battery_model.update_array(tick=time_difference, delta_energy_array=np.array(-energy_array, dtype=float))

    # example usage
    plot_results(soc_array, predicted_ut_array, voltage)


if __name__ == '__main__':
    battery_model()
