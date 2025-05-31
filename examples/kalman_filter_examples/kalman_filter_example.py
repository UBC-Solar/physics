import numpy as np
import pandas as pd
import pathlib
import matplotlib.pyplot as plt
from physics.models.battery.kalman_filter import FilteredBatteryModel
from physics.models.battery.battery_config import BatteryModelConfig, load_battery_config, KalmanFilterConfig


# This test requires a voltage.csv and current.csv in the same directory to run
def csv_to_timeseries_tuples(csv_file):
    path = pathlib.Path(__file__).parent / csv_file
    df = pd.read_csv(path)
    df['Time'] = pd.to_datetime(df['Time'])
    return np.array(list(zip(df['Time'].dt.to_pydatetime(), df['Value'])))


def plot_kalman_results(measured_Ut, predicted_Ut_array, predicted_Uc_array, SOC_array, window=None):
    fig, ax = plt.subplots()

    if window is None:
        window = slice(0, len(predicted_Ut_array), 1)

    # ax.plot(predicted_Uoc_array[window], label="Predicted OCV", color="tab:cyan")
    ax.plot(predicted_Ut_array[window], label=r"Filtered Predicted $U_t$", color="orange")
    ax.plot(measured_Ut[window], label=r"Measured $U_t$", color="tab:red")
    ax.plot(predicted_Uc_array[window] + 100, label=r"Predicted $U_c$", color="magenta")
    ax.axhline(y=100, linestyle='dotted', color="magenta")
    # ax.set_ylim(75, 140)
    ax.set_xticks([])
    ax.set_ylabel("Voltage")

    ax2 = ax.twinx()
    ax2.plot(SOC_array[window], color="tab:cyan", label="Filtered SOC")
    ax2.set_ylabel("SOC")

    ax.legend(loc='upper right')
    ax2.legend(loc='lower left')

    plt.title("Impact of Kalman filtering on equivalent-circuit battery modeling ")
    plt.show()


def kalman_filter():
    
    voltage_data = csv_to_timeseries_tuples('voltage.csv')
    current_data = csv_to_timeseries_tuples('current.csv')

    model_config: BatteryModelConfig = load_battery_config(pathlib.Path(__file__).parent / 'battery_config.toml')

    filter_config: KalmanFilterConfig = KalmanFilterConfig(
        model_config,
        process_noise_matrix=np.diag([
            1e-10 * 0.1,
            1e-6 * 0.1
        ]),
        state_covariance_matrix=np.diag(
            [1e-2 * 0.5,
             1e-1]
        ),
        measurement_noise_vector=np.eye(1, dtype=float) * 1e0 * 0.5
    )

    ekf = FilteredBatteryModel(filter_config, initial_SOC=1.04, initial_Uc=0.0)

    SOC_array = np.zeros(len(voltage_data))
    Ut_array = np.zeros(len(voltage_data))
    current_array = np.zeros(len(voltage_data))
    predicted_Ut_array = np.zeros(len(voltage_data))
    predicted_Uc_array = np.zeros(len(voltage_data))
    predicted_Uoc_array = np.zeros(len(voltage_data))

    # This dataset has 0.1s period between measurements
    time_difference = 0.1

    for i in range(int(len(voltage_data))):
        # for i in range(20000):
        # Calculate time difference between current and previous measurements

        Ut = float(voltage_data[i][1])
        I = float(current_data[i][1])

        ekf.predict_then_update(Ut, I, time_difference)

        SOC_array[i] = ekf.SOC
        Ut_array[i] = Ut
        current_array[i] = I
        predicted_Ut_array[i] = ekf.Ut
        predicted_Uc_array[i] = ekf.Uc
        predicted_Uoc_array[i] = ekf._U_oc(ekf.SOC)

    # example usage
    plot_kalman_results(voltage_data[:, 1], predicted_Ut_array, predicted_Uc_array, SOC_array)


if __name__ == '__main__':
    kalman_filter()
