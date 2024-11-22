from physics.models.battery import BatteryModelConfig, load_battery_config
from physics.models.battery.battery_model import BatteryModel
import pathlib
import numpy as np
import matplotlib.pyplot as plt


if __name__ == "__main__":
    config_file = pathlib.Path(__file__).parent / "battery_config.toml"
    battery_config = load_battery_config(str(config_file))

    battery_model = BatteryModel(battery_config)

    hppc_pulse = np.concat((
        np.full(100, fill_value=0.0),
        np.full(10, fill_value=-80.0),
        np.full(360, fill_value=0.0),
        np.full(10, fill_value=20.0),
        np.full(60, fill_value=0.0),
        np.full(360, fill_value=-80.0),
        np.full(3600, fill_value=0.0),
    ))

    power_array = np.tile(hppc_pulse, 10)

    soc, voltage = battery_model.update_array(power_array, 1.0, rust=False)

    fig, ax = plt.subplots()

    ax2 = ax.twinx()

    ax.plot(voltage)
    # ax2.plot(hppc_pulse)
    plt.savefig("data.png")
    plt.show()