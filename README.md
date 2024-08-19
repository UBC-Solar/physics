# UBC Solar Physics

<!-- marker-index-start -->

[![Documentation Status](https://readthedocs.org/projects/ubc-solar-physics/badge/?version=latest)](https://ubc-solar-physics.readthedocs.io/en/latest/?badge=latest)

UBC Solar's physics and environment models for simulating our groundbreaking solar cars.

The API is currently unstable, and backwards compatibility may not be maintained. 

## Requirements

Versions indicated are recommended

* Git [^1]
* Python >=3.9 [^2]
* Rustc >=1.79.0 [^3]
* Cargo >=1.79.0 [^4]

## Installation

First, clone this repository.

```bash
git clone https://github.com/UBC-Solar/physics.git
```
Then, create and activate a virtual environment.
Next, install dependencies in editable mode.

```bash
pip3 install -e .
```

## Getting Started

Example of calculating solar arrays produced energy

```python
from physics.models.arrays import BasicArray
import numpy as np

efficiency = 0.25  # 25.0% efficient
panel_size = 4.0   # 4.0m^2 of panels
tick = 1.0         # 1.0s interval

arrays = BasicArray(panel_efficiency=efficiency, panel_size=panel_size)

irradiance = np.full([5], 400.0)  # 10 seconds of 400.0W/m^2 irradiance

solar_power_produced = arrays.calculate_produced_energy(solar_irradiance=irradiance, tick=tick)

assert np.array_equal(solar_power_produced, np.array([400.0, 400.0, 400.0, 400.0, 400.0]))
```

## Appendix

[^1]: use `git --version` to verify version

[^2]: use `python3 --version` to verify version

[^3]: use `rustc --version` to verify version

[^4]: use `cargo --version` to verify version

<!-- marker-index-end -->