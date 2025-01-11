use std::f64;
use numpy::ndarray::ArrayViewD;

/// Evaluate a polynomial given coefficients and an input value (x)
fn evaluate_polynomial(coefficients: &[f64], x: f64) -> f64 {
    coefficients.iter().fold(0.0, |acc, &coeff| acc * x + coeff)
}

/// Evolve the battery state for a single step
fn battery_evolve(
    power: f64,                    // Watts
    tick: f64,                     // Seconds
    state_of_charge: f64,          // Dimensionless, 0 < SOC < 1
    polarization_potential: f64,   // Volts
    polarization_resistance: f64,  // Ohms
    internal_resistance: f64,      // Ohms
    open_circuit_voltage: f64,     // Volts
    time_constant: f64,            // Seconds
    nominal_charge_capacity: f64,  // Nominal charge capacity (Coulombs)
) -> (f64, f64, f64) {
    // Compute current (I) based on power input/output
    let current: f64 = power / (open_circuit_voltage + polarization_potential + internal_resistance);

    // Update state of charge and polarization potential
    let new_state_of_charge: f64 = state_of_charge + (current * tick / nominal_charge_capacity);
    let new_polarization_potential: f64 = f64::exp(-tick / time_constant) * polarization_potential
        + current * polarization_resistance * (1.0 - f64::exp(-tick / time_constant));
    let terminal_voltage: f64 = open_circuit_voltage + new_polarization_potential
        + (current * internal_resistance); // Terminal voltage

    (new_state_of_charge, new_polarization_potential, terminal_voltage)
}

pub fn update_battery_array(
    delta_energy_array: ArrayViewD<'_, f64>,            // W*s
    tick: f64,                                          // Seconds
    initial_state_of_charge: f64,                       // dimensionless, 0 < SOC < 1
    initial_polarization_potential: f64,                // Volts
    polarization_resistance: f64,                       // Ohms
    internal_resistance_coeffs: ArrayViewD<'_, f64>,    // Coefficients for internal resistance
    open_circuit_voltage_coeffs: ArrayViewD<'_, f64>,   // Coefficients for open-circuit voltage
    time_constant: f64,                                 // Seconds 
    nominal_charge_capacity: f64,                       // Coulombs
) -> (Vec<f64>, Vec<f64>) {
    let mut state_of_charge: f64 = initial_state_of_charge; 
    let mut polarization_potential: f64 = initial_polarization_potential;
    let mut soc_array: Vec<f64> = Vec::with_capacity(delta_energy_array.len());
    let mut voltage_array: Vec<f64> = Vec::with_capacity(delta_energy_array.len());

    for &power in delta_energy_array.iter() {
        // Interpolate values from coefficient
        let open_circuit_voltage: f64 = evaluate_polynomial(open_circuit_voltage_coeffs.as_slice().unwrap(), state_of_charge);
        let internal_resistance: f64 = evaluate_polynomial(internal_resistance_coeffs.as_slice().unwrap(), state_of_charge);

        let (new_state_of_charge, new_polarization_potential, terminal_voltage) = battery_evolve(
            power,
            tick,
            state_of_charge,
            polarization_potential,
            polarization_resistance,
            internal_resistance,
            open_circuit_voltage,
            time_constant,
            nominal_charge_capacity,
        );

        // Update state for the next iteration
        state_of_charge = new_state_of_charge;
        polarization_potential = new_polarization_potential;

        // Store results
        soc_array.push(new_state_of_charge);
        voltage_array.push(terminal_voltage);
    }

    (soc_array, voltage_array)
}

