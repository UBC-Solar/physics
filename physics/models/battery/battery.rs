use std::f64;
use numpy::ndarray::ArrayViewD;

/// Evaluate a polynomial given coefficients and an input value (x)
fn evaluate_polynomial(coefficients: &[f64], x: f64) -> f64 {
    coefficients.iter().rev().fold(0.0, |acc, &coeff| acc * x + coeff)
}

/// Evolve the battery state for a single step
fn rust_battery_evolve(
    power: f64,                    // Power input/output (W)
    tick: f64,                     // Time step (s)
    state_of_charge: f64,          // State of charge (dimensionless, 0 < SOC < 1)
    polarization_potential: f64,   // Polarization potential (V)
    polarization_resistance: f64,  // Polarization resistance (Ohms)
    internal_resistance: f64,      // Internal resistance (Ohms)
    open_circuit_voltage: f64,     // Open-circuit voltage (V)
    time_constant: f64,            // Time constant
    nominal_charge_capacity: f64,  // Nominal charge capacity (Coulombs)
) -> (f64, f64, f64) {
    // Compute current (I) based on power input/output
    let current: f64 = if power <= 0.0 {
        power / (open_circuit_voltage + polarization_potential + internal_resistance) // Discharge current
    } else {
        power / (open_circuit_voltage + polarization_potential + internal_resistance) // Charge current
    };

    // Update state of charge and polarization potential
    let new_state_of_charge: f64 = state_of_charge + (current * tick / nominal_charge_capacity);
    let new_polarization_potential: f64 = f64::exp(-tick / time_constant) * polarization_potential
        + current * polarization_resistance * (1.0 - f64::exp(-tick / time_constant));
    let terminal_voltage: f64 = open_circuit_voltage + new_polarization_potential
        + (current * internal_resistance); // Terminal voltage

    (new_state_of_charge, new_polarization_potential, terminal_voltage)
}

pub fn rust_update_battery_array(
    delta_energy_array: ArrayViewD<'_, f64>, // Array of energy changes (W*s)
    tick: f64,                               // Time step (s)
    initial_state_of_charge: f64,            // Initial state of charge (dimensionless, 0 < SOC < 1)
    initial_polarization_potential: f64,     // Initial polarization potential (V)
    polarization_resistance: f64,            // Polarization resistance (Ohms)
    internal_resistance_coeffs: ArrayViewD<'_, f64>,  // Polynomial coefficients for internal resistance
    open_circuit_voltage_coeffs: ArrayViewD<'_, f64>, // Polynomial coefficients for open-circuit voltage
    time_constant: f64,                      // Unitless 
    nominal_charge_capacity: f64,            // Coulombs
) -> (Vec<f64>, Vec<f64>) {
    let mut state_of_charge: f64 = initial_state_of_charge; // Track SOC
    let mut polarization_potential: f64 = initial_polarization_potential; // Track polarization potential
    let mut soc_array: Vec<f64> = Vec::with_capacity(delta_energy_array.len());
    let mut voltage_array: Vec<f64> = Vec::with_capacity(delta_energy_array.len());

    // Iterate over the energy array to update SOC and voltage step by step
    for &power in delta_energy_array.iter() {
        // Evaluate polynomials for U_oc and R_0
        let open_circuit_voltage: f64 = evaluate_polynomial(open_circuit_voltage_coeffs.as_slice().unwrap(), state_of_charge);
        let internal_resistance: f64 = evaluate_polynomial(internal_resistance_coeffs.as_slice().unwrap(), state_of_charge);

        // Call evolve function with interpolated values
        let (new_state_of_charge, new_polarization_potential, terminal_voltage) = rust_battery_evolve(
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

