use std::f64;
use numpy::ndarray::{ArrayView1};

fn get_lookup_index(soc: f64, quantization_step: f64, num_indices: usize, min_soc: f64) -> usize {
    // Apply the same formula as in Python
    let index = ((soc - min_soc) / quantization_step).floor() as usize;

    // Clamp the index to be between 0 and num_indices - 1
    index.min(num_indices - 1)  // equivalent to max(0, min(num_indices - 1, index))
}

/// Evaluate a polynomial given coefficients and an input value (x)
fn evaluate_lookup(lookup: &[f64], quantization_step: f64, value: f64, min_soc: f64) -> f64 {
    let index = get_lookup_index(value, quantization_step, lookup.len(), min_soc);
    lookup[index]
}

/// Evolve the battery state for a single step
fn battery_evolve(
    current: f64,                  // Amperes
    tick: f64,                     // Seconds
    state_of_charge: f64,          // Dimensionless, 0 < SOC < 1
    polarization_potential: f64,   // Volts
    polarization_resistance: f64,  // Ohms
    internal_resistance: f64,      // Ohms
    open_circuit_voltage: f64,     // Volts
    time_constant: f64,            // Seconds
    nominal_charge_capacity: f64,  // Nominal charge capacity (Coulombs)
) -> (f64, f64, f64) {
    // Update state of charge and polarization potential
    let new_state_of_charge: f64 = state_of_charge + (current * tick / nominal_charge_capacity);
    let new_polarization_potential: f64 = f64::exp(-tick / time_constant) * polarization_potential
        + current * polarization_resistance * (1.0 - f64::exp(-tick / time_constant));
    let terminal_voltage: f64 = open_circuit_voltage + new_polarization_potential
        + (current * internal_resistance); // Terminal voltage

    (new_state_of_charge, new_polarization_potential, terminal_voltage)
}

// Update battery state, using either energy or current draw
pub fn update_battery_state(
    energy_or_current_array: ArrayView1<'_, f64>,  // Power (W*s) or current (Amperes)
    tick: f64,                                     // Seconds
    initial_state_of_charge: f64,                  // dimensionless, 0 < SOC < 1
    initial_polarization_potential: f64,           // Volts
    internal_resistance_lookup: ArrayView1<'_, f64>,// Coefficients for internal resistance
    open_circuit_voltage_lookup: ArrayView1<'_, f64>, // Coefficients for open-circuit voltage
    polarization_resistance_lookup: ArrayView1<'_, f64>, // Coefficients for polarization resistance
    capacitance_lookup: ArrayView1<'_, f64>,        // Coefficients for polarization capacitance
    nominal_charge_capacity: f64,                   // Coulombs
    is_energy_input: bool,                          // Whether the input is power or current,
    quantization_step: f64,                         // The quantization step size of SOC for lookup tables
    min_soc: f64,

) -> (Vec<f64>, Vec<f64>) {
    let mut state_of_charge: f64 = initial_state_of_charge;
    let mut polarization_potential: f64 = initial_polarization_potential;
    let mut soc_array: Vec<f64> = Vec::with_capacity(energy_or_current_array.len());
    let mut voltage_array: Vec<f64> = Vec::with_capacity(energy_or_current_array.len());

    for &input in energy_or_current_array.iter() {
        // Interpolate values from coefficient
        let open_circuit_voltage = evaluate_lookup(open_circuit_voltage_lookup.as_slice().unwrap(), quantization_step, state_of_charge, min_soc);
        let internal_resistance = evaluate_lookup(internal_resistance_lookup.as_slice().unwrap(), quantization_step, state_of_charge, min_soc);
        let polarization_resistance = evaluate_lookup(polarization_resistance_lookup.as_slice().unwrap(), quantization_step, state_of_charge, min_soc);
        let capacitance = evaluate_lookup(capacitance_lookup.as_slice().unwrap(), quantization_step, state_of_charge, min_soc);
        let time_constant = polarization_resistance * capacitance;

        // Calculate current from power or use the current directly
        let current: f64 = if is_energy_input {
            // Use the last voltage to calculate current, or an absurdly large number if it is the
            // first, because we don't know voltage yet, so we will have a very small initial
            // current, no matter what. We shouldn't be starting to simulate when the battery is
            // in an active state anyway, so this should be an alright compromise.
            input / (tick * voltage_array.last().unwrap_or(&10000.0)) // I = (E / dt) / V
        } else {
            input // Current is directly given in the current input array
        };

        let (new_state_of_charge, new_polarization_potential, terminal_voltage) = battery_evolve(
            current,
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
