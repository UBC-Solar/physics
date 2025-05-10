use numpy::ndarray::{ArrayViewD, ArrayView1};

pub fn rust_closest_gis_indices_loop(
    distances: ArrayViewD<'_, f64>,
    path_distances: ArrayViewD<'_, f64>,
) -> Vec<i64> {
    let mut current_coord_index: usize = 0;
    let mut distance_travelled: f64 = 0.0;
    let mut result: Vec<i64> = Vec::with_capacity(distances.len());

    for &distance in distances {
        distance_travelled += distance;

        while distance_travelled > path_distances[current_coord_index] {
            distance_travelled -= path_distances[current_coord_index];
            current_coord_index += 1;
            if current_coord_index >= path_distances.len() {
                current_coord_index = 0;
            }
        }

        current_coord_index = std::cmp::min(current_coord_index, path_distances.len() - 1);
        result.push(current_coord_index as i64);
    }

    result
}

///
/// Generate valid driving speeds as a simulation-time array given a set of average speeds for each
/// simulated lap.
/// Driving speeds will only be non-zero when we are allowed to drive, and the speed
/// for every tick during a lap will be that lap's corresponding desired average speed for as long
/// as it takes to complete the lap.
/// An average speed of 0m/s for a lap will be interpreted as "sit and charge" for `idle_time`
/// ticks.
///
/// # Arguments
///
/// * `average_speeds`: An array of average speeds in m/s, one for each simulated lap. If there are more
/// speeds given than laps available, the unused speeds will be silently ignored. If there are too
/// few, an error will be returned.
/// * `simulation_dt`: The simulated tick length
/// * `driving_allowed_boolean`: A simulation-time boolean where the `True` elements are when we
/// are allowed to drive, and `False` is when we are not.
/// * `track_length`: The length of the track in meters.
/// * `idle_time`: The number of ticks to "sit and charge" when desired.
///
/// Returns: A simulation-time array of driving speeds in m/s, or an error if there weren't enough
/// laps provided to fill the entire simulation time.
///
pub fn get_driving_speeds(
    average_speeds: ArrayView1<'_, f64>,            // Average speeds in m/s
    simulation_dt: i64,                             // Time step in seconds
    driving_allowed_boolean: ArrayView1<'_, bool>,  // Simulation-time boolean array
    track_length: f64,                              // Track length in meters
    idle_time: i64                                  // Time to idle in seconds
) -> Result<Vec<f64>, &'static str> {
    let ticks_to_complete_lap: Vec<i64> = average_speeds.iter().map(| &average_speed | {
        if average_speed > 0.0 {
            // The number of ticks is the number of seconds, divided by seconds per tick
            (track_length / average_speed / simulation_dt as f64).ceil() as i64
        } else {
            (idle_time as f64 / simulation_dt as f64).ceil() as i64
        }
    }).collect();

    let mut lap_index: usize = 0;
    let mut lap_speed: f64 = average_speeds[lap_index];

    let mut ticks_to_lap_completion: i64 = ticks_to_complete_lap[lap_index];

    let mut driving_speeds: Vec<f64> = Vec::with_capacity(driving_allowed_boolean.len());
    for driving_allowed in driving_allowed_boolean.iter() {
        if !driving_allowed {
            // If we aren't allowed to drive, speed should be zero. Also, we should mark that we are
            // done our lap since it means we ended the day in the middle of the lap, and we will
            // start the next day at the beginning of a new lap, not where we ended off.

            // If it's the first lap, we don't want to skip because we are probably in the morning
            // where we haven't begun driving yet.
            if lap_index > 0 {
                ticks_to_lap_completion = 0;
            }

            driving_speeds.push(0.0)
        } else {
            // If we are driving, we should decrement ticks to lap completion. If its already
            // zero, that means that we are done the lap and should move onto the next lap.
            if ticks_to_lap_completion > 0 {
                ticks_to_lap_completion -= 1;

                driving_speeds.push(lap_speed)
            } else {
                // To advance to the next lap, increment the index and evaluate new variables
                lap_index += 1;
                if lap_index >= average_speeds.len() {
                    return Err("Not enough average speeds!")
                }

                // We subtract 1 since this iteration counts for the next lap, not the one
                // that we just finished
                ticks_to_lap_completion = ticks_to_complete_lap[lap_index] - 1;
                lap_speed = average_speeds[lap_index];

                driving_speeds.push(lap_speed)
            }
        }

    }

    Ok(driving_speeds)
}
