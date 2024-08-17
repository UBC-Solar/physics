use chrono::{Datelike, NaiveDateTime, Timelike};
use numpy::ndarray::{s, Array, Array2, ArrayViewD, ArrayViewMut2, ArrayViewMut3, Axis};

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
        }

        current_coord_index = std::cmp::min(current_coord_index, path_distances.len() - 1);
        result.push(current_coord_index as i64);
    }

    result
}