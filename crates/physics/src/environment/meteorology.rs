use std::fmt;

use chrono::{DateTime, Datelike, Timelike};
use ndarray::{Array2, ArrayView1, ArrayView3, s};

/// Map cumulative distances to the index of the closest weather coordinate.
///
/// # Arguments
///
/// * `cumulative_distances`: The cumulative distance (m) travelled at each tick.
/// * `average_distances`: The distance (m) at which the vehicle becomes closer to each next
///   weather coordinate than the one before it.
///
/// Returns: The weather coordinate index at each tick.
pub fn closest_weather_indices(
    cumulative_distances: ArrayView1<'_, f64>,
    average_distances: ArrayView1<'_, f64>,
) -> Vec<i64> {
    let mut current_coord_index: usize = 0;
    let mut result: Vec<i64> = Vec::with_capacity(cumulative_distances.len());

    for &distance in cumulative_distances {
        current_coord_index = std::cmp::min(current_coord_index, average_distances.len() - 1);

        if distance > average_distances[current_coord_index] {
            current_coord_index += 1;

            current_coord_index = std::cmp::min(current_coord_index, average_distances.len() - 1);
        }

        result.push(current_coord_index as i64);
    }

    result
}

/// Why a weather forecast could not be sampled.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum WeatherError {
    /// The forecast has no coordinates or no times.
    EmptyForecast,
    /// `unix_timestamps` and `indices` must have one entry per tick each.
    LengthMismatch { timestamps: usize, indices: usize },
    /// The field said to hold forecast times does not exist.
    TimeFieldOutOfRange {
        time_field: usize,
        num_fields: usize,
    },
    /// A weather coordinate index is negative or past the last coordinate.
    CoordinateOutOfRange { index: i64, num_coordinates: usize },
}

impl fmt::Display for WeatherError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyForecast => write!(f, "weather forecast has no coordinates or no times"),
            Self::LengthMismatch {
                timestamps,
                indices,
            } => write!(
                f,
                "expected one weather coordinate index per timestamp, got {indices} indices for {timestamps} timestamps"
            ),
            Self::TimeFieldOutOfRange {
                time_field,
                num_fields,
            } => write!(
                f,
                "time field {time_field} is out of range for a forecast with {num_fields} fields"
            ),
            Self::CoordinateOutOfRange {
                index,
                num_coordinates,
            } => write!(
                f,
                "weather coordinate index {index} is out of range for a forecast with {num_coordinates} coordinates"
            ),
        }
    }
}

impl std::error::Error for WeatherError {}

/// Sample a weather forecast at the weather coordinate and time of each tick.
///
/// # Arguments
///
/// * `unix_timestamps`: The time of each tick.
/// * `indices`: The weather coordinate of each tick.
/// * `weather_forecast`: The forecast, shaped `[coordinate, time, field]`.
/// * `time_field`: The field holding each forecast's time. Times are read from coordinate 0 only
///   and truncated to whole seconds, so every coordinate must share the same forecast times.
///
/// Returns: One row per tick: all fields of the forecast at that tick's coordinate whose time is
/// closest to the tick's time (the earlier one on a tie).
pub fn weather_in_time(
    unix_timestamps: ArrayView1<'_, i64>,
    indices: ArrayView1<'_, i64>,
    weather_forecast: ArrayView3<'_, f64>,
    time_field: usize,
) -> Result<Array2<f64>, WeatherError> {
    let (num_coordinates, num_times, num_fields) = weather_forecast.dim();

    if num_coordinates == 0 || num_times == 0 {
        return Err(WeatherError::EmptyForecast);
    }
    if time_field >= num_fields {
        return Err(WeatherError::TimeFieldOutOfRange {
            time_field,
            num_fields,
        });
    }
    if unix_timestamps.len() != indices.len() {
        return Err(WeatherError::LengthMismatch {
            timestamps: unix_timestamps.len(),
            indices: indices.len(),
        });
    }
    if let Some(&index) = indices
        .iter()
        .find(|&&i| i < 0 || i as usize >= num_coordinates)
    {
        return Err(WeatherError::CoordinateOutOfRange {
            index,
            num_coordinates,
        });
    }

    let forecast_times: Vec<i64> = weather_forecast
        .slice(s![0, .., time_field])
        .iter()
        .map(|&time| time as i64)
        .collect();
    let closest_times = closest_timestamp_indices(unix_timestamps, &forecast_times);

    let mut weather = Array2::zeros((indices.len(), num_fields));
    for ((mut row, &coordinate), &time) in weather
        .rows_mut()
        .into_iter()
        .zip(indices)
        .zip(&closest_times)
    {
        row.assign(&weather_forecast.slice(s![coordinate as usize, time, ..]));
    }

    Ok(weather)
}

/// For each timestamp, find the index of the closest forecast time, choosing the lowest index
/// when several are equally close.
///
/// Uses a binary search when `forecast_times` is sorted (as forecasts are), and a linear scan
/// otherwise. `unix_timestamps` may be in any order.
///
/// # Panics
///
/// If `forecast_times` is empty.
pub fn closest_timestamp_indices(
    unix_timestamps: ArrayView1<'_, i64>,
    forecast_times: &[i64],
) -> Vec<usize> {
    assert!(
        !forecast_times.is_empty(),
        "forecast_times must not be empty"
    );

    if forecast_times.is_sorted() {
        unix_timestamps
            .iter()
            .map(|&timestamp| closest_in_sorted(forecast_times, timestamp))
            .collect()
    } else {
        unix_timestamps
            .iter()
            .map(|&timestamp| closest_in_unsorted(forecast_times, timestamp))
            .collect()
    }
}

/// Index of the time in sorted, non-empty `times` closest to `target` (lowest index on a tie).
fn closest_in_sorted(times: &[i64], target: i64) -> usize {
    // `upper` is the first time at or after `target`. Every time before it is earlier than `target`,
    // so it is also the first occurrence of its value.
    let upper = times.partition_point(|&time| time < target);
    if upper == 0 {
        return 0;
    }

    // The closest earlier time may be repeated; take its first occurrence.
    let lower_time = times[upper - 1];
    let lower = times.partition_point(|&time| time < lower_time);
    if upper == times.len() {
        return lower;
    }

    if target.abs_diff(lower_time) <= times[upper].abs_diff(target) {
        lower
    } else {
        upper
    }
}

/// Index of the time in non-empty `times` closest to `target` (lowest index on a tie).
fn closest_in_unsorted(times: &[i64], target: i64) -> usize {
    times
        .iter()
        .enumerate()
        .min_by_key(|&(_, &time)| target.abs_diff(time))
        .map(|(index, _)| index)
        .expect("times is not empty")
}

/// Convert unix timestamps (already shifted to local time) to the day of the year and the hour of
/// the day.
///
/// # Panics
///
/// If a timestamp is beyond the range chrono can represent.
pub fn calculate_array_ghi_times(local_times: ArrayView1<'_, u64>) -> (Vec<f64>, Vec<f64>) {
    local_times
        .iter()
        .map(|&unix_time_stamp| {
            let datetime = DateTime::from_timestamp(unix_time_stamp as i64, 0)
                .expect("timestamp out of range")
                .naive_utc();
            let day_of_year = datetime.date().ordinal() as f64;
            let local_time = datetime.time().num_seconds_from_midnight() as f64 / 3600.0;
            (day_of_year, local_time)
        })
        .unzip()
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{Array1, Array3, array};

    /// The original implementation: compare every timestamp against every forecast time.
    fn closest_brute_force(times: &[i64], target: i64) -> usize {
        let mut best = 0;
        for (index, &time) in times.iter().enumerate() {
            if target.abs_diff(time) < target.abs_diff(times[best]) {
                best = index;
            }
        }
        best
    }

    #[test]
    fn sorted_search_matches_brute_force() {
        let cases: [&[i64]; 5] = [
            &[100],
            &[0, 10, 20, 30],
            &[0, 10, 10, 10, 20, 30, 30],
            &[-50, -50, 0, 5, 5, 1000],
            &[7, 7, 7],
        ];
        for times in cases {
            for target in -100..1100 {
                assert_eq!(
                    closest_in_sorted(times, target),
                    closest_brute_force(times, target),
                    "times {times:?}, target {target}"
                );
            }
        }
    }

    #[test]
    fn ties_choose_earlier_time() {
        let times = [0, 10, 20];
        let targets = array![5, 15, 10];
        assert_eq!(
            closest_timestamp_indices(targets.view(), &times),
            vec![0, 1, 1]
        );
    }

    #[test]
    fn unsorted_times_fall_back_to_scan() {
        let times = [30, 0, 20, 10];
        let targets = array![-5, 4, 6, 26, 100];
        assert_eq!(
            closest_timestamp_indices(targets.view(), &times),
            vec![1, 1, 3, 0, 0]
        );
    }

    fn forecast() -> Array3<f64> {
        // 2 coordinates, 3 times, fields [time, value]
        let mut forecast = Array3::zeros((2, 3, 2));
        for coordinate in 0..2 {
            for time in 0..3 {
                forecast[[coordinate, time, 0]] = (time * 100) as f64;
                forecast[[coordinate, time, 1]] = (coordinate * 10 + time) as f64;
            }
        }
        forecast
    }

    #[test]
    fn samples_coordinate_and_closest_time() {
        let forecast = forecast();
        let timestamps = array![0, 140, 160, 1000];
        let indices = array![1, 0, 1, 0];
        let weather =
            weather_in_time(timestamps.view(), indices.view(), forecast.view(), 0).unwrap();
        assert_eq!(weather.column(1).to_vec(), vec![10.0, 1.0, 12.0, 2.0]);
    }

    #[test]
    fn rejects_invalid_inputs() {
        let forecast = forecast();
        let timestamps = array![0, 100];
        let run = |indices: Array1<i64>, time_field| {
            weather_in_time(
                timestamps.view(),
                indices.view(),
                forecast.view(),
                time_field,
            )
        };

        assert_eq!(
            run(array![0], 0),
            Err(WeatherError::LengthMismatch {
                timestamps: 2,
                indices: 1
            })
        );
        assert_eq!(
            run(array![0, 2], 0),
            Err(WeatherError::CoordinateOutOfRange {
                index: 2,
                num_coordinates: 2
            })
        );
        assert_eq!(
            run(array![-1, 0], 0),
            Err(WeatherError::CoordinateOutOfRange {
                index: -1,
                num_coordinates: 2
            })
        );
        assert_eq!(
            run(array![0, 0], 2),
            Err(WeatherError::TimeFieldOutOfRange {
                time_field: 2,
                num_fields: 2
            })
        );

        let empty = Array3::<f64>::zeros((1, 0, 2));
        assert_eq!(
            weather_in_time(timestamps.view(), array![0, 0].view(), empty.view(), 0),
            Err(WeatherError::EmptyForecast)
        );
    }
}
