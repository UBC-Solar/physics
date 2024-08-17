use chrono::{Datelike, NaiveDateTime, Timelike};
use numpy::ndarray::{s, Array, Array2, ArrayViewD, ArrayViewMut2, ArrayViewMut3, Axis};

pub fn rust_calculate_array_ghi_times<'a>(
    local_times: ArrayViewD<'_, u64>,
) -> (Vec<f64>, Vec<f64>) {
    let mut datetimes: Vec<_> = Vec::with_capacity(local_times.len());

    for &unix_time_stamp in local_times {
        let datetime = NaiveDateTime::from_timestamp_opt(unix_time_stamp as i64, 0).unwrap();
        datetimes.push(datetime);
    }

    let day_of_year_out: Vec<f64> = datetimes
        .iter()
        .map(|&date| date.date().ordinal() as f64)
        .collect();
    let local_time_out: Vec<f64> = datetimes
        .iter()
        .map(|&date| date.time().num_seconds_from_midnight() as f64 / 3600.0)
        .collect();

    (day_of_year_out, local_time_out)
}
