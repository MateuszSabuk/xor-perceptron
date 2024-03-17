use csv::ReaderBuilder;
use std::error::Error;
use std::fs::File;
use plotters::backend::BitMapBackend;
use plotters::chart::ChartBuilder;
use plotters::prelude::{BLUE, LineSeries, WHITE};
use nalgebra::{DMatrix, OMatrix, Dyn, U1};
use plotters::prelude::*;

pub fn get_data_from_csv(filename: &str) -> Result<(DMatrix<f32>, DMatrix<f32>), Box<dyn Error>> {
    let file = File::open(filename)?;

    let mut x_data = Vec::new();
    let mut y_data = Vec::new();

    let mut num_of_records = 0;
    let mut x_length = 0;
    let mut y_length = 0;
        for result in ReaderBuilder::new().has_headers(false).from_reader(file).records() {
        let mut is_x = true;
        num_of_records += 1;
        let record = result?;
        for (i,it) in record.iter().enumerate() {
            if it.is_empty() {
                is_x = false;
                x_length = i;
                y_length = record.len() - i - 1;
                continue;
            }
            if is_x {
                x_data.push(it.parse::<f32>().unwrap());
            } else {
                y_data.push(it.parse::<f32>().unwrap());
            }
        }
    }

    let x_matrix = DMatrix::from_vec(x_length, num_of_records, x_data);
    let y_matrix = DMatrix::from_vec(y_length, num_of_records,  y_data);

    Ok((x_matrix,y_matrix))
}


pub fn display_plot(data: &[f32]) -> Result<(), Box<dyn std::error::Error>> {
    // Create a new drawing area
    let root = BitMapBackend::new("plot.png", (800, 600)).into_drawing_area();
    root.fill(&WHITE)?;

    // Define the range of values for x-axis and y-axis
    let x_min = 0.0;
    let x_max = data.len() as f32;
    let y_min = *data.iter().min_by(|a, b| a.partial_cmp(b).unwrap()).unwrap_or(&0.0);
    let y_max = *data.iter().max_by(|a, b| a.partial_cmp(b).unwrap()).unwrap_or(&0.0);

    // Create a new chart context
    let mut chart = ChartBuilder::on(&root)
        .x_label_area_size(40)
        .y_label_area_size(40)
        .margin(5)
        .caption("MSE", ("sans-serif", 20))
        .build_cartesian_2d(x_min..x_max, y_min..y_max)?;

    // Draw the data as a line plot
    chart
        .configure_mesh()
        .draw()?;
    chart.draw_series(LineSeries::new(
        data.iter().enumerate().map(|(x, y)| (x as f32, *y)),
        &BLUE,
    ))?;

    Ok(())
}

/// Display a scatter plot with points colored based on a vector of boolean values.
/// If the boolean value is true, the point will be plotted in blue; otherwise, it will be plotted in red.
pub fn scatter_plot(x_coords: &OMatrix<f32, Dyn, U1>, y_coords: &OMatrix<f32, Dyn, U1>, good: &OMatrix<bool, Dyn, Dyn>) -> Result<(), Box<dyn std::error::Error>> {
    // Create a new drawing area
    let root = BitMapBackend::new("scatter_plot.png", (800, 600)).into_drawing_area();
    root.fill(&WHITE)?;

    // Define the range of values for x-axis and y-axis
    let x_min = *x_coords.iter().min_by(|a, b| a.partial_cmp(b).unwrap()).unwrap_or(&0.0);
    let x_max = *x_coords.iter().max_by(|a, b| a.partial_cmp(b).unwrap()).unwrap_or(&0.0);
    let y_min = *y_coords.iter().min_by(|a, b| a.partial_cmp(b).unwrap()).unwrap_or(&0.0);
    let y_max = *y_coords.iter().max_by(|a, b| a.partial_cmp(b).unwrap()).unwrap_or(&0.0);

    // Create a new chart context
    let mut chart = ChartBuilder::on(&root)
        .x_label_area_size(40)
        .y_label_area_size(40)
        .margin(5)
        .caption("Scatter Plot", ("sans-serif", 20))
        .build_cartesian_2d(x_min..x_max, y_min..y_max)?;

    // Draw the background
    chart.configure_mesh().draw()?;

    // Draw the data points
    let mut data_good = vec![];
    let mut data_bad = vec![];
    for (i, _) in x_coords.iter().enumerate() {
        if good[i] {
            data_good.push((x_coords[i], y_coords[i]));
        } else {
            data_bad.push((x_coords[i], y_coords[i]))
        }
    }

    // Draw good points in blue
    chart.draw_series(
        data_good.iter().map(|point| Circle::new(*point, 5, &BLUE))
    )?;

    // Draw bad points in red
    chart.draw_series(
        data_bad.iter().map(|point| Circle::new(*point, 5, &RED))
    )?;

    Ok(())
}
