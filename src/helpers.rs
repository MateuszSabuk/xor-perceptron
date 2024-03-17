use csv::ReaderBuilder;
use std::error::Error;
use std::fs::File;
use plotters::backend::BitMapBackend;
use plotters::chart::ChartBuilder;
use plotters::prelude::{BLUE, LineSeries, WHITE};
use nalgebra::{DMatrix};
use plotters::prelude::*;

pub fn get_data_from_csv(filename: &str) -> Result<(DMatrix<f32>, DMatrix<f32>), Box<dyn Error>> {
    let file = File::open(filename)?;

    let mut x_data = Vec::new();
    let mut y_data = Vec::new();

    for result in ReaderBuilder::new().has_headers(false).from_reader(file).records() {
        let mut is_x = true;
        let record = result?;
        let mut x_row: Vec<f32> = vec![];
        let mut y_row: Vec<f32> = vec![];
        for it in record.iter() {
            if it.is_empty() {
                is_x = false;
                continue;
            }
            if is_x {
                x_row.push(it.parse::<f32>().unwrap());
            } else {
                y_row.push(it.parse::<f32>().unwrap());
            }
        }
        x_data.push(x_row);
        y_data.push(y_row);
    }

    let x_matrix = DMatrix::from_vec(x_data.len(), x_data[0].len(), x_data.concat()).transpose();
    let y_matrix = DMatrix::from_vec(y_data.len(), y_data[0].len(), y_data.concat()).transpose();

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