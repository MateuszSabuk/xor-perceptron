use nalgebra::{DMatrix, DVector};
use plotters::prelude::*;

fn step_function(x: f32) -> f32 {
    return if x >= 0.0 {1.0} else {0.0};
}

fn relu(x:f32) -> f32 {
    return if x >= 0.0 {x} else {0.0};
}


pub struct Perceptron {
    layers: Vec<Layer>
}

impl Perceptron {
    fn train(&mut self, x: DMatrix<f32>, y:DMatrix<f32>, learning_rate:f32, epochs:usize) -> Vec<f32> {
        // assert!(x.len() == y.len());
        let m:f32 = 1. / x.row(0).len() as f32;
        let mut total_errors = vec![];
        for _ in 0..epochs {
            let mut total_error = 0.0;
            for (i,col) in x.column_iter().enumerate() {
                // Forward pass
                let mut linears: Vec<DVector<f32>> = vec![col.clone_owned()];
                let mut vals: Vec<DVector<f32>> = vec![col.clone_owned()];
                for layer in &mut self.layers {
                    linears.push(layer.process(vals.last().unwrap().clone()));
                    vals.push(layer.activate(linears.last().unwrap().clone()))
                }
                let mut error:DVector<f32> =  y.column(i) - vals.last().unwrap();
                total_error += error.map(|x| x.powf(2.0)).sum();
                
                // Backward Propagation
                for (ln, layer) in self.layers.iter_mut().enumerate().rev() {
                    error = layer.update(&error, &vals[ln], &linears[ln] ,learning_rate, m);
                }
            }
            total_errors.push(total_error / y.len() as f32);
        }
        return total_errors;
    }
}

fn create_perceptron (inputs:usize, outputs:usize) -> Perceptron {
    // TODO: more control over creation of perceptron
    let layer1: Layer = create_layer(4, inputs, relu);
    let layer2: Layer = create_layer(outputs, 4, relu);

    return Perceptron {
        layers: vec![layer1, layer2],
    };
}

pub struct Layer {
    weights: DMatrix<f32>,
    biases: DVector<f32>,
    activation: fn(f32)->f32,
}

impl Layer {
    
    fn process(&self, inputs:DVector<f32>) -> DVector<f32> {
        return &self.weights*&inputs + &self.biases;
    }

    fn activate(&self, inputs:DVector<f32>) -> DVector<f32> {
        return inputs.map(self.activation);
    } 

    fn update(&mut self, error:&DVector<f32>, vals:&DVector<f32>, linear:&DVector<f32>, learning_rate:f32, m:f32) -> DVector<f32> {
        let next_error = self.weights.transpose() * error;
  
        if self.activation == relu {
            let _ = next_error.component_mul(&linear.clone().map(step_function));
        }

        self.weights += m * learning_rate * error * vals.transpose();
        self.biases += m * learning_rate * error;
        return next_error;
    }
}

fn create_layer (size:usize, input_size:usize, activation: fn(f32)->f32) -> Layer {
    return Layer{
        weights: DMatrix::new_random(size, input_size),
        biases: DVector::new_random(size),
        activation
    };
}

fn display_plot(data: &[f32]) -> Result<(), Box<dyn std::error::Error>> {
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

fn main() {
    let x:DMatrix<f32> = DMatrix::from_row_slice(2, 4, &[
        0.0, 1.0, 0.0, 1.0,
        0.0, 0.0, 1.0, 1.0
    ]);
    let y:DMatrix<f32> = DMatrix::from_row_slice(2, 4, &[
        0.0, 1.0, 1.0, 0.0,
        1.0, 0.0, 0.0, 1.0
    ]);
    let mut perceptron = create_perceptron(x.column(0).len(), y.column(0).len());
    let learning_rate = 0.1;
    let epochs = 1000;
    let error_vec = perceptron.train(x, y, learning_rate, epochs);

    let _ = display_plot(&error_vec);
}
