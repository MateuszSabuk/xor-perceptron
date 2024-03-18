mod helpers;
use helpers::{get_data_from_csv, display_plot, scatter_plot};
use nalgebra::{DMatrix, DVector};
use crate::ActivationFunction::ReLU;

#[derive(Copy, Clone, PartialEq)]
enum ActivationFunction {
    Step,
    ReLU,
    Impulse,
}

impl ActivationFunction {
    fn apply(&self, x: f32) -> f32 {
        return match self {
            ActivationFunction::Step => {
                if x >= 0.0 { 1.0 } else { 0.0 }
            },
            ActivationFunction::ReLU => {
                if x >= 0.0 { x } else { 0.0 }
            },
            ActivationFunction::Impulse => {
                if x == 0.0 { 1.0 } else { 0.0 }
            },
        }
    }
}

pub struct Perceptron {
    layers: Vec<Layer>,
}

impl Perceptron {
    /// Creates a new perceptron \
    /// **Length of both argument vectors should be the same!**
    /// # Arguments
    /// * `num_of_inputs` - Number of inputs for the first layer
    /// * `neuron_num_vec`: Vector of numbers of neurons for each consecutive layer \
    ///     last number is the number of outputs of the perceptron
    /// * `activation_fn_vec`: Vector of references to activation functions for each consecutive layer
    pub fn new(num_of_inputs: usize, mut neuron_num_vec: Vec<usize>, activation_fn_vec: Vec<ActivationFunction>) -> Perceptron {
        assert_eq!(neuron_num_vec.len(), activation_fn_vec.len());
        assert!(neuron_num_vec.len() > 0);

        neuron_num_vec.insert(0,num_of_inputs);
        let mut layers: Vec<Layer> = vec![];
        for i in 1..neuron_num_vec.len() {
            let layer: Layer = Layer::new(neuron_num_vec[i], neuron_num_vec[i-1], activation_fn_vec[i-1]);
            layers.push(layer);
        }

        return Perceptron {
            layers,
        };
    }

    /// Trains the perceptron with x and y data
    /// # Arguments
    /// * `x` - Matrix of f32 inputs with size (m x n) where:
    ///     - m - num of inputs
    ///     - n - length of training data sets
    /// * `y` - Matrix (m x n) of f32 right outputs corresponding to x training data
    ///     - m - num of outputs
    ///     - n - length of training output data sets
    /// * `learning_rate` - Parameter for the speed of learning
    /// * `epochs` - Number of training iterations
    pub fn train(&mut self, x: DMatrix<f32>, y: DMatrix<f32>, learning_rate: f32, epochs: usize) -> Vec<f32> {
        assert_eq!(x.ncols(),y.ncols());

        let m: f32 = 1f32 / x.ncols() as f32;
        let mut total_errors = vec![];
        for ep in 0..epochs {
            let mut total_error = 0.0;
            for (i, col) in x.column_iter().enumerate() {
                // Forward Pass
                let mut linears: Vec<DVector<f32>> = vec![col.clone_owned()];
                let mut vals: Vec<DVector<f32>> = vec![col.clone_owned()];
                for layer in &mut self.layers {
                    linears.push(layer.process(vals.last().unwrap().clone()));
                    vals.push(layer.activate(linears.last().unwrap().clone()))
                }

                // Evaluate error
                let mut error: DVector<f32> = y.column(i) - vals.last().unwrap();
                total_error += error.map(|x| x.powf(2.0)).sum();

                // Backpropagation
                for (ln, layer) in self.layers.iter_mut().enumerate().rev() {
                    error = layer.update(&error, &vals[ln], &linears[ln], learning_rate, m);
                }
            }
            total_errors.push(total_error / (y.ncols() as f32));

            // Display status every 10% done
            if (ep % (epochs/10)) == 0 {
                print!("Training epoch {}/{}, ", ep, epochs);
                println!("MSE: {},", total_errors.last().unwrap());
            }
            // Training stop condition if total_error smaller than threshold
            if 0.01 > total_error / y.len() as f32 {
                println!("MSE: {},", total_errors.last().unwrap());
                break;
            }
        }
        return total_errors;
    }

    // Test perceptron
    fn test(&self, x_test: DMatrix<f32>, y_test: DMatrix<f32>) -> Vec<f32> {
        let mut total_error = vec![];
        for (i, col) in x_test.column_iter().enumerate() {
            // Przejście w przód (forward pass)
            let mut vals: Vec<DVector<f32>> = vec![col.clone_owned()];
            for layer in &self.layers {
                vals.push(layer.process(vals.last().unwrap().clone()));
            }

            let error: DVector<f32> = y_test.column(i) - vals.last().unwrap();
            total_error.push(error.map(|x| x.powf(2.0)).sum());
        }
        return total_error;
    }
}

// Structure of a single perceptron layer
pub struct Layer {
    weights: DMatrix<f32>,
    biases: DVector<f32>,
    activation: ActivationFunction,
}

impl Layer {
    // Create new Layer
    fn new(size: usize, input_size: usize, activation: ActivationFunction) -> Layer {
        return Layer {
            weights: DMatrix::new_random(size, input_size),
            biases: DVector::new_random(size),
            activation,
        };
    }

    // Forward process the data by the layer
    fn process(&self, inputs: DVector<f32>) -> DVector<f32> {
        return &self.weights * &inputs + &self.biases;
    }

    // Run activation function for each of the input values
    fn activate(&self, inputs: DVector<f32>) -> DVector<f32> {
        return inputs.map(|x| self.activation.apply(x));
    }

    // Weights and biases update
    fn update(&mut self, error: &DVector<f32>, vals: &DVector<f32>, linear: &DVector<f32>, learning_rate: f32, m: f32) -> DVector<f32> {
        let next_error = self.weights.transpose() * error;

        // Use of the derivative functions
        if self.activation == ActivationFunction::ReLU {
            let _ = next_error.component_mul(&linear.clone().map(|x| ActivationFunction::Step.apply(x)));
        } else if self.activation == ActivationFunction::Step {
            let _ = next_error.component_mul(&linear.clone().map(|x| ActivationFunction::Impulse.apply(x)));
        }

        self.weights += m * learning_rate * error * vals.transpose();
        self.biases += m * learning_rate * error;
        return next_error;
    }
}


// Divide into training and testing datasets
fn divide_train_test(x: &DMatrix<f32>, y: &DMatrix<f32>, proportion: f32) -> ((DMatrix<f32>, DMatrix<f32>), (DMatrix<f32>, DMatrix<f32>)) {
    let length = x.ncols() as f32;
    let test_num = (length * proportion) as usize;
    let train_x = x.columns(0, test_num).clone_owned();
    let train_y = y.columns(0, test_num).clone_owned();
    let test_x = x.columns(test_num, x.ncols() - test_num).clone_owned();
    let test_y = y.columns(test_num, y.ncols() - test_num).clone_owned();
    return ((train_x, train_y), (test_x, test_y));
}

fn main() {
    let (x,y) = get_data_from_csv("inputs_outputs.csv").unwrap();
    let ((train_x,train_y),(test_x,test_y)) = divide_train_test(&x,&y,0.8);
    let mut perceptron = Perceptron::new(x.nrows(),vec![4,4,1],vec![ReLU,ReLU,ReLU]);
    let learning_rate = 0.01;
    let epochs = 200;
    let error_vec = perceptron.train(train_x, train_y, learning_rate, epochs);

    let test_error_vec = perceptron.test(test_x.clone_owned(),test_y.clone_owned());

    let coords_x = test_x.row(0).clone().transpose();
    let coords_y = test_x.row(1).clone().transpose();
    let good = DMatrix::from_row_slice(test_error_vec.len(),1, &test_error_vec).map(|b| b<0.5);

    if let Err(err) = scatter_plot(&coords_x, &coords_y, &good) {
        eprintln!("Error plotting: {}", err);
    }
    let _ = display_plot(&error_vec);
}
