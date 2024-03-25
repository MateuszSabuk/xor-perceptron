mod helpers;

use std::iter::zip;
use helpers::{get_data_from_csv, display_plot, scatter_plot};
use nalgebra::{DMatrix, DVector, OMatrix, Dyn, U1, zero};
use crate::ActivationFunction::{ReLU, Step, Sigmoid};

#[derive(Copy, Clone, PartialEq)]
enum ActivationFunction {
    Step,
    ReLU,
    Sigmoid,
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
            ActivationFunction::Sigmoid => {
                1.0/(1.0+(-x).exp())
            },
        }
    }

    fn prime(&self, x: f32) -> f32 {
        return match self {
            ActivationFunction::Step => {
                if x == 0.0 { 1.0 } else { 0.0 } // Impulse
            },
            ActivationFunction::ReLU => {
                if x >= 0.0 { 1.0 } else { 0.0 } // Step
            },
            ActivationFunction::Sigmoid => { // Sigmoid prime
                self.apply(x) * (1.0 - self.apply(x))
            }
        }
    }
}

fn mae(predicted_vec: Vec<DVector<f32>>, true_mat: DMatrix<f32>) -> f32 {
    let predicted_mat = DMatrix::from_columns(&predicted_vec);
    let absolute = (predicted_mat.clone() - true_mat.clone()).abs();
    println!("{}",predicted_mat.columns(0,4));
    return absolute.column_mean().mean();
}

#[derive(Clone)]
pub struct Perceptron {
    layers: Vec<Layer>
}

impl Perceptron {
    /// Creates a new perceptron \
    /// **Length of both argument vectors should be the same!**
    /// # Arguments
    /// * `num_of_inputs` - Number of inputs for the first layer
    /// * `neuron_num_vec`: Vector of numbers of neurons for each consecutive layer \
    ///     last number is the number of outputs of the perceptron
    /// * `activation_fn_vec`: Vector of references to activation functions for each consecutive layer
    fn new(num_of_inputs: usize, mut neuron_num_vec: Vec<usize>, activation_fn_vec: Vec<ActivationFunction>) -> Perceptron {
        assert_eq!(neuron_num_vec.len(), activation_fn_vec.len());
        assert!(neuron_num_vec.len() > 0);

        neuron_num_vec.insert(0,num_of_inputs);
        let mut layers: Vec<Layer> = vec![];
        for i in 1..neuron_num_vec.len() {
            let layer: Layer = Layer::new(neuron_num_vec[i], neuron_num_vec[i-1], activation_fn_vec[i-1]);
            layers.push(layer);
        }



        return Perceptron {
            layers
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
    pub fn train(&mut self, x: DMatrix<f32>, y: DMatrix<f32>, learning_rate: f32, epochs: usize, (test_x, test_y): (DMatrix<f32>,DMatrix<f32>)) -> Vec<f32> {
        assert_eq!(x.ncols(),y.ncols());
        let num_of_data_examples = x.ncols() as f32;

        let mut all_costs_vec = vec![];
        for ep in 0..epochs {
            let mut epoch_predicted_vec = vec![];
            let (mut nabla_w, mut nabla_b) = self.get_weights_and_biases();
            for i in 0..nabla_w.len() {
                nabla_w[i] = nabla_w[i].map(|_|0f32);
            }
            for i in 0..nabla_b.len() {
                nabla_b[i] = nabla_b[i].map(|_|0f32);
            }

            let mut mini_batches = vec![];
            for i in (20..x.ncols()).step_by(20) {
                let b_x = x.columns(i - 20, 20).clone_owned();
                let b_y = y.columns(i - 20, 20).clone_owned();

                mini_batches.push((b_x, b_y));
            }

            for batch in mini_batches {
                // Iterate through the input-output data
                for (batch_x, batch_y) in zip(batch.0.column_iter(),batch.1.column_iter()) {
                    // Forward Pass
                    let mut zs: Vec<DVector<f32>> = vec![];
                    let mut activations: Vec<DVector<f32>> = vec![batch_x.clone_owned()];
                    for layer in &mut self.layers {
                        zs.push(layer.process(activations.last().unwrap().clone()));
                        activations.push(layer.activate(zs.last().unwrap().clone()))
                    }

                    // Evaluate cost of last layer neuron
                    epoch_predicted_vec.push(activations.last().unwrap().clone());

                    let diff = (activations.last().unwrap() - batch_y);
                    let prime = zs.last().unwrap().map(|x| self.layers.last().unwrap().activation.prime(x));
                    let delta: DVector<f32> = diff.component_mul(&prime);

                    let (mut w_delta, mut b_delta) = self.get_weights_and_biases();
                    b_delta.pop();
                    b_delta.push(delta.clone());
                    w_delta.pop();
                    w_delta.push(delta.clone() * (activations.get(activations.len()-2).unwrap().transpose()));

                    let (delta_nabla_w, delta_nabla_b) = self.backpropagate((w_delta,b_delta), &zs, &activations);
                    for (i,n) in nabla_w.iter_mut().enumerate() {
                        *n += delta_nabla_w[i].clone();
                    }
                    for (i,n) in nabla_b.iter_mut().enumerate() {
                        *n += delta_nabla_b[i].clone();
                    }
                }
                let m: f32 = learning_rate / num_of_data_examples;
                for (i, layer) in self.layers.iter_mut().enumerate() {
                    layer.weights -= m * &nabla_w[i];
                    layer.biases -= m * &nabla_b[i];
                }
            }

            println!("{ep}");
            self.evaluate((test_x.clone(),test_y.clone()));

            // all_costs_vec.push(mae(epoch_predicted_vec,batch_y.clone()));
            // Display status every 10% done
            if (ep % (epochs/10)) == 0 {
                print!("Training epoch {}/{}, ", ep, epochs);
                // println!("MSE: {},", all_costs_vec.last().unwrap());
            }
        }
        return all_costs_vec;
    }

    fn backpropagate(&mut self, (mut w_delta,mut b_delta): (Vec<OMatrix<f32, Dyn, Dyn>>, Vec<OMatrix<f32, Dyn, U1>>), zs: &Vec<DVector<f32>>, activations: &Vec<DVector<f32>>) -> (Vec<OMatrix<f32, Dyn, Dyn>>, Vec<OMatrix<f32, Dyn, U1>>) {
        let mut d:Vec<DVector<f32>>= vec![b_delta.last().unwrap().clone()];
        for i in (0..(self.layers.len()-1)).rev() {
            let ap = zs[i].map(|x|self.layers[i].activation.prime(x));
            d.push((self.layers[i+1].weights.clone().transpose() * d.last().unwrap()).component_mul(&ap));

            b_delta[i] = d.last().unwrap().clone_owned();
            w_delta[i] = d.last().unwrap().clone_owned() * activations[i].transpose();
        }
        let mut nabla_w = vec![];
        let mut nabla_b = vec![];
        for (i,_) in w_delta.iter().enumerate() {
            nabla_w.push(w_delta[i].clone());
            nabla_b.push(b_delta[i].clone());
        }
        return (nabla_w, nabla_b);
    }

    fn get_weights_and_biases(&self) -> (Vec<DMatrix<f32>>, Vec<DVector<f32>>) {
        let mut w: Vec<DMatrix<f32>> = vec![];
        let mut b: Vec<DVector<f32>> = vec![];
        for layer in &self.layers {
            w.push(layer.weights.clone_owned());
            b.push(layer.biases.clone_owned());
        }
        return (w,b)
    }

    fn evaluate(&self, (test_x, test_y): (DMatrix<f32>,DMatrix<f32>)) -> f32 {
        let mut temp = 0;
        for (x, y) in zip(test_x.column_iter(),test_y.column_iter()) {
            // Przejście w przód (forward pass)
            let mut vals: Vec<DVector<f32>> = vec![x.clone_owned()];
            for layer in &self.layers {
                vals.push(layer.activate(layer.process(vals.last().unwrap().clone())));
            }
            // if temp <4 {
            //     temp += 1;
            //     println!("y {} vals{}",y,vals.last().unwrap());
            // }
            let error: DVector<f32> = y - vals.last().unwrap();
        }
        return 0.0
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
#[derive(Clone)]
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
    let mut perceptron = Perceptron::new(x.nrows(),vec![4,4,2],vec![Sigmoid,Sigmoid,Sigmoid]);
    let learning_rate = 0.1;
    let epochs = 1000;
    let error_vec = perceptron.train(train_x, train_y, learning_rate, epochs, (test_x.clone(), test_y.clone()));

    let test_error_vec = perceptron.test(test_x.clone_owned(),test_y.clone_owned());

    let coords_x = test_x.row(0).clone().transpose();
    let coords_y = test_x.row(1).clone().transpose();
    let good = DMatrix::from_row_slice(test_error_vec.len(),1, &test_error_vec).map(|b| b<0.5);

    if let Err(err) = scatter_plot(&coords_x, &coords_y, &good) {
        eprintln!("Error plotting: {}", err);
    }
    let _ = display_plot(&error_vec);
}
