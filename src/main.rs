mod helpers;
use helpers::{get_data_from_csv, display_plot, scatter_plot};
use nalgebra::{DMatrix, DVector, OMatrix, Dyn, U1};
use crate::ActivationFunction::{ReLU, Step};

#[derive(Copy, Clone, PartialEq)]
enum ActivationFunction {
    Step,
    ReLU,
}

impl ActivationFunction {
    fn apply(&self, x: f32) -> f32 {
        return match self {
            ActivationFunction::Step => {
                if x >= 0.0 { 1.0 } else { 0.0 }
            },
            ActivationFunction::ReLU => {
                if x >= 0.0 { x } else { 0.0 }
            }
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
        }
    }
}

#[derive(Clone)]
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
    pub fn train(&mut self, x: DMatrix<f32>, y: DMatrix<f32>, learning_rate: f32, epochs: usize) -> Vec<f32> {
        assert_eq!(x.ncols(),y.ncols());
        let num_of_data_examples = x.ncols() as f32;

        let mut all_costs_vec = vec![];
        for ep in 0..epochs {
            let mut epoch_cost_vec = vec![];
            let (mut nabla_w, mut nabla_b) = self.get_weights_and_biases();

            // Iterate through the input-output data
            for (i, col) in x.column_iter().enumerate() {
                // Forward Pass
                let mut zs: Vec<DVector<f32>> = vec![col.clone_owned()];
                let mut activations: Vec<DVector<f32>> = vec![col.clone_owned()];
                for layer in &mut self.layers {
                    zs.push(layer.process(activations.last().unwrap().clone()));
                    activations.push(layer.activate(zs.last().unwrap().clone()))
                }
                // Evaluate cost of last layer neuron
                epoch_cost_vec.push((activations.last().unwrap() - y.column(i)).map(|x|x.powf(2f32)).sum());

                let delta: DVector<f32> = (activations.last().unwrap() - y.column(i))
                    .map(|x|self.layers.last().unwrap().activation.prime(x));

                let (mut w_delta, mut b_delta) = self.get_weights_and_biases();
                b_delta.pop();
                b_delta.push(delta.clone());
                w_delta.pop();
                w_delta.push(delta.clone() * (activations.get(activations.len()-2).unwrap().transpose()));

                let (delta_nabla_w, delta_nabla_b) = self.backpropagate((w_delta,b_delta), &zs, &activations, &delta);
                for (i,n) in nabla_w.iter_mut().enumerate() {
                    *n += delta_nabla_w[i].clone();
                }
                for (i,n) in nabla_b.iter_mut().enumerate() {
                    *n += delta_nabla_b[i].clone();
                }
            }
            let m: f32 = learning_rate / num_of_data_examples;
            for (i, layer) in self.layers.iter_mut().enumerate() {
                layer.weights -= m*&nabla_w[i];
                layer.biases -= m*&nabla_b[i];
            }

            let sum_of_costs: f32 = epoch_cost_vec.iter().sum();
            all_costs_vec.push(sum_of_costs/num_of_data_examples);
            // Display status every 10% done
            if (ep % (epochs/10)) == 0 {
                print!("Training epoch {}/{}, ", ep, epochs);
                println!("MSE: {},", all_costs_vec.last().unwrap());
            }
        }
        return all_costs_vec;
    }

    fn backpropagate(&mut self, (mut w_delta,mut b_delta): (Vec<OMatrix<f32, Dyn, Dyn>>, Vec<OMatrix<f32, Dyn, U1>>), zs: &Vec<DVector<f32>>, activations: &Vec<DVector<f32>>, delta: &DVector<f32>) -> (Vec<OMatrix<f32, Dyn, Dyn>>, Vec<OMatrix<f32, Dyn, U1>>) {
        let mut d:Vec<DVector<f32>>= vec![delta.clone()];
        for i in (0..(self.layers.len()-1)).rev() {
            let ap = zs[i].map(|x|self.layers[i].activation.prime(x));
            d.push((self.layers[i].weights.clone() * ap) * delta);

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
    let mut perceptron = Perceptron::new(x.nrows(),vec![4,3,2],vec![ReLU,ReLU,Step]);
    let learning_rate = 0.001;
    let epochs = 1000;
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
