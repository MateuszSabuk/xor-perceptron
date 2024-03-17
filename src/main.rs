mod helpers;
use helpers::{get_data_from_csv, display_plot, scatter_plot};
use nalgebra::{DMatrix, DVector};

// Funkcja skokowa (step function) dla perceptronu
fn step_function(x: f32) -> f32 {
    return if x >= 0.0 { 1.0 } else { 0.0 };
}

// Funkcja ReLU
fn relu(x: f32) -> f32 {
    return if x >= 0.0 { x } else { 0.0 };
}

// Funkcja impulsowa (impulse function)
fn impulse(x: f32) -> f32 {
    return if x == 0.0 { 1.0 } else { 0.0 };
}

// Struktura Perceptron
pub struct Perceptron {
    layers: Vec<Layer>,
}

impl Perceptron {
    // Metoda trenowania perceptronu
    fn train(&mut self, x: DMatrix<f32>, y: DMatrix<f32>, learning_rate: f32, epochs: usize) -> Vec<f32> {
        let m: f32 = 1. / x.row(0).len() as f32;
        let mut total_errors = vec![];
        for ep in 0..epochs {
            let mut total_error = 0.0;
            for (i, col) in x.column_iter().enumerate() {
                // Przejście w przód (forward pass)
                let mut linears: Vec<DVector<f32>> = vec![col.clone_owned()];
                let mut vals: Vec<DVector<f32>> = vec![col.clone_owned()];
                for layer in &mut self.layers {
                    linears.push(layer.process(vals.last().unwrap().clone()));
                    vals.push(layer.activate(linears.last().unwrap().clone()))
                }
                let mut error: DVector<f32> = y.column(i) - vals.last().unwrap();
                total_error += error.map(|x| x.powf(2.0)).sum();

                // Propagacja wsteczna (backward propagation)
                for (ln, layer) in self.layers.iter_mut().enumerate().rev() {
                    error = layer.update(&error, &vals[ln], &linears[ln], learning_rate, m);
                }
            }
            total_errors.push(total_error / y.len() as f32);
            if ep % epochs/10 == 0 {
                print!("Training: {}%,",1000. * ep as f32/epochs as f32);
                println!("MSE: {},", total_errors.last().unwrap());
            }
            // Warunek zakończenia trenowania
            if 0.01 > total_error / y.len() as f32 {
                println!("MSE: {},", total_errors.last().unwrap());
                break;
            }
        }
        return total_errors;
    }

    // Metoda testowania perceptronu
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

// Tworzenie perceptronu
fn create_perceptron(inputs: usize, outputs: usize) -> Perceptron {
    let layer1: Layer = create_layer(4, inputs, relu);
    let layer2: Layer = create_layer(4, 4, relu);
    let layer3: Layer = create_layer(outputs, 4, relu);

    return Perceptron {
        layers: vec![layer1, layer2, layer3],
    };
}

// Struktura warstwy w perceptronie
pub struct Layer {
    weights: DMatrix<f32>,
    biases: DVector<f32>,
    activation: fn(f32) -> f32,
}

impl Layer {
    // Przetwarzanie danych przez warstwę
    fn process(&self, inputs: DVector<f32>) -> DVector<f32> {
        return &self.weights * &inputs + &self.biases;
    }

    // Aktywacja warstwy
    fn activate(&self, inputs: DVector<f32>) -> DVector<f32> {
        return inputs.map(self.activation);
    }

    // Aktualizacja wag i obliczenie błędu wstecznego
    fn update(&mut self, error: &DVector<f32>, vals: &DVector<f32>, linear: &DVector<f32>, learning_rate: f32, m: f32) -> DVector<f32> {
        let next_error = self.weights.transpose() * error;

        // Zastosowanie pochodnych funkcji aktywacji
        if self.activation == relu {
            let _ = next_error.component_mul(&linear.clone().map(step_function));
        } else if self.activation == step_function {
            let _ = next_error.component_mul(&linear.clone().map(impulse));
        }

        self.weights += m * learning_rate * error * vals.transpose();
        self.biases += m * learning_rate * error;
        return next_error;
    }
}

// Tworzenie warstwy perceptronu
fn create_layer(size: usize, input_size: usize, activation: fn(f32) -> f32) -> Layer {
    return Layer {
        weights: DMatrix::new_random(size, input_size),
        biases: DVector::new_random(size),
        activation,
    };
}

// Podział danych na zbiór treningowy i testowy
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
    let ((train_x,train_y),(test_x,test_y)) = divide_train_test(&x,&y,0.7);
    let mut perceptron = create_perceptron(x.nrows(), y.nrows());
    let learning_rate = 0.01;
    let epochs = 200;
    let error_vec = perceptron.train(train_x, train_y, learning_rate, epochs);

    let test_error_vec = perceptron.test(test_x.clone_owned(),test_y.clone_owned());

    let coords_x = test_x.row(0).clone().transpose();
    let coords_y = test_x.row(1).clone().transpose();
    let good = DMatrix::from_row_slice(test_error_vec.len(),1, &test_error_vec).map(|b| b<0.1);

    if let Err(err) = scatter_plot(&coords_x, &coords_y, &good) {
        eprintln!("Error plotting: {}", err);
    }
    let _ = display_plot(&error_vec);
}
