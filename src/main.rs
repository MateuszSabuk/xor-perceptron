use nalgebra::{DMatrix, DVector, SimdPartialOrd};

fn activation_function(values: &DVector<f32>, f: fn(f32)->f32) -> DVector<f32> {
    return values.map(f);
}

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
    fn train(&mut self, x: DMatrix<f32>, y:DMatrix<f32>, learning_rate:f32, epochs:usize) -> Vec<Vec<DVector<f32>>> {
        assert!(x.len() == y.len());
        let m:f32 = 1. / x.row(0).len() as f32;
        let mut total_errors = vec![];
        for _ in 0..epochs {
            let mut total_error = vec![];
            for (i,col) in x.column_iter().enumerate() {
                // Forward pass
                let mut linears: Vec<DVector<f32>> = vec![col.clone_owned()];
                let mut vals: Vec<DVector<f32>> = vec![col.clone_owned()];
                for layer in &mut self.layers {
                    linears.push(layer.process(vals.last().unwrap().clone()));
                    vals.push(layer.activate(linears.last().unwrap().clone()))
                }
                let mut error:DVector<f32> =  y.column(i) - vals.last().unwrap();
                total_error.push(error.clone());
                
                // Backward Propagation
                for (ln, layer) in self.layers.iter_mut().enumerate().rev() {
                    error = layer.update(&error, &vals[ln], &linears[ln] ,learning_rate, m);
                }
            }
            total_errors.push(total_error);
        }
        return total_errors;
    }
}

fn create_perceptron (inputs:usize, outputs:usize) -> Perceptron {
    // TODO: more control over creation of perceptron
    let layer1: Layer = create_layer(4, inputs, step_function);
    let layer2: Layer = create_layer(outputs, 4, step_function);

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
            next_error.component_mul(&linear.clone().map(step_function));
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
        activation: activation
    };
}

fn main() {
    let mut perceptron = create_perceptron(2, 2);
    let x:DMatrix<f32> = DMatrix::from_row_slice(2, 4, &[
        0.0, 1.0, 0.0, 1.0,
        0.0, 0.0, 1.0, 1.0
    ]);
    let y:DMatrix<f32> = DMatrix::from_row_slice(2, 4, &[
        0.0, 1.0, 1.0, 0.0,
        1.0, 0.0, 0.0, 1.0
    ]);
    let learning_rate = 0.1;
    let epochs = 1000;
    let error = perceptron.train(x, y, learning_rate, epochs);
    

    for (i,e) in error.iter().enumerate() {
        for (j,a) in e.iter().enumerate() {
            print!("epoch = {}, n = {}, {}", i,j,a);
        }
    }
}
