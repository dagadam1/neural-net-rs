use ndarray::{Array, Array1, Array2, ArrayView, Dim, IntoDimension};
use rand::{Rng, Fill, thread_rng};

#[derive(Debug)]
struct Network {
    num_layers: usize,
    sizes: Vec<usize>,
    biases: Vec<Array2<f64>>,
    weights: Array2<f64>,
}

impl Network {
    pub fn new(sizes: Vec<usize>) -> Network {
        // Initialize random number generator
        let mut rng = thread_rng();

        // Every Array in this vector ('biases') corresponds to a layer in the network and contains the biases for the neurons in that layer.
        // The first layer is the input layer and does not have any biases.
        let mut biases: Vec<Array2<f64>> = vec![];
        // For every layer in the network except the first one:
        for layer_size in &sizes[1..] {
            // Add an array with the correct number of biases for this layer to the 'biases' vector:
            // Randomize the value of every bias
            biases.push(
                Array::from_shape_fn((*layer_size, 1), |_| rng.gen())
            );
        }

        Network{
            num_layers: sizes.len(), 
            sizes,
            biases,
            weights: ,
        }
    }
}

pub fn run() {
    let test = Network::new(vec![2,3,4]);
    println!("{:?}", test);
}