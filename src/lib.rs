use ndarray::{Array, Array2};
use rand::{Rng, thread_rng};

#[derive(Debug)]
struct Network {
    num_layers: usize,
    sizes: Vec<usize>,
    biases: Vec<Array2<f64>>,
    weights: Vec<Array2<f64>>,
}

impl Network {
    pub fn new(sizes: Vec<usize>) -> Network {
        // Initialize random number generator
        let mut rng = thread_rng();

        // Every Array in this vector ('biases') corresponds to a layer in the network and contains the biases for the neurons in that layer.
        // The first layer is the input layer and does not have any biases.
        // For example: The bias of the first neuron in the second layer (first layer after the input layer) is biases[0][(0, 0)]
        let mut biases: Vec<Array2<f64>> = vec![];
        // For every layer in the network except the first one:
        for layer_size in &sizes[1..] {
            // Add an array with the correct number of biases for this layer to the 'biases' vector:
            // Randomize the value of every bias
            biases.push(
                Array::from_shape_fn((*layer_size, 1), |_| rng.gen())
            );
        }

        let mut weights: Vec<Array2<f64>> = vec![];
        for (x, y) in sizes[..sizes.len() - 1].iter().zip(&sizes[1..]) {
            weights.push(
                Array::from_shape_fn((*y, *x), |_| rng.gen())
            );
        }

        Network{
            num_layers: sizes.len(), 
            sizes,
            biases,
            weights,
        }
    }
}

pub fn run() {
    let test = Network::new(vec![2,3,4]);
    println!("{:?}", test);
    println!("{:?}", test.biases[0][(0, 0)]);
}

fn sigmoid(z: Array2<f64>) -> Array2<f64> {
    z.map(|x| {
        1.0/(1.0+x.exp())
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_network_initialization() {
        let sizes = vec![6, 3, 4, 5];
        let mut net = Network::new(sizes);

        assert_eq!(net.biases.pop().unwrap().dim(), (5, 1));
        assert_eq!(net.biases.pop().unwrap().dim(), (4, 1));
        assert_eq!(net.biases.pop().unwrap().dim(), (3, 1));
        assert!(net.biases.is_empty());

        assert_eq!(net.weights.pop().unwrap().dim(), (5, 4));
        assert_eq!(net.weights.pop().unwrap().dim(), (4, 3));
        assert_eq!(net.weights.pop().unwrap().dim(), (3, 6));
        assert!(net.weights.is_empty());

        assert_eq!(net.sizes, vec![6, 3, 4, 5]);
        assert_eq!(net.num_layers, 4);
    }
}