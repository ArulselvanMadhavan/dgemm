use dam::context_tools::*;
use ndarray::prelude::*;

/// Constants for GEMM
/// link_capacity - Number of elements acceptable in a send/recv
/// buffer_size - Number of receive msgs acceptable before starting a GEMM
pub struct GemmConstants {
    link_capacity: usize,
    buffer_size: usize,
}

impl GemmConstants {
    pub fn new(link_capacity: usize, buffer_size: usize) -> Self {
        Self {
            link_capacity,
            buffer_size,
        }
    }
}

/// Models weight stationary systolic/dataflow GEMM
#[context_macro]
pub struct Gemm<E: Clone, T: Clone> {
    weights: Array2<E>,
    biases: Array1<E>,
    constants: GemmConstants,
    input: Receiver<T>,
    output: Sender<T>,
    initiation_interval: u64,
}

impl<E, T> Gemm<E, T>
where
    E: ndarray::LinalgScalar + Send + Sync + std::fmt::Debug,
    T: DAMType + IntoIterator<Item = E>,
{
    pub fn new(
        weights: Array2<E>,
        biases: Array1<E>,
        constants: GemmConstants,
        input: Receiver<T>,
        output: Sender<T>,
        initiation_interval: u64,
    ) -> Self {
        let result = Self {
            weights,
            biases,
            constants,
            input,
            output,
            initiation_interval,
            context_info: Default::default(),
        };
        result.input.attach_receiver(&result);
        result.output.attach_sender(&result);
        result
    }
}

impl<E, T> Context for Gemm<E, T>
where
    E: ndarray::LinalgScalar + Send + Sync + std::fmt::Debug,
    T: DAMType + IntoIterator<Item = E>,
{
    fn run(&mut self) {
        let link_cap = self.constants.link_capacity;
        let in_features = self.weights.nrows();
        let _factor = link_cap / in_features;
        let bsize = self.constants.buffer_size;
        let mut ibuf = Array::<E, _>::zeros([bsize, link_cap]);
        loop {
            for b in 0..bsize {
                match self.input.dequeue(&self.time) {
                    Ok(data) => {
                        let row = Array::from_iter(data.data.clone().into_iter());
                        ibuf.row_mut(b).assign(&row);
                    }
                    Err(_) if b == 0 => return,
                    Err(_) => panic!("GEMM:Nothing to dequeue"),
                }
                self.time.incr_cycles(1);
            }
            println!("GEMM: X:{:?}", ibuf);
            self.time.incr_cycles(self.initiation_interval);
        }
    }
}
