use dam::context_tools::*;
use ndarray::prelude::*;
#[context_macro]
pub struct GEMV<T: Clone> {
    weights: Array2<T>,
    biases: Array1<T>,
    input: Receiver<T>,
    output: Sender<T>,
    initiation_interval: u64,
}

impl<T: DAMType> GEMV<T>
where
    T: ndarray::LinalgScalar,
{
    pub fn new(
        input: Receiver<T>,
        output: Sender<T>,
        weights: Array2<T>,
        biases: Array1<T>,
        initiation_interval: u64,
    ) -> Self {
        let result = Self {
            input,
            output,
            weights,
            biases,
            initiation_interval,
            context_info: Default::default(),
        };
        result.input.attach_receiver(&result);
        result.output.attach_sender(&result);
        result
    }
}

impl<T> Context for GEMV<T>
where
    T: DAMType + ndarray::LinalgScalar,
{
    fn run(&mut self) {
        let isize = self.weights.ncols();
        let osize = self.weights.nrows();
        loop {
            let mut ibuffer = Vec::with_capacity(isize);
            for i in 0..isize {
                match self.input.dequeue(&self.time) {
                    Ok(data) => {
                        ibuffer.push(data.data);
                    }
                    Err(_) if i == 0 => return,
                    Err(_) => panic!("Nothing to dequeue. Unexpected exit"),
                }
                self.time.incr_cycles(1);
            }

            let input_vec = ndarray::Array::from_vec(ibuffer);
            let output = self.weights.dot(&input_vec);

            for i in 0..osize {
                let cur_time = self.time.tick();
                self.output
                    .enqueue(
                        &self.time,
                        ChannelElement::new(cur_time + 1 + (i as u64), output[i] + self.biases[i]),
                    )
                    .unwrap();
            }
            self.time.incr_cycles(self.initiation_interval);
        }
    }
}
