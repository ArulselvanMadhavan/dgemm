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
    T: DAMType + IntoIterator<Item = E> + From<Array1<E>>,
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
    T: DAMType + IntoIterator<Item = E> + From<Array1<E>>,
{
    fn run(&mut self) {
        let link_cap = self.constants.link_capacity;
        let in_features = self.weights.nrows();
        let out_features = self.weights.ncols();
        let ifactor = link_cap / in_features;
        let ofactor = link_cap / out_features;
        let isize = self.constants.buffer_size;
        let mut ibuf = Array::<E, _>::zeros([isize, link_cap]);
        assert!((isize * ifactor) % ofactor == 0);
        let osize = (isize * ifactor) / ofactor;
        let mut obuf = Array::<E, _>::zeros([osize, link_cap]);
        let mut rd_counter = 0;
        let mut wr_counter = 0;
        loop {
            // for b in 0..bsize {
            match self.input.dequeue(&self.time) {
                Ok(data) => {
                    //dbg!(b, self.time.tick() + 1, data.time);
                    let row = Array::from_iter(data.data.clone().into_iter());
                    ibuf.row_mut(rd_counter).assign(&row);
                    rd_counter += 1;
                }
                Err(_) if wr_counter > 0 => (),
                Err(_) => {
                    dbg!("Nothing to dequeue. Ending sim.");
                    return;
                }
            }
            // self.time.incr_cycles(1);
            // }
            if rd_counter == isize {
                dbg!(
                    "Time:",
                    rd_counter,
                    wr_counter,
                    self.time.tick(),
                    isize * ifactor * in_features * out_features
                );

                let x = ibuf.to_shape((isize * ifactor, in_features)).unwrap();
                let out = x.dot(&self.weights);
                obuf = out.to_shape((osize, link_cap)).unwrap().to_owned();
                wr_counter = osize;
                rd_counter = 0;
                self.time.incr_cycles(1);
            }
            // self.time.incr_cycles(1);
            // for o in 0..bsize {
            if wr_counter > 0 {
                let cur_time = self.time.tick();
                let row = obuf.row(osize - wr_counter).to_owned();
                let ce = ChannelElement::new(cur_time + 1, row).convert::<T>();
                self.output.enqueue(&self.time, ce).unwrap();
                wr_counter -= 1;
                // self.time.incr_cycles(1);
                // }
            }
            self.time.incr_cycles(self.initiation_interval);
        }
    }
}
