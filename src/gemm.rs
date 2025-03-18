use dam::context_tools::*;
use ndarray::prelude::*;

#[context_macro]
pub struct Gemm<T: Clone> {
    weights: Array2<T>,
    biases: Array1<T>,
    input: Receiver<T>,
    output: Sender<T>,
    initiation_interval: u64,
}

impl<T: DAMType> Gemm<T>
where
    T: ndarray::LinalgScalar,
{
    pub fn new(
        weights: Array2<T>,
        biases: Array1<T>,
        input: Receiver<T>,
        output: Sender<T>,
        initiation_interval: u64,
    ) -> Self {
        let result = Self {
            weights,
            biases,
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

impl<T> Context for Gemm<T>
where
    T: DAMType + ndarray::LinalgScalar,
{
    // Simulates Pytorch Linear function
    fn run(&mut self) {
        let osize = self.weights.nrows();
        let isize = self.weights.ncols();
        let tsize = 4;
        let rd_latency = 1;
        let wr_latency = 1;
        let matmul_latency = 1;
        let ibsize = tsize * isize;
        let obsize = tsize * osize;
        let bias_mat = self.biases.to_shape((1, osize)).unwrap();
        loop {
            let mut ibuffer = Vec::with_capacity(ibsize);
            for i in 0..ibsize {
                match self.input.dequeue(&self.time) {
                    Ok(data) => {
                        println!("CT:{:?}|{:?}", self.time.tick(), data.time);
                        ibuffer.push(data.data);
                    }
                    Err(_) if i == 0 => return,
                    Err(_) => panic!("Nothing to dequeue."),
                }
            }
            self.time.incr_cycles(rd_latency);
            let x_in = ndarray::Array2::from_shape_vec((tsize, isize), ibuffer).unwrap();
            let mut output = x_in.dot(&self.weights.t());
            output = output + &bias_mat;
            let output = output.flatten();
            self.time.incr_cycles(matmul_latency);
            let cur_time = self.time.tick();
            for o in 0..obsize {
                println!("Cur_time:{:?}|{:?}", cur_time, self.time.tick());
                self.output
                    .enqueue(
                        &self.time,
                        ChannelElement::new(cur_time + wr_latency, output[o]),
                    )
                    .unwrap();
            }
            self.time.incr_cycles(self.initiation_interval);
        }
    }
}
