use dam::context_tools::*;
use ndarray::prelude::*;

#[context_macro]
pub struct Gemm<E: Clone, T: Clone> {
    weights: Array2<E>,
    biases: Array1<E>,
    input: Receiver<T>,
    output: Sender<T>,
    initiation_interval: u64,
}

impl<E, T: DAMType> Gemm<E, T>
where
    E: ndarray::LinalgScalar + Send + Sync,
{
    pub fn new(
        weights: Array2<E>,
        biases: Array1<E>,
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

impl<E, T> Context for Gemm<E, T>
where
    E: ndarray::LinalgScalar + Send + Sync,
    T: DAMType,
{
    // Simulates Pytorch Linear function
    fn run(&mut self) {
        //     let osize = self.weights.nrows();
        //     let isize = self.weights.ncols();
        //     let tsize = 4;
        //     let link_cap = tsize;
        //     let rd_latency = 1;
        //     let wr_latency = 1;
        //     let matmul_latency = 1;
        //     assert!((tsize * isize) % link_cap == 0);
        //     assert!((tsize * osize) % link_cap == 0);
        //     let ibsize = (tsize * isize) / link_cap;
        //     let obsize = (tsize * osize) / link_cap;
        //     let bias_mat = self.biases.to_shape((1, osize)).unwrap();
        //     loop {
        //         let mut ibuffer = Vec::with_capacity(ibsize);
        //         let mut ibuffer = Array::zeros([ibsize, link_cap]);
        //         for i in 0..ibsize {
        //             match self.input.dequeue(&self.time) {
        //                 Ok(data) => {
        //                     println!("CT:{:?}|{:?}", self.time.tick(), data.time);
        //                     // ibuffer.extend_from_slice(&data.data);
        //                     // let all = Slice::new(0, None, 1);
        //                     // let out = data.data[all];
        //                     // ibuffer.push(data.data);
        //                     ibuffer.slice_mut(s![i, 0..link_cap]).assign(&data.data);
        //                     // ibuffer.row_mut(i).assign(&data.data)
        //                 }
        //                 Err(_) if i == 0 => return,
        //                 Err(_) => panic!("Nothing to dequeue."),
        //             }
        //         }
        //         // let ibx = ibuffer.self.time.incr_cycles(rd_latency);
        //         let x_in = ndarray::Array2::from_shape_vec((tsize, isize), ibuffer).unwrap();
        //         let mut output = x_in.dot(&self.weights.t());
        //         output = output + &bias_mat;
        //         let output = Array::from_iter(output.iter());
        //         self.time.incr_cycles(matmul_latency);
        //         let cur_time = self.time.tick();
        //         for o in 0..obsize {
        //             println!("Cur_time:{:?}|{:?}", cur_time, self.time.tick());
        //             self.output
        //                 .enqueue(
        //                     &self.time,
        //                     ChannelElement::new(cur_time + wr_latency, *output[o]),
        //                 )
        //                 .unwrap();
        //         }
        //         self.time.incr_cycles(self.initiation_interval);
        //     }
    }
}
