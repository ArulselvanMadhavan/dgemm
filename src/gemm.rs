use dam::{context_tools::*, structures::Time};
use ndarray::prelude::*;

use crate::trace::{self, perfetto::TracePacket};

/// Constants for GEMM
/// link_capacity - Number of elements acceptable in a send/recv
/// buffer_size - Number of receive msgs acceptable before starting a GEMM
pub struct GemmConstants {
    link_capacity: usize,
    buffer_size: usize,
    thread_id: u32,
    thread_uuid: u64,
}

impl GemmConstants {
    pub fn new(link_capacity: usize, buffer_size: usize, thread_id: u32, thread_uuid: u64) -> Self {
        Self {
            link_capacity,
            buffer_size,
            thread_id,
            thread_uuid,
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
    fn publish_evt(&self, evt_name: &str, start: u64) -> [TracePacket; 2] {
        trace::mk_time_slice(
            self.constants.thread_id,
            self.constants.thread_uuid,
            evt_name,
            [start, self.time.tick().time()],
        )
    }

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
        // const MAX_EVTS_PER_CYCLE: usize = 3;
        // let mut tpkts = Vec::<TracePacket>::with_capacity(MAX_EVTS_PER_CYCLE * 2);
        let mut tpkts = Vec::<TracePacket>::with_capacity(isize);
        let mut rd_begin = Time::default();
        let mut mm_begin = Time::default();
        let mut wr_begin = Time::default();
        let mut is_mm_ready = false;
        loop {
            if rd_counter < isize {
                match self.input.dequeue(&self.time) {
                    Ok(data) => {
                        let row = Array::from_iter(data.data.clone().into_iter());
                        ibuf.row_mut(rd_counter).assign(&row);
                        if rd_counter == 0 {
                            rd_begin = self.time.tick();
                        }
                        rd_counter += 1;
                    }
                    Err(_) if wr_counter > 0 => (),
                    Err(_) => {
                        trace::write_trace(
                            format!("gemm{tid}.perfetto", tid = self.constants.thread_id).as_str(),
                            tpkts,
                        );
                        dbg!("Nothing to dequeue. Ending sim.");
                        return;
                    }
                }
            } else if rd_counter == isize {
                tpkts.extend_from_slice(&self.publish_evt("RD_BUF_IN", rd_begin.time()));
                is_mm_ready = true;
            } else {
                panic!("GEMM Rd counter > buffer size")
            }

            if is_mm_ready {
                mm_begin = self.time.tick();
                let x = ibuf.to_shape((isize * ifactor, in_features)).unwrap();
                let out = x.dot(&self.weights);
                obuf = out.to_shape((osize, link_cap)).unwrap().to_owned();
                wr_counter = osize;
                rd_counter = 0;
                self.time.incr_cycles((isize + osize) as u64);
                tpkts.extend_from_slice(&self.publish_evt("GEMM", mm_begin.time()));
                is_mm_ready = false;
                wr_begin = self.time.tick();
            }
            if wr_counter > 0 {
                let cur_time = self.time.tick();
                let row = obuf.row(osize - wr_counter).to_owned();
                let ce = ChannelElement::new(cur_time + 1, row).convert::<T>();
                self.output.enqueue(&self.time, ce).unwrap();
                wr_counter -= 1;
                if wr_counter == 0 {
                    tpkts.extend_from_slice(&self.publish_evt("WR_BUF_OUT", wr_begin.time()))
                }
            }
            self.time.incr_cycles(self.initiation_interval);
        }
    }
}
