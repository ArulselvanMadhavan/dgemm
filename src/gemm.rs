use std::fs::File;

use dam::context_tools::*;
use ndarray::prelude::*;
use protobuf::{CodedOutputStream, Message};

use crate::trace::{
    self,
    perfetto::{Trace, TracePacket},
};

/// Constants for GEMM
/// link_capacity - Number of elements acceptable in a send/recv
/// buffer_size - Number of receive msgs acceptable before starting a GEMM
pub struct GemmConstants {
    link_capacity: usize,
    buffer_size: usize,
    thread_id: u32,
    track_ids: [u64; 3],
    num_matmuls: usize,
}

impl GemmConstants {
    pub fn new(
        link_capacity: usize,
        buffer_size: usize,
        thread_id: u32,
        track_ids: [u64; 3],
        num_matmuls: usize,
    ) -> Self {
        Self {
            link_capacity,
            buffer_size,
            thread_id,
            track_ids,
            num_matmuls,
        }
    }
}

/// Models weight stationary systolic/dataflow GEMM
#[context_macro]
pub struct Gemm<E: Clone, T: Clone> {
    weights: Array2<E>,
    biases: Array1<E>,
    constants: GemmConstants,
    input: [Receiver<T>; 2],
    output: [Sender<T>; 2],
    initiation_interval: u64,
}

impl<E, T> Gemm<E, T>
where
    E: ndarray::LinalgScalar + Send + Sync + std::fmt::Debug,
    T: DAMType + IntoIterator<Item = E> + From<Array1<E>>,
{
    fn evt_slice(&self, evt_name: &str, track_idx: usize, count: u64) -> [TracePacket; 2] {
        let cur_time = self.time.tick().time();
        trace::mk_time_slice(
            self.constants.thread_id,
            self.constants.track_ids[track_idx],
            evt_name,
            [cur_time, cur_time + count],
        )
    }

    // fn evt_begin(&self, evt_name: &str) -> TracePacket {
    //     trace::slice_begin(
    //         self.constants.thread_id,
    //         self.constants.thread_uuid,
    //         evt_name,
    //         self.time.tick().time(),
    //     )
    // }

    // fn evt_end(&self) -> TracePacket {
    //     trace::slice_end(
    //         self.constants.thread_id,
    //         self.constants.thread_uuid,
    //         self.time.tick().time(),
    //     )
    // }

    pub fn new(
        weights: Array2<E>,
        biases: Array1<E>,
        constants: GemmConstants,
        input: [Receiver<T>; 2],
        output: [Sender<T>; 2],
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
        result.input.iter().for_each(|x| x.attach_receiver(&result));
        result.output.iter().for_each(|x| x.attach_sender(&result));
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
        let mut is_rd_ctrl = true;
        let mut is_wr_ctrl = false;
        let mut is_mm_ctrl = false;
        let mut trace = Trace::new();
        let mut file = File::create(format!(
            "gemm{tid}.perfetto",
            tid = self.constants.thread_id
        ))
        .unwrap();
        let mut cos = CodedOutputStream::new(&mut file);
        let mut num_matmuls = 0;
        loop {
            let mut tpkts = Vec::<TracePacket>::with_capacity(self.constants.track_ids.len() * 2);
            if is_rd_ctrl {
                match self.input[0].dequeue(&self.time) {
                    Ok(data) => {
                        let row = Array::from_iter(data.data.clone().into_iter());
                        ibuf.row_mut(rd_counter).assign(&row);
                        rd_counter += 1;
                        tpkts.extend_from_slice(&self.evt_slice("RD_BUF_IN", 0, 1));
                    }
                    Err(_) => {
                        let dbg_str = format!(
                            "T={time}|GEMM={tid}|Nothing to read",
                            time = self.time.tick().time(),
                            tid = self.constants.thread_id
                        );
                        dbg!(dbg_str);
                        ()
                    }
                }
            }
            if is_wr_ctrl {
                let row = obuf.row(osize - wr_counter).to_owned();
                let ce = ChannelElement::new(self.time.tick() + 1, row).convert::<T>();
                self.output[0].enqueue(&self.time, ce).unwrap();
                tpkts.extend_from_slice(&self.evt_slice("WR_BUF_OUT", 1, 1));
                wr_counter -= 1;
            }
            if is_mm_ctrl {
                let x = ibuf.to_shape((in_features, isize * ifactor)).unwrap();
                let x = x.t();
                let out = x.dot(&self.weights);
                obuf = out.to_shape((osize, link_cap)).unwrap().to_owned();
                wr_counter = osize;
                rd_counter = 0;
                let mm_cycles = (isize + osize - 1) as u64;
                tpkts.extend_from_slice(&self.evt_slice("GEMM", 2, mm_cycles + 1));
                self.time.incr_cycles(mm_cycles);
                num_matmuls += 1;
            }
            trace.packet = tpkts;
            trace.write_to(&mut cos).unwrap();
            is_rd_ctrl = rd_counter < isize;
            is_wr_ctrl = wr_counter > 0;
            is_mm_ctrl = rd_counter == isize && wr_counter == 0;
            self.time.incr_cycles(self.initiation_interval);
            if num_matmuls == self.constants.num_matmuls && wr_counter == 0 {
                break;
            }
        }
        let dbg_str = format!(
            "T={t}|GEMM={tid}|Ending sim",
            t = self.time.tick().time(),
            tid = self.constants.thread_id
        );
        dbg!(dbg_str);
        cos.flush().unwrap();
    }
}
