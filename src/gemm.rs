use dam::context_tools::*;
use ndarray::prelude::*;
use protobuf::{CodedOutputStream, Message};

use crate::trace::{
    self,
    perfetto::{Trace, TracePacket},
};

#[derive(strum_macros::EnumCount, strum_macros::Display, strum_macros::EnumString)]
pub enum Tracks {
    RdLeft = 0,
    RdUp = 1,
    WrDown = 2,
    WrRight = 3,
    Gemm = 4,
}
/// Constants for GEMM
/// link_capacity - Number of elements acceptable in a send/recv
/// buffer_size - Number of receive msgs acceptable before starting a GEMM
pub struct GemmConstants {
    link_capacity: usize,
    buffer_size: usize,
    thread_id: u32,
    track_ids: [u64; 5],
    num_matmuls: usize,
}

impl GemmConstants {
    pub fn new(
        link_capacity: usize,
        buffer_size: usize,
        thread_id: u32,
        track_ids: [u64; 5], // FIXME: Make it a const generic
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
        let mut ibuf1 = Array::<E, _>::zeros([isize, link_cap]);
        let mut ibuf2 = Array::<E, _>::zeros([isize, link_cap]);
        assert!((isize * ifactor) % ofactor == 0);
        let osize = (isize * ifactor) / ofactor;
        let mut obuf = Array::<E, _>::zeros([osize, link_cap]);
        let mut cbuf = Array::<E, _>::zeros([osize, link_cap]);
        let mut rd_counter1 = 0;
        let mut rd_counter2 = 0;
        let mut wr_counter1 = 0;
        let mut wr_counter2 = 0;
        let mut is_rd_ctrl1 = true;
        let mut is_rd_ctrl2 = true;
        let mut is_wr_ctrl1 = false;
        let mut is_wr_ctrl2 = false;
        let mut is_mm_ctrl = false;
        let mut trace = Trace::new();
        let mut file = trace::mk_trace_file(
            format!("gemm{tid}.perfetto", tid = self.constants.thread_id).as_str(),
        );
        let mut cos = CodedOutputStream::new(&mut file);
        let mut num_matmuls = 0;
        loop {
            let mut tpkts = Vec::<TracePacket>::with_capacity(self.constants.track_ids.len() * 2);
            if is_rd_ctrl1 {
                match self.input[0].dequeue(&self.time) {
                    Ok(data) => {
                        let row = Array::from_iter(data.data.clone().into_iter());
                        ibuf1.row_mut(rd_counter1).assign(&row);
                        rd_counter1 += 1;
                        let evt = Tracks::RdLeft;
                        tpkts.extend_from_slice(&self.evt_slice(
                            evt.to_string().as_str(),
                            evt as usize,
                            1,
                        ));
                    }
                    Err(_) => (),
                }
            }
            if is_rd_ctrl2 {
                match self.input[1].dequeue(&self.time) {
                    Ok(data) => {
                        let row = Array::from_iter(data.data.clone().into_iter());
                        cbuf.row_mut(rd_counter2).assign(&row);
                        rd_counter2 += 1;
                        let evt = Tracks::RdUp;
                        tpkts.extend_from_slice(&self.evt_slice(
                            evt.to_string().as_str(),
                            evt as usize,
                            1,
                        ));
                    }
                    Err(_) => (),
                }
            }
            if is_wr_ctrl1 {
                let row = obuf.row(osize - wr_counter1).to_owned();
                let ce = ChannelElement::new(self.time.tick() + 1, row).convert::<T>();
                self.output[1].enqueue(&self.time, ce).unwrap();
                let evt = Tracks::WrDown;
                tpkts.extend_from_slice(&self.evt_slice(evt.to_string().as_str(), evt as usize, 1));
                wr_counter1 -= 1;
            }
            if is_wr_ctrl2 {
                let row = ibuf2.row(isize - wr_counter2).to_owned();
                let ce = ChannelElement::new(self.time.tick() + 1, row).convert::<T>();
                self.output[0].enqueue(&self.time, ce).unwrap();
                let evt = Tracks::WrRight;
                tpkts.extend_from_slice(&self.evt_slice(evt.to_string().as_str(), evt as usize, 1));
                wr_counter2 -= 1;
            }
            if is_mm_ctrl {
                let x = ibuf1.to_shape((isize * ifactor, in_features)).unwrap();
                let cout = cbuf.to_shape((isize * ifactor, out_features)).unwrap();
                let out = x.dot(&self.weights) + cout;
                // println!("{:?}|{:?}", self.constants.thread_id, x);
                // println!("{:?}|{:?}", self.constants.thread_id, self.weights);
                obuf = out.to_shape((osize, link_cap)).unwrap().to_owned();
                wr_counter1 = osize;
                wr_counter2 = isize;
                ibuf2 = ibuf1.clone();
                rd_counter1 = 0;
                rd_counter2 = 0;
                let mm_cycles = (isize + osize - 1) as u64;
                let evt = Tracks::Gemm;
                tpkts.extend_from_slice(&self.evt_slice(
                    evt.to_string().as_str(),
                    evt as usize,
                    mm_cycles + 1,
                ));
                self.time.incr_cycles(mm_cycles);
                num_matmuls += 1;
            }
            trace.packet = tpkts;
            trace.write_to(&mut cos).unwrap();
            is_rd_ctrl1 = rd_counter1 < isize;
            is_rd_ctrl2 = rd_counter2 < osize;
            is_wr_ctrl1 = wr_counter1 > 0;
            is_wr_ctrl2 = wr_counter2 > 0;
            is_mm_ctrl = rd_counter1 == isize
                && rd_counter2 == osize
                && wr_counter1 == 0
                && wr_counter2 == 0;
            self.time.incr_cycles(self.initiation_interval);
            if num_matmuls == self.constants.num_matmuls && wr_counter1 == 0 && wr_counter2 == 0 {
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
