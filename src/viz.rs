use std::{any::Any, fs, str::FromStr};

use dgemm::{
    gemm::Tracks,
    trace::perfetto::{
        self,
        track_event::{self, Type},
        Trace, TrackEvent,
    },
};
use ndarray::prelude::*;
use protobuf::*;
use raylib::prelude::*;
use strum::EnumCount;
fn main() {
    let files = fs::read_dir("artifacts/trace/").unwrap();
    let vpkts = files.filter_map(|entry| {
        let entry = entry.unwrap();
        let fname = entry.file_name().into_string().unwrap();
        if fname.contains("gemm") {
            let mut file = fs::File::open(entry.path()).unwrap();
            let mut cis = CodedInputStream::new(&mut file);
            let mut trace = Trace::new();
            trace.merge_from(&mut cis).unwrap();
            let tpkts = trace.packet;
            let mut vpkts = Vec::with_capacity(tpkts.len());
            for mut tpkt in tpkts {
                let tevt = tpkt.take_track_event();
                let ts = tpkt.timestamp();
                let slice_type = tevt.type_().value();
                let evt = Tracks::from_str(tevt.name()).unwrap();
                vpkts.push((ts, slice_type, evt));
            }
            vpkts.reverse();
            Some(vpkts)
        } else {
            None
        }
    });
    let mut vpkts = vpkts.collect::<Array1<Vec<(u64, i32, Tracks)>>>();
    // let mut vp = &vpkts[0];
    // let (ts, st, _) = vpkts[0][vp.len() - 1];
    // if ts > 0 {
    //     vpkts[0].pop();
    // }
    // println!("{:?}|{:?}", vpkts[0].len(), vpkts[1].len())

    // let (mut rl, thread) = raylib::init().size(640, 480).title("Hello, World").build();

    // while !rl.window_should_close() {
    //     let mut d = rl.begin_drawing(&thread);

    //     d.clear_background(Color::WHITE);
    //     d.draw_text("Hello, world!", 12, 12, 20, Color::BLACK);
    // }
}
