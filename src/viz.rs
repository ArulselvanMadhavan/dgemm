use std::{any::Any, fs, str::FromStr};

use dgemm::{
    gemm::Tracks,
    trace::perfetto::{
        self,
        track_event::{self, Type},
        Trace, TrackEvent,
    },
};
use itertools::Itertools;
use ndarray::prelude::*;
use protobuf::*;
use raylib::prelude::*;
use strum::{EnumCount, VariantArray};

const SCR_X_OFF: usize = 50;
const SCR_Y_OFF: usize = 50;
const CIR_X_OFF: usize = 60;
const CIR_Y_OFF: usize = 60;
const CIR_RADIUS: usize = 10;
const LINE_LEN: usize = 40;
const WIN_X: usize = 1500;
const WIN_Y: usize = 820;

fn mk_coordinates(dims: [usize; 2]) -> Array3<[usize; 4]> {
    let [row, col] = dims;
    let mut coords = Array3::from_elem([Tracks::COUNT, dims[0], dims[1]], [0, 0, 0, 0]);
    for r in 0..row {
        for c in 0..col {
            let (cx, cy) = (SCR_X_OFF + CIR_X_OFF * r, SCR_Y_OFF + CIR_Y_OFF * c);
            for t in Tracks::VARIANTS {
                let t_idx = *t as usize;
                match t {
                    Tracks::RdLeft => {
                        coords[[t_idx, r, c]] =
                            [cx - CIR_RADIUS, cy, cx - CIR_RADIUS - LINE_LEN, cy]
                    }
                    Tracks::RdUp => {
                        coords[[t_idx, r, c]] =
                            [cx, cy - CIR_RADIUS, cx, cy - CIR_RADIUS - LINE_LEN]
                    }
                    Tracks::WrDown => {
                        coords[[t_idx, r, c]] =
                            [cx, cy + CIR_RADIUS, cx, cy + CIR_RADIUS + LINE_LEN]
                    }
                    Tracks::WrRight => {
                        coords[[t_idx, r, c]] =
                            [cx + CIR_RADIUS, cy, cx + CIR_RADIUS + LINE_LEN, cy]
                    }
                    Tracks::Gemm => {
                        coords[[t_idx, r, c]] = [cx, cy, 0, 0];
                    }
                }
            }
        }
    }
    coords
}
fn main() {
    let files = fs::read_dir("artifacts/trace/").unwrap();
    let files = files.map(|f| f.unwrap().path().into_os_string().into_string().unwrap());
    // let files = files.sorted();
    let files = files.sorted_by(|a, b| {
        let a_splits = a.split(['_']);
        let b_splits = b.split(['_']);
        let mut a_splits = a_splits.skip(1);
        let mut b_splits = b_splits.skip(1);
        match (a_splits.next(), b_splits.next()) {
            (Some(a_id), Some(b_id)) => Ord::cmp(
                &a_id.to_string().parse::<u32>().unwrap(),
                &b_id.to_string().parse::<u32>().unwrap(),
            ),
            (_, _) => Ord::cmp(&a, &b),
        }
    });
    let vpkts = files.filter_map(|path| {
        if path.contains("gemm") {
            let is_first = path.contains("gemm_0");
            let mut file = fs::File::open(path).unwrap();
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
                if is_first {
                    dbg!(ts, slice_type, evt);
                }
                vpkts.push((ts, slice_type, evt));
            }
            vpkts.sort_by(|(ts1, _, _), (ts2, _, _)| Ord::cmp(ts1, ts2));
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

    let (mut rl, thread) = raylib::init()
        .size(WIN_X as i32, WIN_Y as i32)
        .title("DGEMM")
        .build();
    let dims: [usize; 2] = [3, 4];
    let mut state = Array3::from_elem([Tracks::COUNT, dims[0], dims[1]], Color::BLACK);
    while !rl.window_should_close() {
        let cur_time = rl.get_time() as u64;
        let mut d = rl.begin_drawing(&thread);
        d.clear_background(Color::WHITE);
        let coords = mk_coordinates(dims);
        (0..dims[0]).into_iter().for_each(|r| {
            (0..dims[1]).into_iter().for_each(|c| {
                let n = r * dims[1] + c;
                let vp = &vpkts[n];
                if vp.len() > 0 {
                    let (ts, stype, trk) = vp[vp.len() - 1];
                    if cur_time >= ts {
                        //update state
                        if stype == 1 {
                            match trk {
                                Tracks::Gemm => state[[trk as usize, r, c]] = Color::GREEN,
                                _ => state[[trk as usize, r, c]] = Color::ORANGERED,
                            }
                        } else {
                            state[[trk as usize, r, c]] = Color::BLACK;
                        }
                        //pop
                        vpkts[n].pop();
                    }
                }
                Tracks::VARIANTS.into_iter().for_each(|t| {
                    let t_idx = *t as usize;
                    let t_state = state[[t_idx, r, c]];
                    match t {
                        Tracks::RdLeft | Tracks::RdUp | Tracks::WrDown | Tracks::WrRight => {
                            let [sx, sy, ex, ey] = coords[[t_idx, r, c]];
                            d.draw_line_ex(
                                Vector2::new(sy as f32, sx as f32),
                                Vector2::new(ey as f32, ex as f32),
                                4.0,
                                t_state,
                            );
                            d.draw_line(sy as i32, sx as i32, ey as i32, ex as i32, t_state);
                        }
                        Tracks::Gemm => {
                            let [cx, cy, _, _] = coords[[t_idx, r, c]];
                            d.draw_circle(cy as i32, cx as i32, CIR_RADIUS as f32, t_state);
                        }
                    }
                })
            })
        });
    }
}
