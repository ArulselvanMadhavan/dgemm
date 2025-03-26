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

const SCR_X_OFF: usize = 40;
const SCR_Y_OFF: usize = 40;
const CIR_X_OFF: usize = 40;
const CIR_Y_OFF: usize = 40;
const CIR_RADIUS: usize = 10;
const LINE_LEN: usize = 10;
const WIN_X: usize = 1500;
const WIN_Y: usize = 820;

fn mk_coordinates(
    dims: [usize; 2],
) -> (
    Array2<[usize; 2]>,
    Array2<[usize; 4]>,
    Array2<[usize; 4]>,
    Array2<[usize; 4]>,
    Array2<[usize; 4]>,
) {
    let [row, col] = dims;
    let mut circ_centers = Array2::from_elem(dims, [0, 0]);
    let mut top_lines = Array2::from_elem(dims, [0, 0, 0, 0]);
    let mut bot_lines = Array2::from_elem(dims, [0, 0, 0, 0]);
    let mut lft_lines = Array2::from_elem(dims, [0, 0, 0, 0]);
    let mut rgt_lines = Array2::from_elem(dims, [0, 0, 0, 0]);
    for r in 0..row {
        for c in 0..col {
            let (cx, cy) = (SCR_X_OFF + CIR_X_OFF * r, SCR_Y_OFF + CIR_Y_OFF * c);
            circ_centers[[r, c]] = [cx, cy];
            top_lines[[r, c]] = [cx, cy - CIR_RADIUS, cx, cy - CIR_RADIUS - LINE_LEN];
            bot_lines[[r, c]] = [cx, cy + CIR_RADIUS, cx, cy + CIR_RADIUS + LINE_LEN];
            lft_lines[[r, c]] = [cx - CIR_RADIUS, cy, cx - CIR_RADIUS - LINE_LEN, cy];
            rgt_lines[[r, c]] = [cx + CIR_RADIUS, cy, cx + CIR_RADIUS + LINE_LEN, cy];
        }
    }
    (circ_centers, top_lines, bot_lines, lft_lines, rgt_lines)
}
fn main() {
    // let files = fs::read_dir("artifacts/trace/").unwrap();
    // let vpkts = files.filter_map(|entry| {
    //     let entry = entry.unwrap();
    //     let fname = entry.file_name().into_string().unwrap();
    //     if fname.contains("gemm") {
    //         let mut file = fs::File::open(entry.path()).unwrap();
    //         let mut cis = CodedInputStream::new(&mut file);
    //         let mut trace = Trace::new();
    //         trace.merge_from(&mut cis).unwrap();
    //         let tpkts = trace.packet;
    //         let mut vpkts = Vec::with_capacity(tpkts.len());
    //         for mut tpkt in tpkts {
    //             let tevt = tpkt.take_track_event();
    //             let ts = tpkt.timestamp();
    //             let slice_type = tevt.type_().value();
    //             let evt = Tracks::from_str(tevt.name()).unwrap();
    //             vpkts.push((ts, slice_type, evt));
    //         }
    //         vpkts.reverse();
    //         Some(vpkts)
    //     } else {
    //         None
    //     }
    // });
    // let mut vpkts = vpkts.collect::<Array1<Vec<(u64, i32, Tracks)>>>();

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

    while !rl.window_should_close() {
        // if rl.is_window_fullscreen() {
        //     rl.toggle_fullscreen();
        // }
        let mut d = rl.begin_drawing(&thread);

        d.clear_background(Color::WHITE);
        let dims = [30, 8];
        let (circ_centers, top_lines, bot_lines, lft_lines, rgt_lines) = mk_coordinates(dims);
        (0..dims[0]).into_iter().for_each(|r| {
            (0..dims[1]).into_iter().for_each(|c| {
                let [cx, cy] = circ_centers[[r, c]];
                d.draw_circle_lines(cx as i32, cy as i32, CIR_RADIUS as f32, Color::BLACK);
                let [sx, sy, ex, ey] = top_lines[[r, c]];
                d.draw_line(sx as i32, sy as i32, ex as i32, ey as i32, Color::BLACK);
                let [sx, sy, ex, ey] = bot_lines[[r, c]];
                d.draw_line(sx as i32, sy as i32, ex as i32, ey as i32, Color::BLACK);
                let [sx, sy, ex, ey] = lft_lines[[r, c]];
                d.draw_line(sx as i32, sy as i32, ex as i32, ey as i32, Color::BLACK);
                let [sx, sy, ex, ey] = rgt_lines[[r, c]];
                d.draw_line(sx as i32, sy as i32, ex as i32, ey as i32, Color::BLACK);
            })
        });
    }
}
