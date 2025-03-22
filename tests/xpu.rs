use dam::{
    channel::{Receiver, Sender},
    simulation::{InitializationOptionsBuilder, ProgramBuilder, RunOptions},
};
use dgemm::{
    consumer::Consumer,
    gemm::{Gemm, GemmConstants},
    producer::Producer,
};
use ndarray::*;

fn assign_chan<'a, T: Clone + 'a>(
    n: usize,
    buffer_size: usize,
    conn: Array2<bool>,
    ctx: &mut ProgramBuilder<'a>,
) -> (
    Array1<Vec<Sender<Array1<T>>>>,
    Array1<Vec<Receiver<Array1<T>>>>,
    Array1<Vec<Sender<Array1<T>>>>,
    Array1<Vec<Receiver<Array1<T>>>>,
) {
    let mut sd_chan = Array1::<Vec<Sender<Array1<T>>>>::default([n]);
    let mut rx_chan = Array1::<Vec<Receiver<Array1<T>>>>::default([n]);
    let mut rx_cons = Array1::<Vec<Receiver<Array1<T>>>>::default([n]);
    let mut sd_prod = Array1::<Vec<Sender<Array1<T>>>>::default([n]);
    for s in 0..n {
        let mut count = 0;
        for r in 0..n {
            if conn.slice(s![s, r]).eq(&arr0(true)) {
                let (tx, rx) = ctx.bounded::<Array1<T>>(buffer_size);
                sd_chan[s].push(tx);
                rx_chan[r].push(rx);
                count += 1;
            }
        }
        if count == 0 {
            let (tx, rx) = ctx.bounded::<Array1<T>>(buffer_size);
            sd_chan[s].push(tx);
            rx_cons[s].push(rx);
        }
        count = 0;
        for r in 0..n {
            if conn.slice(s![r, s]).eq(&arr0(true)) {
                count += 1;
            }
        }
        if count == 0 {
            let (tx, rx) = ctx.bounded::<Array1<T>>(buffer_size);
            sd_prod[s].push(tx);
            rx_chan[s].push(rx);
        }
    }
    (sd_chan, rx_chan, sd_prod, rx_cons)
}

fn get_chan<T>(i: usize, chan: &mut Array1<Vec<T>>, bkup: &mut Array1<Vec<T>>) -> T {
    match (chan.get_mut(i), bkup.get_mut(i)) {
        (Some(chanvec), _) if chanvec.len() == 1 => chanvec.remove(0),
        (_, Some(convec)) if convec.len() == 1 => convec.remove(0),
        (_, _) => panic!("Chan vec empty"),
    }
}

fn mesh_conn<'a, T: Clone + 'a>(
    dims: [usize; 2],
    buffer_size: usize,
    ctx: &mut ProgramBuilder<'a>,
) -> (Vec<[Receiver<Array1<T>>; 2]>, Vec<[Sender<Array1<T>>; 2]>) {
    let n = dims.iter().fold(1, |prod, val| prod * val);
    let mut conn = Array2::<bool>::default([n, n]);
    for r in 0..n {
        conn.slice_mut(s![r, r]).assign(&arr0(true));
    }
    let conn = conn.to_shape([n, dims[0], dims[1]]).unwrap();
    let mut send_right = conn.clone();
    send_right
        .slice_mut(s![.., .., 1..])
        .assign(&conn.slice(s![.., .., ..-1]));
    send_right
        .slice_mut(s![.., .., 0])
        .assign(&Array1::<bool>::default([dims[0]]));
    let mut send_down = conn.clone().to_owned();
    send_down
        .slice_mut(s![.., 1.., ..])
        .assign(&conn.slice(s![.., ..-1, ..]));
    send_down
        .slice_mut(s![.., 0, ..])
        .assign(&Array1::<bool>::default([dims[1]]));

    let send_right = send_right.to_shape((n, n)).unwrap().to_owned();
    let send_down = send_down.to_shape((n, n)).unwrap().to_owned();
    let (mut rchan, mut lchan, mut rcon, mut lcon) =
        assign_chan::<T>(n, buffer_size, send_right, ctx);
    let (mut dchan, mut uchan, mut dcon, mut ucon) =
        assign_chan::<T>(n, buffer_size, send_down, ctx);
    // let send_chans = Array1::<Vec<Sender<Array1<f64>>>>::default([n]);
    // let rx_chans = Array1::<Vec<Sender<Array1<f64>>>>::default([n]);
    let mut out_chans = Vec::with_capacity(n);
    let mut in_chans = Vec::with_capacity(n);
    for i in 0..n {
        let rsend = get_chan(i, &mut rchan, &mut rcon);
        let dsend = get_chan(i, &mut dchan, &mut dcon);
        let lrecv = get_chan(i, &mut lchan, &mut lcon);
        let urecv = get_chan(i, &mut uchan, &mut ucon);
        out_chans.push([rsend, dsend]);
        in_chans.push([lrecv, urecv]);
    }
    (in_chans, out_chans)
}

// fn mesh_chan(dims: [usize; 2], buffer_size: usize, ctx: &mut ProgramBuilder<'_>) {
//     let n = dims.iter().fold(1, |prod, val| prod * val);
//     let send_conns = mesh_conn(dims);
//     // each node will get [0, 2] senders or receivers
//     // after iter assign prod/consumer channels
//     for src in 0..n {
//         for rec in 0..n {
//             if send_conns.slice(s![src, rec]).eq(&arr0(true)) {
//                 ctx.bounded::<Array1<f64>>(buffer_size);
//             }
//         }
//     }
//     // let send_right = (0..n)
//     //     .into_iter()
//     //     .map(|_x| ctx.bounded::<Array1<f64>>(buffer_size))
//     //     .collect_array()
//     //     .unwrap();
// }

#[test]
fn xpu_linear_test() {
    const LINK_CAPACITY: usize = 4;
    const IN_FEATURES: usize = 4;
    const OUT_FEATURES: usize = 4;
    assert!(LINK_CAPACITY % IN_FEATURES == 0);
    assert!(LINK_CAPACITY % OUT_FEATURES == 0);
    const BUFFER_CAPACITY: usize = 2;
    const NUM_INPUTS: usize = (LINK_CAPACITY / IN_FEATURES) * BUFFER_CAPACITY * 3;
    const W_SIZE: usize = IN_FEATURES * OUT_FEATURES;
    const X_SIZE: usize = NUM_INPUTS * IN_FEATURES;
    const X_SEND_STEPS: usize = X_SIZE / LINK_CAPACITY;
    const TRACKS_PER_THREAD: usize = 3;
    const DIMS: [usize; 2] = [1, 1];
    let processes = vec![("xpu".to_string(), vec!["xpu1".to_string()])];
    let tuuids = dgemm::trace::get_trace_descriptors::<TRACKS_PER_THREAD>(processes, 2, 1);
    let mut ctx = ProgramBuilder::default();
    // let (in_conns, out_conns) = mesh_conn::<f64>(DIMS, BUFFER_CAPACITY, &mut ctx);
    // dbg!("{:?}|{:?}", in_conns.len(), out_conns.len());
    // Build Links
    let (x_send, x_recv) = ctx.bounded::<Array1<f64>>(BUFFER_CAPACITY);
    let (out_send, out_recv) = ctx.bounded::<Array1<f64>>(BUFFER_CAPACITY);
    // let weight_mat = Array::<f64, _>::linspace(0.0, W_SIZE as f64, W_SIZE);
    let weight_mat = Array::range(0., W_SIZE as f64, 1.);
    let weight_mat = weight_mat
        .to_shape((IN_FEATURES, OUT_FEATURES))
        .unwrap()
        .to_owned();
    let x_mat = Array::range(0., X_SIZE as f64, 1.)
        .into_shape([X_SEND_STEPS, LINK_CAPACITY])
        .unwrap();
    let biases = ndarray::Array::<f64, _>::linspace(0.0, OUT_FEATURES as f64, OUT_FEATURES);
    // let mut ref_out = weight_mat.dot(&x_mat.t());
    // let bias_mat = biases
    //     .clone()
    //     .to_shape((OUT_FEATURES, 1))
    //     .unwrap()
    //     .to_owned();
    // ref_out = ref_out + &bias_mat;
    let mut x_mat_vec = Vec::with_capacity(X_SEND_STEPS);
    x_mat.map_axis(Axis(1), |x| x_mat_vec.push(x.to_owned()));
    ctx.add_child(Producer::new(|| x_mat_vec.into_iter(), x_send));

    ctx.add_child(Gemm::new(
        weight_mat,
        biases,
        GemmConstants::new(LINK_CAPACITY, BUFFER_CAPACITY, 0, *tuuids.get(0).unwrap()),
        x_recv,
        out_send,
        1,
    ));
    ctx.add_child(Consumer::new(OUT_FEATURES as u64, out_recv));
    // println!("Ref out:{:?}", ref_out.t());

    let executed = ctx
        .initialize(
            InitializationOptionsBuilder::default()
                .run_flavor_inference(true)
                .build()
                .unwrap(),
        )
        .unwrap()
        .run(RunOptions::default());
    println!("Took {:?} cycles", executed.elapsed_cycles());
}
