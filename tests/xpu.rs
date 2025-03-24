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
    Vec<Sender<Array1<T>>>,
    Vec<Receiver<Array1<T>>>,
    Vec<Option<Sender<Array1<T>>>>,
    Vec<Option<Receiver<Array1<T>>>>,
) {
    let mut sd_chan = Vec::<Sender<Array1<T>>>::with_capacity(n);
    let mut rx_chan = Vec::<Receiver<Array1<T>>>::with_capacity(n);
    let mut rx_cons = Vec::<Option<Receiver<Array1<T>>>>::with_capacity(n);
    let mut sd_prod = Vec::<Option<Sender<Array1<T>>>>::with_capacity(n);

    for s in 0..n {
        rx_cons.push(None);
        sd_prod.push(None);
        let mut count = 0;
        for r in 0..n {
            if conn.slice(s![s, r]).eq(&arr0(true)) {
                let (tx, rx) = ctx.bounded::<Array1<T>>(buffer_size);
                sd_chan.push(tx);
                rx_chan.push(rx);
                count += 1;
            }
        }
        if count == 0 {
            let (tx, rx) = ctx.bounded::<Array1<T>>(buffer_size);
            sd_chan.push(tx);
            rx_cons[s] = Some(rx); // consumer
        }
        count = 0;
        for r in 0..n {
            if conn.slice(s![r, s]).eq(&arr0(true)) {
                count += 1;
            }
        }
        if count == 0 {
            let (tx, rx) = ctx.bounded::<Array1<T>>(buffer_size);
            sd_prod[s] = Some(tx); // producer
            rx_chan.push(rx);
        }
    }
    assert!(sd_chan.len() == n && rx_chan.len() == n);
    (sd_chan, rx_chan, sd_prod, rx_cons)
}

fn mesh_conn<'a, T: Clone + 'a>(
    dims: [usize; 2],
    buffer_size: usize,
    ctx: &mut ProgramBuilder<'a>,
) -> (
    Vec<[Receiver<Array1<T>>; 2]>,
    Vec<[Sender<Array1<T>>; 2]>,
    Vec<[Option<Sender<Array1<T>>>; 2]>,
    Vec<[Option<Receiver<Array1<T>>>; 2]>,
) {
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
    let (mut rchan, mut lchan, mut rprod, mut lcon) =
        assign_chan::<T>(n, buffer_size, send_right, ctx);
    let (mut dchan, mut uchan, mut dprod, mut ucon) =
        assign_chan::<T>(n, buffer_size, send_down, ctx);
    let mut out_chans = Vec::with_capacity(n);
    let mut in_chans = Vec::with_capacity(n);
    let mut in_prods = Vec::with_capacity(n);
    let mut out_cons = Vec::with_capacity(n);
    dbg!(
        "Node to Node Links:",
        dims[0] * (dims[1] - 1) + dims[1] * (dims[0] - 1)
    );
    dbg!("Prod|Cons links", dims[0] * 2 + dims[1] * 2);
    dbg!(rprod.len(), lcon.len(), dprod.len(), ucon.len());
    (0..n).for_each(|_| {
        out_chans.push([rchan.remove(0), dchan.remove(0)]);
        in_chans.push([lchan.remove(0), uchan.remove(0)]);
        in_prods.push([rprod.remove(0), dprod.remove(0)]);
        out_cons.push([lcon.remove(0), ucon.remove(0)]);
    });

    assert!(
        rchan.len() == 0
            && uchan.len() == 0
            && dchan.len() == 0
            && lchan.len() == 0
            && rprod.len() == 0
            && lcon.len() == 0
            && dprod.len() == 0
            && ucon.len() == 0
    );
    (in_chans, out_chans, in_prods, out_cons)
}

#[test]
fn xpu_linear_test() {
    const LINK_CAPACITY: usize = 4;
    const IN_FEATURES: usize = 4;
    const OUT_FEATURES: usize = 4;
    assert!(LINK_CAPACITY % IN_FEATURES == 0);
    assert!(LINK_CAPACITY % OUT_FEATURES == 0);
    const BUFFER_CAPACITY: usize = 2;
    const NUM_MATMULS: usize = 3;
    const NUM_INPUTS: usize = (LINK_CAPACITY / IN_FEATURES) * BUFFER_CAPACITY * NUM_MATMULS;
    const W_SIZE: usize = IN_FEATURES * OUT_FEATURES;
    const X_SIZE: usize = NUM_INPUTS * IN_FEATURES;
    const X_SEND_STEPS: usize = X_SIZE / LINK_CAPACITY;
    const TRACKS_PER_THREAD: usize = 3;
    const DIMS: [usize; 2] = [1, 1];

    let num_nodes: usize = DIMS.iter().fold(1, |prod, x| prod * x);
    // Trace descriptors
    let thread_names = (0..num_nodes).map(|n| format!("xpu{n}", n = n));
    let thread_names = Vec::from_iter(thread_names);
    let thread_count = thread_names.len();
    let processes = vec![("xpu".to_string(), thread_names)];
    let tuuids = dgemm::trace::get_trace_descriptors::<TRACKS_PER_THREAD>(
        processes,
        thread_count + 1,
        thread_count,
    );
    // Build Mesh
    let mut ctx = ProgramBuilder::default();
    let (mut in_conns, mut out_conns, mut in_prods, mut out_cons) =
        mesh_conn::<f64>(DIMS, BUFFER_CAPACITY, &mut ctx);
    // Inputs
    let weight_mat = Array::range(0., (num_nodes * W_SIZE) as f64, 1.);
    let weight_mat = weight_mat
        .to_shape((DIMS[0], IN_FEATURES, DIMS[1], OUT_FEATURES))
        .unwrap()
        .to_owned();
    let x_mat = Array::range(0., (DIMS[0] * X_SIZE) as f64, 1.)
        .into_shape([NUM_INPUTS, DIMS[0], IN_FEATURES])
        .unwrap();
    let biases = ndarray::Array::<f64, _>::linspace(0.0, OUT_FEATURES as f64, OUT_FEATURES);

    // Build contexts
    (0..num_nodes).for_each(|node_id| {
        let row_id = node_id / DIMS[1];
        let col_id = node_id - (row_id * DIMS[1]);
        let wmat = weight_mat.select(Axis(2), &[col_id]).remove_axis(Axis(2));
        let wmat = wmat.select(Axis(0), &[row_id]).remove_axis(Axis(0));
        ctx.add_child(Gemm::new(
            wmat,
            biases.clone(),
            GemmConstants::new(
                LINK_CAPACITY,
                BUFFER_CAPACITY,
                0,
                *tuuids.get(0).unwrap(),
                NUM_MATMULS,
            ),
            in_conns.remove(0),
            out_conns.remove(0),
            1,
        ));
        let pdelay = 5;
        let build_input = |node_id: usize| {
            let x_dim_id = node_id / DIMS[1];
            let xmat = x_mat.select(Axis(1), &[x_dim_id]).remove_axis(Axis(1));
            let xmat = xmat.to_shape((X_SEND_STEPS, LINK_CAPACITY)).unwrap();
            let mut x_mat_vec = Vec::with_capacity(X_SEND_STEPS);
            xmat.map_axis(Axis(1), |x| x_mat_vec.push(x.to_owned()));
            x_mat_vec
        };
        match in_prods.remove(0) {
            [Some(x_send), None] => {
                let x_mat_vec = build_input(node_id);
                ctx.add_child(Producer::new(
                    || x_mat_vec.into_iter(),
                    x_send,
                    node_id,
                    pdelay,
                ));
            }
            [None, Some(x_send)] => ctx.add_child(Producer::new(
                || (0..X_SEND_STEPS).map(|_x| Array1::zeros(LINK_CAPACITY)),
                x_send,
                node_id,
                pdelay,
            )),
            [Some(r_send), Some(d_send)] => {
                let x_mat_vec = build_input(node_id);
                ctx.add_child(Producer::new(
                    || x_mat_vec.into_iter(),
                    r_send,
                    node_id,
                    pdelay,
                ));
                ctx.add_child(Producer::new(
                    || (0..X_SEND_STEPS).map(|_x| Array1::zeros(LINK_CAPACITY)),
                    d_send,
                    node_id,
                    pdelay,
                ));
            }
            [None, None] => (),
        }
        match out_cons.remove(0) {
            [Some(out_recv), None] => {
                // Skip Consumer
                ctx.add_child(Consumer::new(OUT_FEATURES as u64, out_recv, node_id))
            }
            [None, Some(out_recv)] => {
                // Verify Consumer
                ctx.add_child(Consumer::new(OUT_FEATURES as u64, out_recv, node_id))
            }
            [Some(l_recv), Some(u_recv)] => {
                ctx.add_child(Consumer::new(OUT_FEATURES as u64, l_recv, node_id));
                ctx.add_child(Consumer::new(OUT_FEATURES as u64, u_recv, node_id));
            }
            _ => (),
        }
    });
    let w_ref = weight_mat
        .to_shape([DIMS[0] * IN_FEATURES, DIMS[1] * OUT_FEATURES])
        .unwrap();
    let ref_out = x_mat
        .to_shape((NUM_INPUTS, DIMS[0] * IN_FEATURES))
        .unwrap()
        .dot(&w_ref);
    println!("Ref out:{:?}@{:?}={:?}", x_mat, w_ref, ref_out);
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
