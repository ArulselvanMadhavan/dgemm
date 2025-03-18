use dam::simulation::{InitializationOptionsBuilder, ProgramBuilder, RunOptions};
use dgemm::{consumer::Consumer, gemm::Gemm, producer::Producer};
use itertools::Itertools;
use ndarray::*;

#[test]
fn xpu_linear_test() {
    const IN_FEATURES: usize = 3;
    const OUT_FEATURES: usize = 3;
    const TILE_SIZE: usize = 4;
    const NUM_INPUTS: usize = 2;
    const W_SIZE: usize = OUT_FEATURES * IN_FEATURES;
    const X_SIZE: usize = NUM_INPUTS * TILE_SIZE * IN_FEATURES;
    const LINK_CAPACITY: usize = TILE_SIZE;
    const IB_SIZE: usize = TILE_SIZE * IN_FEATURES;
    const OB_SIZE: usize = TILE_SIZE * OUT_FEATURES;
    assert!(X_SIZE % LINK_CAPACITY == 0);
    assert!(IB_SIZE % LINK_CAPACITY == 0);
    assert!(OB_SIZE % LINK_CAPACITY == 0);
    const X_SEND_STEPS: usize = X_SIZE / LINK_CAPACITY;

    let mut ctx = ProgramBuilder::default();
    let x_i: Array1<f64> = array![0.0];
    let (x_send, x_recv) = ctx.bounded::<Array1<f64>>(IB_SIZE / LINK_CAPACITY);
    let (out_send, out_recv) = ctx.bounded::<Array1<f64>>(OB_SIZE / LINK_CAPACITY);
    let weight_mat = Array::<f64, _>::linspace(0.0, W_SIZE as f64, W_SIZE);
    let weight_mat = weight_mat
        .to_shape((OUT_FEATURES, IN_FEATURES))
        .unwrap()
        .to_owned();
    let x_mat = Array::range(0., X_SIZE as f64, 1.)
        .into_shape([X_SIZE / LINK_CAPACITY, LINK_CAPACITY])
        .unwrap();
    let biases = ndarray::Array::<f64, _>::linspace(0.0, OUT_FEATURES as f64, OUT_FEATURES);
    let mut ref_out = weight_mat.dot(&x_mat.t());
    let bias_mat = biases
        .clone()
        .to_shape((OUT_FEATURES, 1))
        .unwrap()
        .to_owned();
    ref_out = ref_out + &bias_mat;
    let mut x_mat_vec = Vec::with_capacity(X_SEND_STEPS);
    x_mat.map_axis(Axis(0), |x| x_mat_vec.push(x.to_owned()));
    ctx.add_child(Producer::new(|| x_mat_vec.into_iter(), x_send));

    // ctx.add_child(Gemm::new(
    //     weight_mat,
    //     biases,
    //     x_recv,
    //     out_send,
    //     XPU_INITIATION_INTERVAL,
    // ));
    ctx.add_child(Consumer::new(OUT_FEATURES as u64, out_recv));
    println!("Ref out:{:?}", ref_out.t());

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
