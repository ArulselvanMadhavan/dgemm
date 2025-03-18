use dam::{
    simulation::{InitializationOptionsBuilder, ProgramBuilder, RunOptions},
    utility_contexts::*,
};
use dgemm::gemm::Gemm;
use dgemm::producer::Producer;

#[test]
fn xpu_linear_test() {
    const IN_FEATURES: usize = 3;
    const OUT_FEATURES: usize = 3;
    const TILE_SIZE: usize = 4;
    const NUM_INPUTS: usize = 2;
    const W_SIZE: usize = OUT_FEATURES * IN_FEATURES;
    const X_SIZE: usize = NUM_INPUTS * TILE_SIZE * IN_FEATURES;
    const XPU_INITIATION_INTERVAL: u64 = 1;
    let mut ctx = ProgramBuilder::default();
    let (x_send, x_recv) = ctx.bounded::<f64>(TILE_SIZE * IN_FEATURES);
    let (out_send, out_recv) = ctx.bounded::<f64>(TILE_SIZE * OUT_FEATURES);
    let weight_mat = ndarray::Array::<f64, _>::linspace(0.0, W_SIZE as f64, W_SIZE);
    let weight_mat = weight_mat
        .to_shape((OUT_FEATURES, IN_FEATURES))
        .unwrap()
        .to_owned();
    let x_mat = ndarray::Array::<f64, _>::linspace(0.0, X_SIZE as f64, X_SIZE);
    let x_mat = x_mat
        .to_shape((NUM_INPUTS * TILE_SIZE, IN_FEATURES))
        .unwrap()
        .to_owned();
    let biases = ndarray::Array::<f64, _>::linspace(0.0, OUT_FEATURES as f64, OUT_FEATURES);
    let mut ref_out = weight_mat.dot(&x_mat.t());
    let bias_mat = biases
        .clone()
        .to_shape((OUT_FEATURES, 1))
        .unwrap()
        .to_owned();
    ref_out = ref_out + &bias_mat;
    ctx.add_child(Producer::new(
        || x_mat.into_iter(),
        IN_FEATURES as u64,
        x_send,
    ));
    ctx.add_child(Gemm::new(
        weight_mat,
        biases,
        x_recv,
        out_send,
        XPU_INITIATION_INTERVAL,
    ));
    ctx.add_child(PrinterContext::new(out_recv));
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
