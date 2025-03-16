use dam::{simulation::ProgramBuilder, utility_contexts::GeneratorContext};
use itertools::enumerate;

fn relu(input: f64) -> f64 {
    input.max(0.0)
}

#[test]
fn matmul_relu_test() {
    const NUM_INPUTS: usize = 32;
    const NUM_FEATURES: usize = 1024;
    const NUM_OUTPUTS: usize = 24;
    let mut ctx = ProgramBuilder::default();
    let (x_send, _x_recv) = ctx.bounded::<f64>(1024);
    let mut input_vec = Vec::with_capacity(NUM_INPUTS * NUM_FEATURES);
    for i in 0..NUM_INPUTS * NUM_FEATURES {
        input_vec.push(i as f64);
    }
    ctx.add_child(GeneratorContext::new(|| input_vec.into_iter(), x_send));
    let (mm_send, mm_recv) = ctx.bounded::<f64>(1024);
    let weights = ndarray::Array2::from_elem((NUM_OUTPUTS, NUM_FEATURES), 0.5);
    let mut bias_vec = Vec::with_capacity(NUM_OUTPUTS);
    for i in 0..NUM_OUTPUTS {
        bias_vec.push((i as f64) - 2.0);
    }
    let biases = ndarray::Array::from_vec(bias_vec);
}
