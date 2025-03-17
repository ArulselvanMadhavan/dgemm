use dam::{
    simulation::{InitializationOptionsBuilder, ProgramBuilder, RunOptions},
    utility_contexts::{CheckerContext, GeneratorContext, PrinterContext},
};
use dgemm::{actfn::Actfn, gemv::GEMV};

fn relu(input: f64) -> f64 {
    input.max(0.0)
}

#[test]
fn matmul_relu_test() {
    const NUM_INPUTS: usize = 8;
    const NUM_FEATURES: usize = 128;
    const NUM_OUTPUTS: usize = 8;
    const IS_PRINT: bool = false;
    let mut ctx = ProgramBuilder::default();
    let (x_send, x_recv) = ctx.bounded::<f64>(NUM_FEATURES);
    let mut input_vec = Vec::with_capacity(NUM_INPUTS * NUM_FEATURES);
    for i in 0..NUM_INPUTS * NUM_FEATURES {
        input_vec.push(i as f64);
    }
    let input_mat =
        ndarray::Array::from_shape_vec((NUM_INPUTS, NUM_FEATURES), input_vec.clone()).unwrap();
    // Context1: Input
    ctx.add_child(GeneratorContext::new(|| input_vec.into_iter(), x_send));
    let (mm_send, mm_recv) = ctx.bounded::<f64>(NUM_FEATURES);
    let weights = ndarray::Array2::from_elem((NUM_OUTPUTS, NUM_FEATURES), 0.5);
    let mut bias_vec = Vec::with_capacity(NUM_OUTPUTS);
    for i in 0..NUM_OUTPUTS {
        bias_vec.push((i as f64) - 2.0);
    }
    let biases = ndarray::Array::from_vec(bias_vec);
    let bias_mat = biases.clone();
    let bias_mat = bias_mat.to_shape((NUM_OUTPUTS, 1)).unwrap();
    // Context2: GEMV
    ctx.add_child(GEMV::new(x_recv, mm_send, weights.clone(), biases, 1));
    let (act_send, act_recv) = ctx.bounded::<f64>(NUM_FEATURES);
    // Context3: Act fn
    ctx.add_child(Actfn::new(mm_recv, act_send, 1, relu));
    // Build Reference output
    let mut ref_out = weights.dot(&input_mat.t());
    ref_out = ref_out + &bias_mat;
    ref_out.mapv_inplace(relu);
    // TraceContext::new()
    if IS_PRINT {
        ctx.add_child(PrinterContext::new(act_recv));
        println!("Ref:{:?}", ref_out);
    } else {
        ctx.add_child(CheckerContext::new(
            move || ref_out.t().to_owned().into_iter(),
            act_recv,
        ));
    }

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
    // assert!(1 == 2);
}
